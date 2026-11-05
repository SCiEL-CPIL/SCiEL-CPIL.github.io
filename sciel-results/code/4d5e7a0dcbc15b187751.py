"""Simulated-annealing covering construction for C(v,k,t)."""
import time
import random
import itertools
import math
import sys

# Use a fixed seed for reproducibility
_SEED = 42


def _all_t_subsets(v, t):
    """Generate all t-subsets as frozensets."""
    return [frozenset(c) for c in itertools.combinations(range(v), t)]


def _block_t_subsets(block, t):
    """All t-subsets contained in a block."""
    return set(itertools.combinations(block, t))


def _greedy_covering(v, k, t, rng, deadline, weight_mode='uniform'):
    """Greedy covering with optional weight-based point selection."""
    all_points = list(range(v))
    # Track uncovered t-subsets
    uncovered = set(itertools.combinations(range(v), t))
    if not uncovered:
        return []

    blocks = []
    start = time.time()

    while uncovered:
        if time.time() > deadline:
            break
        # Pick the first uncovered t-subset (deterministic choice for speed)
        seed_tuple = min(uncovered)
        block = list(seed_tuple)

        while len(block) < k:
            best_gain = -1
            best_point = None
            for x in all_points:
                if x in block:
                    continue
                # Count how many uncovered t-subsets this point would cover
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    if tuple(sorted(rest + (x,))) in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point is None:
                # Pick any unused point
                for x in all_points:
                    if x not in block:
                        best_point = x
                        break
            block.append(best_point)

        block.sort()
        blocks.append(block)
        # Remove covered t-subsets
        for ts in itertools.combinations(block, t):
            uncovered.discard(ts)

    return blocks


def _count_covered(blocks, v, t):
    """Count how many t-subsets are covered by the given blocks."""
    covered = set()
    for block in blocks:
        for ts in itertools.combinations(block, t):
            covered.add(ts)
    return len(covered)


def _local_search(blocks, v, k, t, rng, deadline, max_iterations=200):
    """Local search: try to replace a block with a better one."""
    all_t_subs = list(itertools.combinations(range(v), t))
    total_t = len(all_t_subs)

    # Build coverage map: for each t-subset, how many blocks cover it
    # For efficiency, track which t-subsets are uncovered
    uncovered = set(all_t_subs)
    for block in blocks:
        for ts in itertools.combinations(block, t):
            uncovered.discard(ts)

    if not uncovered:
        return blocks

    best_blocks = [list(b) for b in blocks]
    best_uncovered_count = len(uncovered)

    start = time.time()
    iterations = 0

    while time.time() < deadline and iterations < max_iterations:
        iterations += 1

        # Pick a random block to try replacing
        idx = rng.randint(0, len(best_blocks) - 1)
        old_block = best_blocks[idx]

        # Find t-subsets that are ONLY covered by this block
        # Actually, let's try a different approach: 
        # Remove this block, see what becomes uncovered, then try to find a better block

        # Count how many t-subsets are exclusively covered by this block
        # For simplicity, let's just try to find a block that covers more uncovered t-subsets

        # Generate candidate block: start from a random uncovered t-subset
        if uncovered:
            seed_ts = list(uncovered)[rng.randint(0, len(uncovered) - 1)]
            candidate = list(seed_ts)
            
            # Grow the candidate block greedily
            while len(candidate) < k:
                best_gain = -1
                best_point = None
                for x in range(v):
                    if x in candidate:
                        continue
                    gain = 0
                    for rest in itertools.combinations(candidate, t - 1):
                        if tuple(sorted(rest + (x,))) in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is None:
                    for x in range(v):
                        if x not in candidate:
                            best_point = x
                            break
                if best_point is None:
                    break
                candidate.append(best_point)
            
            candidate.sort()

            # Compare: does replacing old_block with candidate reduce uncovered count?
            # Simulate the replacement
            new_uncovered = set(uncovered)
            # Add back t-subsets from old_block that might become uncovered
            for ts in itertools.combinations(old_block, t):
                new_uncovered.add(ts)
            # Remove t-subsets from candidate
            for ts in itertools.combinations(candidate, t):
                new_uncovered.discard(ts)

            if len(new_uncovered) < best_uncovered_count:
                best_blocks[idx] = candidate
                uncovered = new_uncovered
                best_uncovered_count = len(uncovered)
                if not uncovered:
                    break

    return best_blocks


def _simulated_annealing(blocks, v, k, t, rng, deadline):
    """Simulated annealing to improve the covering."""
    all_t_subs = list(itertools.combinations(range(v), t))
    total_t = len(all_t_subs)

    # Initial energy: number of uncovered t-subsets
    def energy(blk):
        covered = set()
        for b in blk:
            for ts in itertools.combinations(b, t):
                covered.add(ts)
        return total_t - len(covered)

    current_blocks = [list(b) for b in blocks]
    current_energy = energy(current_blocks)
    best_blocks = [list(b) for b in current_blocks]
    best_energy = current_energy

    start_temp = 10.0
    end_temp = 0.01
    start = time.time()
    total_time = deadline - start
    if total_time <= 0:
        return current_blocks

    # Cooling schedule: exponential
    n_steps = 5000  # We'll adapt based on time
    step_time = total_time / n_steps

    for step in range(n_steps):
        if time.time() > deadline:
            break

        temperature = start_temp * (end_temp / start_temp) ** (step / n_steps)

        # Generate a move: replace one block with a new random-ish block
        idx = rng.randint(0, len(current_blocks) - 1)
        old_block = current_blocks[idx]

        # Create candidate by perturbing the old block
        candidate = list(old_block)
        
        # Randomly swap out 1-3 points
        n_swaps = rng.randint(1, 3)
        for _ in range(n_swaps):
            if len(candidate) <= 1:
                break
            pos = rng.randint(0, len(candidate) - 1)
            old_point = candidate[pos]
            # Find a point not in candidate
            available = [x for x in range(v) if x not in candidate]
            if not available:
                continue
            new_point = rng.choice(available)
            candidate[pos] = new_point
        
        candidate.sort()

        # Avoid duplicate blocks
        if any(set(c) == set(b) for b in current_blocks if b != candidate):
            continue

        # Compute new energy
        new_blocks = [list(b) for b in current_blocks]
        new_blocks[idx] = candidate
        new_energy = energy(new_blocks)

        delta_e = new_energy - current_energy

        # Accept or reject
        if delta_e <= 0 or rng.random() < math.exp(-delta_e / max(temperature, 1e-10)):
            current_blocks = new_blocks
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_blocks = [list(b) for b in current_blocks]
                if best_energy == 0:
                    break

    return best_blocks


def construct(v, k, t):
    """Construct a covering design C(v,k,t)."""
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    rng = random.Random(_SEED)

    # Phase 1: Greedy initialization
    blocks = _greedy_covering(v, k, t, rng, deadline - 100.0)

    if not blocks:
        return []

    # Phase 2: Local search
    local_deadline = deadline - 40.0
    if time.time() < local_deadline:
        blocks = _local_search(blocks, v, k, t, rng, local_deadline)

    # Phase 3: Simulated annealing
    sa_deadline = deadline - 5.0
    if time.time() < sa_deadline:
        blocks = _simulated_annealing(blocks, v, k, t, rng, sa_deadline)

    # Final check: ensure all t-subsets are covered
    # If not, add more blocks greedily
    uncovered = set(itertools.combinations(range(v), t))
    for block in blocks:
        for ts in itertools.combinations(block, t):
            uncovered.discard(ts)

    if uncovered and time.time() < deadline - 1.0:
        # Add greedy blocks to cover remaining
        while uncovered and time.time() < deadline - 0.5:
            seed_ts = min(uncovered)
            block = list(seed_ts)
            while len(block) < k:
                best_gain = -1
                best_point = None
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        if tuple(sorted(rest + (x,))) in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is None:
                    for x in range(v):
                        if x not in block:
                            best_point = x
                            break
                if best_point is None:
                    break
                block.append(best_point)
            block.sort()
            # Avoid duplicates
            if any(set(block) == set(b) for b in blocks):
                continue
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                uncovered.discard(ts)

    return [list(b) for b in blocks]
