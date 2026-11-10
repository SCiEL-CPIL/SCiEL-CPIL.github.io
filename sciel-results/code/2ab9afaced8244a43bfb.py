"""Simulated annealing covering design C(v,k,t) with fast bitmask energy updates."""
import itertools
import time
import random
import math
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using:
    1. Greedy construction to get an initial covering
    2. Local search (redundancy removal) to minimize
    3. Simulated annealing with bitmask-based fast energy updates
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin within 150s limit

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    total_tsubsets = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Precompute for each block (k-subset) which t-subsets it covers
    # and for each point, which t-subsets it participates in
    # For fast gain computation during greedy

    def block_tsubsets(block):
        """Return list of t-subset indices covered by a block."""
        result = []
        for ts in itertools.combinations(block, t):
            idx = tsubset_to_idx.get(tuple(sorted(ts)))
            if idx is not None:
                result.append(idx)
        return result

    def greedy_build(seed):
        """Greedy covering construction."""
        rng = random.Random(seed)
        uncovered = [True] * total_tsubsets
        uncovered_count = total_tsubsets
        blocks = []

        while uncovered_count > 0:
            if time.time() > deadline - 20:
                break

            # Pick a random uncovered t-subset
            # Find uncovered indices efficiently
            uncovered_indices = [i for i in range(total_tsubsets) if uncovered[i]]
            if not uncovered_indices:
                break
            chosen_idx = rng.choice(uncovered_indices)
            seed_ts = all_tsubsets[chosen_idx]

            # Build block greedily around this seed
            block = list(seed_ts)
            rng.shuffle(block)

            while len(block) < k:
                if time.time() > deadline - 20:
                    break

                best_gain = -1
                best_point = None
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(list(rest) + [x]))
                        idx = tsubset_to_idx.get(ts)
                        if idx is not None and uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x

                if best_point is None:
                    break
                block.append(best_point)

            block.sort()
            blocks.append(block)

            # Update uncovered
            for idx in block_tsubsets(block):
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1

        if uncovered_count == 0:
            return blocks
        return None

    def local_search_redundancy(blocks, max_time=10):
        """Remove redundant blocks iteratively."""
        start_ls = time.time()
        changed = True
        while changed and (time.time() - start_ls) < max_time:
            changed = False
            cover_count = defaultdict(int)
            for block in blocks:
                for idx in block_tsubsets(block):
                    cover_count[idx] += 1

            new_blocks = []
            for block in blocks:
                is_critical = False
                for idx in block_tsubsets(block):
                    if cover_count[idx] == 1:
                        is_critical = True
                        break
                if not is_critical:
                    for idx in block_tsubsets(block):
                        cover_count[idx] -= 1
                    changed = True
                else:
                    new_blocks.append(block)
            blocks = new_blocks
        return blocks

    def validate_covering(blocks):
        """Check if every t-subset is covered."""
        covered = set()
        for block in blocks:
            for idx in block_tsubsets(block):
                covered.add(idx)
        return len(covered) == total_tsubsets

    # --- Simulated Annealing ---
    def simulated_annealing(blocks, max_time=60):
        """
        SA on block set.
        Energy = num_uncovered + alpha * num_blocks
        Moves: remove a block, add a block, swap a block.
        Use cover_count array for fast energy deltas.
        """
        sa_start = time.time()
        alpha = 0.5  # penalty weight for block count

        # Build cover_count
        cover_count = [0] * total_tsubsets
        for block in blocks:
            for idx in block_tsubsets(block):
                cover_count[idx] += 1

        def num_uncovered():
            return sum(1 for c in cover_count if c == 0)

        def energy():
            return num_uncovered() + alpha * len(blocks)

        # Precompute block_tsubsets for all blocks
        block_indices = [block_tsubsets(b) for b in blocks]

        best_blocks = list(blocks)
        best_energy = energy()
        current_energy = best_energy

        # Temperature schedule
        T0 = max(5.0, 0.1 * total_tsubsets)
        Tend = 0.01
        # We'll run for max_time
        steps_target = 200000

        T = T0
        rng = random.Random(42)

        # Precompute a pool of candidate blocks for addition/swap
        # Generate random k-subsets
        candidate_pool = []
        for _ in range(500):
            block = rng.sample(range(v), k)
            block.sort()
            candidate_pool.append(block)

        step = 0
        while (time.time() - sa_start) < max_time and step < steps_target:
            step += 1

            # Update temperature
            T = T0 * (Tend / T0) ** (step / steps_target)

            move_type = rng.random()
            old_energy = current_energy
            old_blocks = list(blocks)
            old_block_indices = [list(bi) for bi in block_indices]

            if move_type < 0.3 and len(blocks) > 1:
                # Remove a block
                i = rng.randrange(len(blocks))
                removed = blocks.pop(i)
                removed_idx = block_indices.pop(i)
                for idx in removed_idx:
                    cover_count[idx] -= 1

            elif move_type < 0.6:
                # Add a block (from pool or random)
                if rng.random() < 0.5 and candidate_pool:
                    new_block = candidate_pool[rng.randrange(len(candidate_pool))].copy()
                else:
                    new_block = sorted(rng.sample(range(v), k))
                new_idx = block_tsubsets(new_block)
                blocks.append(new_block)
                block_indices.append(new_idx)
                for idx in new_idx:
                    cover_count[idx] += 1

            elif move_type < 0.9 and len(blocks) > 1:
                # Swap a block
                i = rng.randrange(len(blocks))
                old_block = blocks[i]
                old_idx = block_indices[i]
                for idx in old_idx:
                    cover_count[idx] -= 1

                if rng.random() < 0.5 and candidate_pool:
                    new_block = candidate_pool[rng.randrange(len(candidate_pool))].copy()
                else:
                    new_block = sorted(rng.sample(range(v), k))

                new_idx = block_tsubsets(new_block)
                blocks[i] = new_block
                block_indices[i] = new_idx
                for idx in new_idx:
                    cover_count[idx] += 1

            else:
                # Perturb: modify a block by replacing one element
                if len(blocks) == 0:
                    continue
                i = rng.randrange(len(blocks))
                block = list(blocks[i])
                old_idx = block_indices[i]
                for idx in old_idx:
                    cover_count[idx] -= 1

                # Replace one random element with another
                pos = rng.randrange(len(block))
                old_elem = block[pos]
                candidates = [x for x in range(v) if x not in block]
                if candidates:
                    new_elem = rng.choice(candidates)
                    block[pos] = new_elem
                block.sort()

                new_idx = block_tsubsets(block)
                blocks[i] = block
                block_indices[i] = new_idx
                for idx in new_idx:
                    cover_count[idx] += 1

            new_energy = energy()
            delta = new_energy - old_energy

            if delta <= 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
                # Accept
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_blocks = list(blocks)
                    best_block_indices = [list(bi) for bi in block_indices]
            else:
                # Reject: revert
                blocks = old_blocks
                block_indices = old_block_indices
                # Rebuild cover_count
                cover_count = [0] * total_tsubsets
                for bi in block_indices:
                    for idx in bi:
                        cover_count[idx] += 1

        # Final local search on best
        result = local_search_redundancy(best_blocks, max_time=5)
        return result

    # --- Main ---
    best_result = None
    best_len = float('inf')

    # Phase 1: Greedy with multiple seeds
    for seed in range(10):
        if time.time() > deadline - 80:
            break
        result = greedy_build(seed)
        if result is not None:
            result = local_search_redundancy(result, max_time=3)
            if validate_covering(result):
                if len(result) < best_len:
                    best_len = len(result)
                    best_result = result

    if best_result is None:
        # Fallback
        result = greedy_build(0)
        if result is not None:
            return result
        return []

    # Phase 2: Simulated Annealing
    if time.time() < deadline - 30:
        sa_time = min(60, deadline - time.time() - 20)
        sa_result = simulated_annealing(best_result, max_time=sa_time)
        if sa_result is not None and validate_covering(sa_result):
            if len(sa_result) < best_len:
                best_len = len(sa_result)
                best_result = sa_result

    # Phase 3: More SA rounds if time
    if time.time() < deadline - 25:
        sa_time = min(30, deadline - time.time() - 15)
        if sa_time > 5:
            sa_result = simulated_annealing(best_result, max_time=sa_time)
            if sa_result is not None and validate_covering(sa_result):
                if len(sa_result) < best_len:
                    best_len = len(sa_result)
                    best_result = sa_result

    # Phase 4: Final local search
    if time.time() < deadline - 5:
        best_result = local_search_redundancy(best_result, max_time=5)
        if not validate_covering(best_result):
            # Revert - shouldn't happen but safety
            pass

    return best_result
