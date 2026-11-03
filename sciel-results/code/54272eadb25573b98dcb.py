"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def _precompute(v, k, t):
    """Precompute all t-subsets, their indices, and per-point membership."""
    all_subsets = [tuple(c) for c in itertools.combinations(range(v), t)]
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    point_to_subsets = [[] for _ in range(v)]
    for idx, s in enumerate(all_subsets):
        for p in s:
            point_to_subsets[p].append(idx)
    return all_subsets, subset_to_idx, point_to_subsets


def _block_covered_indices(block, subset_to_idx):
    """Return frozenset of t-subset indices covered by this block."""
    covered = set()
    for c in itertools.combinations(block, t):
        covered.add(subset_to_idx[c])
    return covered


def _greedy_cover(v, k, t, rng, all_subsets, subset_to_idx):
    """Fast greedy construction to get a valid starting covering."""
    uncovered = set(range(len(all_subsets)))
    blocks = []
    block_set = set()

    while uncovered:
        seed_idx = rng.choice(list(uncovered))
        seed = all_subsets[seed_idx]
        block = list(seed)
        rng.shuffle(block)
        block_set_local = set(block)

        while len(block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block_set_local]
            rng.shuffle(cands)

            for x in cands:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = tuple(sorted(rest + (x,)))
                    idx = subset_to_idx.get(fs)
                    if idx is not None and idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            block_set_local.add(best_x)

        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in block_set:
            block_set.add(b_tuple)
            blocks.append(b_tuple)
            for c in itertools.combinations(block, t):
                uncovered.discard(subset_to_idx[c])

    return blocks


def _try_remove_redundant(blocks, block_covered, counts, rng):
    """Try to remove a redundant block. Returns True if removed."""
    indices = list(range(len(blocks)))
    rng.shuffle(indices)
    for i in indices:
        redundant = True
        for idx in block_covered[i]:
            if counts[idx] <= 1:
                redundant = False
                break
        if redundant:
            for idx in block_covered[i]:
                counts[idx] -= 1
            blocks.pop(i)
            block_covered.pop(i)
            return True
    return False


def _polish(blocks, block_covered, counts, v, k, t, all_subsets, subset_to_idx, rng, deadline):
    """Final exact polish: exhaustive single-block swap search at temp 0."""
    n_blocks = len(blocks)
    all_blocks_pool = list(itertools.combinations(range(v), k))
    pool_set = set(all_blocks_pool)
    # Precompute covered indices for all possible blocks (24C7 = 346104)
    # That's too many to precompute all. Instead, compute on demand.
    # But we can precompute for a smaller candidate pool.
    
    # Actually, let's do a smarter approach: for each block position,
    # find all blocks that are valid replacements (don't break coverage).
    # A block B is a valid replacement for position i if:
    #   for every subset s in old_block_covered[i] with count[s] == 1,
    #   s must also be in B.
    
    # This is expensive to check for all 346104 blocks per position.
    # Instead, do multiple passes of random swaps + redundancy removal.
    
    improved = True
    passes = 0
    while improved and time.time() < deadline:
        improved = False
        passes += 1
        
        # Pass 1: remove all redundant blocks
        while _try_remove_redundant(blocks, block_covered, counts, rng):
            improved = True
        
        # Pass 2: try random swaps
        for _ in range(min(500, max(50, n_blocks * 5))):
            if time.time() > deadline:
                break
            i = rng.randrange(len(blocks))
            old_block = blocks[i]
            old_covered = block_covered[i]
            
            # Find critical subsets
            critical = [idx for idx in old_covered if counts[idx] == 1]
            
            # Try random replacement blocks
            for attempt in range(20):
                new_block = tuple(rng.sample(range(v), k))
                if new_block == old_block:
                    continue
                new_block_sorted = tuple(sorted(new_block))
                new_covered = _block_covered_indices(new_block_sorted, subset_to_idx)
                
                # Check validity: no critical subset loses coverage
                valid = True
                for idx in critical:
                    if idx not in new_covered:
                        valid = False
                        break
                
                if valid:
                    # Apply swap
                    for idx in old_covered:
                        counts[idx] -= 1
                    for idx in new_covered:
                        counts[idx] += 1
                    blocks[i] = new_block_sorted
                    block_covered[i] = new_covered
                    improved = True
                    break
    
    return blocks, block_covered, counts


def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115.0  # leave 5s buffer

    # Precompute
    all_subsets, subset_to_idx, point_to_subsets = _precompute(v, k, t)
    n_subsets = len(all_subsets)

    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng, all_subsets, subset_to_idx)

    # Precompute covered subsets for each block
    block_covered = [_block_covered_indices(b, subset_to_idx) for b in blocks]

    # counts[i] = number of blocks covering subset i
    counts = [0] * n_subsets
    for covered in block_covered:
        for idx in covered:
            counts[idx] += 1

    best_blocks = [list(b) for b in blocks]
    best_count = len(blocks)

    # Simulated Annealing parameters
    T = 2.0
    T_min = 0.01
    cooling = 0.999

    no_improve = 0
    max_no_improve = 200

    deadline = start_time + time_limit
    polish_deadline = start_time + time_limit - 10.0  # leave 10s for polish

    while time.time() < polish_deadline:
        # 1. Try to remove redundant blocks
        if _try_remove_redundant(blocks, block_covered, counts, rng):
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = [list(b) for b in blocks]
            no_improve = 0
            T = max(T * cooling, T_min)
            continue

        # 2. Try block replacement with simulated annealing
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            old_block = blocks[i]
            old_covered = block_covered[i]

            # Identify critical subsets (count == 1) in old_block
            critical = [idx for idx in old_covered if counts[idx] == 1]

            # Collect points needed for critical subsets
            points_needed = set()
            for idx in critical:
                points_needed.update(all_subsets[idx])

            if len(points_needed) > k:
                no_improve += 1
                T = max(T * cooling, T_min)
                continue

            # Build new block containing points_needed
            new_block_set = set(points_needed)
            cands = [x for x in range(v) if x not in new_block_set]
            rng.shuffle(cands)

            # Greedy fill to maximize coverage of low-count subsets
            for x in cands:
                if len(new_block_set) >= k:
                    break
                gain = 0
                for rest in itertools.combinations(new_block_set, t - 1):
                    fs = tuple(sorted(rest + (x,)))
                    idx = subset_to_idx.get(fs)
                    if idx is not None:
                        c = counts[idx]
                        if c == 0:
                            gain += 10
                        elif c == 1:
                            gain += 1
                if gain > 0:
                    new_block_set.add(x)
                else:
                    new_block_set.add(x)

            if len(new_block_set) == k:
                new_block = tuple(sorted(new_block_set))

                if new_block != old_block:
                    new_covered = _block_covered_indices(new_block, subset_to_idx)

                    # Check if swap is valid: no subset loses its last cover
                    valid = True
                    for idx in old_covered:
                        if idx not in new_covered and counts[idx] <= 1:
                            valid = False
                            break

                    if valid:
                        # Apply swap
                        for idx in old_covered:
                            counts[idx] -= 1
                        for idx in new_covered:
                            counts[idx] += 1

                        blocks[i] = new_block
                        block_covered[i] = new_covered
                        no_improve = 0
                    else:
                        no_improve += 1
        else:
            no_improve += 1

        # 3. Perturbation if stuck
        if no_improve > max_no_improve:
            if len(blocks) > 1:
                old_blocks = list(blocks)
                old_block_covered = [set(c) for c in block_covered]
                old_counts = list(counts)

                # Remove 2-3 random blocks and try to recover
                num_remove = min(rng.randint(2, 3), len(blocks) - 1)
                remove_indices = rng.sample(range(len(blocks)), num_remove)
                remove_indices.sort(reverse=True)

                for idx in remove_indices:
                    for s_idx in block_covered[idx]:
                        counts[s_idx] -= 1
                    blocks.pop(idx)
                    block_covered.pop(idx)

                # Try to add blocks to cover uncovered sets
                attempts = 0
                while attempts < 50:
                    attempts += 1
                    # Find an uncovered subset
                    uncovered_list = [i for i in range(n_subsets) if counts[i] == 0]
                    if not uncovered_list:
                        break
                    seed_idx = rng.choice(uncovered_list)
                    seed = all_subsets[seed_idx]
                    new_b = list(seed)
                    rng.shuffle(new_b)

                    while len(new_b) < k:
                        best_x = None
                        best_gain = -1
                        cands = [x for x in range(v) if x not in new_b]
                        rng.shuffle(cands)
                        for x in cands:
                            gain = 0
                            for rest in itertools.combinations(new_b, t - 1):
                                fs = tuple(sorted(rest + (x,)))
                                idx = subset_to_idx.get(fs)
                                if idx is not None and counts[idx] == 0:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                        if best_x is None:
                            break
                        new_b.append(best_x)

                    new_b.sort()
                    new_b_tuple = tuple(new_b)

                    if new_b_tuple not in blocks:
                        blocks.append(new_b_tuple)
                        new_covered = _block_covered_indices(new_b_tuple, subset_to_idx)
                        block_covered.append(new_covered)
                        for idx in new_covered:
                            counts[idx] += 1

                # Check if we recovered a valid covering
                valid = all(counts[i] > 0 for i in range(n_subsets))
                if valid:
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = [list(b) for b in blocks]
                    no_improve = 0
                    T = max(T * 1.5, 1.0)
                else:
                    blocks = old_blocks
                    block_covered = old_block_covered
                    counts = old_counts
                    no_improve += 1
                    T = max(T * 0.8, T_min)
            else:
                no_improve += 1
                T = max(T * 0.9, T_min)
        else:
            no_improve += 1

        # Cool down
        T = max(T * cooling, T_min)

    # Final polish pass
    blocks, block_covered, counts = _polish(
        blocks, block_covered, counts, v, k, t,
        all_subsets, subset_to_idx, rng, deadline
    )

    if len(blocks) < best_count:
        best_count = len(blocks)
        best_blocks = [list(b) for b in blocks]

    return best_blocks
# EVOLVE-BLOCK-END
