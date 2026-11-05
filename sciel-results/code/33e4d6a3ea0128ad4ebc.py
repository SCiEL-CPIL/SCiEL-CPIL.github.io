import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}

    # Precompute all k-blocks and their t-subset coverage
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, store the list of subset indices it covers
    block_to_subsets = [[] for _ in range(n_blocks)]
    # For each subset, store which blocks cover it
    subset_to_blocks = [[] for _ in range(n_subsets)]

    for b_idx, block in enumerate(all_blocks):
        for subset in itertools.combinations(block, t):
            u_idx = subset_to_idx[subset]
            block_to_subsets[b_idx].append(u_idx)
            subset_to_blocks[u_idx].append(b_idx)

    # Convert to numpy arrays for fast operations
    block_subsets_arr = [np.array(bs, dtype=np.int32) for bs in block_to_subsets]
    subset_blocks_arr = [np.array(sb, dtype=np.int32) for sb in subset_to_blocks]

    def greedy_cover(seed_val):
        """Run greedy set cover with optional randomization. Returns list of block indices."""
        rng = random.Random(seed_val)
        uncovered = np.ones(n_subsets, dtype=bool)
        gains = np.zeros(n_blocks, dtype=np.int32)

        # Initial gains
        for u in range(n_subsets):
            for b in subset_to_blocks[u]:
                gains[b] += 1

        chosen = []
        used = set()

        while True:
            if time.time() > deadline:
                break
            if not uncovered.any():
                break

            # Randomized tie-breaking: pick from top-k candidates with some probability
            max_gain = gains.max()
            if max_gain <= 0:
                break

            if seed_val is not None and rng.random() < 0.3:
                # Pick randomly among blocks with gain close to max
                threshold = max_gain * 0.8
                candidates = np.where(gains >= threshold)[0]
                candidates = [c for c in candidates if c not in used]
                if candidates:
                    b_star = rng.choice(candidates)
                else:
                    b_star = int(np.argmax(gains))
            else:
                b_star = int(np.argmax(gains))

            if b_star in used:
                # Find next best unused
                gains[b_star] = -1
                b_star = int(np.argmax(gains))
                if gains[b_star] <= 0:
                    break

            chosen.append(b_star)
            used.add(b_star)

            for u in block_to_subsets[b_star]:
                if uncovered[u]:
                    uncovered[u] = False
                    for b in subset_to_blocks[u]:
                        gains[b] -= 1

            if not uncovered.any():
                break

        return chosen

    def remove_redundant(block_indices):
        """Remove blocks that are not needed for coverage. Returns reduced list."""
        if len(block_indices) <= 1:
            return list(block_indices)

        # Count how many times each subset is covered
        cover_count = np.zeros(n_subsets, dtype=np.int32)
        for b in block_indices:
            for u in block_to_subsets[b]:
                cover_count[u] += 1

        # Try removing each block if all its subsets are still covered
        remaining = list(block_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(remaining)):
                b = remaining[i]
                # Check if removing b leaves all its subsets covered
                removable = True
                for u in block_to_subsets[b]:
                    if cover_count[u] <= 1:
                        removable = False
                        break
                if removable:
                    for u in block_to_subsets[b]:
                        cover_count[u] -= 1
                    remaining.pop(i)
                    changed = True
                    break  # Restart scan after removal

        return remaining

    def local_search(block_indices):
        """Try to improve by swapping blocks. Returns improved list."""
        if len(block_indices) <= 1:
            return list(block_indices)

        # Build coverage counts
        cover_count = np.zeros(n_subsets, dtype=np.int32)
        for b in block_indices:
            for u in block_to_subsets[b]:
                cover_count[u] += 1

        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            remaining = list(block_indices)
            for i in range(len(remaining)):
                if time.time() > deadline - 2.0:
                    break
                b_remove = remaining[i]
                # Check if b_remove can be removed
                can_remove = True
                for u in block_to_subsets[b_remove]:
                    if cover_count[u] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    # Try to find a replacement that covers all uniquely-covered subsets
                    # Actually just remove it
                    for u in block_to_subsets[b_remove]:
                        cover_count[u] -= 1
                    remaining.pop(i)
                    improved = True
                    break
                else:
                    # Try swapping: remove b_remove, add a block that covers the critical subsets
                    critical = [u for u in block_to_subsets[b_remove] if cover_count[u] == 1]
                    if not critical:
                        continue
                    # Find candidate blocks that cover all critical subsets
                    best_candidate = -1
                    best_gain = 0
                    for b_add in subset_to_blocks[critical[0]]:
                        if b_add == b_remove or b_add in remaining:
                            continue
                        # Check if b_add covers all critical subsets
                        covers_all = True
                        for cu in critical:
                            if cu not in block_to_subsets[b_add]:
                                covers_all = False
                                break
                        if not covers_all:
                            continue
                        # Calculate gain: how many new subsets does b_add cover
                        new_gain = 0
                        for u in block_to_subsets[b_add]:
                            if cover_count[u] == 0:
                                new_gain += 1
                        if new_gain > best_gain:
                            best_gain = new_gain
                            best_candidate = b_add
                    if best_candidate >= 0 and best_gain > 0:
                        # Perform swap
                        for u in block_to_subsets[b_remove]:
                            cover_count[u] -= 1
                        for u in block_to_subsets[best_candidate]:
                            cover_count[u] += 1
                        remaining[i] = best_candidate
                        improved = True
                        break

        return remaining

    # Run multiple greedy restarts with different seeds
    best_blocks = None
    best_count = float('inf')

    # First, deterministic greedy
    chosen = greedy_cover(None)
    chosen = remove_redundant(chosen)
    chosen = local_search(chosen)
    best_blocks = chosen
    best_count = len(chosen)

    # Then randomized restarts
    for seed_val in range(20):
        if time.time() > deadline - 10.0:
            break
        chosen = greedy_cover(seed_val)
        if len(chosen) >= best_count:
            continue  # No point polishing if already worse
        chosen = remove_redundant(chosen)
        chosen = local_search(chosen)
        if len(chosen) < best_count:
            best_count = len(chosen)
            best_blocks = chosen

    return [list(all_blocks[b]) for b in best_blocks]
