import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 150s limit

    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    N_t = len(all_t_subsets)
    t_to_idx = {s: i for i, s in enumerate(all_t_subsets)}

    all_blocks = list(itertools.combinations(points, k))
    N_b = len(all_blocks)

    # Precompute: for each block, which t-subsets it covers
    block_cover = []
    for b in all_blocks:
        covered = [t_to_idx[s] for s in itertools.combinations(b, t)]
        block_cover.append(covered)

    # For each t-subset, which blocks cover it
    t_to_blocks = [[] for _ in range(N_t)]
    for bi, covered in enumerate(block_cover):
        for ti in covered:
            t_to_blocks[ti].append(bi)

    # Convert to numpy arrays for faster operations
    # block_cover as list of arrays
    block_cover_np = [np.array(c, dtype=np.int32) for c in block_cover]

    def fast_greedy(rng, alpha=1.0):
        """
        Greedy set cover with randomization.
        alpha controls the randomness in tie-breaking.
        """
        uncovered = np.ones(N_t, dtype=bool)
        num_uncovered = N_t
        selected = []

        while num_uncovered > 0:
            # Pick a random uncovered t-subset
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            ti_target = rng.choice(uncovered_indices.tolist())

            candidates = t_to_blocks[ti_target]
            if not candidates:
                break

            # Score each candidate by gain
            best_block = -1
            best_gain = -1
            best_score = -1

            # For speed, sample if too many candidates
            if len(candidates) > 200:
                sample = rng.sample(candidates, 200)
            else:
                sample = candidates

            for bi in sample:
                gain = np.sum(uncovered[block_cover_np[bi]])
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
                    best_score = gain

            if best_block == -1:
                # Fallback: pick first candidate
                best_block = candidates[0]
                best_gain = np.sum(uncovered[block_cover_np[best_block]])

            selected.append(best_block)
            covered = block_cover_np[best_block]
            newly_covered = uncovered[covered]
            num_uncovered -= np.sum(newly_covered)
            uncovered[covered] = False

        return selected

    def prune(blocks):
        """Remove redundant blocks."""
        if not blocks:
            return []

        counts = np.zeros(N_t, dtype=np.int32)
        for bi in blocks:
            counts[block_cover_np[bi]] += 1

        current_blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for bi in current_blocks:
                # Check if any t-subset is uniquely covered by this block
                if np.any(counts[block_cover_np[bi]] == 1):
                    new_blocks.append(bi)
                else:
                    counts[block_cover_np[bi]] -= 1
                    changed = True
            current_blocks = new_blocks
        return current_blocks

    def local_search(blocks, rng, time_limit):
        """
        Local search: try to remove or swap blocks.
        """
        current_blocks = list(blocks)
        if not current_blocks:
            return current_blocks

        counts = np.zeros(N_t, dtype=np.int32)
        for bi in current_blocks:
            counts[block_cover_np[bi]] += 1

        best_blocks = list(current_blocks)
        best_size = len(current_blocks)
        no_improve_count = 0
        max_no_improve = 50

        def is_redundant(bi):
            return not np.any(counts[block_cover_np[bi]] == 1)

        while time.time() < time_limit:
            if no_improve_count > max_no_improve:
                break

            idx = rng.randrange(len(current_blocks))
            bi = current_blocks[idx]

            # Try to remove block bi if it's redundant
            if is_redundant(bi):
                current_blocks.pop(idx)
                counts[block_cover_np[bi]] -= 1
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = list(current_blocks)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                continue

            # Try to swap block bi with another block
            # Find t-subsets uniquely covered by bi
            unique_mask = counts[block_cover_np[bi]] == 1
            if not np.any(unique_mask):
                no_improve_count += 1
                continue

            unique_ts = block_cover_np[bi][unique_mask]
            if len(unique_ts) == 0:
                no_improve_count += 1
                continue

            # Find blocks that cover all unique_ts
            candidates = set(t_to_blocks[unique_ts[0]])
            for ti in unique_ts[1:]:
                candidates &= set(t_to_blocks[ti])

            candidates.discard(bi)

            if not candidates:
                no_improve_count += 1
                continue

            # Pick a random candidate to swap with
            cand_list = list(candidates)
            cand = rng.choice(cand_list)

            # Check if swap is valid
            # After swap, no t-subset should become uncovered
            new_counts = counts.copy()
            new_counts[block_cover_np[bi]] -= 1
            new_counts[block_cover_np[cand]] += 1

            if np.any(new_counts < 0):
                no_improve_count += 1
                continue

            # Perform swap
            counts = new_counts
            current_blocks[idx] = cand

            # Try to remove any newly redundant blocks
            new_blocks = []
            removed_any = False
            for b in current_blocks:
                if is_redundant(b):
                    counts[block_cover_np[b]] -= 1
                    removed_any = True
                else:
                    new_blocks.append(b)

            current_blocks = new_blocks
            if len(current_blocks) < best_size:
                best_size = len(current_blocks)
                best_blocks = list(current_blocks)
                no_improve_count = 0
            else:
                no_improve_count += 1

        return best_blocks

    def iterated_greedy(rng, time_limit):
        """
        Iterated greedy: generate a greedy solution, prune it,
        then perturb and re-greedy.
        """
        best_blocks = None
        best_len = float('inf')

        while time.time() < time_limit:
            # Generate a greedy solution with randomization
            blocks = fast_greedy(rng)
            blocks = prune(blocks)

            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = list(blocks)

            # Perturb: remove some random blocks and re-greedy
            if best_blocks and time.time() < time_limit - 10:
                perturbed = list(best_blocks)
                # Remove 5-15% of blocks randomly
                num_remove = max(1, int(len(perturbed) * rng.uniform(0.05, 0.15)))
                to_remove = rng.sample(range(len(perturbed)), min(num_remove, len(perturbed)))
                to_remove_set = set(to_remove)
                perturbed = [b for i, b in enumerate(perturbed) if i not in to_remove_set]

                # Re-greedy to fill gaps
                uncovered = np.ones(N_t, dtype=bool)
                for bi in perturbed:
                    uncovered[block_cover_np[bi]] = False

                num_uncovered = np.sum(uncovered)
                if num_uncovered > 0:
                    # Add blocks to cover remaining
                    while num_uncovered > 0 and time.time() < time_limit - 5:
                        uncovered_indices = np.where(uncovered)[0]
                        if len(uncovered_indices) == 0:
                            break
                        ti_target = rng.choice(uncovered_indices.tolist())
                        candidates = t_to_blocks[ti_target]
                        if not candidates:
                            break

                        best_block = -1
                        best_gain = -1
                        for bi in candidates[:50]:  # Limit candidates for speed
                            gain = np.sum(uncovered[block_cover_np[bi]])
                            if gain > best_gain:
                                best_gain = gain
                                best_block = bi

                        if best_block == -1:
                            break

                        perturbed.append(best_block)
                        newly = uncovered[block_cover_np[best_block]]
                        num_uncovered -= np.sum(newly)
                        uncovered[block_cover_np[best_block]] = False

                perturbed = prune(perturbed)
                if len(perturbed) < best_len:
                    best_len = len(perturbed)
                    best_blocks = list(perturbed)

        return best_blocks

    # Main Execution
    best_solution = None
    best_len = float('inf')

    # Phase 1: Multi-start Greedy
    num_starts = 8
    for i in range(num_starts):
        if time.time() > deadline - 40.0:
            break
        rng = random.Random(123 + i * 7)
        blocks = fast_greedy(rng)
        blocks = prune(blocks)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_solution = list(blocks)

    if not best_solution:
        rng = random.Random(1)
        best_solution = fast_greedy(rng)
        best_solution = prune(best_solution)
        best_len = len(best_solution)

    # Phase 2: Local Search
    if best_solution and time.time() < deadline - 10.0:
        rng = random.Random(999)
        optimized = local_search(best_solution, rng, deadline - 5.0)
        if len(optimized) < best_len:
            best_len = len(optimized)
            best_solution = optimized

    # Phase 3: Iterated Greedy for further improvement
    if best_solution and time.time() < deadline - 10.0:
        rng = random.Random(42)
        ig_result = iterated_greedy(rng, deadline - 5.0)
        if ig_result and len(ig_result) < best_len:
            best_len = len(ig_result)
            best_solution = ig_result

    return [list(all_blocks[bi]) for bi in best_solution]
