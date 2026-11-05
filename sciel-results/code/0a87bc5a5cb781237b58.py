import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118  # Safety margin for 120s limit

    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    num_triples = len(triples)
    triple_to_idx = {tri: i for i, tri in enumerate(triples)}

    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)

    # Precompute: for each block, the list of triple indices it covers
    block_triple_list = []
    for b in all_blocks:
        tids = [triple_to_idx[tri] for tri in itertools.combinations(b, t)]
        block_triple_list.append(tids)

    # Reverse index: for each triple, list of block indices that cover it
    triple_block_list = [[] for _ in range(num_triples)]
    for b_idx, tids in enumerate(block_triple_list):
        for t_idx in tids:
            triple_block_list[t_idx].append(b_idx)

    def greedy_construct(seed, max_blocks=120):
        """Weighted greedy with incremental gain updates."""
        rng = random.Random(seed)
        gain = np.array([len(ts) for ts in block_triple_list], dtype=np.int32)
        is_uncovered = np.ones(num_triples, dtype=np.bool_)

        selected = []

        while is_uncovered.any():
            if time.time() > deadline - 25:
                break
            if len(selected) >= max_blocks:
                break

            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break

            # Find candidates with max gain
            candidates = np.where(gain == max_gain)[0]
            b_idx = int(rng.choice(candidates))

            selected.append(b_idx)

            # Update: for each triple in this block, if it was uncovered,
            # mark it covered and decrement gain for all blocks covering it
            for t_idx in block_triple_list[b_idx]:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                    for other_b in triple_block_list[t_idx]:
                        gain[other_b] -= 1

        return selected, is_uncovered

    def prune_solution(block_indices):
        """Remove redundant blocks iteratively."""
        if not block_indices:
            return []

        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in block_indices:
            for t_idx in block_triple_list[b_idx]:
                cover_count[t_idx] += 1

        current = list(block_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(current)):
                b_idx = current[i]
                redundant = True
                for t_idx in block_triple_list[b_idx]:
                    if cover_count[t_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    for t_idx in block_triple_list[b_idx]:
                        cover_count[t_idx] -= 1
                    current.pop(i)
                    changed = True
                    if time.time() > deadline - 15:
                        return current
                    break

        return current

    def local_search(block_indices, max_iter=200):
        """Try 1-for-1 swaps to reduce block count."""
        if len(block_indices) < 2:
            return block_indices

        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in block_indices:
            for t_idx in block_triple_list[b_idx]:
                cover_count[t_idx] += 1

        current = list(block_indices)
        best = list(current)
        best_size = len(current)

        rng = random.Random(42)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Try removing each block and see if we can add a cheaper one
            for i in range(len(current)):
                if time.time() > deadline - 10:
                    break
                b_remove = current[i]
                # Tentatively remove
                for t_idx in block_triple_list[b_remove]:
                    cover_count[t_idx] -= 1

                # Check if any triple is now uncovered
                uncovered = False
                for t_idx in block_triple_list[b_remove]:
                    if cover_count[t_idx] < 1:
                        uncovered = True
                        break

                if not uncovered:
                    # Removal is valid - just remove it
                    current.pop(i)
                    improved = True
                    if len(current) < best_size:
                        best = list(current)
                        best_size = len(current)
                    break
                else:
                    # Restore
                    for t_idx in block_triple_list[b_remove]:
                        cover_count[t_idx] += 1

                    # Try to find a block that covers all uncovered triples
                    # and has minimal "cost" (number of already-covered triples it adds)
                    uncovered_triples = []
                    for t_idx in block_triple_list[b_remove]:
                        if cover_count[t_idx] < 1:
                            uncovered_triples.append(t_idx)

                    if not uncovered_triples:
                        continue

                    # Find blocks that cover ALL uncovered triples
                    # and minimize the number of new triples they cover
                    best_add = -1
                    best_new_covered = float('inf')

                    # Sample a subset of blocks for speed
                    sample_size = min(5000, num_blocks)
                    sample = rng.sample(range(num_blocks), sample_size)

                    for b_add in sample:
                        if b_add == b_remove:
                            continue
                        # Check if b_add covers all uncovered triples
                        adds = set(block_triple_list[b_add])
                        if not all(t in adds for t in uncovered_triples):
                            continue
                        # Count how many new triples it covers
                        new_count = sum(1 for t in adds if cover_count[t] < 1)
                        if new_count < best_new_covered:
                            best_new_covered = new_count
                            best_add = b_add

                    if best_add >= 0:
                        # Perform the swap
                        for t_idx in block_triple_list[best_add]:
                            cover_count[t_idx] += 1
                        current[i] = best_add
                        improved = True
                        if len(current) < best_size:
                            best = list(current)
                            best_size = len(current)
                        break

        return best

    def is_valid_solution(block_indices):
        """Verify all triples are covered."""
        if not block_indices:
            return num_triples == 0
        covered = np.zeros(num_triples, dtype=np.int32)
        for b_idx in block_indices:
            for t_idx in block_triple_list[b_idx]:
                covered[t_idx] += 1
        return np.all(covered >= 1)

    best_solution = None
    best_size = float('inf')

    # Phase 1: Multiple greedy constructions with different seeds
    num_starts = 30
    for seed in range(num_starts):
        if time.time() > deadline - 30:
            break

        raw, still_uncovered = greedy_construct(seed)

        if still_uncovered.any():
            # Greedy didn't finish - try to complete with a fast scan
            if time.time() > deadline - 25:
                continue
            # Complete with simple greedy on remaining
            gain = np.array([len(ts) for ts in block_triple_list], dtype=np.int32)
            for b_idx in raw:
                for t_idx in block_triple_list[b_idx]:
                    if still_uncovered[t_idx]:
                        still_uncovered[t_idx] = False
                        for other_b in triple_block_list[t_idx]:
                            gain[other_b] -= 1

            while still_uncovered.any():
                if time.time() > deadline - 25:
                    break
                max_gain = int(np.max(gain))
                if max_gain <= 0:
                    break
                candidates = np.where(gain == max_gain)[0]
                b_idx = int(candidates[0])
                raw.append(b_idx)
                for t_idx in block_triple_list[b_idx]:
                    if still_uncovered[t_idx]:
                        still_uncovered[t_idx] = False
                        for other_b in triple_block_list[t_idx]:
                            gain[other_b] -= 1

        if not raw:
            continue

        pruned = prune_solution(raw)

        if len(pruned) < best_size:
            if is_valid_solution(pruned):
                best_solution = pruned
                best_size = len(pruned)
                if best_size <= 72:
                    break

    # Phase 2: Local search on the best solution
    if best_solution is not None and time.time() < deadline - 15:
        refined = local_search(best_solution)
        if len(refined) < best_size:
            if is_valid_solution(refined):
                best_solution = refined
                best_size = len(refined)

    # Phase 3: If we have time, try more local search passes
    if best_solution is not None and time.time() < deadline - 5:
        for _ in range(3):
            if time.time() > deadline - 3:
                break
            refined = local_search(best_solution)
            if len(refined) < best_size:
                if is_valid_solution(refined):
                    best_solution = refined
                    best_size = len(refined)

    if best_solution is None:
        # Fallback: single greedy without time limit concerns
        raw, _ = greedy_construct(0)
        if raw:
            best_solution = prune_solution(raw)
        else:
            # Absolute fallback
            best_solution = list(range(min(100, num_blocks)))

    result_blocks = [list(all_blocks[i]) for i in best_solution]
    return result_blocks

if __name__ == "__main__":
    pass
