import itertools
import random
import time
import heapq
from collections import deque

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t = len(all_t_subsets)

    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k = len(all_k_subsets)

    k_to_t_indices = []
    t_to_k_indices = [[] for _ in range(num_t)]

    for ki, k_sub in enumerate(all_k_subsets):
        t_indices = []
        for t_sub in itertools.combinations(k_sub, t):
            ti = t_subset_to_idx[t_sub]
            t_indices.append(ti)
            t_to_k_indices[ti].append(ki)
        k_to_t_indices.append(t_indices)

    block_masks = [0] * num_k
    for ki, t_indices in enumerate(k_to_t_indices):
        mask = 0
        for ti in t_indices:
            mask |= (1 << ti)
        block_masks[ki] = mask

    def run_greedy(seed=None):
        rng = random.Random(seed) if seed is not None else random.Random()

        uncovered_mask = (1 << num_t) - 1
        uncovered_count = num_t
        selected_blocks = []

        heap = []
        for ki in range(num_k):
            gain = (uncovered_mask & block_masks[ki]).bit_count()
            if gain > 0:
                noise = rng.random() * 0.01
                heapq.heappush(heap, (-(gain + noise), ki))

        while uncovered_count > 0:
            if time.time() > deadline:
                break

            best_ki = -1

            while heap:
                neg_gain, ki = heapq.heappop(heap)
                actual_gain = (uncovered_mask & block_masks[ki]).bit_count()
                if actual_gain == 0:
                    continue
                if actual_gain != int(-neg_gain):
                    noise = rng.random() * 0.01
                    heapq.heappush(heap, (-(actual_gain + noise), ki))
                    continue
                best_ki = ki
                break

            if best_ki == -1:
                break

            selected_blocks.append(best_ki)
            block_mask = block_masks[best_ki]
            newly_covered = uncovered_mask & block_mask
            if newly_covered:
                uncovered_count -= newly_covered.bit_count()
                uncovered_mask &= ~newly_covered

        return selected_blocks

    def local_search(blocks_list, time_budget=5.0):
        current_blocks = list(blocks_list)
        current_set = set(current_blocks)
        end_time = time.time() + time_budget

        def compute_counts(blks):
            counts = [0] * num_t
            for ki in blks:
                for ti in k_to_t_indices[ki]:
                    counts[ti] += 1
            return counts

        improved = True
        while improved and time.time() < end_time:
            improved = False
            counts = compute_counts(current_blocks)

            # Remove blocks with no unique subsets
            removed = False
            for i in range(len(current_blocks)):
                ki = current_blocks[i]
                has_unique = any(counts[ti] == 1 for ti in k_to_t_indices[ki])
                if not has_unique:
                    current_blocks.pop(i)
                    current_set.discard(ki)
                    improved = True
                    removed = True
                    break
            if removed:
                continue

            # Single-block replacement
            for i in range(len(current_blocks)):
                ki = current_blocks[i]
                unique_subs = [ti for ti in k_to_t_indices[ki] if counts[ti] == 1]
                if not unique_subs:
                    continue

                if len(unique_subs) == 1:
                    ti = unique_subs[0]
                    for new_ki in t_to_k_indices[ti]:
                        if new_ki == ki or new_ki in current_set:
                            continue
                        current_blocks[i] = new_ki
                        current_set.discard(ki)
                        current_set.add(new_ki)
                        improved = True
                        break
                else:
                    us_sorted = sorted(unique_subs, key=lambda ti: len(t_to_k_indices[ti]))
                    candidates = set(t_to_k_indices[us_sorted[0]])
                    for ti in us_sorted[1:]:
                        candidates &= set(t_to_k_indices[ti])
                        if not candidates:
                            break
                    candidates.discard(ki)
                    candidates -= current_set
                    if candidates:
                        new_ki = random.choice(list(candidates))
                        current_blocks[i] = new_ki
                        current_set.discard(ki)
                        current_set.add(new_ki)
                        improved = True
                        break

                if improved:
                    break

            # 2-exchange: replace two blocks simultaneously
            if not improved and len(current_blocks) >= 2:
                blocks_with_unique = []
                for i, ki in enumerate(current_blocks):
                    unique_subs = [ti for ti in k_to_t_indices[ki] if counts[ti] == 1]
                    if unique_subs:
                        blocks_with_unique.append((i, ki, unique_subs))

                if len(blocks_with_unique) >= 2:
                    for a in range(min(len(blocks_with_unique), 4)):
                        for b in range(a + 1, min(len(blocks_with_unique), 4)):
                            ia, ki_a, unique_a = blocks_with_unique[a]
                            ib, ki_b, unique_b = blocks_with_unique[b]
                            all_unique = unique_a + unique_b
                            au_sorted = sorted(all_unique, key=lambda ti: len(t_to_k_indices[ti]))
                            cand_a = set(t_to_k_indices[au_sorted[0]])
                            cand_b = set(t_to_k_indices[au_sorted[0]])
                            for ti in au_sorted[1:]:
                                cand_a &= set(t_to_k_indices[ti])
                                cand_b &= set(t_to_k_indices[ti])
                                if not cand_a or not cand_b:
                                    break
                            cand_a.discard(ki_a)
                            cand_a.discard(ki_b)
                            cand_b.discard(ki_a)
                            cand_b.discard(ki_b)
                            cand_a -= current_set
                            cand_b -= current_set
                            if cand_a and cand_b:
                                new_a = random.choice(list(cand_a))
                                new_b = random.choice(list(cand_b))
                                if new_a != new_b:
                                    current_blocks[ia] = new_a
                                    current_blocks[ib] = new_b
                                    current_set.discard(ki_a)
                                    current_set.discard(ki_b)
                                    current_set.add(new_a)
                                    current_set.add(new_b)
                                    improved = True
                                    break
                        if improved:
                            break

        return current_blocks

    def maxflow_refine(blocks_list, time_budget=3.0):
        """Use max-flow to find minimum block subset, then use min-cut to identify forced blocks."""
        if not blocks_list:
            return blocks_list
        
        current_blocks = list(blocks_list)
        current_set = set(current_blocks)
        end_time = time.time() + time_budget
        
        # Build incidence: for each block in current set, which t-subsets does it cover?
        # We want to find a minimum-size subset of current_blocks that still covers all t-subsets.
        
        # Max-flow formulation:
        # Source -> block nodes (capacity 1) -> t-subset nodes (capacity INF) -> Sink
        # If max flow < num_t, we can't cover all t-subsets with the current blocks.
        # The min-cut tells us which blocks are "forced" (on the source side of the cut).
        
        # Actually, let's think differently. We have a set of blocks that covers all t-subsets.
        # We want to remove as many as possible while maintaining coverage.
        # This is a hitting set / set cover problem on the t-subsets.
        
        # Simpler approach: iteratively try to remove blocks.
        # For each block, check if removing it still maintains coverage.
        
        def is_valid(blocks):
            counts = [0] * num_t
            for ki in blocks:
                for ti in k_to_t_indices[ki]:
                    counts[ti] += 1
            return all(c >= 1 for c in counts)
        
        # Greedy removal: try to remove blocks that are least "important"
        improved = True
        while improved and time.time() < end_time:
            improved = False
            counts = [0] * num_t
            for ki in current_blocks:
                for ti in k_to_t_indices[ki]:
                    counts[ti] += 1
            
            # Find blocks with no unique coverage
            for i in range(len(current_blocks)):
                ki = current_blocks[i]
                has_unique = any(counts[ti] == 1 for ti in k_to_t_indices[ki])
                if not has_unique:
                    # Try removing this block
                    new_blocks = current_blocks[:i] + current_blocks[i+1:]
                    if is_valid(new_blocks):
                        current_blocks = new_blocks
                        current_set.discard(ki)
                        improved = True
                        break
        
        return current_blocks

    def flow_based_replacement(blocks_list, time_budget=5.0):
        """Use flow analysis to identify blocks that can be replaced to reduce total count."""
        if not blocks_list:
            return blocks_list
        
        current_blocks = list(blocks_list)
        current_set = set(current_blocks)
        end_time = time.time() + time_budget
        
        def compute_counts(blks):
            counts = [0] * num_t
            for ki in blks:
                for ti in k_to_t_indices[ki]:
                    counts[ti] += 1
            return counts
        
        improved = True
        while improved and time.time() < end_time:
            improved = False
            counts = compute_counts(current_blocks)
            
            # Identify blocks with unique coverage
            blocks_with_unique = []
            for i, ki in enumerate(current_blocks):
                unique_subs = [ti for ti in k_to_t_indices[ki] if counts[ti] == 1]
                if unique_subs:
                    blocks_with_unique.append((i, ki, unique_subs))
            
            # For each block with unique coverage, try to find a replacement
            # that covers all its unique subsets AND some other subsets,
            # potentially allowing removal of another block.
            for i, ki, unique_subs in blocks_with_unique:
                if time.time() > end_time:
                    break
                
                # Find candidates that cover all unique_subs
                if len(unique_subs) == 1:
                    ti = unique_subs[0]
                    candidates = [new_ki for new_ki in t_to_k_indices[ti] 
                                  if new_ki != ki and new_ki not in current_set]
                else:
                    us_sorted = sorted(unique_subs, key=lambda ti: len(t_to_k_indices[ti]))
                    candidates = set(t_to_k_indices[us_sorted[0]])
                    for ti in us_sorted[1:]:
                        candidates &= set(t_to_k_indices[ti])
                        if not candidates:
                            break
                    candidates.discard(ki)
                    candidates -= current_set
                    candidates = list(candidates)
                
                if not candidates:
                    continue
                
                # Try each candidate: after replacement, can we remove another block?
                best_candidate = None
                best_new_blocks = None
                
                for new_ki in candidates[:20]:  # Limit candidates to keep fast
                    # Simulate replacement
                    test_blocks = current_blocks.copy()
                    test_blocks[i] = new_ki
                    test_set = current_set.copy()
                    test_set.discard(ki)
                    test_set.add(new_ki)
                    
                    # Recompute counts
                    test_counts = [0] * num_t
                    for bki in test_blocks:
                        for ti in k_to_t_indices[bki]:
                            test_counts[ti] += 1
                    
                    # Check if we can now remove any other block
                    for j, bki in enumerate(test_blocks):
                        if j == i:
                            continue
                        has_unique = any(test_counts[ti] == 1 for ti in k_to_t_indices[bki])
                        if not has_unique:
                            # Check if removing bki still maintains coverage
                            new_test_blocks = test_blocks[:j] + test_blocks[j+1:]
                            new_counts = [0] * num_t
                            for bki2 in new_test_blocks:
                                for ti in k_to_t_indices[bki2]:
                                    new_counts[ti] += 1
                            if all(c >= 1 for c in new_counts):
                                best_candidate = new_ki
                                best_new_blocks = new_test_blocks
                                break
                    
                    if best_new_blocks is not None:
                        break
                
                if best_new_blocks is not None:
                    current_blocks = best_new_blocks
                    current_set = set(current_blocks)
                    improved = True
                    break
        
        return current_blocks

    best_blocks = None
    best_count = float('inf')
    max_restarts = 40

    for seed in range(max_restarts):
        if time.time() > deadline - 20:
            break

        blocks_idx = run_greedy(seed)
        if not blocks_idx:
            continue

        # Local search with time budget
        ls_time = min(6.0, max(2.0, (deadline - 10 - time.time()) * 0.25))
        result = local_search(blocks_idx, ls_time)

        # Flow-based refinement
        if time.time() < deadline - 5:
            flow_time = min(4.0, max(1.0, (deadline - 5 - time.time()) * 0.3))
            result = flow_based_replacement(result, flow_time)

        if len(result) < best_count:
            best_count = len(result)
            best_blocks = result

    if best_blocks is None:
        blocks_idx = run_greedy(0)
        if blocks_idx:
            best_blocks = blocks_idx
        else:
            return []

    return [list(all_k_subsets[ki]) for ki in best_blocks]
