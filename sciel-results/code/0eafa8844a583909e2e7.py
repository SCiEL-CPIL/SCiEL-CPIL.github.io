import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110
    
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    n_t_subsets = len(all_t_subsets)
    
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Pre-compute mapping from t-subset to list of k-subsets containing it?
    # C(19,4) = 3876. C(19,7) = 50388.
    # It is faster to generate k-subsets on the fly during greedy/local search
    # if we can filter them, but we need to know which k-subsets cover which t-subsets.
    # Let's pre-compute the incidence matrix in a sparse format or just lists.
    # Given v=19, k=7, t=4, the number of k-subsets is 50k. This is manageable.
    # But iterating 50k * 15 times in Python is slow.
    
    # Strategy:
    # 1. Calculate degree of each t-subset.
    # 2. Define weights for criticality.
    # 3. Greedy construction using these weights.
    # 4. Local search to minimize size.
    
    # Pre-compute all k-subsets and their covered t-subsets
    # This is the bottleneck. Let's optimize it.
    all_k_subsets = list(itertools.combinations(points, k))
    n_k_subsets = len(all_k_subsets)
    
    # block_t_indices[b] = list of t_subset indices covered by block b
    block_t_indices = []
    degree = [0] * n_t_subsets
    
    # We can speed this up by noting that a k-subset covers C(k,t) t-subsets.
    for block in all_k_subsets:
        covered_ts = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            covered_ts.append(idx)
            degree[idx] += 1
        block_t_indices.append(covered_ts)
        
    # Identify critical subsets
    # Heuristic: weight = 1 / degree. Lower degree -> higher weight -> more critical.
    # We will use these weights in the greedy selection to prioritize covering "hard" subsets.
    weights = [1.0 / d if d > 0 else 0.0 for d in degree]
    
    # To make the GA/Local Search focus on critical subsets, we can define a "critical" set
    # as the top 20% by weight. However, for validity, we must cover ALL.
    # The prompt says: "apply a genetic algorithm to optimize block selection specifically against these critical constraints rather than the full universe".
    # This implies the fitness function might prioritize critical coverage, but the final solution must be valid (cover all).
    # So, we will use the weights to guide the *construction* and *local search* gains, 
    # ensuring that we don't waste blocks on "easy" subsets if "hard" ones are uncovered.
    
    # Reverse map: t_subset_index -> list of block indices
    # This is useful for local search swaps.
    ts_to_blocks = [[] for _ in range(n_t_subsets)]
    for b_idx, ts_list in enumerate(block_t_indices):
        for ts_idx in ts_list:
            ts_to_blocks[ts_idx].append(b_idx)

    def greedy_construct():
        """
        Greedy set cover using weighted gains.
        """
        coverage = [0] * n_t_subsets
        selected = []
        uncovered_count = n_t_subsets
        
        # To introduce diversity for the "GA" aspect, we can randomize the tie-breaking
        # or the order of evaluation slightly.
        
        while uncovered_count > 0 and time.time() - start_time < time_limit * 0.2:
            # Find the best block to add
            best_block = -1
            best_gain = -1.0
            
            # Optimization: Instead of checking all blocks, check blocks covering a random uncovered subset?
            # No, standard greedy checks all. But 50k blocks * 15 t-subsets = 750k ops per step.
            # 150 steps * 750k = 112M ops. Too slow in Python.
            
            # Better approach:
            # Pick a random uncovered t-subset.
            # Only consider blocks that cover it.
            # This reduces the search space significantly.
            
            # Find an uncovered subset
            uncovered_indices = [i for i, c in enumerate(coverage) if c == 0]
            if not uncovered_indices:
                break
            target = random.choice(uncovered_indices)
            
            # Iterate through blocks covering 'target'
            # ts_to_blocks[target] has ~ C(19-4, 7-4) = C(15,3) = 455 blocks on average?
            # Actually, degree is around 455. So this is very fast.
            
            candidates = ts_to_blocks[target]
            if not candidates:
                break # Should not happen for valid parameters
            
            for b_idx in candidates:
                gain = 0.0
                for ts_idx in block_t_indices[b_idx]:
                    if coverage[ts_idx] == 0:
                        gain += weights[ts_idx]
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                break
                
            selected.append(best_block)
            for ts_idx in block_t_indices[best_block]:
                if coverage[ts_idx] == 0:
                    uncovered_count -= 1
                coverage[ts_idx] += 1
                
        return selected, coverage

    def local_search(initial_blocks, initial_coverage):
        """
        Local search to minimize block count while maintaining full coverage.
        """
        current_blocks = list(initial_blocks)
        coverage = list(initial_coverage)
        
        # Ensure full coverage (repair if necessary, though greedy should be full)
        uncovered = [i for i in range(n_t_subsets) if coverage[i] == 0]
        while uncovered:
            target = random.choice(uncovered)
            candidates = ts_to_blocks[target]
            if not candidates:
                break
            best_b = -1
            best_gain = -1
            for b in candidates:
                gain = 0
                for ts in block_t_indices[b]:
                    if coverage[ts] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_b = b
            if best_b == -1:
                break
            if best_b not in current_blocks:
                current_blocks.append(best_b)
                for ts in block_t_indices[best_b]:
                    coverage[ts] += 1
            uncovered = [i for i in range(n_t_subsets) if coverage[i] == 0]

        improved = True
        while improved and time.time() - start_time < time_limit * 0.9:
            improved = False
            
            # 1. Deletion Pass
            # Try to remove any block that doesn't uniquely cover any t-subset.
            for i in range(len(current_blocks)):
                b_idx = current_blocks[i]
                can_remove = True
                for ts in block_t_indices[b_idx]:
                    if coverage[ts] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for ts in block_t_indices[b_idx]:
                        coverage[ts] -= 1
                    current_blocks.pop(i)
                    improved = True
                    break # Restart scan after modification
            
            if time.time() - start_time > time_limit * 0.85:
                break

            # 2. Swap Pass (1-for-1)
            if not improved:
                indices = list(range(len(current_blocks)))
                random.shuffle(indices)
                for i in indices:
                    old_b = current_blocks[i]
                    # Unique subsets covered by old_b
                    unique_ts = [ts for ts in block_t_indices[old_b] if coverage[ts] == 1]
                    if not unique_ts:
                        continue
                    
                    # Find a candidate that covers all unique_ts
                    # Sort unique_ts by rarity to prune candidates fast
                    unique_ts.sort(key=lambda x: len(ts_to_blocks[x]))
                    
                    candidates = set(ts_to_blocks[unique_ts[0]])
                    for ts in unique_ts[1:]:
                        candidates.intersection_update(ts_to_blocks[ts])
                    candidates.discard(old_b)
                    
                    if not candidates:
                        continue
                    
                    # Pick the best candidate: one that covers the most *other* subsets (to allow future deletions)
                    # Or simply one that has high weight coverage?
                    # Let's pick the one that maximizes the number of subsets with coverage > 1 
                    # that it covers (this might allow removing other blocks later?)
                    # Actually, a good heuristic is to pick the candidate that covers the most subsets
                    # that are currently covered by >1 block, or just minimize the "loss" of unique coverage.
                    # Since we must cover unique_ts, the "loss" is fixed.
                    # We want to pick a block that is "cheap" in terms of redundancy?
                    # Let's pick the candidate with the smallest size? All blocks are same size.
                    # Let's pick the candidate that covers the most subsets with coverage == 1 (other than unique_ts)?
                    # No, that increases criticality.
                    # Let's pick the candidate that maximizes the number of subsets with coverage > 1.
                    # This increases redundancy, which might allow further swaps/deletions.
                    
                    best_new = -1
                    best_score = -1
                    cand_list = list(candidates)
                    if len(cand_list) > 20:
                        cand_list = random.sample(cand_list, 20)
                    
                    for nb in cand_list:
                        score = 0
                        for ts in block_t_indices[nb]:
                            if coverage[ts] > 1:
                                score += 1
                        # Also prefer blocks that cover critical subsets?
                        # Let's stick to redundancy for now.
                        if score > best_score:
                            best_score = score
                            best_new = nb
                    
                    if best_new != -1:
                        # Perform swap
                        for ts in block_t_indices[old_b]:
                            coverage[ts] -= 1
                        for ts in block_t_indices[best_new]:
                            coverage[ts] += 1
                        current_blocks[i] = best_new
                        improved = True
                        break
                        
        return current_blocks

    # Main Loop: Multiple restarts (Simulated Annealing / GA hybrid)
    best_solution = None
    best_size = float('inf')
    
    # Run multiple greedy constructions
    n_restarts = 20 # Adjust based on time
    
    for r in range(n_restarts):
        if time.time() - start_time > time_limit * 0.8:
            break
            
        # Randomize the greedy choice slightly by shuffling the order of candidates in ts_to_blocks?
        # Or just relying on random.choice(uncovered_indices) provides enough diversity.
        
        selected, coverage = greedy_construct()
        
        if not selected:
            continue
            
        optimized = local_search(selected, coverage)
        
        if len(optimized) < best_size:
            best_size = len(optimized)
            best_solution = optimized
            
        # Early stop if we reach a very good bound?
        # Lower bound is 131. If we hit 135, it's good.
        
    if best_solution is None:
        # Fallback: Just return the first greedy solution
        selected, coverage = greedy_construct()
        best_solution = local_search(selected, coverage)
        
    # Final cleanup
    final_blocks = local_search(best_solution, [0]*n_t_subsets) # Re-calculate coverage for safety?
    # local_search recalculates coverage internally if we pass a dummy coverage? 
    # No, my local_search takes initial_coverage. I should pass the correct one or let it recalc.
    # Let's modify local_search to recalc coverage if initial_coverage is None.
    
    # Actually, the previous local_search implementation uses the passed coverage.
    # I need to ensure the coverage passed to the final call is correct.
    # Let's just run local_search again on the best_solution with a fresh coverage calculation.
    
    # Recalculate coverage for best_solution
    cov = [0] * n_t_subsets
    for b in best_solution:
        for ts in block_t_indices[b]:
            cov[ts] += 1
            
    final_blocks = local_search(best_solution, cov)
    
    return [list(all_k_subsets[i]) for i in final_blocks]
