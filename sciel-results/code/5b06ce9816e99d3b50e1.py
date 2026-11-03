import itertools
import numpy as np
import time
import random
import heapq
from math import comb

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    num_t = len(all_t_subsets)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    t_choose_k = comb(k, t)
    
    # Precompute for each block: which t-subsets it covers
    # And for each t-subset: which blocks cover it
    block_t_indices = np.zeros((num_blocks, t_choose_k), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        ts_list = list(itertools.combinations(block, t))
        for j, ts in enumerate(ts_list):
            block_t_indices[i, j] = t_subset_to_idx[ts]
    
    # Build inverse: for each t-subset, list of blocks that cover it
    # This is expensive to build fully, so we'll use it selectively
    # Actually, let's build it as a list of lists
    t_subset_blocks = [[] for _ in range(num_t)]
    for i in range(num_blocks):
        for j in range(t_choose_k):
            ts_idx = block_t_indices[i, j]
            t_subset_blocks[ts_idx].append(i)
    
    best_solution = None
    best_len = float('inf')
    
    # Primal-dual greedy approach:
    # 1. Maintain dual variables y[s] for each t-subset s (initially 0)
    # 2. Repeatedly:
    #    a. Find uncovered t-subsets
    #    b. Increase y[s] for uncovered s at a rate proportional to 1/|uncovered in block|
    #    c. When some block's total dual weight equals its cost (1), select it
    #    
    # Simplified primal-dual for set cover with unit costs:
    # - All blocks have cost 1
    # - We increase duals for uncovered elements
    # - A block is selected when sum of duals of its uncovered elements >= 1
    #
    # This is essentially a weighted greedy where the weight of a block is
    # determined by how much "dual pressure" it accumulates.
    
    # Alternative: Use the LP dual weights from a simple iteration
    # Let's implement a practical primal-dual:
    
    def primal_dual_greedy(seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered = np.ones(num_t, dtype=bool)
        dual = np.zeros(num_t, dtype=np.float64)
        solution = []
        used_blocks = set()
        
        while True:
            if time.time() > deadline:
                return None
            
            if not np.any(uncovered):
                break
            
            # Find all blocks that cover at least one uncovered t-subset
            # For each such block, compute its "dual value" = sum of duals of uncovered t-subsets in it
            # We want to pick the block that maximizes this dual value / cost (cost=1)
            
            # Efficient approach: maintain for each block the sum of duals of its uncovered elements
            # When we select a block, we mark its elements as covered and update duals
            
            # For simplicity in this iteration, let's do:
            # 1. Compute gain for each block (number of new uncovered elements it covers)
            # 2. Compute dual weight = sum of dual[uncovered elements in block]
            # 3. Pick block maximizing dual_weight / gain (or just dual_weight)
            
            # Actually, the standard primal-dual for set cover:
            # - Increase duals for all uncovered elements at rate 1
            # - When a block's total dual (over its elements) reaches its cost, add it
            # - With unit costs, a block is added when sum of duals of its elements >= 1
            # - But we only count duals of elements that are still uncovered at the time...
            
            # Let me implement a cleaner version:
            
            # Phase 1: Increase duals for uncovered elements
            # Phase 2: Select blocks whose dual sum >= 1
            
            # For practical purposes with unit costs, this reduces to:
            # Pick the block that covers the most uncovered elements (greedy)
            # but with a twist: we can use the dual values to break ties or weight choices
            
            # Let's try a different structural approach:
            # Use the primal-dual to get initial weights, then do local search
            
            # Compute gains
            gains = np.zeros(num_blocks, dtype=np.int32)
            for i in range(num_blocks):
                if i in used_blocks:
                    continue
                ts_idx = block_t_indices[i]
                gains[i] = np.sum(uncovered[ts_idx])
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Among blocks with max gain, pick one that maximizes dual weight
            candidates = np.where(gains == max_gain)[0]
            
            if len(candidates) == 1:
                best_block = candidates[0]
            else:
                # Compute dual weight for each candidate
                best_block = candidates[0]
                best_dual_weight = -1
                for c in candidates:
                    ts_idx = block_t_indices[c]
                    dw = np.sum(dual[ts_idx[uncovered[ts_idx]]])
                    if dw > best_dual_weight:
                        best_dual_weight = dw
                        best_block = c
            
            # Select the block
            solution.append(best_block)
            used_blocks.add(best_block)
            
            # Update: mark covered
            covered_ts = block_t_indices[best_block]
            uncovered[covered_ts] = False
            
            # Increase duals for all currently uncovered elements
            # In primal-dual, we increase duals at a rate such that
            # the constraint is tight when we select a block
            # For unit cost blocks, we increase duals of uncovered elements
            # until some block's dual sum reaches 1
            # 
            # Simplification: just increase all uncovered duals by a small amount
            # This creates a preference for blocks covering elements with higher duals
            uncovered_mask = uncovered
            if np.any(uncovered_mask):
                dual[uncovered_mask] += 1.0 / max_gain  # Scale by current max gain
        
        return solution
    
    # Local search: 1-exchange (swap) and deletion
    def local_search(solution, block_t_indices, uncovered_initial, deadline):
        if solution is None:
            return None
        
        sol = list(solution)
        
        # Build coverage counts
        cov = np.zeros(num_t, dtype=np.int32)
        for b_idx in sol:
            cov[block_t_indices[b_idx]] += 1
        
        # Deletion phase: remove blocks that are not essential
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(sol) - 1, -1, -1):
                if time.time() > deadline:
                    return sol
                b_idx = sol[i]
                ts_idx = block_t_indices[b_idx]
                if np.min(cov[ts_idx]) > 1:
                    cov[ts_idx] -= 1
                    sol.pop(i)
                    changed = True
        
        # 1-exchange phase: try to swap a block for a better one
        # For each block in the solution, try replacing it with a block that
        # covers more uncovered t-subsets (relative to the rest of the solution)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(sol)):
                if time.time() > deadline:
                    return sol
                b_old = sol[i]
                ts_old = block_t_indices[b_old]
                
                # Coverage if we remove b_old
                cov_temp = cov.copy()
                cov_temp[ts_old] -= 1
                
                # Find t-subsets that would become uncovered
                would_uncover = cov_temp[ts_old] == 0
                
                if not np.any(would_uncover):
                    # b_old is redundant, just remove it
                    cov = cov_temp
                    sol.pop(i)
                    improved = True
                    break
                else:
                    # Need to find a block that covers all would_uncover t-subsets
                    # and doesn't make things worse
                    need_cover = set(np.where(would_uncover)[0])
                    
                    # Find blocks that cover all of need_cover
                    best_block = -1
                    best_score = -1
                    
                    # For efficiency, only consider blocks that cover at least one of need_cover
                    # and check if they cover all
                    candidate_blocks = set()
                    for ts in need_cover:
                        candidate_blocks.update(t_subset_blocks[ts])
                    
                    # Remove the old block from consideration
                    candidate_blocks.discard(b_old)
                    
                    for c in candidate_blocks:
                        if c in sol:
                            continue
                        ts_c = block_t_indices[c]
                        # Check if c covers all of need_cover
                        if all(ts in ts_c for ts in need_cover):
                            # Score: how many new t-subsets does c cover that aren't already well-covered?
                            # We want to minimize the "damage" - i.e., t-subsets that go from cov>=2 to cov=1
                            # Actually, we want to maximize the number of t-subsets with cov==1 that c covers
                            # because those are the ones at risk
                            new_cov = cov_temp + np.zeros(num_t, dtype=np.int32)
                            new_cov[ts_c] += 1
                            # Count t-subsets that are now covered by exactly 1 block (were >=1 before)
                            # Actually, let's just check validity and prefer blocks that cover more
                            gain = np.sum(new_cov[ts_c] == 1)
                            if gain > best_score:
                                best_score = gain
                                best_block = c
                    
                    if best_block >= 0:
                        # Perform the swap
                        cov[ts_old] -= 1
                        cov[block_t_indices[best_block]] += 1
                        sol[i] = best_block
                        improved = True
                        break
        
        return sol
    
    # Run multiple restarts with different seeds
    seed_counter = 0
    while time.time() < deadline:
        if time.time() > deadline:
            break
        
        # Primal-dual greedy
        sol = primal_dual_greedy(seed=seed_counter)
        seed_counter += 1
        
        if sol is None or len(sol) == 0:
            continue
        
        # Local search
        sol = local_search(sol, block_t_indices, None, deadline)
        
        if sol is None:
            continue
        
        # Validate
        cov_check = np.zeros(num_t, dtype=np.int32)
        for b_idx in sol:
            cov_check[block_t_indices[b_idx]] += 1
        if np.any(cov_check == 0):
            continue
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = [list(all_blocks[b_idx]) for b_idx in sol]
            
            if best_len <= 131:
                break
    
    if best_solution is None:
        # Fallback: simple greedy
        uncovered = np.ones(num_t, dtype=bool)
        solution_blocks = []
        while np.any(uncovered) and time.time() < deadline:
            gains = np.zeros(num_blocks, dtype=np.int32)
            for i in range(num_blocks):
                ts_idx = block_t_indices[i]
                gains[i] = np.sum(uncovered[ts_idx])
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            best_block = np.argmax(gains)
            solution_blocks.append(best_block)
            uncovered[block_t_indices[best_block]] = False
        
        if solution_blocks:
            cov_final = np.zeros(num_t, dtype=np.int32)
            for b_idx in solution_blocks:
                cov_final[block_t_indices[b_idx]] += 1
            if not np.any(cov_final == 0):
                return [list(all_blocks[b_idx]) for b_idx in solution_blocks]
        
        # Last resort: return empty (invalid, but better than exception)
        return []
    
    return best_solution
