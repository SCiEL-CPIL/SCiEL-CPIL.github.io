import itertools
import numpy as np
import time
import random
import heapq
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for safety

    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}
    
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # Map each t-subset to the list of k-blocks that contain it
    blocks_of_subset = [[] for _ in range(n_t)]
    # Map each k-block to the list of t-subsets it contains
    subsets_of_block = [[] for _ in range(n_k)]
    
    # Build adjacency
    for j, B in enumerate(k_subsets):
        for S in itertools.combinations(B, t):
            i = t_index[S]
            blocks_of_subset[i].append(j)
            subsets_of_block[j].append(i)
            
    # Convert subsets_of_block to numpy arrays for faster indexing in some contexts?
    # Lists are fine for iteration.
    
    best_solution = None
    best_size = float('inf')
    
    # 2. LP Relaxation for initial weights
    duals = None
    try:
        if time.time() > deadline - 30:
            raise TimeoutError()
            
        rows = []
        cols = []
        for j in range(n_k):
            for i in subsets_of_block[j]:
                rows.append(i)
                cols.append(j)
        
        M = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_t, n_k))
        
        c = np.ones(n_k)
        A_ub = -M.T.tocsr()
        b_ub = -np.ones(n_t)
        bounds = [(0, 1)] * n_k
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', 
                     options={'presolve': True, 'time_limit': 10})
        
        if res.success and res.ineqlin is not None:
            duals = np.abs(res.ineqlin.marginals)
            if np.sum(duals) < 1e-10:
                duals = None
    except Exception:
        duals = None
        
    if duals is None:
        duals = np.ones(n_t)
        
    # Precompute block weights based on duals
    base_weights = np.zeros(n_k)
    for j in range(n_k):
        if subsets_of_block[j]:
            base_weights[j] = np.sum(duals[subsets_of_block[j]])
            
    # Helper to check if a solution is valid (should be, but for safety in local search)
    def is_valid(sol):
        covered = np.zeros(n_t, dtype=bool)
        for b_idx in sol:
            covered[subsets_of_block[b_idx]] = True
        return np.all(covered)

    def calc_cover_counts(sol):
        counts = np.zeros(n_t, dtype=np.int32)
        for b_idx in sol:
            counts[subsets_of_block[b_idx]] += 1
        return counts

    def local_search(sol):
        if not sol:
            return sol
        sol = sol.copy()
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            # 1. Deletion
            cover_count = calc_cover_counts(sol)
            new_sol = []
            for b_idx in sol:
                subs = subsets_of_block[b_idx]
                if np.all(cover_count[subs] > 1):
                    cover_count[subs] -= 1
                    changed = True
                else:
                    new_sol.append(b_idx)
            sol = new_sol
            
            # 2. 2-Swap
            # Try to replace two blocks with one
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                cover_count = calc_cover_counts(sol)
                sol_set = set(sol)
                n_sol = len(sol)
                
                for i in range(n_sol):
                    if time.time() > deadline - 5:
                        break
                    b1 = sol[i]
                    for j in range(i + 1, n_sol):
                        if time.time() > deadline - 5:
                            break
                        b2 = sol[j]
                        
                        # Find critical subsets: those with count 1 or 2 that become 0 if b1, b2 removed
                        # Actually, if count[s] == 1 and s in b1, removing b1 makes it 0.
                        # If count[s] == 2 and s in b1 and b2, removing both makes it 0.
                        critical = []
                        # Check subsets of b1
                        for s in subsets_of_block[b1]:
                            c = cover_count[s]
                            if c == 1:
                                # Must be in b1 only? If in b2, c would be at least 2? No, c is total count.
                                # If s in b1 and b2, and c=1, impossible.
                                # So if c=1, s is in exactly one block in sol. Since s in b1, it's only in b1.
                                # Removing b1 makes it 0.
                                critical.append(s)
                            elif c == 2:
                                # If s is in b2, removing both makes it 0.
                                if b2 in blocks_of_subset[s]:
                                    critical.append(s)
                        
                        # Check subsets of b2 that are not in b1
                        # We only need to check if they become 0.
                        # If s in b2, not in b1.
                        # If c==1, removing b2 makes it 0.
                        # If c==2, removing b2 leaves 1 (since not in b1). So not critical.
                        for s in subsets_of_block[b2]:
                            if b1 not in blocks_of_subset[s]:
                                if cover_count[s] == 1:
                                    critical.append(s)
                        
                        if not critical:
                            # Remove both
                            new_sol = [b for idx, b in enumerate(sol) if idx not in (i, j)]
                            sol = new_sol
                            improved = True
                            break
                        
                        # Try to find a block k covering all critical subsets
                        # Intersect blocks_of_subset[s] for s in critical
                        # Use the smallest list to start
                        if critical:
                            # Sort critical subsets by frequency of blocks covering them?
                            # Or just take the first one with smallest block list
                            min_len = float('inf')
                            min_s = -1
                            for s in critical:
                                if len(blocks_of_subset[s]) < min_len:
                                    min_len = len(blocks_of_subset[s])
                                    min_s = s
                            
                            candidates = set(blocks_of_subset[min_s])
                            for s in critical:
                                if s != min_s:
                                    candidates &= set(blocks_of_subset[s])
                                    if not candidates:
                                        break
                            
                            if candidates:
                                # Filter out b1, b2 and existing blocks?
                                # We are replacing b1, b2 with k. k should not be b1 or b2.
                                # k can be any other block.
                                candidates.discard(b1)
                                candidates.discard(b2)
                                # We don't need to discard other blocks in sol, because if k is in sol,
                                # we are effectively adding a duplicate? No, we remove b1, b2 and add k.
                                # If k is already in sol, we end up with same count?
                                # If k in sol, and we remove b1, b2, and add k (which is already there),
                                # we effectively just remove b1, b2.
                                # But we checked if we could remove b1, b2 without adding anything (critical empty).
                                # Here critical is not empty. So we need k to cover them.
                                # If k is in sol, it already covers them?
                                # If k in sol, and k covers critical, then critical subsets are covered by k.
                                # So removing b1, b2 leaves them covered by k.
                                # So we can just remove b1, b2.
                                # Wait, if k is in sol, and k covers critical, then the "critical" logic was slightly off?
                                # No, critical means count becomes 0 if b1, b2 removed.
                                # If k is in sol and covers critical, then count was at least 1 from k.
                                # So count would not become 0.
                                # So if k is in sol, it implies critical was empty?
                                # Not necessarily. If k is in sol, and k covers s, then cover_count[s] includes k.
                                # If cover_count[s] == 1, and k covers s, then k is the only block.
                                # But we are removing b1, b2. If k is not b1 or b2, it stays.
                                # So s remains covered.
                                # So if k is in sol (and k != b1, b2), then s is NOT critical.
                                # Thus, if critical is not empty, no block in sol (except b1, b2) can cover them.
                                # So candidates must be outside sol.
                                candidates -= sol_set
                                
                                if candidates:
                                    # Pick a candidate
                                    # Heuristic: pick one that covers the most other subsets?
                                    # Or just pick the first one.
                                    k = next(iter(candidates))
                                    new_sol = [b for idx, b in enumerate(sol) if idx not in (i, j)]
                                    new_sol.append(k)
                                    sol = new_sol
                                    improved = True
                                    break
                    if improved:
                        break
                if improved:
                    # Restart deletion
                    changed = True
            
            # 3. 3-Swap (Replace 3 with 2)
            # This is more expensive.
            # Try if time permits
            if time.time() < deadline - 10:
                improved = True
                while improved and time.time() < deadline - 5:
                    improved = False
                    cover_count = calc_cover_counts(sol)
                    sol_set = set(sol)
                    n_sol = len(sol)
                    
                    # Limit triple checks to avoid O(n^3) explosion if n_sol is large
                    # n_sol is around 150. 150^3 / 6 = 562,500. Might be too slow in Python.
                    # We will only try this if n_sol is small or we have lots of time.
                    if n_sol > 100:
                        break
                        
                    for i in range(n_sol):
                        if time.time() > deadline - 5:
                            break
                        b1 = sol[i]
                        for j in range(i + 1, n_sol):
                            if time.time() > deadline - 5:
                                break
                            b2 = sol[j]
                            for l in range(j + 1, n_sol):
                                if time.time() > deadline - 5:
                                    break
                                b3 = sol[l]
                                
                                # Find critical subsets for removing b1, b2, b3
                                critical = []
                                # We need to check all subsets in b1 U b2 U b3
                                # A subset s is critical if cover_count[s] - (s in b1) - (s in b2) - (s in b3) == 0
                                
                                # Efficiently check:
                                # For each s in b1:
                                #   c = cover_count[s]
                                #   in_b2 = s in b2?
                                #   in_b3 = s in b3?
                                #   new_c = c - 1 - in_b2 - in_b3
                                #   if new_c == 0: critical.append(s)
                                
                                # To speed up, we can precompute sets of subsets for b1, b2, b3?
                                # They are lists.
                                # Let's use sets for intersection logic if needed, but here we just iterate.
                                
                                set_b2 = set(subsets_of_block[b2])
                                set_b3 = set(subsets_of_block[b3])
                                
                                for s in subsets_of_block[b1]:
                                    c = cover_count[s]
                                    in_b2 = 1 if s in set_b2 else 0
                                    in_b3 = 1 if s in set_b3 else 0
                                    if c - 1 - in_b2 - in_b3 == 0:
                                        critical.append(s)
                                        
                                # Check subsets in b2 not in b1
                                set_b1 = set(subsets_of_block[b1])
                                for s in subsets_of_block[b2]:
                                    if s not in set_b1:
                                        c = cover_count[s]
                                        in_b3 = 1 if s in set_b3 else 0
                                        if c - 1 - in_b3 == 0:
                                            critical.append(s)
                                            
                                # Check subsets in b3 not in b1 or b2
                                for s in subsets_of_block[b3]:
                                    if s not in set_b1 and s not in set_b2:
                                        c = cover_count[s]
                                        if c - 1 == 0:
                                            critical.append(s)
                                            
                                if not critical:
                                    new_sol = [b for idx, b in enumerate(sol) if idx not in (i, j, l)]
                                    sol = new_sol
                                    improved = True
                                    break
                                    
                                # Try to find 2 blocks k1, k2 covering all critical
                                # This is harder.
                                # For each k1 in candidates (intersection of blocks_of_subset[s] for s in critical):
                                #   For each k2 in candidates:
                                #     Check if k1 U k2 covers critical.
                                
                                if critical:
                                    min_len = float('inf')
                                    min_s = -1
                                    for s in critical:
                                        if len(blocks_of_subset[s]) < min_len:
                                            min_len = len(blocks_of_subset[s])
                                            min_s = s
                                    
                                    candidates = set(blocks_of_subset[min_s])
                                    for s in critical:
                                        if s != min_s:
                                            candidates &= set(blocks_of_subset[s])
                                            if not candidates:
                                                break
                                    
                                    if candidates:
                                        candidates -= sol_set
                                        candidates.discard(b1)
                                        candidates.discard(b2)
                                        candidates.discard(b3)
                                        
                                        # If we have candidates, we need to find a pair (k1, k2) from candidates
                                        # such that union of subsets_of_block[k1] and subsets_of_block[k2] covers critical.
                                        # This is a set cover problem on a small set of critical subsets.
                                        # Since critical is small, we can just check pairs.
                                        cand_list = list(candidates)
                                        found_pair = False
                                        for idx1 in range(len(cand_list)):
                                            if time.time() > deadline - 5:
                                                break
                                            k1 = cand_list[idx1]
                                            # Check if k1 alone covers all?
                                            # If yes, we could have done 3->1?
                                            # But we are in 3->2 loop.
                                            # If k1 covers all, we can replace 3 with 1.
                                            # But 2-swap would have found 2->1?
                                            # 3->1 is possible if 3 blocks are redundant.
                                            # Let's just try to find 2 blocks.
                                            
                                            # Optimization: Check if k1 covers all critical
                                            covers_all = True
                                            for s in critical:
                                                if k1 not in blocks_of_subset[s]:
                                                    covers_all = False
                                                    break
                                            if covers_all:
                                                # Replace 3 with 1
                                                new_sol = [b for idx, b in enumerate(sol) if idx not in (i, j, l)]
                                                new_sol.append(k1)
                                                sol = new_sol
                                                improved = True
                                                found_pair = True
                                                break
                                            
                                            # Otherwise, find k2 that covers the rest
                                            # Rest = critical not covered by k1
                                            # We can just check if there exists k2 in candidates
                                            # such that k2 covers all s in critical where k1 doesn't.
                                            
                                            for idx2 in range(idx1 + 1, len(cand_list)):
                                                k2 = cand_list[idx2]
                                                ok = True
                                                for s in critical:
                                                    if k1 not in blocks_of_subset[s] and k2 not in blocks_of_subset[s]:
                                                        ok = False
                                                        break
                                                if ok:
                                                    new_sol = [b for idx, b in enumerate(sol) if idx not in (i, j, l)]
                                                    new_sol.append(k1)
                                                    new_sol.append(k2)
                                                    sol = new_sol
                                                    improved = True
                                                    found_pair = True
                                                    break
                                            if found_pair:
                                                break
                    if improved:
                        break
                if improved:
                    changed = True

        return sol

    # 3. Greedy Construction with Restarts
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Initialize weights
        block_weights = base_weights.copy()
        
        # Perturbation
        if restart > 0:
            noise = np.random.uniform(0, 0.05 * np.max(block_weights), n_k)
            block_weights += noise
            
        # Greedy Selection
        covered = np.zeros(n_t, dtype=bool)
        picked = np.zeros(n_k, dtype=bool)
        solution_blocks = []
        
        heap = [(-block_weights[j], j) for j in range(n_k) if block_weights[j] > 1e-12]
        heapq.heapify(heap)
        
        while heap:
            if time.time() > deadline - 10:
                break
                
            neg_w, j = heapq.heappop(heap)
            w = -neg_w
            
            if picked[j]:
                continue
            if w > block_weights[j] + 1e-9:
                continue
            if w <= 1e-12:
                break
                
            # Select block j
            picked[j] = True
            solution_blocks.append(j)
            
            # Update weights of other blocks
            for s_idx in subsets_of_block[j]:
                if not covered[s_idx]:
                    covered[s_idx] = True
                    y_s = duals[s_idx]
                    if y_s > 1e-15:
                        for b_prime in blocks_of_subset[s_idx]:
                            if not picked[b_prime]:
                                block_weights[b_prime] -= y_s
                                if block_weights[b_prime] > 1e-12:
                                    heapq.heappush(heap, (-block_weights[b_prime], b_prime))
                                
            if np.all(covered):
                break
                
        if np.all(covered) and len(solution_blocks) < best_size:
            best_size = len(solution_blocks)
            # Run local search
            local_sol = local_search(solution_blocks)
            if len(local_sol) < best_size:
                best_size = len(local_sol)
                best_solution = local_sol

    # Fallback if no solution found
    if best_solution is None:
        best_solution = []
        covered = np.zeros(n_t, dtype=bool)
        while not np.all(covered):
            best_b = -1
            max_new = -1
            for j in range(n_k):
                if not picked[j]:
                    new_cov = sum(1 for s in subsets_of_block[j] if not covered[s])
                    if new_cov > max_new:
                        max_new = new_cov
                        best_b = j
            if best_b == -1:
                break
            best_solution.append(best_b)
            for s in subsets_of_block[best_b]:
                covered[s] = True
        best_size = len(best_solution)
        # Run local search on fallback
        local_sol = local_search(best_solution)
        if len(local_sol) < best_size:
            best_size = len(local_sol)
            best_solution = local_sol

    return [list(k_subsets[idx]) for idx in best_solution]
