import numpy as np
import itertools
import time
import random
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

def construct(v, k, t):
    start_time = time.time()
    TIMEOUT = 145  # Safety margin before 150s kill

    # Precompute all t-subsets and k-blocks
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_id = {ts: i for i, ts in enumerate(all_tsubs)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Build incidence matrix: A[block, tsub] = 1 if tsub ⊆ block
    # Each block of size k covers C(k,t) t-subsets
    tsubs_per_block = len(list(itertools.combinations(range(k), t)))
    
    # Build sparse incidence matrix
    rows, cols = [], []
    for b_idx, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            rows.append(b_idx)
            cols.append(tsub_to_id[ts])
    
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_blocks, n_tsubs))
    
    # Block t-subset lists for fast access
    block_tsubs_list = []
    for b_idx, block in enumerate(all_blocks):
        block_tsubs_list.append([tsub_to_id[ts] for ts in itertools.combinations(block, t)])
    
    # Reverse index: tsub -> list of blocks containing it
    tsub_blocks = [[] for _ in range(n_tsubs)]
    for b_idx in range(n_blocks):
        for ts_id in block_tsubs_list[b_idx]:
            tsub_blocks[ts_id].append(b_idx)
    
    def time_left():
        return TIMEOUT - (time.time() - start_time)
    
    def solve_lp():
        """Solve the LP relaxation to get dual variables (difficulty weights)."""
        # Primal: min 1'x s.t. A'x >= 1, x >= 0
        # Dual:   max 1'y s.t. Ay <= 1, y >= 0
        # We solve the dual for speed (fewer constraints if n_tsubs < n_blocks)
        # Dual: max sum(y) s.t. A@y <= 1, y >= 0
        # Convert to standard form: min -sum(y) s.t. -A@y >= -1, y >= 0
        
        # Use scipy's linprog via milp for LP
        # Actually, let's use a simpler approach: scipy.optimize.linprog
        from scipy.optimize import linprog
        
        c = -np.ones(n_tsubs)  # minimize -sum(y)
        A_ub = A  # A@y <= 1
        b_ub = np.ones(n_blocks)
        bounds = [(0, None)] * n_tsubs
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                return res.x  # dual variables = difficulty weights
            else:
                return None
        except:
            return None
    
    def greedy_cover(weights=None, seed_order=None):
        """Greedy set cover with weighted scoring."""
        if weights is None:
            weights = np.ones(n_tsubs)
        
        uncovered = np.ones(n_tsubs, dtype=bool)
        chosen = []
        
        if seed_order is not None:
            tie_order = {b: i for i, b in enumerate(seed_order)}
        else:
            tie_order = None
        
        # Precompute weighted scores incrementally
        # Score of block b = sum of weights of uncovered tsubs it covers
        current_weights = weights.copy()
        
        while np.any(uncovered):
            if time_left() < 5:
                break
            
            # Compute scores: for each block, sum of weights of uncovered tsubs
            # A @ (weights * uncovered) gives this
            effective_weights = weights * uncovered
            scores = A @ effective_weights
            
            max_score = scores.max()
            if max_score < 1e-10:
                break
            
            best_candidates = np.where(scores == max_score)[0]
            
            if tie_order is not None and len(best_candidates) > 1:
                best_idx = min(best_candidates, key=lambda x: tie_order.get(x, float('inf')))
            else:
                best_idx = best_candidates[0]
            
            chosen.append(best_idx)
            uncovered[block_tsubs_list[best_idx]] = False
        
        return chosen
    
    def local_search_1exchange(chosen_indices):
        """1-exchange: remove one block, add one block covering all uniquely-covered tsubs."""
        chosen_set = set(chosen_indices)
        cover_counts = np.zeros(n_tsubs, dtype=int)
        for idx in chosen_indices:
            cover_counts[block_tsubs_list[idx]] += 1
        
        improved = True
        while improved and time_left() > 10:
            improved = False
            for idx in list(chosen_set):
                ts_list = block_tsubs_list[idx]
                unique_ts = [ts for ts in ts_list if cover_counts[ts] == 1]
                
                if len(unique_ts) == 0:
                    chosen_set.discard(idx)
                    cover_counts[ts_list] -= 1
                    improved = True
                    continue
                
                # Find a replacement covering all unique_ts
                cand_set = set()
                for ts in unique_ts:
                    cand_set.update(tsub_blocks[ts])
                
                for cand in cand_set:
                    if cand in chosen_set:
                        continue
                    cand_ts = set(block_tsubs_list[cand])
                    if all(ts in cand_ts for ts in unique_ts):
                        chosen_set.discard(idx)
                        cover_counts[ts_list] -= 1
                        chosen_set.add(cand)
                        cover_counts[block_tsubs_list[cand]] += 1
                        improved = True
                        break
                if improved:
                    break
        return list(chosen_set)
    
    def local_search_2exchange(chosen_indices):
        """2-for-1 exchange: remove two blocks, add one block covering all newly-uncovered tsubs."""
        chosen_set = set(chosen_indices)
        cover_counts = np.zeros(n_tsubs, dtype=int)
        for idx in chosen_indices:
            cover_counts[block_tsubs_list[idx]] += 1
        
        improved = True
        while improved and time_left() > 15:
            improved = False
            chosen_list = list(chosen_set)
            n = len(chosen_list)
            
            for i in range(n):
                if time_left() < 15:
                    break
                for j in range(i + 1, n):
                    if time_left() < 15:
                        break
                    idx1, idx2 = chosen_list[i], chosen_list[j]
                    
                    ts1 = block_tsubs_list[idx1]
                    ts2 = block_tsubs_list[idx2]
                    
                    # Temporarily remove both
                    cover_counts[ts1] -= 1
                    cover_counts[ts2] -= 1
                    
                    # Find newly uncovered tsubs
                    uncovered_ts = set()
                    for ts in ts1:
                        if cover_counts[ts] == 0:
                            uncovered_ts.add(ts)
                    for ts in ts2:
                        if cover_counts[ts] == 0:
                            uncovered_ts.add(ts)
                    
                    if len(uncovered_ts) == 0:
                        # Both redundant, remove one
                        chosen_set.discard(idx1)
                        cover_counts[ts1] += 1
                        improved = True
                        break
                    elif len(uncovered_ts) > 0:
                        # Find a block covering all uncovered_ts
                        cand_set = set()
                        for ts in uncovered_ts:
                            cand_set.update(tsub_blocks[ts])
                        
                        for cand in cand_set:
                            if cand in chosen_set:
                                continue
                            cand_ts = set(block_tsubs_list[cand])
                            if all(ts in cand_ts for ts in uncovered_ts):
                                chosen_set.discard(idx1)
                                chosen_set.discard(idx2)
                                chosen_set.add(cand)
                                cover_counts[ts1] += 1
                                cover_counts[ts2] += 1
                                cover_counts[block_tsubs_list[cand]] += 1
                                improved = True
                                break
                        if improved:
                            break
                    else:
                        cover_counts[ts1] += 1
                        cover_counts[ts2] += 1
                        break
                
                if improved:
                    break
        return list(chosen_set)
    
    def remove_redundant(chosen_indices):
        """Remove blocks that are fully covered by others."""
        chosen_set = set(chosen_indices)
        cover_counts = np.zeros(n_tsubs, dtype=int)
        for idx in chosen_indices:
            cover_counts[block_tsubs_list[idx]] += 1
        
        final = []
        for idx in list(chosen_set):
            ts_list = block_tsubs_list[idx]
            if np.min(cover_counts[ts_list]) > 1:
                cover_counts[ts_list] -= 1
            else:
                final.append(idx)
        return final
    
    # Phase 1: Solve LP to get difficulty weights
    weights = solve_lp()
    if weights is None or time_left() < 20:
        weights = np.ones(n_tsubs)
    
    # Phase 2: Weighted greedy
    initial = greedy_cover(weights=weights)
    
    # Phase 3: Local search
    current = local_search_1exchange(initial)
    current = local_search_2exchange(current)
    
    best_count = len(current)
    best_solution = current
    
    # Phase 4: Multi-start with perturbation
    remaining = time_left()
    if remaining > 20:
        n_restarts = min(30, int(remaining / 4))
        rng = random.Random(42)
        
        for r in range(n_restarts):
            if time_left() < 10:
                break
            
            # Perturb weights
            if r % 3 == 0:
                # Use LP weights
                w = weights.copy()
            elif r % 3 == 1:
                # Uniform
                w = np.ones(n_tsubs)
            else:
                # Random perturbation of LP weights
                w = weights * (1 + 0.5 * rng.random())
            
            # Random tie-breaking order
            block_order = list(range(n_blocks))
            rng.shuffle(block_order)
            
            candidate = greedy_cover(weights=w, seed_order=block_order)
            candidate = local_search_1exchange(candidate)
            candidate = local_search_2exchange(candidate)
            
            if len(candidate) < best_count:
                best_count = len(candidate)
                best_solution = candidate
    
    # Final cleanup
    best_solution = remove_redundant(best_solution)
    
    result_blocks = [list(all_blocks[i]) for i in best_solution]
    return result_blocks
