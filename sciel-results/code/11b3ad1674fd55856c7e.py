import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0

    # 1. Precomputation
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsub = len(all_tsubsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # block_tsets[bi] = list of tset indices covered by block bi
    block_tsets = []
    for b in all_blocks:
        ts_indices = []
        for ts in itertools.combinations(b, t):
            ts_indices.append(tset_index[ts])
        block_tsets.append(ts_indices)
        
    # tset_to_blocks[ti] = list of block indices covering tset ti
    tset_to_blocks = [[] for _ in range(n_tsub)]
    for bi, ts_list in enumerate(block_tsets):
        for ti in ts_list:
            tset_to_blocks[ti].append(bi)

    # Precompute block_tsets as sets for fast subset checks
    block_tsets_set = [set(ts_list) for ts_list in block_tsets]

    # Helper: Compute coverage counts
    def compute_cov(sel):
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in sel:
            for ti in block_tsets[bi]:
                cov[ti] += 1
        return cov

    # Helper: Reduce solution (remove redundant blocks)
    def reduce_solution(sel):
        if not sel: return []
        cov = compute_cov(sel)
        
        new_sel = []
        for bi in sel:
            essential = False
            for ti in block_tsets[bi]:
                if cov[ti] == 1:
                    essential = True
                    break
            if not essential:
                for ti in block_tsets[bi]:
                    cov[ti] -= 1
            else:
                new_sel.append(bi)
                
        if len(new_sel) < len(sel):
            return reduce_solution(new_sel)
        return new_sel

    def greedy_cover(randomize=False):
        """Greedy set cover with optional randomization in tie-breaking."""
        uncovered = np.ones(n_tsub, dtype=bool)
        gain = np.zeros(n_blocks, dtype=np.int32)
        for bi in range(n_blocks):
            gain[bi] = len(block_tsets[bi])
            
        selected = []
        remaining = n_tsub
        
        while remaining > 0:
            if time.time() > deadline - 10.0:
                break
            if randomize:
                # Add small random perturbation to break ties
                mask = gain > 0
                if not np.any(mask):
                    break
                # Among max-gain blocks, pick randomly
                max_gain = gain[mask].max()
                candidates = np.where(mask & (gain == max_gain))[0]
                bi = int(random.choice(candidates))
            else:
                bi = int(np.argmax(gain))
            if gain[bi] == 0:
                break
            selected.append(bi)
            gain[bi] = 0
            
            for ti in block_tsets[bi]:
                if uncovered[ti]:
                    uncovered[ti] = False
                    remaining -= 1
                    for bj in tset_to_blocks[ti]:
                        if gain[bj] > 0:
                            gain[bj] -= 1
        return selected

    def local_search(sel, max_iter=200):
        """Local search: 1-for-1 swaps and 1-for-2 swaps to reduce block count."""
        sel = list(sel)
        sel_set = set(sel)
        cov = compute_cov(sel)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            
            # First, remove any redundant blocks
            new_sel = []
            for bi in sel:
                essential = False
                for ti in block_tsets[bi]:
                    if cov[ti] == 1:
                        essential = True
                        break
                if not essential:
                    for ti in block_tsets[bi]:
                        cov[ti] -= 1
                else:
                    new_sel.append(bi)
            
            if len(new_sel) < len(sel):
                sel = new_sel
                sel_set = set(sel)
                cov = compute_cov(sel)
                improved = True
                continue
            
            # Compute unique t-sets for each selected block
            unique_sets = {}
            for bi in sel:
                u = frozenset(ti for ti in block_tsets[bi] if cov[ti] == 1)
                unique_sets[bi] = u
            
            # Try 1-for-2 swap: find b_new not in sel that covers unique sets of 2+ blocks
            # Group by unique set size to prioritize small ones
            sorted_blocks = sorted(sel, key=lambda bi: len(unique_sets[bi]))
            
            found_move = False
            for i in range(len(sorted_blocks)):
                if time.time() > deadline - 5.0:
                    break
                bi = sorted_blocks[i]
                u_i = unique_sets[bi]
                if len(u_i) > 4:  # Skip large unique sets for 1-for-2
                    continue
                
                for j in range(i + 1, len(sorted_blocks)):
                    bj = sorted_blocks[j]
                    u_j = unique_sets[bj]
                    if len(u_j) > 4:
                        continue
                    
                    target = u_i | u_j
                    if not target:
                        continue
                    
                    # Find blocks covering all t in target
                    first_t = next(iter(target))
                    candidates = set(tset_to_blocks[first_t])
                    for t in list(target)[1:]:
                        candidates.intersection_update(tset_to_blocks[t])
                        if not candidates:
                            break
                    
                    if not candidates:
                        continue
                    
                    candidates.difference_update(sel_set)
                    if not candidates:
                        continue
                    
                    # Pick a candidate
                    b_new = next(iter(candidates))
                    
                    # Verify it makes both bi and bj redundant
                    # Check if b_new covers all unique t-sets of bi and bj
                    new_block_tsets = block_tsets_set[b_new]
                    if not (u_i <= new_block_tsets and u_j <= new_block_tsets):
                        continue
                    
                    # Check how many other blocks become redundant
                    redundant_count = 2  # bi and bj
                    for bk in sel:
                        if bk == bi or bk == bj:
                            continue
                        u_k = unique_sets[bk]
                        if u_k and u_k <= new_block_tsets:
                            redundant_count += 1
                    
                    # Net gain: remove redundant_count blocks, add 1
                    if redundant_count > 1:
                        # Apply the move
                        sel.remove(bi)
                        sel.remove(bj)
                        # Remove other redundant blocks
                        for bk in list(sel):
                            if bk != b_new and unique_sets[bk] and unique_sets[bk] <= new_block_tsets:
                                sel.remove(bk)
                        sel.append(b_new)
                        sel_set = set(sel)
                        cov = compute_cov(sel)
                        improved = True
                        found_move = True
                        break
                
                if found_move:
                    break
            
            if not found_move:
                # Try 1-for-1 swap: replace bi with b_new that covers all unique t-sets of bi
                # and also covers some unique t-sets of another block
                for bi in sel:
                    if time.time() > deadline - 5.0:
                        break
                    u_i = unique_sets[bi]
                    if not u_i:
                        continue
                    if len(u_i) > 5:
                        continue
                    
                    # Find blocks covering all t in u_i
                    first_t = next(iter(u_i))
                    candidates = set(tset_to_blocks[first_t])
                    for t in list(u_i)[1:]:
                        candidates.intersection_update(tset_to_blocks[t])
                        if not candidates:
                            break
                    
                    if not candidates:
                        continue
                    
                    # Remove bi from consideration (we're replacing it)
                    candidates.discard(bi)
                    
                    if not candidates:
                        continue
                    
                    # Pick a candidate that also covers unique t-sets of another block
                    best_b_new = None
                    best_extra = 0
                    for b_new in list(candidates)[:20]:  # Limit candidates
                        extra = 0
                        for bk in sel:
                            if bk == bi:
                                continue
                            u_k = unique_sets[bk]
                            if u_k and u_k <= block_tsets_set[b_new]:
                                extra += 1
                        if extra > best_extra:
                            best_extra = extra
                            best_b_new = b_new
                    
                    if best_b_new is not None and best_extra > 0:
                        # Apply: remove bi and any blocks made redundant, add best_b_new
                        redundant = [bi]
                        for bk in sel:
                            if bk == bi:
                                continue
                            u_k = unique_sets[bk]
                            if u_k and u_k <= block_tsets_set[best_b_new]:
                                redundant.append(bk)
                        
                        for bk in redundant:
                            sel.remove(bk)
                        sel.append(best_b_new)
                        sel_set = set(sel)
                        cov = compute_cov(sel)
                        improved = True
                        break
            
            if not improved:
                break
        
        return sel

    # Main: Multiple greedy passes with local search
    best_sol = None
    best_size = float('inf')
    
    # First greedy pass (deterministic)
    sol = greedy_cover(randomize=False)
    sol = reduce_solution(sol)
    sol = local_search(sol)
    sol = reduce_solution(sol)
    
    if len(sol) < best_size:
        best_size = len(sol)
        best_sol = list(sol)
    
    # Additional randomized greedy passes
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        random.seed(42 + r)
        sol = greedy_cover(randomize=True)
        sol = reduce_solution(sol)
        sol = local_search(sol, max_iter=100)
        sol = reduce_solution(sol)
        
        if len(sol) < best_size:
            best_size = len(sol)
            best_sol = list(sol)
    
    # Final local search on best solution with more iterations
    if best_sol is not None and time.time() < deadline - 5.0:
        best_sol = local_search(best_sol, max_iter=300)
        best_sol = reduce_solution(best_sol)
    
    if best_sol is None:
        # Fallback: return all blocks (should not happen)
        best_sol = list(range(n_blocks))
    
    return [list(all_blocks[bi]) for bi in best_sol]
