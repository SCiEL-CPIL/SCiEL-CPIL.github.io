import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 110.0  # Safety margin
    
    # Precomputation
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

    def time_left():
        return deadline - time.time()

    # Algorithm X with dancing links style recursion
    def algorithm_x():
        """Find minimum covering using Algorithm X with dynamic variable ordering"""
        # State: which tsets are uncovered
        uncovered = [True] * n_tsub
        selected_blocks = []
        best_size = float('inf')
        
        # Precompute: for each tset, how many blocks cover it
        tset_block_count = [len(tset_to_blocks[ti]) for ti in range(n_tsub)]
        
        # We'll use a recursive backtracking with pruning
        # Key insight: at each step, pick the uncovered tset with fewest available blocks
        # This is the "minimum remaining values" heuristic
        
        def backtrack(depth, remaining_uncovered):
            nonlocal best_size
            
            if remaining_uncovered == 0:
                if depth < best_size:
                    best_size = depth
                    # Store best solution
                return True
            
            if depth >= best_size:
                return False
            
            if time_left() < 1.0:
                return False
            
            # Find the uncovered tset with the fewest available blocks
            min_count = float('inf')
            min_tset = -1
            for ti in range(n_tsub):
                if uncovered[ti]:
                    count = 0
                    for bi in tset_to_blocks[ti]:
                        if bi not in selected_set:
                            count += 1
                    if count < min_count:
                        min_count = count
                        min_tset = ti
                    if count == 0:
                        return False  # Dead end
            
            if min_tset == -1:
                return True  # All covered
            
            # Try each block that covers min_tset
            for bi in tset_to_blocks[min_tset]:
                if bi in selected_set:
                    continue
                
                # Select block bi
                selected_set.add(bi)
                selected_blocks.append(bi)
                
                # Update uncovered status
                new_remaining = remaining_uncovered
                for ti in block_tsets[bi]:
                    if uncovered[ti]:
                        uncovered[ti] = False
                        new_remaining -= 1
                
                # Recurse
                if backtrack(depth + 1, new_remaining):
                    # Unselect
                    selected_blocks.pop()
                    selected_set.discard(bi)
                    for ti in block_tsets[bi]:
                        if not uncovered[ti]:
                            uncovered[ti] = True
                        else:
                            # Was already uncovered, stays uncovered
                            pass
                    # Check if we found a complete solution
                    if new_remaining == 0 and depth + 1 < best_size:
                        best_size = depth + 1
                        best_solution = list(selected_blocks)
                        # Continue searching for better
                        continue
                    else:
                        # Restore and continue
                        for ti in block_tsets[bi]:
                            if uncovered[ti]:
                                pass
                            else:
                                # Need to check if this tset is covered by other selected blocks
                                # This is complex, so let's use a different approach
                                pass
                    break  # Found a solution, stop
                else:
                    # Unselect
                    selected_blocks.pop()
                    selected_set.discard(bi)
                    for ti in block_tsets[bi]:
                        if not uncovered[ti]:
                            # Check if covered by other blocks
                            covered = False
                            for other_bi in selected_set:
                                if ti in block_tsets[other_bi]:
                                    covered = True
                                    break
                            if not covered:
                                uncovered[ti] = True
        
        # Initialize
        selected_set = set()
        best_solution = []
        
        # Start with a greedy solution as upper bound
        greedy_sol = greedy_cover()
        best_size = len(greedy_sol)
        best_solution = list(greedy_sol)
        
        # Try to improve with Algorithm X
        if time_left() > 20.0:
            # Use a modified approach: try to find solutions smaller than current best
            # This is still exponential, but we have a good upper bound
            pass
        
        return best_solution

    def greedy_cover():
        uncovered = np.ones(n_tsub, dtype=bool)
        gain = np.zeros(n_blocks, dtype=np.int32)
        for bi in range(n_blocks):
            gain[bi] = len(block_tsets[bi])
            
        selected = []
        remaining = n_tsub
        
        while remaining > 0 and time_left() > 10.0:
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

    def reduce_solution(sel):
        if not sel: return []
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in sel:
            for ti in block_tsets[bi]:
                cov[ti] += 1
                
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

    def local_search(sel):
        if not sel: return sel
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in sel:
            for ti in block_tsets[bi]:
                cov[ti] += 1
                
        sel_set = set(sel)
        improved = True
        while improved and time_left() > 5.0:
            improved = False
            for i in range(len(sel)):
                if time_left() < 5.0: break
                bi_remove = sel[i]
                
                newly_uncovered = []
                for ti in block_tsets[bi_remove]:
                    if cov[ti] == 1:
                        newly_uncovered.append(ti)
                        
                if not newly_uncovered:
                    sel.pop(i)
                    for ti in block_tsets[bi_remove]:
                        cov[ti] -= 1
                    sel_set.discard(bi_remove)
                    improved = True
                    break
                    
                candidates = None
                for ti in newly_uncovered:
                    if candidates is None:
                        candidates = set(tset_to_blocks[ti])
                    else:
                        candidates.intersection_update(tset_to_blocks[ti])
                        
                if not candidates:
                    continue
                    
                candidates.difference_update(sel_set)
                
                if not candidates:
                    continue
                    
                best_bj = random.choice(list(candidates))
                
                sel[i] = best_bj
                sel_set.discard(bi_remove)
                sel_set.add(best_bj)
                
                for ti in block_tsets[bi_remove]:
                    cov[ti] -= 1
                for ti in block_tsets[best_bj]:
                    cov[ti] += 1
                    
                improved = True
                break
                
        return sel

    # Main: Use greedy as baseline, then try to improve with local search iterations
    best_sol = greedy_cover()
    best_sol = reduce_solution(best_sol)
    best_sol = local_search(best_sol)
    best_size = len(best_sol)
    
    while time_left() > 15.0:
        current = list(best_sol)
        if not current:
            break
            
        n_remove = random.randint(2, 4)
        if len(current) <= n_remove:
            current = []
        else:
            remove_idx = random.sample(range(len(current)), n_remove)
            for idx in sorted(remove_idx, reverse=True):
                current.pop(idx)
                
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in current:
            for ti in block_tsets[bi]:
                cov[ti] += 1
                
        uncovered_mask = (cov == 0)
        remaining = np.sum(uncovered_mask)
        
        if remaining == 0:
            new_sol = local_search(current)
            if len(new_sol) < best_size:
                best_sol = new_sol
                best_size = len(new_sol)
            continue
            
        current_set = set(current)
        gain = np.zeros(n_blocks, dtype=np.int32)
        for bi in range(n_blocks):
            if bi in current_set:
                continue
            cnt = 0
            for ti in block_tsets[bi]:
                if uncovered_mask[ti]:
                    cnt += 1
            gain[bi] = cnt
            
        while remaining > 0 and time_left() > 10.0:
            bi = int(np.argmax(gain))
            if gain[bi] == 0:
                break
            current.append(bi)
            current_set.add(bi)
            gain[bi] = 0
            
            for ti in block_tsets[bi]:
                if uncovered_mask[ti]:
                    uncovered_mask[ti] = False
                    remaining -= 1
                    for bj in tset_to_blocks[ti]:
                        if gain[bj] > 0:
                            gain[bj] -= 1
                            
        new_sol = reduce_solution(current)
        new_sol = local_search(new_sol)
        
        if len(new_sol) < best_size:
            best_sol = new_sol
            best_size = len(new_sol)
            
    return [list(all_blocks[bi]) for bi in best_sol]
