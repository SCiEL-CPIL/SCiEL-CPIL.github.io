<<<<<<< SEARCH
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0  # Safety margin before 120s kill
=======
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 140.0  # Safety margin before 150s kill
>>>>>>> REPLACE
<<<<<<< SEARCH
    # 3. Reduction Pass - iterative with O(1) set membership
    def reduce_solution(sel):
        if not sel:
            return []
        
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in sel:
            for ti in block_tsets_np[bi]:
                cov[ti] += 1
                
        new_sel = []
        for bi in sel:
            essential = False
            for ti in block_tsets_np[bi]:
                if cov[ti] == 1:
                    essential = True
                    break
            if not essential:
                for ti in block_tsets_np[bi]:
                    cov[ti] -= 1
            else:
                new_sel.append(bi)
                
        if len(new_sel) < len(sel):
            return reduce_solution(new_sel)
        return new_sel
        
    selected = reduce_solution(selected)
=======
    # 3. Reduction Pass - iterative with O(1) set membership
    def reduce_solution(sel):
        if not sel:
            return []
        
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in sel:
            for ti in block_tsets_np[bi]:
                cov[ti] += 1
                
        new_sel = []
        for bi in sel:
            essential = False
            for ti in block_tsets_np[bi]:
                if cov[ti] == 1:
                    essential = True
                    break
            if not essential:
                for ti in block_tsets_np[bi]:
                    cov[ti] -= 1
            else:
                new_sel.append(bi)
                
        if len(new_sel) < len(sel):
            return reduce_solution(new_sel)
        return new_sel
        
    def swap_improve(sel, cov, max_iter=200):
        """Try 1-swap improvements: replace a block with another to enable further reductions."""
        if not sel:
            return sel, cov
        
        improved = True
        iters = 0
        while improved and iters < max_iter and time.time() < deadline - 10.0:
            improved = False
            iters += 1
            
            # Recompute coverage if needed
            cov_check = np.zeros(n_tsub, dtype=np.int32)
            for bi in sel:
                for ti in block_tsets_np[bi]:
                    cov_check[ti] += 1
            
            # Try each block for swap
            for idx, bi_old in enumerate(sel):
                if time.time() > deadline - 10.0:
                    return sel, cov
                
                # Find tsets uniquely covered by bi_old
                essential_tsets = [ti for ti in block_tsets_np[bi_old] if cov_check[ti] == 1]
                
                if not essential_tsets:
                    # Not essential, should have been removed by reduce_solution
                    continue
                
                # Find candidate blocks that cover all essential tsets
                candidates = set(tset_to_blocks[essential_tsets[0]])
                for ti in essential_tsets[1:]:
                    candidates &= set(tset_to_blocks[ti])
                candidates.discard(bi_old)
                
                if not candidates:
                    continue
                
                # Try each candidate
                for bi_new in candidates:
                    if bi_new in sel:
                        continue
                    
                    # Check if swap is valid
                    test_cov = cov_check.copy()
                    for ti in block_tsets_np[bi_old]:
                        test_cov[ti] -= 1
                    for ti in block_tsets_np[bi_new]:
                        test_cov[ti] += 1
                    
                    # Check if all tsets are still covered
                    if np.any(test_cov == 0):
                        continue
                    
                    # Perform swap
                    sel[idx] = bi_new
                    cov = test_cov
                    improved = True
                    break
                
                if improved:
                    break
        
        return sel, cov
    
    selected = reduce_solution(selected)
    cov_selected = compute_cov(selected)
    selected, cov_selected = swap_improve(selected, cov_selected)
    selected = reduce_solution(selected)
>>>>>>> REPLACE
<<<<<<< SEARCH
    # 4. Iterated Local Search (ILS)
    # State: selected (list of block indices), cov (coverage count array)
    
    def compute_cov(sel):
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in sel:
            for ti in block_tsets_np[bi]:
                cov[ti] += 1
        return cov
    
    best_selected = list(selected)
    best_size = len(selected)
    current_selected = list(selected)
    current_cov = compute_cov(current_selected)
    
    T = 5.0
    cooling_rate = 0.99
    T_min = 0.1
=======
    # 4. Iterated Local Search (ILS)
    # State: selected (list of block indices), cov (coverage count array)
    
    def compute_cov(sel):
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in sel:
            for ti in block_tsets_np[bi]:
                cov[ti] += 1
        return cov
    
    best_selected = list(selected)
    best_size = len(selected)
    current_selected = list(selected)
    current_cov = cov_selected.copy()
    
    T = 5.0
    cooling_rate = 0.99
    T_min = 0.1
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Helper: Greedy repair for uncovered tsets
    def repair(sel, cov, uncovered_tsets_list):
        """
        Greedily add blocks to cover uncovered_tsets_list.
        Returns (new_sel, new_cov, added_blocks)
        """
        if not uncovered_tsets_list:
            return sel, cov, []
        
        uncovered_set = set(uncovered_tsets_list)
        added = []
        
        while uncovered_set and time.time() < deadline - 10.0:
            # Find block that covers the most uncovered tsets
            best_bj = -1
            max_cover = 0
            candidates = set()
            for ti in uncovered_set:
                candidates.update(tset_to_blocks[ti])
            
            # Optimization: If candidates is huge, sampling might be faster, 
            # but for v=19, candidates is at most 75k. 
            # We can optimize by only checking blocks that cover *any* uncovered tset.
            for bj in candidates:
                cnt = 0
                for ti in block_tsets_np[bj]:
                    if ti in uncovered_set:
                        cnt += 1
                if cnt > max_cover:
                    max_cover = cnt
                    best_bj = bj
                    
            if best_bj == -1:
                break
            
            sel.append(best_bj)
            added.append(best_bj)
            for ti in block_tsets_np[best_bj]:
                cov[ti] += 1
                if ti in uncovered_set:
                    uncovered_set.remove(ti)
                    
        return sel, cov, added
=======
    # Helper: Greedy repair for uncovered tsets
    def repair(sel, cov, uncovered_tsets_list):
        """
        Greedily add blocks to cover uncovered_tsets_list.
        Returns (new_sel, new_cov, added_blocks)
        """
        if not uncovered_tsets_list:
            return sel, cov, []
        
        uncovered_set = set(uncovered_tsets_list)
        added = []
        
        while uncovered_set and time.time() < deadline - 10.0:
            # Find block that covers the most uncovered tsets
            best_bj = -1
            max_cover = 0
            candidates = set()
            for ti in uncovered_set:
                candidates.update(tset_to_blocks[ti])
            
            # Optimization: If candidates is huge, sampling might be faster, 
            # but for v=19, candidates is at most 75k. 
            # We can optimize by only checking blocks that cover *any* uncovered tset.
            for bj in candidates:
                cnt = 0
                for ti in block_tsets_np[bj]:
                    if ti in uncovered_set:
                        cnt += 1
                if cnt > max_cover:
                    max_cover = cnt
                    best_bj = bj
                    
            if best_bj == -1:
                break
            
            sel.append(best_bj)
            added.append(best_bj)
            for ti in block_tsets_np[best_bj]:
                cov[ti] += 1
                if ti in uncovered_set:
                    uncovered_set.remove(ti)
                    
        return sel, cov, added
>>>>>>> REPLACE
<<<<<<< SEARCH
    while time.time() < deadline - 5.0 and T > T_min:
        
        # --- Perturbation (Kick) ---
        # Remove 1 or 2 random blocks
        kick_size = 1 if random.random() < 0.7 else 2
        if len(current_selected) <= kick_size:
            T *= cooling_rate
            continue
            
        kick_indices = random.sample(range(len(current_selected)), kick_size)
        kick_blocks = [current_selected[i] for i in kick_indices]
        
        # Save state
        saved_selected = list(current_selected)
        saved_cov = current_cov.copy()
        
        # Remove kicked blocks
        for bi in kick_blocks:
            # Remove from list (careful with indices if multiple)
            # Since we sampled indices, we should remove by value or sort indices
            # Simpler: create a new list excluding these blocks
            pass
            
        # Let's do it properly:
        kick_set = set(kick_blocks)
        new_selected = [bi for bi in current_selected if bi not in kick_set]
        
        # Update coverage
        new_cov = saved_cov.copy()
        uncovered_tsets = []
        for bi in kick_blocks:
            for ti in block_tsets_np[bi]:
                new_cov[ti] -= 1
                if new_cov[ti] == 0:
                    uncovered_tsets.append(ti)
        
        # --- Repair ---
        new_selected, new_cov, _ = repair(new_selected, new_cov, uncovered_tsets)
        
        # --- Local Search (Refinement) ---
        # 1. Reduce
        new_selected = reduce_solution(new_selected)
        new_cov = compute_cov(new_selected)
        
        # 2. 1-Swap Local Search (limited iterations)
        # Try to swap a block with another to reduce size or enable reduction
        # We only do this if we have time
        if time.time() < deadline - 2.0:
            improved = True
            while improved and time.time() < deadline - 2.0:
                improved = False
                # Try removing each block and seeing if we can add a better one?
                # Simpler: Just try 1-swap that reduces size? 
                # Actually, reduce_solution already handles removals.
                # 1-swap is useful if a block is "essential" but swapping it 
                # for another non-essential block (relative to the rest) allows 
                # further reductions.
                
                # For now, rely on reduce_solution which is very effective.
                # If we want to be more aggressive, we can try random swaps.
                # But reduce_solution is O(N * K * T) which is fast enough.
                
                # Let's add a simple 1-swap: 
                # Pick a random block, try to replace it with a random other block 
                # if the new set is still a cover.
                if len(new_selected) > 1:
                    idx = random.randint(0, len(new_selected) - 1)
                    bi_old = new_selected[idx]
                    # Find candidates for bi_new
                    # We need bi_new to cover all tsets that bi_old uniquely covers?
                    # No, we just need the whole set to remain a cover.
                    # If we remove bi_old, some tsets might become uncovered.
                    # If so, bi_new must cover them.
                    
                    # Check if bi_old is essential
                    essential_tsets = [ti for ti in block_tsets_np[bi_old] if new_cov[ti] == 1]
                    
                    if essential_tsets:
                        # bi_new must cover all essential_tsets
                        candidates = set()
                        for ti in essential_tsets:
                            candidates.update(tset_to_blocks[ti])
                        candidates.discard(bi_old)
                        if candidates:
                            bi_new = random.choice(list(candidates))
                            if bi_new not in new_selected:
                                # Try the swap
                                test_selected = new_selected.copy()
                                test_selected[idx] = bi_new
                                test_cov = new_cov.copy()
                                for ti in block_tsets_np[bi_old]:
                                    test_cov[ti] -= 1
                                for ti in block_tsets_np[bi_new]:
                                    test_cov[ti] += 1
                                
                                # Check validity
                                valid = True
                                for ti in block_tsets_np[bi_old]:
                                    if test_cov[ti] == 0:
                                        valid = False
                                        break
                                if valid:
                                    # Accept swap, then reduce
                                    new_selected = test_selected
                                    new_cov = test_cov
                                    new_selected = reduce_solution(new_selected)
                                    new_cov = compute_cov(new_selected)
                                    improved = True
                    else:
                        # bi_old is not essential, so reduce_solution would have removed it.
                        # This case shouldn't happen after reduce_solution.
                        pass
                        
        # --- Acceptance ---
        delta = len(new_selected) - len(current_selected)
        
        if delta <= 0:
            # Better or equal
            current_selected = new_selected
            current_cov = new_cov
            if len(current_selected) < best_size:
                best_size = len(current_selected)
                best_selected = list(current_selected)
        else:
            # Worse, accept with probability exp(-delta / T)
            if random.random() < np.exp(-delta / T):
                current_selected = new_selected
                current_cov = new_cov
            else:
                # Reject, revert to current
                pass
                # current_selected and current_cov are already the old ones (we didn't modify them in place)
        
        T *= cooling_rate
=======
    while time.time() < deadline - 5.0 and T > T_min:
        
        # --- Perturbation (Kick) ---
        # Remove 1 or 2 random blocks
        kick_size = 1 if random.random() < 0.7 else 2
        if len(current_selected) <= kick_size:
            T *= cooling_rate
            continue
            
        kick_indices = random.sample(range(len(current_selected)), kick_size)
        kick_blocks = [current_selected[i] for i in kick_indices]
        
        # Save state
        saved_selected = list(current_selected)
        saved_cov = current_cov.copy()
        
        # Remove kicked blocks
        kick_set = set(kick_blocks)
        new_selected = [bi for bi in current_selected if bi not in kick_set]
        
        # Update coverage
        new_cov = saved_cov.copy()
        uncovered_tsets = []
        for bi in kick_blocks:
            for ti in block_tsets_np[bi]:
                new_cov[ti] -= 1
                if new_cov[ti] == 0:
                    uncovered_tsets.append(ti)
        
        # --- Repair ---
        new_selected, new_cov, _ = repair(new_selected, new_cov, uncovered_tsets)
        
        # --- Local Search (Refinement) ---
        # 1. Reduce
        new_selected = reduce_solution(new_selected)
        new_cov = compute_cov(new_selected)
        
        # 2. 1-Swap Local Search (limited iterations)
        if time.time() < deadline - 2.0:
            new_selected, new_cov = swap_improve(new_selected, new_cov, max_iter=50)
            new_selected = reduce_solution(new_selected)
            new_cov = compute_cov(new_selected)
                        
        # --- Acceptance ---
        delta = len(new_selected) - len(current_selected)
        
        if delta <= 0:
            # Better or equal
            current_selected = new_selected
            current_cov = new_cov
            if len(current_selected) < best_size:
                best_size = len(current_selected)
                best_selected = list(current_selected)
        else:
            # Worse, accept with probability exp(-delta / T)
            if random.random() < np.exp(-delta / T):
                current_selected = new_selected
                current_cov = new_cov
            else:
                # Reject, revert to current
                pass
                # current_selected and current_cov are already the old ones (we didn't modify them in place)
        
        T *= cooling_rate
>>>>>>> REPLACE
<<<<<<< SEARCH
    # 5. Final Polish
    best_selected = reduce_solution(best_selected)
    
    # Additional polish: try removing blocks in random order
    rng_polish = random.Random(12345)
    for _ in range(5):
        if time.time() > deadline - 2.0:
            break
        order = list(range(len(best_selected)))
        rng_polish.shuffle(order)
        new_sel = []
        cov_check = np.zeros(n_tsub, dtype=np.int32)
        for bi in best_selected:
            for ti in block_tsets_np[bi]:
                cov_check[ti] += 1
        
        for idx in order:
            bi = best_selected[idx]
            essential = False
            for ti in block_tsets_np[bi]:
                if cov_check[ti] == 1:
                    essential = True
                    break
            if not essential:
                for ti in block_tsets_np[bi]:
                    cov_check[ti] -= 1
            else:
                new_sel.append(bi)
        
        if len(new_sel) < len(best_selected):
            best_selected = new_sel
        else:
            break
    
    return [list(all_blocks[bi]) for bi in best_selected]
=======
    # 5. Final Polish
    best_selected = reduce_solution(best_selected)
    
    # Additional polish: try removing blocks in random order
    rng_polish = random.Random(12345)
    for _ in range(5):
        if time.time() > deadline - 2.0:
            break
        order = list(range(len(best_selected)))
        rng_polish.shuffle(order)
        new_sel = []
        cov_check = np.zeros(n_tsub, dtype=np.int32)
        for bi in best_selected:
            for ti in block_tsets_np[bi]:
                cov_check[ti] += 1
        
        for idx in order:
            bi = best_selected[idx]
            essential = False
            for ti in block_tsets_np[bi]:
                if cov_check[ti] == 1:
                    essential = True
                    break
            if not essential:
                for ti in block_tsets_np[bi]:
                    cov_check[ti] -= 1
            else:
                new_sel.append(bi)
        
        if len(new_sel) < len(best_selected):
            best_selected = new_sel
        else:
            break
    
    # Final swap improvement
    if time.time() < deadline - 1.0:
        cov_best = compute_cov(best_selected)
        best_selected, cov_best = swap_improve(best_selected, cov_best, max_iter=100)
        best_selected = reduce_solution(best_selected)
    
    return [list(all_blocks[bi]) for bi in best_selected]
>>>>>>> REPLACE
