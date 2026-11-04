import numpy as np
import itertools
import time
import random

D = 11

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Generate all 4-subsets of range(11)
    subsets = list(itertools.combinations(range(D), 4))
    n = len(subsets)  # 330
    
    # Represent each subset as a bitmask
    masks = np.zeros(n, dtype=np.int64)
    for i, s in enumerate(subsets):
        m = 0
        for pos in s:
            m |= (1 << pos)
        masks[i] = m
    
    # Compute compatibility: two 4-sets are compatible if |S ∩ T| <= 2
    # Build compatibility matrix
    m1 = masks[:, np.newaxis]  # (n, 1)
    m2 = masks[np.newaxis, :]  # (1, n)
    and_matrix = m1 & m2
    
    # Popcount table for values 0 to 2^11-1
    popcount_table = np.zeros(2048, dtype=np.int32)
    for i in range(2048):
        popcount_table[i] = bin(i).count('1')
    
    popcount_matrix = popcount_table[and_matrix]  # (n, n)
    compatible = (popcount_matrix <= 2)  # (n, n) boolean matrix
    # Diagonal: popcount of 4-set is 4 > 2, so compatible[i][i] = False. Good.
    
    # Conflict graph: edge between i and j if NOT compatible
    conflict = ~compatible
    
    # Greedy with minimum remaining values (MRV) heuristic
    # For each candidate, score by number of compatible unselected supports it preserves
    
    def greedy_mrv():
        """Greedy algorithm with MRV heuristic for maximum independent set."""
        remaining = np.ones(n, dtype=bool)
        selected = np.zeros(n, dtype=bool)
        
        while remaining.any():
            candidates = np.where(remaining)[0]
            if len(candidates) == 0:
                break
            
            # For each candidate, count how many remaining supports it conflicts with
            # We want to pick the one that conflicts with the fewest remaining
            conflicts_with_remaining = np.sum(conflict[np.ix_(candidates, candidates)], axis=1)
            
            # Pick the one with minimum conflicts
            min_idx = candidates[np.argmin(conflicts_with_remaining)]
            selected[min_idx] = True
            remaining[min_idx] = False
            # Remove all that conflict with min_idx from remaining
            remaining[conflict[min_idx, :]] = False
        
        return selected
    
    # Multi-start with randomization
    best_selected = None
    best_score = 0
    
    rng = np.random.default_rng(42)
    
    # First, do a deterministic greedy
    sel = greedy_mrv()
    score = np.sum(sel)
    if score > best_score:
        best_selected = sel.copy()
        best_score = score
    
    # Local search refinement
    def is_valid(x):
        """Check if selection is valid (no conflicts)."""
        sel_idx = np.where(x)[0]
        if len(sel_idx) <= 1:
            return True
        sub_conflict = conflict[np.ix_(sel_idx, sel_idx)]
        n_sel = len(sel_idx)
        iu = np.triu_indices(n_sel, k=1)
        return not np.any(sub_conflict[iu])
    
    def local_search(current, max_iters=5000):
        """Local search: try adding compatible vertices, removing conflicting ones."""
        current = current.copy()
        current_score = np.sum(current)
        improved = True
        
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            
            # Try adding each unselected vertex that is compatible with all selected
            unselected = np.where(~current)[0]
            if len(unselected) == 0:
                break
            
            selected_idx = np.where(current)[0]
            if len(selected_idx) == 0:
                # All compatible, add all
                current[:] = True
                current_score = n
                break
            
            # For each unselected, check if compatible with all selected
            # compatible with all selected: conflict[u, selected].all() == False
            sub_conflict = conflict[np.ix_(unselected, selected_idx)]
            # A vertex u can be added if it has no conflicts with any selected
            can_add = np.all(~sub_conflict, axis=1)
            addable = unselected[can_add]
            
            if len(addable) > 0:
                # Add all addable at once (they're mutually compatible? Not necessarily)
                # Check mutual compatibility among addable
                if len(addable) > 1:
                    sub_sub_conflict = conflict[np.ix_(addable, addable)]
                    iu = np.triu_indices(len(addable), k=1)
                    mutually_compatible = not np.any(sub_sub_conflict[iu])
                    if mutually_compatible:
                        current[addable] = True
                        current_score += len(addable)
                        improved = True
                    else:
                        # Add one at a time
                        for u in addable:
                            if not np.any(conflict[u, current]):
                                current[u] = True
                                current_score += 1
                                improved = True
                else:
                    current[addable[0]] = True
                    current_score += 1
                    improved = True
            
            # If we couldn't add anything, try swap moves
            if not improved:
                # Try: remove one selected, add one unselected
                for s in selected_idx:
                    for u in unselected:
                        if np.any(conflict[u, current]) and not np.any(conflict[u, np.delete(current, s)]):
                            new_x = current.copy()
                            new_x[s] = False
                            new_x[u] = True
                            if is_valid(new_x) and np.sum(new_x) > current_score:
                                current = new_x
                                current_score = np.sum(current)
                                improved = True
                                break
                    if improved:
                        break
        
        return current, current_score
    
    # Multi-start: randomize the order of greedy
    n_restarts = 50
    for r in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        # Randomized greedy: shuffle candidate order
        order = np.arange(n)
        rng.shuffle(order)
        
        remaining = np.ones(n, dtype=bool)
        selected = np.zeros(n, dtype=bool)
        
        for idx in order:
            if not remaining[idx]:
                continue
            # Check if idx is compatible with all currently selected
            sel_idx = np.where(selected)[0]
            if len(sel_idx) == 0:
                selected[idx] = True
                remaining[conflict[idx, :]] = False
                remaining[idx] = False
            else:
                if not np.any(conflict[idx, sel_idx]):
                    selected[idx] = True
                    remaining[conflict[idx, :]] = False
                    remaining[idx] = False
        
        score = np.sum(selected)
        
        # Local search refinement
        refined, refined_score = local_search(selected, max_iters=2000)
        
        if refined_score > best_score:
            best_selected = refined.copy()
            best_score = refined_score
    
    # Final local search on best
    if best_selected is not None:
        final, final_score = local_search(best_selected, max_iters=10000)
        if final_score > best_score:
            best_selected = final.copy()
            best_score = final_score
    
    # Build vectors from selected supports
    sel_indices = np.where(best_selected)[0]
    
    vectors = []
    for idx in sel_indices:
        s = subsets[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            vectors.append(v)
    
    if len(vectors) == 0:
        # Fallback: return a single vector
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    return np.array(vectors, dtype=np.int64)
