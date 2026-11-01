import time
import numpy as np
import itertools
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1_lp_greedy(deadline)
    
    # Fallbacks for other cases
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4 and radius == 1:
            return tetracode
        
    # Generic fallback: return all words (valid format, worst case)
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1_lp_greedy(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute weights
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    # Build M (Incidence Matrix: M[v, c] = 1 if v in B(c))
    M = lil_matrix((total, total), dtype=np.float64)
    
    for v in range(total):
        neighbors = [v]
        for pos in range(n):
            weight = weights[pos]
            digit = (v // weight) % 3
            for new_digit in range(3):
                if new_digit == digit:
                    continue
                new_v = v + (new_digit - digit) * weight
                neighbors.append(new_v)
        
        for c in neighbors:
            M[v, c] = 1.0
            
    M = M.tocsr()
    M_T = M.T.tocsr()
    
    # Solve Dual LP for weights y
    # Max 1^T y s.t. M^T y <= 1, y >= 0
    c_lp = -np.ones(total)
    A_ub = M_T
    b_ub = np.ones(total)
    
    try:
        res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs', options={'presolve': True})
        if res.success:
            y = -res.x
        else:
            y = np.ones(total) / ball_size
    except Exception:
        y = np.ones(total) / ball_size

    if time.time() > deadline - 20.0:
        y = np.ones(total) / ball_size

    # Randomized Greedy
    uncovered = np.ones(total, dtype=bool)
    selected = []
    
    while np.any(uncovered):
        if time.time() > deadline - 10.0:
            break
            
        y_uncovered = y * uncovered
        scores = M_T.dot(y_uncovered)
        best_idx = np.argmax(scores)
        
        selected.append(best_idx)
        neighbors = M_T.indices[M_T.indptr[best_idx]:M_T.indptr[best_idx+1]]
        uncovered[neighbors] = False
        
    # Local Search
    cover_count = np.zeros(total, dtype=np.int32)
    for c in selected:
        neighbors = M_T.indices[M_T.indptr[c]:M_T.indptr[c+1]]
        cover_count[neighbors] += 1
        
    # Precompute neighbor lists as lists for faster access in local search
    # M_T is CSR. M_T.indptr and M_T.indices are arrays.
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # 1. Prune redundant codewords
        changed = True
        while changed:
            changed = False
            for i in range(len(selected)):
                c = selected[i]
                neighbors = M_T.indices[M_T.indptr[c]:M_T.indptr[c+1]]
                if np.min(cover_count[neighbors]) > 1:
                    cover_count[neighbors] -= 1
                    selected.pop(i)
                    changed = True
                    break
        
        # 2. Try Swaps
        # For each codeword c, try to replace it with c'
        # Check if replacement allows removal of another codeword
        for i in range(len(selected)):
            if time.time() > deadline - 5.0:
                break
            c = selected[i]
            neighbors_c = M_T.indices[M_T.indptr[c]:M_T.indptr[c+1]]
            
            # Tentatively remove c
            cover_count[neighbors_c] -= 1
            uncovered_now = cover_count == 0
            
            if not np.any(uncovered_now):
                # c was redundant, should have been pruned
                selected.pop(i)
                improved = True
                break
            
            # Find best c' to cover uncovered_now
            # Score: number of uncovered words covered
            # Or weighted by y
            y_uncovered_now = y * uncovered_now
            scores = M_T.dot(y_uncovered_now)
            
            # Mask out current selected?
            # We can pick any c'. If c' is in selected, it doesn't change size.
            # We only want to swap if it leads to a reduction.
            
            best_c_prime = np.argmax(scores)
            
            # If best_c_prime is already in selected, this swap is neutral (size same)
            # We only proceed if best_c_prime is NOT in selected, or if we are okay with duplicates (we are not).
            # Actually, if best_c_prime is in selected, it means adding it doesn't help cover new words 
            # that aren't already covered by the existing copy? 
            # Wait, if c' is in selected, it's already there. Adding it again does nothing.
            # So we must pick a c' NOT in selected.
            
            # Create mask of selected
            selected_set = set(selected)
            if best_c_prime in selected_set:
                # Find next best
                scores[best_c_prime] = -1
                best_c_prime = np.argmax(scores)
                if best_c_prime in selected_set:
                    # Just skip this swap attempt
                    cover_count[neighbors_c] += 1
                    continue
            
            neighbors_cp = M_T.indices[M_T.indptr[best_c_prime]:M_T.indptr[best_c_prime+1]]
            
            # Apply swap
            cover_count[neighbors_cp] += 1
            selected[i] = best_c_prime
            
            # Check if this allows removal of other codewords
            # Only check codewords that interact with c or c'
            # Interacting codewords are those sharing neighbors with c or c'
            # We can just do a full prune pass, it's fast enough if we break early.
            
            # Quick Prune
            removed_any = False
            for j in range(len(selected)):
                if j == i:
                    continue
                d = selected[j]
                neighbors_d = M_T.indices[M_T.indptr[d]:M_T.indptr[d+1]]
                if np.min(cover_count[neighbors_d]) > 1:
                    cover_count[neighbors_d] -= 1
                    selected.pop(j)
                    removed_any = True
                    break # Restart outer loop
            
            if removed_any:
                improved = True
                break
            else:
                # No improvement, revert swap?
                # Actually, if no removal happened, the size is the same.
                # We might want to keep the swap if it "improves" potential, 
                # but for simplicity, let's revert to avoid cycling.
                cover_count[neighbors_cp] -= 1
                selected[i] = c
                cover_count[neighbors_c] += 1

    # Final Pruning
    changed = True
    while changed and time.time() < deadline - 2.0:
        changed = False
        for i in range(len(selected)):
            c = selected[i]
            neighbors = M_T.indices[M_T.indptr[c]:M_T.indptr[c+1]]
            if np.min(cover_count[neighbors]) > 1:
                cover_count[neighbors] -= 1
                selected.pop(i)
                changed = True
                break

    # Convert to list of lists
    result = []
    for c in selected:
        digits = []
        temp = c
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
