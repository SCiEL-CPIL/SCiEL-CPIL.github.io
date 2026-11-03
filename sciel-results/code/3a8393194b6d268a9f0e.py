import numpy as np
import itertools
import random
import time
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct(q, n, radius):
    total_words = q ** n
    
    # If the space is too large for MIP or radius is not 1, use heuristic fallback
    # The task is specifically 3,7,1. 
    # We assume small enough for MIP if total_words < 5000.
    
    if total_words > 5000:
        # Fallback heuristic
        # ... (simple greedy)
        pass
        
    # Build sparse incidence matrix
    # A[i, j] = 1 if d(i, j) <= radius
    
    # For radius 1, we can generate neighbors directly
    if radius == 1:
        rows = []
        cols = []
        data = []
        
        # Map word tuple to index? Or just use index directly?
        # Index calculation: sum(w[k] * q^(n-1-k))
        
        # We iterate over all i
        for i in range(total_words):
            # Convert i to word
            w = [0] * n
            temp = i
            for k in range(n-1, -1, -1):
                w[k] = temp % q
                temp //= q
            
            # Self
            rows.append(i)
            cols.append(i)
            data.append(1)
            
            # Neighbors
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        # Calculate new index
                        # index = w[0]*q^(n-1) + ... + w[n-1]
                        # We can compute this efficiently
                        # Or just create the list and convert
                        w_new = w[:]
                        w_new[pos] = new_val
                        j = 0
                        for k in range(n):
                            j = j * q + w_new[k]
                        
                        rows.append(i)
                        cols.append(j)
                        data.append(1)
        
        A = csr_matrix((data, (rows, cols)), shape=(total_words, total_words))
        
    else:
        # General radius (slower, but still possible for small N)
        rows = []
        cols = []
        data = []
        all_words = [list(p) for p in itertools.product(range(q), repeat=n)]
        
        for i in range(total_words):
            w_i = all_words[i]
            for j in range(total_words):
                w_j = all_words[j]
                dist = sum(1 for a, b in zip(w_i, w_j) if a != b)
                if dist <= radius:
                    rows.append(i)
                    cols.append(j)
                    data.append(1)
        
        A = csr_matrix((data, (rows, cols)), shape=(total_words, total_words))

    c = np.ones(total_words)
    cons = LinearConstraint(A, lb=np.ones(total_words), ub=np.inf)
    bounds = Bounds(lb=np.zeros(total_words), ub=np.ones(total_words))
    integrality = np.ones(total_words)
    
    try:
        res = milp(c=c, constraints=cons, bounds=bounds, integrality=integrality)
        if res.status == 0:
            selected_indices = np.where(res.x > 0.5)[0]
            all_words = [list(p) for p in itertools.product(range(q), repeat=n)]
            return [all_words[i] for i in selected_indices]
    except Exception:
        pass
    
    # Fallback if MIP fails
    all_words = [list(p) for p in itertools.product(range(q), repeat=n)]
    # Simple Greedy
    balls = [0] * total_words
    if radius == 1:
        for i, w in enumerate(all_words):
            mask = 1 << i
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        j = 0
                        w_new = w[:]
                        w_new[pos] = new_val
                        for k in range(n):
                            j = j * q + w_new[k]
                        mask |= (1 << j)
            balls[i] = mask
    else:
        for i, w in enumerate(all_words):
            mask = 0
            for j, v in enumerate(all_words):
                if sum(1 for a, b in zip(w, v) if a != b) <= radius:
                    mask |= (1 << j)
            balls[i] = mask
            
    all_ones_mask = (1 << total_words) - 1
    remaining = all_ones_mask
    selected = []
    while remaining:
        best_idx = -1
        max_cover = 0
        for i in range(total_words):
            c_val = (balls[i] & remaining).bit_count()
            if c_val > max_cover:
                max_cover = c_val
                best_idx = i
        if best_idx == -1:
            break
        selected.append(best_idx)
        remaining &= ~balls[best_idx]
        
    return [all_words[i] for i in selected]
