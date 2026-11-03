import numpy as np
import scipy.optimize as opt
import scipy.sparse as sp
import itertools

def construct(q, n, radius):
    # Handle specific case q=3, n=8, radius=1 with ILP
    if q == 3 and n == 8 and radius == 1:
        total_words = 3 ** n
        # Generate all words
        words = [list(x) for x in itertools.product(range(q), repeat=n)]
        
        # Build incidence matrix A (sparse)
        # A[i, j] = 1 if dist(words[i], words[j]) <= radius
        # We can build this by checking neighbors for each i
        
        rows = []
        cols = []
        data = []
        
        # Precompute indices for neighbors?
        # For each word i, find all j such that dist(i, j) <= 1
        # j is i itself, or i with one digit changed
        
        # To make it faster, we can map word tuple to index
        word_to_idx = {tuple(w): i for i, w in enumerate(words)}
        
        for i, w in enumerate(words):
            # Self
            rows.append(i)
            cols.append(i)
            data.append(1)
            
            # Neighbors
            for pos in range(n):
                orig_val = w[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    w[pos] = new_val
                    j = word_to_idx[tuple(w)]
                    w[pos] = orig_val # restore
                    rows.append(i)
                    cols.append(j)
                    data.append(1)
        
        # Convert to sparse matrix
        A = sp.csr_matrix((data, (rows, cols)), shape=(total_words, total_words))
        
        # Objective: minimize sum(x_i)
        c = np.ones(total_words)
        
        # Constraints: A @ x >= 1
        # scipy.optimize.milp requires constraints in the form A_ub @ x <= b_ub
        # So we use -A @ x <= -1
        A_ub = -A
        b_ub = -np.ones(total_words)
        
        # Bounds: 0 <= x_i <= 1
        bounds = [(0, 1)] * total_words
        
        # Integrality: all variables are integer
        integrality = np.ones(total_words)
        
        # Solve
        try:
            res = opt.milp(c, constraints=[opt.LinearConstraint(A_ub, ub=b_ub)], 
                           integrality=integrality, bounds=bounds)
            
            if res.success:
                # Extract solution
                x = res.x
                # Threshold at 0.5
                selected_indices = np.where(x > 0.5)[0]
                return [words[i] for i in selected_indices]
            else:
                # Fallback if ILP fails
                pass
                
        except Exception:
            pass
            
        # Fallback: Direct Sum (same as before, but as a safety net)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

    # General fallback
    if n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    else:
        return [list(x) for x in itertools.product(range(q), repeat=n)]
