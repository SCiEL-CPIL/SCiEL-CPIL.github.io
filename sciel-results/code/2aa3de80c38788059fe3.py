import numpy as np
import time

def solve(a, b):
    n = len(a)
    # Convert to float64 for faster arithmetic and to avoid integer overflow issues in intermediate sums
    # though tai256c values are small enough for int64, float64 is safer for delta calculations
    A = np.array(a, dtype=np.float64)
    B = np.array(b, dtype=np.float64)
    
    # Ensure symmetry if needed? QAPLIB tai256c is symmetric.
    # A = (A + A.T) / 2.0
    # B = (B + B.T) / 2.0
    
    # Initial solution: Rank matching
    heavy = np.argsort(-A.sum(axis=1), kind="stable")
    central = np.argsort(B.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    def calc_cost(p):
        return np.sum(A * B[np.ix_(p, p)])
    
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Precompute A rows for faster access? A[i] is already a view.
    
    while time.time() < deadline:
        # 2-Opt Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                Ai = A[i]
                pi_loc = p[i]
                Bi_row = B[pi_loc]
                
                for j in range(i + 1, n):
                    # Calculate Delta
                    # W = A[i] - A[j]
                    # D = B[p[j]] - B[p[i]]
                    # S = dot(W, D)
                    # Term_i = (A[i,i] - A[j,i]) * (B[p[j], p[i]] - B[p[i], p[i]])
                    # Term_j = (A[i,j] - A[j,j]) * (B[p[j], p[j]] - B[p[i], p[j]])
                    # Cross = (A[i,j] - A[j,i]) * (B[p[j], p[i]] - B[p[i], p[j]])
                    
                    Aj = A[j]
                    pj_loc = p[j]
                    Bj_row = B[pj_loc]
                    
                    W = Ai - Aj
                    D = Bj_row - Bi_row
                    S = np.dot(W, D)
                    
                    # Subtract k=i term
                    # W[i] = A[i,i] - A[j,i]
                    # D[i] = B[p[j], p[i]] - B[p[i], p[i]]
                    term_i = W[i] * (Bj_row[pi_loc] - Bi_row[pi_loc])
                    
                    # Subtract k=j term
                    # W[j] = A[i,j] - A[j,j]
                    # D[j] = B[p[j], p[j]] - B[p[i], p[j]]
                    term_j = W[j] * (Bj_row[pj_loc] - Bi_row[pj_loc])
                    
                    # Cross term
                    cross = (Ai[j] - Aj[i]) * (Bj_row[pi_loc] - Bi_row[pj_loc])
                    
                    delta = S - term_i - term_j + cross
                    
                    if delta < -1e-9: # Threshold to avoid floating point noise
                        # Swap
                        p[i], p[j] = pj_loc, pi_loc
                        current_cost += delta
                        improved = True
                        # Update cached values if we were keeping them, but here we recalc rows
                        # Since we swapped p[i] and p[j], the rows for i and j change
                        # But we are in the inner loop for fixed i, j.
                        # The next j will use the new p[i] (which is old p[j])?
                        # No, p[i] is now pj_loc.
                        # We should update Bi_row and pi_loc for the next j?
                        # Yes.
                        pi_loc = p[i]
                        Bi_row = B[pi_loc]
                        # Ai is unchanged.
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        # Perturbation: Swap 2 random pairs
        if time.time() >= deadline:
            break
            
        idx1 = np.random.randint(0, n)
        idx2 = np.random.randint(0, n)
        if idx1 == idx2:
            idx2 = (idx2 + 1) % n
        p[idx1], p[idx2] = p[idx2], p[idx1]
        
        idx3 = np.random.randint(0, n)
        idx4 = np.random.randint(0, n)
        if idx3 == idx4:
            idx4 = (idx4 + 1) % n
        p[idx3], p[idx4] = p[idx4], p[idx3]
        
        # Recalculate cost after perturbation
        current_cost = calc_cost(p)

    return best_p.tolist()
