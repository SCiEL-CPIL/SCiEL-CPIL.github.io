import random
import numpy as np
from fractions import Fraction
import time

def construct():
    n, m, p = 2, 3, 7
    rank_school = n * m * p
    start_time = time.time()
    deadline = start_time + 140.0

    # Precompute valid indices
    valid_indices = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_indices.add((a, b, c))

    def verify_float(U, V, W, rank):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        T = np.zeros((n*m, m*p, p*n), dtype=float)
        for r in range(rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            T += np.einsum('a,b,c->abc', u, v, w)
        
        # Check valid
        for (a, b, c) in valid_indices:
            if abs(T[a, b, c] - 1.0) > 1e-6:
                return False
        
        # Check invalid
        mask = np.zeros((n*m, m*p, p*n), dtype=bool)
        for (a, b, c) in valid_indices:
            mask[a, b, c] = True
        invalid_vals = T[~mask]
        if np.any(np.abs(invalid_vals) > 1e-6):
            return False
        return True

    def verify_exact(U, V, W, rank):
        Uf = [[Fraction(U[i][r]) for r in range(rank)] for i in range(n*m)]
        Vf = [[Fraction(V[i][r]) for r in range(rank)] for i in range(m*p)]
        Wf = [[Fraction(W[i][r]) for r in range(rank)] for i in range(p*n)]
        
        for (a, b, c) in valid_indices:
            s = Fraction(0)
            for r in range(rank):
                s += Uf[a][r] * Vf[b][r] * Wf[c][r]
            if s != 1:
                return False
        
        # Check invalids
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    if (a, b, c) in valid_indices:
                        continue
                    s = Fraction(0)
                    for r in range(rank):
                        s += Uf[a][r] * Vf[b][r] * Wf[c][r]
                    if s != 0:
                        return False
        return True

    # Initialize Schoolbook
    U_sb = [[0]*rank_school for _ in range(n*m)]
    V_sb = [[0]*rank_school for _ in range(m*p)]
    W_sb = [[0]*rank_school for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j
                b = j*p+k
                c = k*n+i
                U_sb[a][col] = 1
                V_sb[b][col] = 1
                W_sb[c][col] = 1
                col += 1
                
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_school

    # Helper to round to nearest half-integer
    def round_to_half(x):
        return int(np.round(x * 2)) / 2.0

    # Strategy 1: HOSVD / Rank Peeling
    def try_hosvd_peel():
        if time.time() > deadline - 20:
            return None
        
        T = np.zeros((n*m, m*p, p*n), dtype=float)
        for (a, b, c) in valid_indices:
            T[a, b, c] = 1.0
            
        best_res = None
        # Try ranks from 30 down to 20
        for rank in range(30, 19, -1):
            if time.time() > deadline - 15:
                break
            
            # Attempt HOSVD truncation
            # U, S, Vt = np.linalg.svd(...) is not directly HOSVD.
            # We can use scipy if available, but standard library + numpy is safer.
            # Let's try a simple greedy peeling or random init + gradient descent?
            # Gradient descent on non-convex problem is risky.
            # Let's try a specific structured initialization based on "splitting" terms.
            
            # If HOSVD is too complex to implement from scratch quickly, 
            # we can try "Random Init + Local Search" or "Structured Init".
            
            # Let's try a "Split" strategy:
            # Combine pairs of schoolbook terms.
            # Term (i,j,k) and (i,j,k')?
            # A[i,j] is shared. B[j,k] and B[j,k'] are different.
            # C[i,k] and C[i,k'] are different.
            
            # Let's try a "Strassen-like" block approach if possible?
            # For 2x3 and 3x7, it's not a square power of 2.
            
            # Let's try a heuristic:
            # Randomly initialize U, V, W with 0, 0.5, 1
            # and try to refine? No, verification is strict.
            
            # Let's stick to Random Search but with a smarter distribution.
            # Or, try to construct a solution by solving a linear system for a subset of terms?
            
            # Given the time, let's just do a robust Random Search with more iterations
            # and a wider range of ranks, using the float filter.
            pass
            
        return best_res

    # Strategy 2: Enhanced Random Search
    # Use entries from {-2, -1, -0.5, 0, 0.5, 1, 2}
    # But maybe restrict to {-1, 0, 1} first as it's more likely to yield cancellations?
    
    # Let's try a specific "structured" random search.
    # Initialize with schoolbook, then try to merge columns?
    
    # Actually, let's just run a strong random search loop.
    
    entries = [-1, -0.5, 0, 0.5, 1] # Smaller set for better chance of exactness
    
    # Try ranks 41 down to 30
    for target_rank in range(41, 29, -1):
        if time.time() > deadline - 10.0:
            break
            
        # Number of iterations
        # We have ~130s left.
        # 1000 iterations per rank * 11 ranks = 11000 iterations.
        # Each iteration: float check (fast) + maybe exact check (slow).
        # If we find a candidate, we do exact check.
        
        iters = 5000
        for _ in range(iters):
            if time.time() > deadline - 5.0:
                break
                
            U = [[random.choice(entries) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice(entries) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.choice(entries) for _ in range(target_rank)] for _ in range(p*n)]
            
            if verify_float(U, V, W, target_rank):
                if verify_exact(U, V, W, target_rank):
                    best_rank = target_rank
                    best_U, best_V, best_W = U, V, W
                    # If we found a lower rank, we should try to go even lower?
                    # The loop structure already goes down.
                    # But we break out of the inner loop to save time?
                    # Yes, if we found rank R, we can try R-1 next.
                    break
        
        # If we found a solution at target_rank, continue to target_rank-1
        # If not, we might still try lower ranks? 
        # Usually if you can't find R, you can't find R-1.
        # But random search is stochastic. Maybe a different random seed works for R-1?
        # Unlikely but possible.
        # However, to be safe and fast, if we fail R, we probably should stop?
        # No, the current code breaks. Let's keep that behavior but maybe allow 1 more step?
        # Let's stick to breaking on failure to save time for verification of found solutions.
        
        if best_rank == target_rank:
            continue
        else:
            break

    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
