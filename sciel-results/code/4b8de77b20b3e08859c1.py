"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor T[i, j, k] = 1 for all valid i, j, k
    T = np.ones((n, m, p), dtype=float)
    
    # Fallback: Schoolbook decomposition (Rank = n * m * p)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_s, V_s, W_s, rank_s = schoolbook()
    best_result = (U_s, V_s, W_s, rank_s)
    
    # Function to verify a decomposition
    def verify(U, V, W, r):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err = abs(val - T[i, j, k])
                    if err > max_err:
                        max_err = err
        return max_err
    
    # Strategy: For a fixed rank r, we have variables U (nm x r), V (mp x r), W (pn x r)
    # Constraint: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i, j, k
    
    # We can reformulate this.
    # For fixed i, k, let u_i = [U[i*m+0, :], ..., U[i*m+m-1, :]] (m x r)
    # Let v_k = [V[0*p+k, :], ..., V[(m-1)*p+k, :]] (m x r)
    # Let w_ik = W[k*n+i, :] (1 x r)
    # The condition is: sum_j (u_i[j, :] * v_k[j, :]) * w_ik = 1
    # Let M_ik = u_i * v_k (element-wise product, m x r)
    # Then sum_j M_ik[j, :] * w_ik = 1
    # This is a linear system in w_ik given u_i and v_k.
    # Let A_ik be the matrix where A_ik[j, c] = u_i[j, c] * v_k[j, c]
    # Then A_ik @ w_ik = ones(m)
    
    # So if we fix U and V, we can solve for W for each (i, k) pair.
    # This reduces the problem to finding U and V such that the resulting linear systems for W have solutions.
    # Since the systems are independent for each (i, k), we just need A_ik to have rank >= m (or rather, the system to be consistent).
    # Since the RHS is ones(m), we need ones(m) to be in the column space of A_ik.
    
    # Even better: We can just try to find U, V such that the least squares solution for W gives small residual.
    # Actually, since we want EXACT decomposition, we need the residual to be 0.
    
    # Let's try a grid search or random search for U and V.
    # For small ranks, this might be feasible.
    
    # Lower bound on rank:
    # Rank >= max(n*p, m*p, n*m) ? No, that's not right.
    # Rank >= max(n, m, p) ? No.
    # For matrix multiplication, rank is at least max(n, m, p) is not true.
    # Actually, rank >= max(n*p, m*p, n*m) is not true either.
    # The rank of the matrix multiplication tensor is at least max(n, m, p) is a common lower bound but not always tight.
    # For 2x3x10, lower bound is at least 10? Or max(2,3,10)=10?
    # Let's start from r = 10.
    
    # Actually, a better lower bound:
    # The tensor has slices of size n x m, m x p, p x n.
    # The rank must be at least the max of the ranks of these slices? No.
    # But the rank must be at least the max of the dimensions of the slices?
    # Slice 1: n x m = 2 x 3. Rank 3.
    # Slice 2: m x p = 3 x 10. Rank 10.
    # Slice 3: p x n = 10 x 2. Rank 2.
    # So rank >= 10.
    
    # Let's try ranks from 10 upwards.
    
    for r in range(10, min(61, n*m*p + 1)):
        if time.time() > deadline - 15:
            break
            
        found = False
        # Number of random restarts
        # For small r, we need many restarts. For larger r, fewer.
        max_restarts = 20 if r < 15 else 5
        
        for trial in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize U and V
            # Use random values
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            
            # Solve for W
            W = np.zeros((pn, r))
            success = True
            
            for k in range(p):
                for i in range(n):
                    # Construct A_ki of shape (m, r)
                    # A_ki[j, c] = U[i*m + j, c] * V[j*p + k, c]
                    A_ki = np.zeros((m, r))
                    for j in range(m):
                        A_ki[j, :] = U[i*m + j, :] * V[j*p + k, :]
                    
                    # Solve A_ki @ w = ones(m)
                    try:
                        # Use least squares
                        w, residuals, rank_A, sv = np.linalg.lstsq(A_ki, np.ones(m), rcond=None)
                        W[k*n + i, :] = w
                    except np.linalg.LinAlgError:
                        success = False
                        break
                if not success:
                    break
            
            if not success:
                continue
                
            # Verify
            err = verify(U, V, W, r)
            if err < 1e-9:
                best_result = (U, V, W, r)
                found = True
                break
        
        if found:
            break

    U, V, W, rank = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
