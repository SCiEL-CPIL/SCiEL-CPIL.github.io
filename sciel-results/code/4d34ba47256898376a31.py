"""Matrix multiplication 3,3,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 5]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        rank = len(U[0])
        nm = n * m
        mp = m * p
        pn = p * n
        
        # Build the target tensor
        T = np.zeros((nm, mp, pn), dtype=int)
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    i = a * m + b
                    j = b * p + c
                    k = c * n + a
                    T[i, j, k] = 1
        
        # Reconstruct from decomposition
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # T_reconstructed = sum_c U[:,c] outer V[:,c] outer W[:,c]
        # This is expensive for large rank, but for our sizes it's fine
        T_rec = np.zeros((nm, mp, pn), dtype=float)
        for c in range(rank):
            T_rec += np.einsum('i,j,k->ijk', U_arr[:, c], V_arr[:, c], W_arr[:, c])
        
        # Check exactness (allow small floating point error)
        if not np.allclose(T_rec, T, atol=1e-9):
            return False
        return True
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower-rank decomposition
    # Strategy: Use alternating least squares with rational rounding
    
    def try_rank_als(r, max_iters=500, seed=42):
        """Try to find a rank-r decomposition using ALS with rational rounding."""
        if time.time() > deadline:
            return None
        
        nm = n * m
        mp = m * p
        pn = p * n
        
        # Build the tensor
        T = np.zeros((nm, mp, pn), dtype=float)
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    i = a * m + b
                    j = b * p + c
                    k = c * n + a
                    T[i, j, k] = 1
        
        np.random.seed(seed)
        
        best_err = float('inf')
        best_result = None
        
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            # Initialize
            U = np.random.randn(nm, r) * 0.5
            V = np.random.randn(mp, r) * 0.5
            W = np.random.randn(pn, r) * 0.5
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Update U
                for i in range(nm):
                    # U[i,c] = sum_{j,k} T[i,j,k] * V[j,c] * W[k,c] / sum_{j,k} V[j,c]^2 * W[k,c]^2
                    num = np.zeros(r)
                    den = np.zeros(r)
                    for j in range(mp):
                        for k in range(pn):
                            if T[i, j, k] != 0:
                                num += V[j, :] * W[k, :]
                                den += V[j, :] ** 2 * W[k, :] ** 2
                    mask = den > 1e-12
                    U[i, mask] = num[mask] / den[mask]
                
                # Update V
                for j in range(mp):
                    num = np.zeros(r)
                    den = np.zeros(r)
                    for i in range(nm):
                        for k in range(pn):
                            if T[i, j, k] != 0:
                                num += U[i, :] * W[k, :]
                                den += U[i, :] ** 2 * W[k, :] ** 2
                    mask = den > 1e-12
                    V[j, mask] = num[mask] / den[mask]
                
                # Update W
                for k in range(pn):
                    num = np.zeros(r)
                    den = np.zeros(r)
                    for i in range(nm):
                        for j in range(mp):
                            if T[i, j, k] != 0:
                                num += U[i, :] * V[j, :]
                                den += U[i, :] ** 2 * V[j, :] ** 2
                    mask = den > 1e-12
                    W[k, mask] = num[mask] / den[mask]
                
                # Compute error
                if it % 50 == 0 or it == max_iters - 1:
                    T_rec = np.zeros((nm, mp, pn), dtype=float)
                    for c in range(r):
                        T_rec += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                    err = np.max(np.abs(T_rec - T))
                    if err < best_err:
                        best_err = err
                        if err < 1e-8:
                            # Try to round to rationals
                            U_frac = [[Fraction(x).limit_denominator(10000) for x in U[i, :]] for i in range(nm)]
                            V_frac = [[Fraction(x).limit_denominator(10000) for x in V[j, :]] for j in range(mp)]
                            W_frac = [[Fraction(x).limit_denominator(10000) for x in W[k, :]] for k in range(pn)]
                            
                            # Convert to lists
                            U_list = [[float(f) for f in row] for row in U_frac]
                            V_list = [[float(f) for f in row] for row in V_frac]
                            W_list = [[float(f) for f in row] for row in W_frac]
                            
                            # Verify with fractions
                            T_rec2 = np.zeros((nm, mp, pn), dtype=float)
                            for c in range(r):
                                T_rec2 += np.einsum('i,j,k->ijk', 
                                    np.array([float(f) for f in U_frac[:, c]] if hasattr(U_frac, '__getitem__') else U_frac[i][c] for i in range(nm)) if False else np.array([float(U_frac[i][c]) for i in range(nm)]),
                                    np.array([float(V_frac[j][c]) for j in range(mp)]),
                                    np.array([float(W_frac[k][c]) for k in range(pn)]))
                            if np.max(np.abs(T_rec2 - T)) < 1e-8:
                                return {"U": [[float(f) for f in row] for row in U_frac],
                                        "V": [[float(f) for f in row] for row in V_frac],
                                        "W": [[float(f) for f in row] for row in W_frac]}
        
        return None
    
    # First, get the schoolbook as fallback
    result = schoolbook()
    
    if verify(result["U"], result["V"], result["W"]):
        best_rank = n * m * p
    else:
        raise ValueError("Schoolbook verification failed")
    
    # Try lower ranks
    # For (3,3,5), try ranks from 30 up to 44
    for r in range(30, best_rank):
        if time.time() > deadline:
            break
        
        res = try_rank_als(r, max_iters=200, seed=42 + r)
        if res is not None:
            if verify(res["U"], res["V"], res["W"]):
                result = res
                best_rank = r
                # Continue trying even lower ranks
                continue
    
    return result
# EVOLVE-BLOCK-END
