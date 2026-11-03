"""Matrix multiplication 3,3,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 5]

import time
import numpy as np
from fractions import Fraction
import scipy.sparse as sp

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 5
    nm, mp, pn = n*m, m*p, p*n
    
    # Build target tensor
    T = np.zeros((nm, mp, pn), dtype=float)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T[i, j, k] = 1.0

    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        T_rec = np.einsum('ic,jc,kc->ijk', U_arr, V_arr, W_arr)
        
        if not np.allclose(T_rec, T, atol=1e-9):
            return False
        return True

    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(nm)]
        V = [[0.0] * rank for _ in range(mp)]
        W = [[0.0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try to find a lower-rank decomposition using vectorized ALS
    def try_rank_als(r, max_iters=100, seed=42):
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        
        # Initialize
        U = np.random.randn(nm, r) * 0.5
        V = np.random.randn(mp, r) * 0.5
        W = np.random.randn(pn, r) * 0.5
        
        # Pre-compute non-zero indices for faster updates
        # T is sparse, but for 3x3x5, dense is okay if we vectorize the sums
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Update U
            # U[i, c] = sum_{j,k} T[i,j,k] V[j,c] W[k,c] / sum_{j,k} T[i,j,k] V[j,c]^2 W[k,c]^2
            # Vectorized:
            # We can compute the numerator and denominator for all i at once using matrix multiplications
            # but T is a 3D tensor.
            # Let's use the loop version but with NumPy vectorization inside the inner loop
            
            # Update U
            for i in range(nm):
                # Get indices where T[i, :, :] is non-zero
                # This is still slow.
                pass # Fallback to simple loop if vectorization is too complex
            
            # Simple loop update (optimized)
            for i in range(nm):
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
            if it % 10 == 0 or it == max_iters - 1:
                T_rec = np.einsum('ic,jc,kc->ijk', U, V, W)
                err = np.max(np.abs(T_rec - T))
                if err < 1e-8:
                    # Try to round to rationals
                    U_frac = [[Fraction(x).limit_denominator(10000) for x in U[i, :]] for i in range(nm)]
                    V_frac = [[Fraction(x).limit_denominator(10000) for x in V[j, :]] for j in range(mp)]
                    W_frac = [[Fraction(x).limit_denominator(10000) for x in W[k, :]] for k in range(pn)]
                    
                    U_list = [[float(f) for f in row] for row in U_frac]
                    V_list = [[float(f) for f in row] for row in V_frac]
                    W_list = [[float(f) for f in row] for row in W_frac]
                    
                    if verify(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # First, get the schoolbook as fallback
    result = schoolbook()
    
    # Try lower ranks
    # For (3,3,5), try ranks from 40 down to 35
    for r in range(40, 34, -1):
        if time.time() > deadline - 5:
            break
        
        res = try_rank_als(r, max_iters=50, seed=42 + r)
        if res is not None:
            if verify(res["U"], res["V"], res["W"]):
                result = res
                # Continue trying even lower ranks
                continue
    
    return result
# EVOLVE-BLOCK-END
