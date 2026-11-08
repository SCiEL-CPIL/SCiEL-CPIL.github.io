"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                expected = 1.0 if True else 0.0  # T[i,j,k]=1 for all valid (i,j,k)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(rank, n, m, p, seed=0, max_iters=500, tol=1e-6):
    """Try to find a rank-r decomposition using ALS."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n*m, rank) * 0.1
    V = rng.randn(m*p, rank) * 0.1
    W = rng.randn(p*n, rank) * 0.1
    
    for it in range(max_iters):
        # Update U
        for idx in range(n*m):
            i, j = divmod(idx, m)
            # U[idx, c] * sum_k V[j*p+k, c] * W[k*n+i, c] should contribute to T[i,j,k]
            # For fixed (i,j), we need: sum_c U[idx,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
            # This is: U[idx,:] @ (V[j*p:(j+1)*p,:] * W[0:(p*n):n, :].T) ... let me reindex
            # V[j*p+k, c] for k=0..p-1, W[k*n+i, c] for k=0..p-1
            # So the "target" vector for U[idx,:] is all ones of length p
            # And the "design" matrix has entries D[k,c] = V[j*p+k,c] * W[k*n+i,c]
            D = np.zeros((p, rank))
            for k in range(p):
                D[k, :] = V[j*p+k, :] * W[k*n+i, :]
            # Solve: U[idx,:] @ D.T = ones(p)
            # i.e., D @ U[idx,:] = ones(p)
            try:
                sol = np.linalg.lstsq(D, np.ones(p), rcond=None)[0]
                U[idx, :] = sol
            except:
                pass
        
        # Update V
        for idx in range(m*p):
            j, k = divmod(idx, p)
            D = np.zeros((n, rank))
            for i in range(n):
                D[i, :] = U[i*m+j, :] * W[k*n+i, :]
            try:
                sol = np.linalg.lstsq(D, np.ones(n), rcond=None)[0]
                V[idx, :] = sol
            except:
                pass
        
        # Update W
        for idx in range(p*n):
            k, i = divmod(idx, n)
            D = np.zeros((m, rank))
            for j in range(m):
                D[j, :] = U[i*m+j, :] * V[j*p+k, :]
            try:
                sol = np.linalg.lstsq(D, np.ones(m), rcond=None)[0]
                W[idx, :] = sol
            except:
                pass
        
        # Check convergence
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] for c in range(rank))
                    err += (val - 1.0)**2
        if err < tol:
            return U, V, W, True
    
    return U, V, W, False

def _round_to_rational(arr, max_denom=100):
    """Try to round entries to small rationals."""
    result = np.zeros_like(arr)
    for idx in np.ndindex(arr.shape):
        val = arr[idx]
        best_frac = Fraction(0, 1)
        best_diff = abs(val)
        for den in range(1, max_denom+1):
            num = round(val * den)
            frac = Fraction(num, den)
            diff = abs(float(frac) - val)
            if diff < best_diff:
                best_diff = diff
                best_frac = frac
        result[idx] = float(best_frac)
    return result

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Lower bound on rank: at least max(n*p, m*p, n*m) = 21
    # Try ranks from 21 upward
    for rank in range(21, min(rank_school, 45)):  # Try up to rank 44
        if time.time() > deadline - 30:
            break
        
        found = False
        for seed in range(20):
            if time.time() > deadline - 30:
                break
            
            U, V, W, success = _try_rank(rank, n, m, p, seed=seed, max_iters=300, tol=1e-8)
            if success:
                # Try to round to rationals
                U_r = _round_to_rational(U, max_denom=50)
                V_r = _round_to_rational(V, max_denom=50)
                W_r = _round_to_rational(W, max_denom=50)
                
                # Verify the rounded version
                if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                    best_rank = rank
                    best_result = {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
                    found = True
                    break
                else:
                    # Try with higher precision
                    U_r = _round_to_rational(U, max_denom=200)
                    V_r = _round_to_rational(V, max_denom=200)
                    W_r = _round_to_rational(W, max_denom=200)
                    if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                        best_rank = rank
                        best_result = {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
                        found = True
                        break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
