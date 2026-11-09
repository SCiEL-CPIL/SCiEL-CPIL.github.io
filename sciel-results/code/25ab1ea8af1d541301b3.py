"""Search for a low-rank exact tensor decomposition of the 3x4 by 4x4 matmul."""
import time
import random
import itertools
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 4]
DEADLINE = time.time() + 140.0  # safety margin under the 150s kill

def _verify(U, V, W):
    """Check whether factor matrices exactly decompose the matmul tensor.
    U: (n*m, r), V: (m*p, r), W: (p*n, r).
    Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1.
    """
    n, m, p = DIMENSIONS
    if len(U) == 0:
        return False
    r = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Reshape U to (n,m,r), V to (m,p,r), W to (p,n,r)
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)

def _schoolbook():
    """Return the schoolbook rank-48 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _als_optimize_fast(U, V, W, n, m, p, max_iter=200, tol=1e-12):
    """Faster ALS using pre-computed matrices."""
    r = U.shape[1]
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 10:
            break
            
        # Update U
        # A_U[i, j, k, c] = V3[j, k, c] * W3[k, i, c]
        # We want U3[i, j, :] to satisfy sum_c U3[i, j, c] A_U[i, j, k, c] = 1 for all k
        # This is a linear system for each (i, j).
        # Instead of looping, we can batch if we reshape, but the RHS is 1 for all k.
        # Let's stick to loops for clarity but optimize inner ops.
        
        # Precompute V3 * W3 terms?
        # V3 is (m, p, r), W3 is (p, n, r)
        # We need for each i, j: M_kc = sum_c V3[j, k, c] W3[k, i, c] ? No.
        # The term is product of V and W for the SAME c.
        # So A_U[i, j, k, c] = V3[j, k, c] * W3[k, i, c].
        
        # Vectorize the construction of A_U?
        # V3[j, :, :] * W3[:, i, :] -> (p, r) for fixed j, i.
        
        for i in range(n):
            for j in range(m):
                # A_U[i, j] is (p, r)
                # V3[j] is (p, r)
                # W3[:, i, :] is (p, r)
                A_U_ij = V3[j] * W3[:, i, :]
                b = np.ones(p)
                try:
                    U[i*m+j], _, _, _ = np.linalg.lstsq(A_U_ij, b, rcond=None)
                except:
                    pass
        
        U3 = U.reshape(n, m, r)
        
        # Update V
        for j in range(m):
            for k in range(p):
                # A_V[j, k] is (n, r)
                # U3[:, j, :] is (n, r)
                # W3[k, :, :] is (n, r)
                A_V_jk = U3[:, j, :] * W3[k]
                b = np.ones(n)
                try:
                    V[j*p+k], _, _, _ = np.linalg.lstsq(A_V_jk, b, rcond=None)
                except:
                    pass
        
        V3 = V.reshape(m, p, r)
        
        # Update W
        for k in range(p):
            for i in range(n):
                # A_W[k, i] is (m, r)
                # U3[:, :, i]? No.
                # U3[i, :, :] is (m, r)
                # V3[:, k, :] is (m, r)
                A_W_ki = U3[i] * V3[:, k, :]
                b = np.ones(m)
                try:
                    W[k*n+i], _, _, _ = np.linalg.lstsq(A_W_ki, b, rcond=None)
                except:
                    pass
        
        W3 = W.reshape(p, n, r)
        
        # Check convergence
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        resid = T - 1.0
        norm = np.linalg.norm(resid)
        if norm < tol:
            break
    
    return U, V, W

def _try_rank_reduction(U, V, W, target_rank, n, m, p):
    """
    Try to reduce the rank of a valid decomposition to target_rank.
    Strategy: Zero out columns until we have target_rank columns, then re-optimize.
    """
    current_rank = U.shape[1]
    if current_rank < target_rank:
        return None
    
    # Select columns to keep. Keep the first target_rank columns?
    # Better: Keep columns with largest norm?
    # For simplicity, let's just take the first target_rank columns.
    U_t = U[:, :target_rank].copy()
    V_t = V[:, :target_rank].copy()
    W_t = W[:, :target_rank].copy()
    
    # Optimize
    U_t, V_t, W_t = _als_optimize_fast(U_t, V_t, W_t, n, m, p, max_iter=300)
    
    # Polish: Try to snap to integers/half-integers
    # 1. Round to nearest integer
    U_r = np.round(U_t)
    V_r = np.round(V_t)
    W_r = np.round(W_t)
    if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
        
    # 2. Round to nearest half-integer
    U_r = np.round(U_t * 2) / 2
    V_r = np.round(V_t * 2) / 2
    W_r = np.round(W_t * 2) / 2
    if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
        
    # 3. Check if the optimized solution is already close enough to be considered valid (floating point)
    # The task requires "integer/half-integer entries". So we MUST snap.
    # If snapping fails, this rank reduction attempt failed.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook()
    best_rank = n * m * p  # 48
    
    # Start from 48 and try to go down
    # We know 48 works.
    # Try 47, 46, ... down to 30 (or 16)
    
    # To save time, we only try a few reductions.
    # If 47 fails, 46 is likely to fail too unless we are lucky.
    # But let's try.
    
    for target_rank in range(best_rank - 1, 15, -1):
        if time.time() > DEADLINE - 20:
            break
        
        # Initialize with the best known decomposition (currently schoolbook or a found one)
        # We need a valid decomposition of rank >= target_rank to start from.
        # If we found a rank 47 solution, we use it to try 46.
        # If we only have 48, we use 48.
        
        # Convert best to numpy
        U_best = np.array(best["U"], dtype=float)
        V_best = np.array(best["V"], dtype=float)
        W_best = np.array(best["W"], dtype=float)
        
        # If best rank is smaller than target_rank, skip? No, we are going down.
        # If best rank is 48, and we try 47, we use U_best.
        # If we found 47, best_rank becomes 47, and we use the new best.
        
        # Try a few random column subsets if simple truncation fails?
        # For now, just truncate.
        
        result = _try_rank_reduction(U_best, V_best, W_best, target_rank, n, m, p)
        
        if result is not None:
            U_l, V_l, W_l = result
            if _verify(U_l, V_l, W_l):
                best = {"U": U_l, "V": V_l, "W": W_l}
                best_rank = target_rank
                # Continue to try lower ranks
                continue
        else:
            # If truncation fails, try a random restart at this rank?
            # This is expensive. Maybe just break if we fail 47?
            # Let's try one random restart for each rank if we have time.
            if time.time() < DEADLINE - 30:
                # Random restart
                for _ in range(5):
                    if time.time() > DEADLINE - 10:
                        break
                    U_rand = np.random.randn(n*m, target_rank) * 0.5
                    V_rand = np.random.randn(m*p, target_rank) * 0.5
                    W_rand = np.random.randn(p*n, target_rank) * 0.5
                    U_rand, V_rand, W_rand = _als_optimize_fast(U_rand, V_rand, W_rand, n, m, p, max_iter=100)
                    
                    # Polish
                    U_r = np.round(U_rand)
                    V_r = np.round(V_rand)
                    W_r = np.round(W_rand)
                    if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                        best = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        best_rank = target_rank
                        break
                    
                    U_r = np.round(U_rand * 2) / 2
                    V_r = np.round(V_rand * 2) / 2
                    W_r = np.round(W_rand * 2) / 2
                    if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                        best = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        best_rank = target_rank
                        break
                        
            # If we failed to find a lower rank, do we continue?
            # It's possible 47 fails but 46 succeeds? Unlikely.
            # But let's keep trying until time runs out or we hit a floor.
            
    return best

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result["U"][0]))
    print("Verified:", _verify(result["U"], result["V"], result["W"]))
