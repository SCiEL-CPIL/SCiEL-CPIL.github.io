"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = len(W[0])
    rng = np.random.default_rng(12345)
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((m, p))
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for r in range(rank):
                lA = sum(U[i*m+j, r] * A[i, j] for j in range(m))
                lB = sum(V[j*p+k, r] * B[j, k] for j in range(m))
                C_computed[i, k] += W[k*n+i, r] * lA * lB
    return np.allclose(C_computed, C_expected, atol=1e-10)

def _residual_norm(U, V, W):
    """Compute the residual norm of the tensor decomposition."""
    n, m, p = DIMENSIONS
    rank = len(W[0])
    residual = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                residual += (val - 1.0) ** 2
    return np.sqrt(residual)

def _try_rank_10_optimized():
    """
    Try to find a rank-10 decomposition using optimized numerical search.
    
    Strategy: Use the border rank interpolation idea - treat 2x2x3 as a 
    specialization of 2x2x4 and search for a rank-10 decomposition over
    reals using scipy least_squares with multiple initializations, then
    apply rational reconstruction for exact verification.
    """
    n, m, p = DIMENSIONS
    rank = 10
    
    start_time = time.time()
    deadline = start_time + 55  # Leave time for other methods
    
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return None
    
    def _flatten(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def _unflatten(x):
        nU = n*m*rank
        nV = m*p*rank
        U = x[:nU].reshape(n*m, rank)
        V = x[nU:nU+nV].reshape(m*p, rank)
        W = x[nU+nV:].reshape(p*n, rank)
        return U, V, W
    
    def _residual_vec(x):
        U, V, W = _unflatten(x)
        # Vectorized residual computation
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    res[idx] = val - 1.0
                    idx += 1
        return res
    
    def _try_round_and_verify(U, V, W):
        """Try to round to integer/half-integer and verify exactness."""
        # Try integer rounding
        candidates = [
            (np.round(U), np.round(V), np.round(W)),
            # Try half-integer rounding
            (np.round(U*2)/2, np.round(V*2)/2, np.round(W*2)/2),
            # Try with some entries at 0.5
            (np.round(U*4)/4, np.round(V*4)/4, np.round(W*4)/4),
        ]
        
        for U_r, V_r, W_r in candidates:
            res = _residual_norm(U_r, V_r, W_r)
            if res < 1e-12:
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
        
        return None
    
    best_residual = np.inf
    best_params = None
    
    # Multiple initialization strategies
    for attempt in range(80):
        if time.time() > deadline:
            break
        
        rng = np.random.default_rng(attempt * 9973 + 7)
        
        U_init = np.zeros((n*m, rank))
        V_init = np.zeros((m*p, rank))
        W_init = np.zeros((p*n, rank))
        
        # Strategy 1: Start from Strassen + perturb schoolbook terms
        if attempt < 30:
            # Strassen terms 0-6
            U_init[0,0] = 1; U_init[3,0] = 1
            V_init[0,0] = 1; V_init[4,0] = 1
            U_init[2,1] = 1; U_init[3,1] = 1
            V_init[0,1] = 1
            U_init[0,2] = 1
            V_init[1,2] = 1; V_init[4,2] = -1
            U_init[3,3] = 1
            V_init[3,3] = 1; V_init[0,3] = -1
            U_init[0,4] = 1; U_init[1,4] = 1
            V_init[4,4] = 1
            U_init[2,5] = 1; U_init[0,5] = -1
            V_init[0,5] = 1; V_init[1,5] = 1
            U_init[1,6] = 1; U_init[3,6] = -1
            V_init[3,6] = 1; V_init[4,6] = 1
            
            W_init[0,0] = 1; W_init[0,3] = 1; W_init[0,4] = -1; W_init[0,6] = 1
            W_init[2,2] = 1; W_init[2,4] = 1
            W_init[1,1] = 1; W_init[1,3] = 1
            W_init[3,0] = 1; W_init[3,1] = -1; W_init[3,2] = 1; W_init[3,5] = 1
            
            # Terms 7-9: linear combinations for column 2
            # Try different patterns
            if attempt % 3 == 0:
                # Pattern: (a+αc)*b02, (b+βd)*b12, correction
                U_init[0,7] = 1; U_init[2,7] = rng.normal(0, 1)
                V_init[2,7] = 1
                W_init[4,7] = 1; W_init[5,7] = rng.normal(0, 0.5)
                
                U_init[1,8] = 1; U_init[3,8] = rng.normal(0, 1)
                V_init[5,8] = 1
                W_init[4,8] = 1; W_init[5,8] = rng.normal(0, 0.5)
                
                U_init[0,9] = rng.normal(0, 1); U_init[1,9] = rng.normal(0, 1)
                U_init[2,9] = rng.normal(0, 1); U_init[3,9] = rng.normal(0, 1)
                V_init[2,9] = rng.normal(0, 1); V_init[5,9] = rng.normal(0, 1)
                W_init[4,9] = rng.normal(0, 1); W_init[5,9] = rng.normal(0, 1)
            elif attempt % 3 == 1:
                # Pattern: share more structure
                U_init[0,7] = 1; U_init[2,7] = 1
                V_init[2,7] = 1
                W_init[4,7] = 1; W_init[5,7] = 1
                
                U_init[1,8] = 1; U_init[3,8] = 1
                V_init[5,8] = 1
                W_init[4,8] = 1; W_init[5,8] = 1
                
                U_init[0,9] = -1; U_init[1,9] = 1
                U_init[2,9] = -1; U_init[3,9] = 1
                V_init[2,9] = 1; V_init[5,9] = -1
                W_init[4,9] = rng.normal(0, 1); W_init[5,9] = rng.normal(0, 1)
            else:
                # Random but structured
                for r in range(7, 10):
                    U_init[:, r] = rng.normal(0, 1, n*m)
                    V_init[:, r] = rng.normal(0, 1, m*p)
                    W_init[:, r] = rng.normal(0, 1, p*n)
                    # Make sparse
                    mask_U = rng.random(n*m) > 0.5
                    mask_V = rng.random(m*p) > 0.5
                    mask_W = rng.random(p*n) > 0.5
                    U_init[:, r] *= mask_U
                    V_init[:, r] *= mask_V
                    W_init[:, r] *= mask_W
        
        # Strategy 2: Fully random with sparsity
        elif attempt < 60:
            for r in range(rank):
                n_nonzero_U = rng.integers(2, 5)
                n_nonzero_V = rng.integers(2, 5)
                n_nonzero_W = rng.integers(2, 5)
                
                idx_U = rng.choice(n*m, size=n_nonzero_U, replace=False)
                idx_V = rng.choice(m*p, size=n_nonzero_V, replace=False)
                idx_W = rng.choice(p*n, size=n_nonzero_W, replace=False)
                
                U_init[idx_U, r] = rng.choice([-2, -1, 1, 2])
                V_init[idx_V, r] = rng.choice([-2, -1, 1, 2])
                W_init[idx_W, r] = rng.choice([-2, -1, 1, 2])
        
        # Strategy 3: Border rank interpolation - start from 2x2x4 Strassen and specialize
        else:
            # 2x2x4 Strassen has rank 7, we need rank 10 for 2x2x3
            # Initialize with 7 Strassen-like terms + 3 random
            U_init[0,0] = 1; U_init[3,0] = 1
            V_init[0,0] = 1; V_init[4,0] = 1
            U_init[2,1] = 1; U_init[3,1] = 1
            V_init[0,1] = 1
            U_init[0,2] = 1
            V_init[1,2] = 1; V_init[4,2] = -1
            U_init[3,3] = 1
            V_init[3,3] = 1; V_init[0,3] = -1
            U_init[0,4] = 1; U_init[1,4] = 1
            V_init[4,4] = 1
            U_init[2,5] = 1; U_init[0,5] = -1
            V_init[0,5] = 1; V_init[1,5] = 1
            U_init[1,6] = 1; U_init[3,6] = -1
            V_init[3,6] = 1; V_init[4,6] = 1
            
            W_init[0,0] = 1; W_init[0,3] = 1; W_init[0,4] = -1; W_init[0,6] = 1
            W_init[2,2] = 1; W_init[2,4] = 1
            W_init[1,1] = 1; W_init[1,3] = 1
            W_init[3,0] = 1; W_init[3,1] = -1; W_init[3,2] = 1; W_init[3,5] = 1
            
            for r in range(7, 10):
                U_init[:, r] = rng.normal(0, 0.5, n*m)
                V_init[:, r] = rng.normal(0, 0.5, m*p)
                W_init[:, r] = rng.normal(0, 0.5, p*n)
        
        x0 = _flatten(U_init, V_init, W_init)
        
        try:
            result = least_squares(_residual_vec, x0, method='lm', max_nfev=2000, ftol=1e-15)
            res = np.sqrt(np.sum(result.fun**2))
            
            if res < best_residual:
                best_residual = res
                U, V, W = _unflatten(result.x)
                best_params = (U.copy(), V.copy(), W.copy())
            
            # If close to zero, try to find exact solution
            if res < 1e-10:
                U_opt, V_opt, W_opt = _unflatten(result.x)
                rounded = _try_round_and_verify(U_opt, V_opt, W_opt)
                if rounded is not None:
                    return rounded
            
            # Also try rounding the best found so far if residual is small
            if best_residual < 1e-8 and best_params is not None:
                U_b, V_b, W_b = best_params
                rounded = _try_round_and_verify(U_b, V_b, W_b)
                if rounded is not None:
                    return rounded
                    
        except Exception:
            pass
    
    # Final check: if we have a very good solution, try harder to round
    if best_residual < 1e-6 and best_params is not None:
        U_b, V_b, W_b = best_params
        # Try multiple rounding strategies
        for scale in [1, 2, 4, 8]:
            U_r = np.round(U_b * scale) / scale
            V_r = np.round(V_b * scale) / scale
            W_r = np.round(W_b * scale) / scale
            res_r = _residual_norm(U_r, V_r, W_r)
            if res_r < 1e-12:
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
        
        # Try local optimization around rounded values
        for scale in [1, 2]:
            U_r = np.round(U_b * scale) / scale
            V_r = np.round(V_b * scale) / scale
            W_r = np.round(W_b * scale) / scale
            x_r = _flatten(U_r, V_r, W_r)
            try:
                result2 = least_squares(_residual_vec, x_r, method='lm', max_nfev=500, ftol=1e-15)
                res2 = np.sqrt(np.sum(result2.fun**2))
                if res2 < 1e-12:
                    U2, V2, W2 = _unflatten(result2.x)
                    return U2.tolist(), V2.tolist(), W2.tolist()
            except Exception:
                pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Try optimized rank-10 search with border rank interpolation
    result_10 = _try_rank_10_optimized()
    if result_10 is not None:
        U, V, W = result_10
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # If we have time, try additional random restarts
    if time.time() < deadline - 30:
        # Quick additional search with different seed
        start_time2 = time.time()
        deadline2 = start_time2 + 30
        for extra_attempt in range(20):
            if time.time() > deadline2:
                break
            rng = np.random.default_rng(extra_attempt * 48611 + 13)
            rank = 10
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            for r in range(rank):
                n_nonzero_U = rng.integers(2, 4)
                n_nonzero_V = rng.integers(2, 4)
                n_nonzero_W = rng.integers(2, 4)
                idx_U = rng.choice(n*m, size=n_nonzero_U, replace=False)
                idx_V = rng.choice(m*p, size=n_nonzero_V, replace=False)
                idx_W = rng.choice(p*n, size=n_nonzero_W, replace=False)
                U[idx_U, r] = rng.choice([-2, -1, 1, 2])
                V[idx_V, r] = rng.choice([-2, -1, 1, 2])
                W[idx_W, r] = rng.choice([-2, -1, 1, 2])
            
            res = _residual_norm(U, V, W)
            if res < 1e-12:
                if _verify(U, V, W):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to rank 11 (Strassen + schoolbook)
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen-style part for the 2x2 block (columns 0 and 1 of B)
    # M1 = (a+d)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    
    # M2 = (c+d)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    
    # M3 = a(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    
    # M4 = d(b10-b00)
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    
    # M5 = (a+b)b11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    
    # M6 = (c-a)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    
    # M7 = (b-d)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    
    # W for Strassen outputs
    # C00 = M1 + M4 - M5 + M7
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    # C01 = M3 + M5
    W[2][2] = 1; W[2][4] = 1
    # C10 = M2 + M4
    W[1][1] = 1; W[1][3] = 1
    # C11 = M1 - M2 + M3 + M6
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    # Schoolbook part for column 2 of B
    # M8 = a * b02
    U[0][7] = 1
    V[2][7] = 1
    
    # M9 = b * b12
    U[1][8] = 1
    V[5][8] = 1
    
    # M10 = c * b02
    U[2][9] = 1
    V[2][9] = 1
    
    # M11 = d * b12
    U[3][10] = 1
    V[5][10] = 1
    
    # W for Schoolbook outputs
    # C02 = M8 + M9
    W[4][7] = 1; W[4][8] = 1
    # C12 = M10 + M11
    W[5][9] = 1; W[5][10] = 1
    
    # Verify the decomposition
    if not _verify(U, V, W):
        # Fallback to schoolbook if verification fails (should not happen)
        rank = 12
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
