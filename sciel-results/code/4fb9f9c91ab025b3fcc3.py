"""Optimized search for rank < 39 using vectorized ALS and snapping to half-integers."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify_exact(U, V, W, n, m, p):
    """
    Verify decomposition using exact Fraction arithmetic.
    U, V, W are lists of lists of floats (snapped to half-integers).
    """
    rank = len(U[0])
    # Convert to Fractions for exactness
    U_frac = [[Fraction(val).limit_denominator(100) for val in row] for row in U]
    V_frac = [[Fraction(val).limit_denominator(100) for val in row] for row in V]
    W_frac = [[Fraction(val).limit_denominator(100) for val in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U_frac[i * m + j][c] * V_frac[j * p + k][c] * W_frac[k * n + i][c]
                if s != 1:
                    return False
    return True

def _snap_to_half_integers(mat, tol=1e-6):
    """
    Snap matrix values to the nearest half-integer if close.
    Returns the snapped matrix and a boolean indicating if all values were close.
    """
    mat_snapped = np.zeros_like(mat)
    valid = True
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            # Check if 2*val is close to an integer
            twice_val = 2.0 * val
            nearest_int = round(twice_val)
            if abs(twice_val - nearest_int) < tol:
                mat_snapped[i, j] = nearest_int / 2.0
            else:
                valid = False
                mat_snapped[i, j] = val
    return mat_snapped, valid

def _construct_rank39():
    """
    Explicit rank-39 decomposition for M_{2,2,11}.
    """
    n, m, p = 2, 2, 11
    rank = 39
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        
        w_c00 = k0 * n
        w_c10 = k0 * n + 1
        w_c01 = k1 * n
        w_c11 = k1 * n + 1
        
        # m1
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1
        # m2
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1
        # m3
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1
        # m4
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1
        # m5
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1
        # m6
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1
        # m7
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1
    
    k2 = 10
    v_e2 = k2
    v_g2 = p + k2
    w_c02 = k2 * n
    w_c12 = k2 * n + 1
    
    U[0][r] = 1; V[v_e2][r] = 1; W[w_c02][r] = 1; r += 1
    U[1][r] = 1; V[v_g2][r] = 1; W[w_c02][r] = 1; r += 1
    U[2][r] = 1; V[v_e2][r] = 1; W[w_c12][r] = 1; r += 1
    U[3][r] = 1; V[v_g2][r] = 1; W[w_c12][r] = 1; r += 1
    
    assert r == rank
    return U, V, W

def _vectorized_als(U, V, W, n, m, p, max_iters=1000, tol=1e-10):
    """
    Vectorized Alternating Least Squares for tensor decomposition.
    """
    rank = U.shape[1]
    
    # Pre-compute index mappings
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # For each row a in U, find (b, d) pairs
    U_indices = {}
    for i in range(n):
        for j in range(m):
            a = i * m + j
            pairs = []
            for k in range(p):
                b = j * p + k
                d = k * n + i
                pairs.append((b, d))
            U_indices[a] = pairs
            
    V_indices = {}
    for j in range(m):
        for k in range(p):
            b = j * p + k
            pairs = []
            for i in range(n):
                a = i * m + j
                d = k * n + i
                pairs.append((a, d))
            V_indices[b] = pairs
            
    W_indices = {}
    for k in range(p):
        for i in range(n):
            d = k * n + i
            pairs = []
            for j in range(m):
                a = i * m + j
                b = j * p + k
                pairs.append((a, b))
            W_indices[d] = pairs

    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iters):
        # Update U
        for a in range(n * m):
            pairs = U_indices[a]
            if not pairs: continue
            bs = [x[0] for x in pairs]
            ds = [x[1] for x in pairs]
            # M matrix: len(pairs) x rank
            M = np.zeros((len(pairs), rank))
            for idx, (b, d) in enumerate(pairs):
                M[idx, :] = V[b, :] * W[d, :]
            # Target: all 1s
            t = np.ones(len(pairs))
            # Solve M u = t
            try:
                U[a, :] = np.linalg.lstsq(M, t, rcond=None)[0]
            except:
                pass
                
        # Update V
        for b in range(m * p):
            pairs = V_indices[b]
            if not pairs: continue
            as_ = [x[0] for x in pairs]
            ds = [x[1] for x in pairs]
            M = np.zeros((len(pairs), rank))
            for idx, (a, d) in enumerate(pairs):
                M[idx, :] = U[a, :] * W[d, :]
            t = np.ones(len(pairs))
            try:
                V[b, :] = np.linalg.lstsq(M, t, rcond=None)[0]
            except:
                pass
                
        # Update W
        for d in range(p * n):
            pairs = W_indices[d]
            if not pairs: continue
            as_ = [x[0] for x in pairs]
            bs = [x[1] for x in pairs]
            M = np.zeros((len(pairs), rank))
            for idx, (a, b) in enumerate(pairs):
                M[idx, :] = U[a, :] * V[b, :]
            t = np.ones(len(pairs))
            try:
                W[d, :] = np.linalg.lstsq(M, t, rcond=None)[0]
            except:
                pass
                
        # Check error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    d = k * n + i
                    s = np.dot(U[a, :], V[b, :] * W[d, :])
                    err += (s - 1.0) ** 2
                    
        if err < tol:
            break
            
    return U, V, W, err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    # 1. Get Rank 39 fallback
    U39, V39, W39 = _construct_rank39()
    # Verify Rank 39 is valid (it should be)
    if not _verify_exact(U39, V39, W39, n, m, p):
        # Fallback to schoolbook if construction fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
        
    best_result = {"U": U39, "V": V39, "W": W39}
    best_rank = 39
    
    # 2. Try to find Rank 38
    if time.time() < deadline - 10:
        U39_np = np.array(U39)
        V39_np = np.array(V39)
        W39_np = np.array(W39)
        
        # Try dropping each column
        for drop_idx in range(39):
            if time.time() > deadline - 5:
                break
                
            U_init = np.delete(U39_np, drop_idx, axis=1)
            V_init = np.delete(V39_np, drop_idx, axis=1)
            W_init = np.delete(W39_np, drop_idx, axis=1)
            
            # Run ALS
            U_sol, V_sol, W_sol, err = _vectorized_als(U_init, V_init, W_sol, n, m, p, max_iters=500, tol=1e-8)
            
            if err < 1e-6:
                # Try snapping
                U_snapped, valid_U = _snap_to_half_integers(U_sol)
                V_snapped, valid_V = _snap_to_half_integers(V_sol)
                W_snapped, valid_W = _snap_to_half_integers(W_sol)
                
                if valid_U and valid_V and valid_W:
                    # Verify exact
                    if _verify_exact(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), n, m, p):
                        return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
                        
        # Try random initializations for Rank 38
        if time.time() < deadline - 5:
            for seed in range(20):
                if time.time() > deadline - 2:
                    break
                rng = np.random.RandomState(seed)
                U_init = rng.randn(4, 38) * 0.5
                V_init = rng.randn(22, 38) * 0.5
                W_init = rng.randn(22, 38) * 0.5
                
                U_sol, V_sol, W_sol, err = _vectorized_als(U_init, V_init, W_init, n, m, p, max_iters=500, tol=1e-8)
                
                if err < 1e-6:
                    U_snapped, valid_U = _snap_to_half_integers(U_sol)
                    V_snapped, valid_V = _snap_to_half_integers(V_sol)
                    W_snapped, valid_W = _snap_to_half_integers(W_sol)
                    
                    if valid_U and valid_V and valid_W:
                        if _verify_exact(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), n, m, p):
                            return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
                            
    return best_result
# EVOLVE-BLOCK-END
