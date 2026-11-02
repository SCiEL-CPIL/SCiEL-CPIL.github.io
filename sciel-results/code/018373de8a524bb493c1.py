import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 4]

def _strassen_2x2():
    """Return the 7 Strassen rank-1 terms for 2x2 matrix multiplication.
    Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    Each term: (u_coeffs, v_coeffs, w_coeffs) as lists of (local_idx, value)
    """
    return [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

def _build_rank28():
    """Build the rank-28 Strassen-based decomposition."""
    n, m, p = DIMENSIONS
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    strassen = _strassen_2x2()
    
    u_map_L = [0, 1, 4, 5]
    u_map_R = [2, 3, 6, 7]
    v_map_TL = [0, 1, 4, 5]
    v_map_TR = [2, 3, 6, 7]
    v_map_BL = [8, 9, 12, 13]
    v_map_BR = [10, 11, 14, 15]
    w_map_L = [0, 2, 1, 3]
    w_map_R = [4, 6, 5, 7]
    
    col = 0
    for u_map, v_map, w_map in [(u_map_L, v_map_TL, w_map_L),
                                  (u_map_R, v_map_BL, w_map_L),
                                  (u_map_L, v_map_TR, w_map_R),
                                  (u_map_R, v_map_BR, w_map_R)]:
        for u_t, v_t, w_t in strassen:
            for idx, val in u_t:
                U[u_map[idx], col] += val
            for idx, val in v_t:
                V[v_map[idx], col] += val
            for idx, val in w_t:
                W[w_map[idx], col] += val
            col += 1
    
    return U, V, W

def _verify(U, V, W, n=2, m=4, p=4, tol=1e-9):
    """Verify the tensor decomposition."""
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    
    Cf = np.zeros((n, p))
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    return np.allclose(C, Cf, atol=tol)

def _tensor_from_factors(U, V, W, n, m, p):
    """Compute tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]"""
    T = np.zeros((n, m, p), dtype=float)
    for c in range(U.shape[1]):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    return T

def _try_rank_reduction(U, V, W, target_rank, start_time, time_limit=145.0):
    """Try to reduce rank from current to target_rank.
    
    Strategy: For each set of columns to eliminate, try to absorb their
    tensor contribution into the remaining columns via least squares.
    """
    n, m, p = DIMENSIONS
    r = U.shape[1]
    if r <= target_rank:
        return None
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build target tensor
    T_target = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i, j, k] = 1.0 if j == k else 0.0
    
    # Try different subsets of columns to keep
    # Use random sampling for efficiency
    np.random.seed(42)
    
    max_attempts = 80
    attempts = 0
    
    while attempts < max_attempts and time.time() - start_time < time_limit - 10.0:
        attempts += 1
        
        # Randomly select which columns to keep
        keep_cols = np.random.choice(r, size=target_rank, replace=False)
        elim_cols = np.setdiff1d(np.arange(r), keep_cols)
        
        U_keep = U[:, keep_cols].copy()
        V_keep = V[:, keep_cols].copy()
        W_keep = W[:, keep_cols].copy()
        
        U_elim = U[:, elim_cols]
        V_elim = V[:, elim_cols]
        W_elim = W[:, elim_cols]
        
        # Compute residual tensor from eliminated columns
        T_resid = _tensor_from_factors(U_elim, V_elim, W_elim, n, m, p)
        
        # We need: tensor(U_keep, V_keep, W_keep) = T_target - T_resid
        T_needed = T_target - T_resid
        
        # Parameters: flatten U_keep, V_keep, W_keep
        n_params = U_keep.size + V_keep.size + W_keep.size
        params = np.concatenate([U_keep.ravel(), V_keep.ravel(), W_keep.ravel()])
        
        def residuals(params):
            Uf = params[:U_keep.size].reshape(U_keep.shape)
            Vf = params[U_keep.size:U_keep.size+V_keep.size].reshape(V_keep.shape)
            Wf = params[U_keep.size+V_keep.size:].reshape(W_keep.shape)
            Tt = _tensor_from_factors(Uf, Vf, Wf, n, m, p)
            return (Tt - T_needed).ravel()
        
        # Use least squares
        try:
            from scipy.optimize import least_squares
            result = least_squares(residuals, params, method='lm', 
                                    max_nfev=5000, xtol=1e-12, ftol=1e-12)
            
            if result.success or np.max(np.abs(result.fun)) < 1e-7:
                Uf = result.x[:U_keep.size].reshape(U_keep.shape)
                Vf = result.x[U_keep.size:U_keep.size+V_keep.size].reshape(V_keep.shape)
                Wf = result.x[U_keep.size+V_keep.size:].reshape(W_keep.shape)
                
                # Verify
                if _verify(Uf, Vf, Wf, tol=1e-6):
                    return Uf, Vf, Wf
        except Exception:
            pass
    
    return None

def _try_circulant_decomposition(start_time, time_limit=145.0):
    """Try a circulant-based structured decomposition exploiting the
    cyclic symmetry of the middle dimension (j in 0..3).
    
    The idea: model the 2x4x4 multiplication as a convolutional operation
    on the intermediate dimension j. For fixed (i,k), the contribution is
    sum_j A[i,j] B[j,k]. We can try to exploit structure where the
    Strassen terms for different blocks share rank-1 factors through
    circulant (Fourier) mixing of the j-index.
    
    Specifically, we try to construct a rank-27 decomposition by:
    1. Starting from the rank-28 Strassen solution
    2. Perturbing with circulant structure in the V matrix (middle dim)
    3. Re-optimizing to see if one rank-1 term can be eliminated
    """
    n, m, p = DIMENSIONS
    
    T_target = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i, j, k] = 1.0 if j == k else 0.0
    
    # Try rank 27 with circulant-inspired initialization
    target_rank = 27
    U0, V0, W0 = _build_rank28()
    U0 = np.array(U0, dtype=float)
    V0 = np.array(V0, dtype=float)
    W0 = np.array(W0, dtype=float)
    
    # Remove one column (the last one) and try to optimize the rest
    # But first, try a more structured approach:
    # Use the Fourier transform of the middle dimension to create circulant structure
    
    # For each of the 28 Strassen terms, the V factor has support on specific
    # j*k indices. We can try to mix these using DFT of size 4 on the j-dimension.
    
    # Actually, let's try a direct optimization approach with multiple random
    # restarts, but with a better initialization strategy:
    # Start from rank-28, remove a random column, and use LM to fix the rest.
    
    n_restarts = 20
    best_result = None
    best_residual = float('inf')
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit - 15.0:
            break
        
        np.random.seed(1000 + restart)
        
        # Remove a random column from the rank-28 solution
        remove_idx = np.random.randint(0, 28)
        keep_idx = [i for i in range(28) if i != remove_idx]
        
        U_init = U0[:, keep_idx].copy()
        V_init = V0[:, keep_idx].copy()
        W_init = W0[:, keep_idx].copy()
        
        # Add small random perturbation to help escape local minima
        U_init += np.random.randn(*U_init.shape) * 0.01
        V_init += np.random.randn(*V_init.shape) * 0.01
        W_init += np.random.randn(*W_init.shape) * 0.01
        
        params = np.concatenate([U_init.ravel(), V_init.ravel(), W_init.ravel()])
        
        def residuals(params):
            Uf = params[:U_init.size].reshape(U_init.shape)
            Vf = params[U_init.size:U_init.size+V_init.size].reshape(V_init.shape)
            Wf = params[U_init.size+V_init.size:].reshape(W_init.shape)
            Tt = _tensor_from_factors(Uf, Vf, Wf, n, m, p)
            return (Tt - T_target).ravel()
        
        try:
            from scipy.optimize import least_squares
            result = least_squares(residuals, params, method='lm', 
                                    max_nfev=5000, xtol=1e-12, ftol=1e-12)
            
            max_res = np.max(np.abs(result.fun))
            if max_res < best_residual:
                best_residual = max_res
                Uf = result.x[:U_init.size].reshape(U_init.shape)
                Vf = result.x[U_init.size:U_init.size+V_init.size].reshape(V_init.shape)
                Wf = result.x[U_init.size+V_init.size:].reshape(W_init.shape)
                
                if max_res < 1e-7:
                    if _verify(Uf, Vf, Wf, tol=1e-6):
                        return Uf, Vf, Wf
                    else:
                        best_result = (Uf, Vf, Wf)
                        
                    if max_res < 1e-8:
                        break
        except Exception:
            pass
    
    if best_result is not None and best_residual < 1e-6:
        Uf, Vf, Wf = best_result
        if _verify(Uf, Vf, Wf, tol=1e-5):
            return Uf, Vf, Wf
    
    return None

def _try_ilp_reduction(U, V, W, start_time, time_limit=145.0):
    """Try to find a lower-rank decomposition using a structured approach.
    
    For 2x4x4, we can exploit the structure more carefully.
    The key insight: C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can try various block decompositions and optimize within them.
    """
    n, m, p = DIMENSIONS
    
    T_target = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i, j, k] = 1.0 if j == k else 0.0
    
    # Try rank 24-27 with random initialization
    for target_rank in [27, 26, 25, 24]:
        if time.time() - start_time > time_limit - 15.0:
            break
        
        best_result = None
        best_residual = float('inf')
        
        n_restarts = 8
        for restart in range(n_restarts):
            if time.time() - start_time > time_limit - 15.0:
                break
            
            np.random.seed(42 + restart)
            
            # Random initialization with small values
            U_init = np.random.randn(n * m, target_rank) * 0.5
            V_init = np.random.randn(m * p, target_rank) * 0.5
            W_init = np.random.randn(p * n, target_rank) * 0.5
            
            params = np.concatenate([U_init.ravel(), V_init.ravel(), W_init.ravel()])
            
            def residuals(params):
                Uf = params[:U_init.size].reshape(U_init.shape)
                Vf = params[U_init.size:U_init.size+V_init.size].reshape(V_init.shape)
                Wf = params[U_init.size+V_init.size:].reshape(W_init.shape)
                Tt = _tensor_from_factors(Uf, Vf, Wf, n, m, p)
                return (Tt - T_target).ravel()
            
            try:
                from scipy.optimize import least_squares
                result = least_squares(residuals, params, method='lm', 
                                        max_nfev=3000, xtol=1e-10)
                
                if np.max(np.abs(result.fun)) < best_residual:
                    best_residual = np.max(np.abs(result.fun))
                    if best_residual < 1e-7:
                        Uf = result.x[:U_init.size].reshape(U_init.shape)
                        Vf = result.x[U_init.size:U_init.size+V_init.size].reshape(V_init.shape)
                        Wf = result.x[U_init.size+V_init.size:].reshape(W_init.shape)
                        
                        if _verify(Uf, Vf, Wf, tol=1e-6):
                            return Uf, Vf, Wf
                        else:
                            best_result = (Uf, Vf, Wf)
                        
                        if best_residual < 1e-8:
                            break
            except Exception:
                pass
        
        if best_result is not None and best_residual < 1e-6:
            Uf, Vf, Wf = best_result
            if _verify(Uf, Vf, Wf, tol=1e-5):
                return Uf, Vf, Wf
    
    return None

def _polish(U, V, W, start_time, time_limit=145.0):
    """Final high-precision polish pass: tighten tolerance and do a
    final least-squares refinement to ensure exact decomposition."""
    n, m, p = DIMENSIONS
    
    T_target = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i, j, k] = 1.0 if j == k else 0.0
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    params = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
    
    def residuals(params):
        Uf = params[:U.size].reshape(U.shape)
        Vf = params[U.size:U.size+V.size].reshape(V.shape)
        Wf = params[U.size+V.size:].reshape(W.shape)
        Tt = _tensor_from_factors(Uf, Vf, Wf, n, m, p)
        return (Tt - T_target).ravel()
    
    try:
        from scipy.optimize import least_squares
        result = least_squares(residuals, params, method='lm', 
                                max_nfev=10000, xtol=1e-14, ftol=1e-14)
        
        Uf = result.x[:U.size].reshape(U.shape)
        Vf = result.x[U.size:U.size+V.size].reshape(V.shape)
        Wf = result.x[U.size+V.size:].reshape(W.shape)
        
        if _verify(Uf, Vf, Wf, tol=1e-8):
            return Uf, Vf, Wf
    except Exception:
        pass
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 145.0  # safety margin
    
    # Start with rank-28 Strassen decomposition
    U, V, W = _build_rank28()
    
    # Verify it's correct
    if not _verify(U, V, W):
        if time.time() - start_time > time_limit:
            raise TimeoutError("Construction took too long")
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = 28
    
    # Try to reduce rank using column elimination
    for target_rank in [27, 26, 25]:
        if time.time() - start_time > time_limit - 20.0:
            break
        
        result = _try_rank_reduction(U, V, W, target_rank, start_time, time_limit)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = target_rank
    
    # Try circulant-inspired structured decomposition
    if time.time() - start_time < time_limit - 25.0:
        result = _try_circulant_decomposition(start_time, time_limit)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = U.shape[1]
    
    # Try ILP-style structured optimization
    if time.time() - start_time < time_limit - 20.0:
        result = _try_ilp_reduction(U, V, W, start_time, time_limit)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = U.shape[1]
    
    # Final polish pass
    if time.time() - start_time < time_limit - 5.0:
        best_U, best_V, best_W = _polish(best_U, best_V, best_W, start_time, time_limit)
    
    if time.time() - start_time > time_limit:
        raise TimeoutError("Construction took too long")
    
    # Round to nearest half-integer if close
    def round_half(x):
        return np.round(x * 2) / 2
    
    U_out = best_U
    V_out = best_V
    W_out = best_W
    
    # Try rounding
    U_r = round_half(U_out)
    V_r = round_half(V_out)
    W_r = round_half(W_out)
    
    if _verify(U_r, V_r, W_r, tol=1e-6):
        U_out, V_out, W_out = U_r, V_r, W_r
    
    # Convert to list
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    # Test with random matrices to ensure correctness
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
            
    if not np.allclose(C, Cf):
        print("Verification Failed")
        print(C)
        print(Cf)
    else:
        print("Verification Passed")
    print("Rank:", U.shape[1])
