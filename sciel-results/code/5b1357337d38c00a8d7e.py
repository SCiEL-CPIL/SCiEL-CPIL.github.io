"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _rank17_decomposition():
    """Return a verified rank-17 decomposition using Strassen + direct terms."""
    n, m, p = 2, 3, 3
    n_a, n_b, n_c = n*m, m*p, p*n  # 6, 9, 6
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    def set_term(r, u_dict, v_dict, w_dict):
        for a, val in u_dict.items():
            U[a, r] = val
        for b, val in v_dict.items():
            V[b, r] = val
        for c, val in w_dict.items():
            W[c, r] = val
    
    # Index mapping:
    # a: A[i,j] -> i*3+j  (a0..a5)
    # b: B[j,k] -> j*3+k  (b0..b8)
    # c: C[i,k] -> k*2+i  (c0..c5)
    
    # Part 1: Strassen for A1(2x2) x B1_left(2x2) -> C[0:2, 0:2]
    set_term(0, {0:1, 4:1}, {0:1, 4:1}, {0:1, 3:1})
    set_term(1, {3:1, 4:1}, {0:1}, {1:1, 3:-1})
    set_term(2, {0:1}, {1:1, 4:-1}, {2:1, 3:1})
    set_term(3, {4:1}, {3:1, 0:-1}, {0:1, 1:1})
    set_term(4, {0:1, 1:1}, {4:1}, {0:-1, 2:1})
    set_term(5, {3:1, 0:-1}, {0:1, 1:1}, {3:1})
    set_term(6, {1:1, 4:-1}, {3:1, 4:1}, {0:1})
    
    # Part 2: Direct for A1(2x2) x B1_right(2x1) -> C[:, 2]
    set_term(7, {0:1}, {2:1}, {4:1})
    set_term(8, {1:1}, {5:1}, {4:1})
    set_term(9, {3:1}, {2:1}, {5:1})
    set_term(10, {4:1}, {5:1}, {5:1})
    
    # Part 3: Direct for A2(2x1) x B2(1x3) -> C[0:2, 0:3]
    set_term(11, {2:1}, {6:1}, {0:1})
    set_term(12, {2:1}, {7:1}, {2:1})
    set_term(13, {2:1}, {8:1}, {4:1})
    set_term(14, {5:1}, {6:1}, {1:1})
    set_term(15, {5:1}, {7:1}, {3:1})
    set_term(16, {5:1}, {8:1}, {5:1})
    
    T = _build_tensor(n, m, p)
    if not _verify_decomposition(U, V, W, T):
        return None
    return U, V, W

def _round_to_rational(vec, max_den=10000):
    """Round each element to a rational with small denominator."""
    from fractions import Fraction
    result = np.zeros_like(vec)
    for idx in np.ndindex(vec.shape):
        val = vec[idx]
        if abs(val) < 1e-12:
            result[idx] = 0.0
            continue
        # Try to find a good rational approximation
        best_frac = Fraction(val).limit_denominator(max_den)
        result[idx] = float(best_frac)
    return result

def _try_rank16_structured(deadline):
    """
    Attempt to find a rank-16 decomposition by starting from the rank-17
    decomposition and trying to merge/eliminate one term.
    """
    n, m, p = 2, 3, 3
    n_a, n_b, n_c = n*m, m*p, p*n
    T = _build_tensor(n, m, p)
    
    base = _rank17_decomposition()
    if base is None:
        return None
    U0, V0, W0 = base
    
    # Try to find a linear combination that allows dropping one term
    # by adding a small perturbation to two terms and eliminating one.
    # We'll try all pairs (r1, r2) to merge, and try to express term r2
    # as a linear combination involving the remaining 16 terms plus a
    # modified r1.
    
    # Alternative approach: use the known result that 2x3x3 has rank 16
    # (Baur-Strassen or other results). Let's try a direct parameterized
    # search around the rank-17 solution.
    
    # We'll do a local optimization: start with 17 terms, try to drive one
    # term to zero by adjusting the others.
    
    from scipy.optimize import minimize
    
    def objective(params):
        # params: flattened U, V, W for rank 16
        size_U = n_a * 16
        size_V = n_b * 16
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:]
        U = U_flat.reshape(n_a, 16)
        V = V_flat.reshape(n_b, 16)
        W = W_flat.reshape(n_c, 16)
        
        result = np.zeros(T.shape)
        for r in range(16):
            result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        err = np.sum((result - T)**2)
        return err
    
    # Initialize from rank-17 by dropping the last term
    U16 = U0[:, :16]
    V16 = V0[:, :16]
    W16 = W0[:, :16]
    
    x0 = np.concatenate([U16.flatten(), V16.flatten(), W16.flatten()])
    
    size_U = n_a * 16
    size_V = n_b * 16
    size_W = n_c * 16
    total_size = size_U + size_V + size_W
    
    attempts = 0
    max_attempts = 20
    best_err = np.inf
    best_params = x0.copy()
    
    while time.time() < deadline - 20 and attempts < max_attempts:
        if time.time() > deadline - 20:
            break
        res = minimize(objective, x0, method='L-BFGS-B', maxiter=500, 
                      options={'ftol': 1e-10, 'gtol': 1e-8})
        attempts += 1
        
        if res.fun < best_err:
            best_err = res.fun
            best_params = res.x.copy()
        
        if best_err < 1e-8:
            break
        
        # Perturb the initial guess
        np.random.seed(attempts * 42)
        noise_scale = 0.1 / (attempts + 1)
        x0 = best_params + np.random.randn(total_size) * noise_scale
    
    if best_err < 1e-6:
        U_out = best_params[:size_U].reshape(n_a, 16)
        V_out = best_params[size_U:size_U+size_V].reshape(n_b, 16)
        W_out = best_params[size_U+size_V:].reshape(n_c, 16)
        
        # Round to rationals
        U_out = _round_to_rational(U_out)
        V_out = _round_to_rational(V_out)
        W_out = _round_to_rational(W_out)
        
        if _verify_decomposition(U_out, V_out, W_out, T, tol=1e-6):
            # Try to get even tighter solution
            if _verify_decomposition(U_out, V_out, W_out, T, tol=1e-9):
                return U_out, V_out, W_out
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 second deadline with safety margin
    
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    T = _build_tensor(n, m, p)
    
    # Attempt rank 16 search
    if time.time() < deadline - 30:
        rank16_result = _try_rank16_structured(deadline)
        if rank16_result is not None:
            U, V, W = rank16_result
            if _verify_decomposition(U, V, W, T, tol=1e-9):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to rank 17
    if time.time() < deadline - 10:
        U, V, W = _rank17_decomposition()
        if U is not None and _verify_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank 18
    rank_fb = n * m * p
    U_fb = np.zeros((n_a, rank_fb))
    V_fb = np.zeros((n_b, rank_fb))
    W_fb = np.zeros((n_c, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
