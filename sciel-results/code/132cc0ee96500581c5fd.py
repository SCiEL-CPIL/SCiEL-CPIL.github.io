import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _build_target_tensor():
    """Build the 3x3 matrix multiplication tensor in the required index convention."""
    n = m = p = 3
    T = np.zeros((9, 9, 9), dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j  # U index
                b = j * 3 + k  # V index
                c = k * 3 + i  # W index
                T[c, a, b] = 1.0
    return T

def _verify(U, V, W):
    """Check that sum_r U[a,r]*V[b,r]*W[c,r] == T[c,a,b]."""
    n = m = p = 3
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    T_calc = np.einsum('ar,br,cr->cab', U, V, W)
    T_target = _build_target_tensor()
    return np.allclose(T_calc, T_target, atol=1e-9), np.max(np.abs(T_calc - T_target))

def _get_rank_26():
    """Hardcoded rank-26 decomposition (Strassen 2x2 + linear tail)."""
    rank = 26
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    col = 0

    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    # Strassen 2x2 (7 terms) for the top-left 2x2 block
    U[0][col]=1; U[4][col]=1; V[0][col]=1; V[4][col]=1; W[0][col]=1; W[4][col]=1; col+=1
    U[3][col]=1; U[4][col]=1; V[0][col]=1; W[1][col]=1; W[4][col]=-1; col+=1
    U[0][col]=1; V[1][col]=1; V[4][col]=-1; W[3][col]=1; W[4][col]=1; col+=1
    U[4][col]=1; V[3][col]=1; V[0][col]=-1; W[0][col]=1; W[1][col]=1; col+=1
    U[0][col]=1; U[1][col]=1; V[4][col]=1; W[0][col]=-1; W[3][col]=1; col+=1
    U[3][col]=1; U[0][col]=-1; V[0][col]=1; V[1][col]=1; W[4][col]=1; col+=1
    U[1][col]=1; U[4][col]=-1; V[3][col]=1; V[4][col]=1; W[0][col]=1; col+=1

    # Linear tail terms (19 more)
    add_term(2,6,0,1); add_term(2,7,3,1)
    add_term(5,6,1,1); add_term(5,7,4,1)
    add_term(0,2,6,1); add_term(1,5,6,1)
    add_term(3,2,7,1); add_term(4,5,7,1)
    add_term(2,8,6,1); add_term(5,8,7,1)
    add_term(6,0,2,1); add_term(7,3,2,1)
    add_term(6,1,5,1); add_term(7,4,5,1)
    add_term(8,6,2,1); add_term(8,7,5,1)
    add_term(6,2,8,1); add_term(7,5,8,1)
    add_term(8,8,8,1)

    return {"U": U, "V": V, "W": W}

def _polish_to_half_integers(U, V, W, T_target, max_iter=50, tol=1e-12):
    """
    Try to polish a real-valued decomposition to have half-integer entries
    while preserving the tensor identity. Uses iterative rounding + correction.
    """
    U = U.copy().astype(float)
    V = V.copy().astype(float)
    W = W.copy().astype(float)
    rank = U.shape[1]

    for iteration in range(max_iter):
        # Round to nearest half-integer
        U_rounded = np.round(U * 2) / 2.0
        V_rounded = np.round(V * 2) / 2.0
        W_rounded = np.round(W * 2) / 2.0

        # Check if already good
        T_calc = np.einsum('ar,br,cr->cab', U_rounded, V_rounded, W_rounded)
        err = np.max(np.abs(T_calc - T_target))
        if err < tol:
            return U_rounded, V_rounded, W_rounded, True

        # Compute the residual
        residual = T_target - T_calc

        # Try a small correction: adjust one factor at a time
        # Use least squares to find small corrections delta_U, delta_V, delta_W
        # such that sum_r (delta_U[a,r]*V[b,r]*W[c,r] + U[a,r]*delta_V[b,r]*W[c,r]
        #                        + U[a,r]*V[b,r]*delta_W[c,r]) ≈ residual[c,a,b]

        # This is linear in each delta when the others are fixed.
        # We'll do alternating least squares for the correction.

        # Fix V, W: solve for delta_U
        # residual[c,a,b] = sum_r delta_U[a,r] * V[b,r] * W[c,r]
        # For each a: residual[:,a,:] = sum_r delta_U[a,r] * (V[:,r] * W[:,r])^T
        # This is: for each a, we need delta_U[a,:] such that
        # sum_r delta_U[a,r] * V[:,r] * W[:,r] ≈ residual[:,a,:]
        # Let M_a = outer(V[:,r], W[:,r]) summed... actually:
        # For fixed a: residual[:,a,:] is a 9x9 matrix (c,b)
        # We need sum_r delta_U[a,r] * v_r * w_r^T where v_r = V[:,r], w_r = W[:,r]
        # So: residual[:,a,:] ≈ sum_r delta_U[a,r] * np.outer(V[:,r], W[:,r])
        # This is a linear system: A_a * delta_U[a,:] = residual[:,a,:].flatten()
        # where A_a has columns = np.outer(V[:,r], W[:,r]).flatten() for r in range(rank)

        # Similarly for V and W.

        # For efficiency, precompute the Gram-like matrices
        # Actually, let's just do a simple approach:
        # Compute gradient of ||T_calc - T_target||^2 w.r.t. U, V, W
        # and take a small step.

        T_calc = np.einsum('ar,br,cr->cab', U, V, W)
        diff = T_calc - T_target

        # Gradient w.r.t. U: dU[a,r] = sum_{c,b} diff[c,a,b] * V[b,r] * W[c,r]
        dU = np.einsum('cab,br,cr->ar', diff, V, W)
        dV = np.einsum('cab,ar,cr->br', diff, U, W)
        dW = np.einsum('cab,ar,br->cr', diff, U, V)

        # Step size: use a simple adaptive step
        step = 0.1 / (1 + iteration * 0.5)
        U -= step * dU
        V -= step * dV
        W -= step * dW

        # Periodically re-round to keep values near half-integers
        if iteration % 5 == 4:
            # Soft pull towards half-integers
            U = 0.5 * U + 0.5 * np.round(U * 2) / 2.0
            V = 0.5 * V + 0.5 * np.round(V * 2) / 2.0
            W = 0.5 * W + 0.5 * np.round(W * 2) / 2.0

    # Final check
    U_rounded = np.round(U * 2) / 2.0
    V_rounded = np.round(V * 2) / 2.0
    W_rounded = np.round(W * 2) / 2.0
    T_calc = np.einsum('ar,br,cr->cab', U_rounded, V_rounded, W_rounded)
    err = np.max(np.abs(T_calc - T_target))
    return U_rounded, V_rounded, W_rounded, err < 1e-9

def _try_rank_23_numeric(T_target, deadline):
    """
    Attempt to find a rank-23 decomposition via numerical optimization.
    Uses multiple random restarts with gradient-based polishing.
    Returns (U, V, W) if successful, None otherwise.
    """
    rank = 23
    T_flat = T_target.flatten()
    dim = 27 * rank  # 9*23 * 3

    def residuals(x):
        U = x[0:9*rank].reshape(9, rank)
        V = x[9*rank:18*rank].reshape(9, rank)
        W = x[18*rank:27*rank].reshape(9, rank)
        Tc = np.einsum('ar,br,cr->cab', U, V, W)
        return Tc.flatten() - T_flat

    best_result = None
    best_cost = float('inf')

    seeds = [42, 123, 456, 789, 1000, 2024, 31337, 999, 7777, 12345]
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        np.random.seed(seed)
        x0 = np.random.randn(dim) * 0.3

        try:
            from scipy.optimize import least_squares
            res = least_squares(residuals, x0, method='lm', max_nfev=2000, ftol=1e-14, xtol=1e-14, gtol=1e-14)
            if res.cost < best_cost:
                best_cost = res.cost
                best_result = res
                if res.cost < 1e-10:
                    break
        except Exception:
            continue

    if best_result is None:
        return None

    U = best_result.x[0:9*rank].reshape(9, rank)
    V = best_result.x[9*rank:18*rank].reshape(9, rank)
    W = best_result.x[18*rank:27*rank].reshape(9, rank)

    # Polish to half-integers
    U_r, V_r, W_r, ok = _polish_to_half_integers(U, V, W, T_target, max_iter=100, tol=1e-11)
    if ok:
        return U_r, V_r, W_r

    # Also try direct rounding of the optimized solution
    U_d = np.round(U * 2) / 2.0
    V_d = np.round(V * 2) / 2.0
    W_d = np.round(W * 2) / 2.0
    T_calc = np.einsum('ar,br,cr->cab', U_d, V_d, W_d)
    if np.max(np.abs(T_calc - T_target)) < 1e-9:
        return U_d, V_d, W_d

    return None

def construct():
    _t0 = time.time()
    deadline = _t0 + 130  # Safety margin before 150s kill

    T_target = _build_target_tensor()

    # First, try to find a rank-23 solution
    try:
        result = _try_rank_23_numeric(T_target, deadline)
        if result is not None:
            U, V, W = result
            ok, err = _verify(U, V, W)
            if ok:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # Fallback to rank 26
    r26 = _get_rank_26()
    ok, err = _verify(r26["U"], r26["V"], r26["W"])
    return r26
# EVOLVE-BLOCK-END
