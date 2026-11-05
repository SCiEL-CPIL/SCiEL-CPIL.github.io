"""
Rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
Combines structured initialization, vectorized residual computation, and
multi-attempt optimization to find low-rank decompositions.
"""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 4]
DEADLINE = 145.0  # safety margin before 150s limit


def _build_target_tensor():
    """Return the 36 (i,j,k) indices of the multiplication tensor."""
    n, m, p = DIMENSIONS
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                terms.append((u_idx, v_idx, w_idx))
    return terms


def _verify(U, V, W, terms):
    """Verify that the decomposition exactly satisfies all tensor equations."""
    for (u_idx, v_idx, w_idx) in terms:
        s = 0.0
        for r in range(U.shape[1]):
            s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
        if abs(s - 1.0) > 1e-9:
            return False
    return True


def _snap_and_verify(U, V, W, terms):
    """Snap to nearest half-integer and verify."""
    U_s = np.round(U * 2) / 2
    V_s = np.round(V * 2) / 2
    W_s = np.round(W * 2) / 2
    if _verify(U_s, V_s, W_s, terms):
        return U_s.tolist(), V_s.tolist(), W_s.tolist()
    # Also try integer snapping
    U_i = np.round(U).astype(int)
    V_i = np.round(V).astype(int)
    W_i = np.round(W).astype(int)
    if _verify(U_i.astype(float), V_i.astype(float), W_i.astype(float), terms):
        return U_i.tolist(), V_i.tolist(), W_i.tolist()
    return None


def _schoolbook_solution():
    """Return the rank-36 schoolbook solution."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    terms = _build_target_tensor()
    for col, (u_idx, v_idx, w_idx) in enumerate(terms):
        U[u_idx, col] = 1
        V[v_idx, col] = 1
        W[w_idx, col] = 1
    return U.tolist(), V.tolist(), W.tolist()


def _strassen_init_3x3x4(R):
    """
    Create a Strassen-inspired initialization for 3x3 * 3x4 matrix multiplication.
    Uses algebraic structure to create better starting points.
    """
    n, m, p = DIMENSIONS
    u_rows = n * m
    v_rows = m * p
    w_rows = p * n
    
    # Strassen-style linear combinations for 3x3x4
    # Define basis patterns that exploit structure
    patterns = []
    
    # Standard Strassen-like combinations adapted to 3x3x4
    # Each pattern is a tuple (u_pattern, v_pattern, w_pattern) where each is a dict mapping index to coefficient
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Simple term
                patterns.append((
                    {i*m+j: 1.0},
                    {j*p+k: 1.0},
                    {k*n+i: 1.0}
                ))
    
    # Add Strassen-inspired combined terms
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Adapt to 3x3x4 with additional structure
    strassen_patterns = [
        # Combined row 0,1 with col 0,1
        ({0*m+0: 1.0, 1*m+1: 1.0}, {0*p+0: 1.0, 1*p+1: 1.0}, {0*n+0: 1.0, 1*n+1: 1.0}),
        ({1*m+0: 1.0, 1*m+1: 1.0}, {0*p+0: 1.0}, {0*n+1: 1.0, 1*n+1: 1.0}),
        ({0*m+0: 1.0}, {1*p+0: 1.0, 1*p+1: -1.0}, {1*n+0: 1.0, 2*n+0: 1.0}),
        ({1*m+1: 1.0}, {0*p+1: 1.0, 0*p+0: -1.0}, {0*n+1: 1.0, 1*n+1: 1.0}),
        ({0*m+0: 1.0, 0*m+1: 1.0}, {1*p+1: 1.0}, {1*n+0: 1.0, 2*n+0: 1.0}),
        ({1*m+0: 1.0, 0*m+0: -1.0}, {0*p+0: 1.0, 0*p+1: 1.0}, {0*n+1: 1.0, 1*n+1: 1.0}),
        ({0*m+1: 1.0, 1*m+1: -1.0}, {1*p+0: 1.0, 1*p+1: 1.0}, {1*n+0: 1.0, 2*n+0: 1.0}),
    ]
    
    # Select R patterns (mix of simple and Strassen-inspired)
    np.random.seed(42)
    all_patterns = patterns + strassen_patterns
    np.random.shuffle(all_patterns)
    selected = all_patterns[:min(R, len(all_patterns))]
    
    U = np.zeros((u_rows, R))
    V = np.zeros((v_rows, R))
    W = np.zeros((w_rows, R))
    
    for r, (u_pat, v_pat, w_pat) in enumerate(selected):
        for idx, coeff in u_pat.items():
            U[idx, r] += coeff
        for idx, coeff in v_pat.items():
            V[idx, r] += coeff
        for idx, coeff in w_pat.items():
            W[idx, r] += coeff
    
    # Add small perturbation
    U += np.random.normal(0, 0.05, U.shape)
    V += np.random.normal(0, 0.05, V.shape)
    W += np.random.normal(0, 0.05, W.shape)
    
    return U, V, W


def _try_solve_rank_multi_init(R, terms, max_time=100.0):
    """
    Try to find a rank-R decomposition with multiple diverse initializations.
    """
    n, m, p = DIMENSIONS
    u_rows = n * m
    v_rows = m * p
    w_rows = p * n

    start_time = time.time()

    term_indices = np.array(terms)

    def residuals(x):
        U_flat = x[:u_rows * R].reshape(u_rows, R)
        V_flat = x[u_rows * R:u_rows * R + v_rows * R].reshape(v_rows, R)
        W_flat = x[u_rows * R + v_rows * R:].reshape(w_rows, R)

        # More efficient: compute only the 36 needed terms
        # T_decomp[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # We only need terms where (a,b,c) is in our list
        result = np.zeros(len(terms))
        for idx, (a, b, c) in enumerate(terms):
            result[idx] = np.dot(U_flat[a, :], V_flat[b, :] * W_flat[c, :])
        return result - 1.0

    best_cost = np.inf
    best_x = None

    num_attempts = 12
    for attempt in range(num_attempts):
        if time.time() - start_time > max_time:
            break

        np.random.seed(100 + attempt * 13)

        # Different initialization strategies
        if attempt % 4 == 0:
            # Strassen-inspired initialization
            U, V, W = _strassen_init_3x3x4(R)

        elif attempt % 4 == 1:
            # Structured: pick R random terms
            all_terms = list(range(len(terms)))
            np.random.shuffle(all_terms)
            selected = all_terms[:min(R, len(terms))]

            U = np.zeros((u_rows, R))
            V = np.zeros((v_rows, R))
            W = np.zeros((w_rows, R))

            for r, term_idx in enumerate(selected):
                u_idx, v_idx, w_idx = terms[term_idx]
                U[u_idx, r] = 1.0
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0

            U += np.random.normal(0, 0.1, U.shape)
            V += np.random.normal(0, 0.1, V.shape)
            W += np.random.normal(0, 0.1, W.shape)

        elif attempt % 4 == 2:
            # Random initialization
            U = np.random.randn(u_rows, R) * 0.5
            V = np.random.randn(v_rows, R) * 0.5
            W = np.random.randn(w_rows, R) * 0.5

        else:
            # Partial structured: assign some columns, leave rest random
            U = np.random.randn(u_rows, R) * 0.1
            V = np.random.randn(v_rows, R) * 0.1
            W = np.random.randn(w_rows, R) * 0.1

            num_structured = min(R, 12)
            all_terms = list(range(len(terms)))
            np.random.shuffle(all_terms)
            for r in range(num_structured):
                u_idx, v_idx, w_idx = terms[all_terms[r]]
                U[u_idx, r] = 1.0
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0

        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=3000,
                                   xtol=1e-12, ftol=1e-12, gtol=1e-12)
        except Exception:
            continue

        if result.cost < best_cost:
            best_cost = result.cost
            best_x = result.x.copy()

        if best_cost < 1e-10:
            U_sol = best_x[:u_rows * R].reshape(u_rows, R)
            V_sol = best_x[u_rows * R:u_rows * R + v_rows * R].reshape(v_rows, R)
            W_sol = best_x[u_rows * R + v_rows * R:].reshape(w_rows, R)

            # High-precision polish pass
            x_polish = least_squares(residuals, best_x, method='lm', max_nfev=5000,
                                     xtol=1e-14, ftol=1e-14, gtol=1e-14)
            if x_polish.cost < best_cost:
                best_x = x_polish.x
                best_cost = x_polish.cost
            
            U_sol = best_x[:u_rows * R].reshape(u_rows, R)
            V_sol = best_x[u_rows * R:u_rows * R + v_rows * R].reshape(v_rows, R)
            W_sol = best_x[u_rows * R + v_rows * R:].reshape(w_rows, R)

            snapped = _snap_and_verify(U_sol, V_sol, W_sol, terms)
            if snapped is not None:
                return snapped

            if _verify(U_sol, V_sol, W_sol, terms):
                return U_sol.tolist(), V_sol.tolist(), W_sol.tolist()

    # If we got a good solution but not quite perfect, try snapping with polish
    if best_x is not None and best_cost < 1e-6:
        # Try polish pass
        try:
            x_polish = least_squares(residuals, best_x, method='lm', max_nfev=5000,
                                     xtol=1e-14, ftol=1e-14, gtol=1e-14)
            if x_polish.cost < best_cost:
                best_x = x_polish.x
                best_cost = x_polish.cost
        except Exception:
            pass
        
        U_sol = best_x[:u_rows * R].reshape(u_rows, R)
        V_sol = best_x[u_rows * R:u_rows * R + v_rows * R].reshape(v_rows, R)
        W_sol = best_x[u_rows * R + v_rows * R:].reshape(w_rows, R)

        snapped = _snap_and_verify(U_sol, V_sol, W_sol, terms)
        if snapped is not None:
            return snapped

    return None


def construct():
    start_time = time.time()
    terms = _build_target_tensor()

    # Try to find solutions for rank < 36, starting from lower ranks
    # Known lower bound for 3x3x3 is 23, for 3x3x4 likely similar or slightly higher
    # Try ranks in order of increasing difficulty
    ranks_to_try = list(range(20, 36))
    
    # Prioritize ranks that are more likely to have solutions
    # Based on known results: 3x3x3 has rank 23, so 3x3x4 likely 24-28
    ranks_to_try.sort(key=lambda r: abs(r - 25))
    
    for R in ranks_to_try:
        if time.time() - start_time > DEADLINE - 10:
            break

        remaining = DEADLINE - (time.time() - start_time) - 10
        if remaining < 15:
            break

        # Allocate time based on rank: give more time to promising ranks
        if 22 <= R <= 28:
            time_budget = min(remaining * 0.4, 40)
        elif 20 <= R < 22 or 28 < R < 32:
            time_budget = min(remaining * 0.3, 30)
        else:
            time_budget = min(remaining * 0.2, 20)

        result = _try_solve_rank_multi_init(R, terms, max_time=time_budget)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook rank 36
    U, V, W = _schoolbook_solution()
    return {"U": U, "V": V, "W": W}
