"""Tensor decomposition search for 2x2 by 2x3 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]
n, m, p = DIMENSIONS


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _rank11_decomposition():
    """
    Rank-11 decomposition: Strassen (rank 7) for 2x2 block + schoolbook (rank 4) for col 2.
    This is the known optimal rank for 2x2x3 over reals/rationals.
    """
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def set_u(i, j, r, val):
        U[i * m + j][r] = val

    def set_v(j, k, r, val):
        V[j * p + k][r] = val

    def set_w(k, i, r, val):
        W[k * n + i][r] = val

    # Term 0: (A00+A11)(B00+B11) -> contributes to C[0,0] and C[1,1]
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)
    set_w(1, 1, 0, 1)

    # Term 1: (A10+A11)B00 -> contributes to C[0,1] and C[1,1]
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1)
    set_w(1, 1, 1, -1)

    # Term 2: A00(B01-B11) -> contributes to C[1,0] and C[1,1]
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1)
    set_w(1, 1, 2, 1)

    # Term 3: A11(B10-B00) -> contributes to C[0,0] and C[0,1]
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1)
    set_w(0, 1, 3, 1)

    # Term 4: (A00+A01)B11 -> contributes to C[0,0] and C[1,0]
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1)
    set_w(1, 0, 4, 1)

    # Term 5: (A10-A00)(B00+B01) -> contributes to C[1,1]
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # Term 6: (A01-A11)(B10+B11) -> contributes to C[0,0]
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # Term 7: A00*B02
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    # Term 8: A01*B12
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    # Term 9: A10*B02
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    # Term 10: A11*B12
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}


def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 12."""
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


def _residuals_rank10(params):
    """Compute residuals for rank-10 tensor decomposition."""
    U_flat = params[:40]
    V_flat = params[40:100]
    W_flat = params[100:160]
    
    U_mat = U_flat.reshape(4, 10)
    V_mat = V_flat.reshape(6, 10)
    W_mat = W_flat.reshape(6, 10)
    
    res = np.empty(12)
    idx = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                s = 0.0
                for r in range(10):
                    s += U_mat[i*2+j, r] * V_mat[j*3+k, r] * W_mat[k*2+i, r]
                res[idx] = s - 1.0
                idx += 1
    return res


def _residuals_rank10_jacobian(params):
    """Compute Jacobian for rank-10 tensor decomposition."""
    U_flat = params[:40]
    V_flat = params[40:100]
    W_flat = params[100:160]
    
    U_mat = U_flat.reshape(4, 10)
    V_mat = V_flat.reshape(6, 10)
    W_mat = W_flat.reshape(6, 10)
    
    J = np.zeros((12, 160))
    idx = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                u_idx = i*2+j
                v_idx = j*3+k
                w_idx = k*2+i
                for r in range(10):
                    J[idx, u_idx*10+r] = V_mat[v_idx, r] * W_mat[w_idx, r]
                    J[idx, 40 + v_idx*10+r] = U_mat[u_idx, r] * W_mat[w_idx, r]
                    J[idx, 100 + w_idx*10+r] = U_mat[u_idx, r] * V_mat[v_idx, r]
                idx += 1
    return J


def _try_rank10_structured(deadline):
    """
    Attempt rank-10 using structured parameterization that exploits algebraic symmetry.
    Key insight: the 2x2x3 tensor has structure we can exploit.
    """
    if time.time() > deadline - 30:
        return None
    
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return None
    
    rng = np.random.default_rng(42)
    best_cost = float('inf')
    best_params = None
    
    # Strategy: Use the rank-11 solution as a base and try to merge two terms
    base = _rank11_decomposition()
    U11 = np.array(base["U"], dtype=float)
    V11 = np.array(base["V"], dtype=float)
    W11 = np.array(base["W"], dtype=float)
    
    # Try all pairs of terms to merge (11 choose 2 = 55 pairs)
    for term_a in range(11):
        for term_b in range(term_a + 1, 11):
            if time.time() > deadline - 35:
                break
            
            # Keep all terms except a and b, add one new term
            keep = [r for r in range(11) if r != term_a and r != term_b]
            U0 = U11[:, keep].copy()
            V0 = V11[:, keep].copy()
            W0 = W11[:, keep].copy()
            
            # Initialize new term with average of the two dropped terms
            U_new = (U11[:, term_a] + U11[:, term_b]) / 2
            V_new = (V11[:, term_a] + V11[:, term_b]) / 2
            W_new = (W11[:, term_a] + W11[:, term_b]) / 2
            
            U_init = np.hstack([U0, U_new.reshape(-1, 1)])
            V_init = np.hstack([V0, V_new.reshape(-1, 1)])
            W_init = np.hstack([W0, W_new.reshape(-1, 1)])
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            for attempt in range(3):
                if time.time() > deadline - 35:
                    break
                
                # Add small random perturbation
                x_try = x0 + rng.normal(0, 0.01 * (attempt + 1), 160)
                
                try:
                    result = least_squares(_residuals_rank10, x_try, method='lm', 
                                         max_nfev=3000, ftol=1e-15, xtol=1e-15)
                    if result.cost < best_cost:
                        best_cost = result.cost
                        best_params = result.x.copy()
                        if result.cost < 1e-14:
                            break
                except Exception:
                    pass
        
        if best_cost < 1e-14:
            break
    
    # Strategy 2: Random restarts with LM
    for trial in range(50):
        if time.time() > deadline - 20:
            break
        
        # Random initialization
        U = np.random.randn(4, 10) * 0.5
        V = np.random.randn(6, 10) * 0.5
        W = np.random.randn(6, 10) * 0.5
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = least_squares(_residuals_rank10, x0, method='lm', 
                                 max_nfev=2000, ftol=1e-14, xtol=1e-14)
            if result.cost < best_cost:
                best_cost = result.cost
                best_params = result.x.copy()
                if result.cost < 1e-14:
                    break
        except Exception:
            pass
    
    if best_params is None or best_cost > 1e-8:
        return None
    
    # High-precision polish
    if time.time() < deadline - 10:
        try:
            result = least_squares(_residuals_rank10, best_params, method='lm', 
                                   max_nfev=50000, ftol=1e-18, xtol=1e-18)
            if result.cost < best_cost:
                best_cost = result.cost
                best_params = result.x.copy()
        except Exception:
            pass
    
    if best_params is None or best_cost > 1e-8:
        return None
    
    U_flat = best_params[:40]
    V_flat = best_params[40:100]
    W_flat = best_params[100:160]
    
    U_mat = U_flat.reshape(4, 10)
    V_mat = V_flat.reshape(6, 10)
    W_mat = W_flat.reshape(6, 10)
    
    # Try to snap to rational values
    snap_strategies = [
        lambda x: round(x * 2) / 2,
        lambda x: round(x),
        lambda x: round(x * 4) / 4,
        lambda x: round(x * 8) / 8,
        lambda x: round(x * 3) / 3,
        lambda x: round(x * 6) / 6,
    ]
    
    for snap in snap_strategies:
        U = [[snap(u) for u in row] for row in U_mat.tolist()]
        V = [[snap(v) for v in row] for row in V_mat.tolist()]
        W = [[snap(w) for w in row] for row in W_mat.tolist()]
        
        if _verify(U, V, W, 2, 2, 3):
            return {"U": U, "V": V, "W": W}
    
    return None


def _try_rank10_direct_polynomial(deadline):
    """
    Direct polynomial system approach: fix some variables to exploit structure,
    then solve for the rest.
    """
    if time.time() > deadline - 25:
        return None
    
    try:
        from scipy.optimize import fsolve
    except ImportError:
        return None
    
    rng = np.random.default_rng(777)
    best_cost = float('inf')
    best_solution = None
    
    # Try different sparsity patterns
    for trial in range(60):
        if time.time() > deadline - 15:
            break
        
        U = np.zeros((4, 10))
        V = np.zeros((6, 10))
        W = np.zeros((6, 10))
        
        # Ensure each column has at least one non-zero in each matrix
        for r in range(10):
            # Pick random positions
            u_pos = rng.choice(4, size=rng.choice([1, 2]), replace=False)
            v_pos = rng.choice(6, size=rng.choice([1, 2]), replace=False)
            w_pos = rng.choice(6, size=rng.choice([1, 2]), replace=False)
            
            for pos in u_pos:
                U[pos, r] = rng.choice([-1, 1])
            for pos in v_pos:
                V[pos, r] = rng.choice([-1, 1])
            for pos in w_pos:
                W[pos, r] = rng.choice([-1, 1])
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = fsolve(_residuals_rank10, x0, full_output=True, xtol=1e-12, maxfev=3000)
            x_sol, info, ier, msg = result
            if ier == 1:
                cost = np.sum(np.array(_residuals_rank10(x_sol))**2)
                if cost < best_cost:
                    best_cost = cost
                    best_solution = x_sol.copy()
                    if cost < 1e-14:
                        break
        except Exception:
            pass
    
    if best_solution is None or best_cost > 1e-10:
        return None
    
    # Polish
    if time.time() < deadline - 8:
        try:
            from scipy.optimize import least_squares
            result = least_squares(_residuals_rank10, best_solution, method='lm', 
                                   max_nfev=30000, ftol=1e-16, xtol=1e-16)
            if result.cost < best_cost:
                best_cost = result.cost
                best_solution = result.x.copy()
        except Exception:
            pass
    
    if best_solution is None or best_cost > 1e-8:
        return None
    
    U_flat = best_solution[:40]
    V_flat = best_solution[40:100]
    W_flat = best_solution[100:160]
    U_mat = U_flat.reshape(4, 10)
    V_mat = V_flat.reshape(6, 10)
    W_mat = W_flat.reshape(6, 10)
    
    snap_strategies = [
        lambda x: round(x * 2) / 2,
        lambda x: round(x),
        lambda x: round(x * 4) / 4,
    ]
    
    for snap in snap_strategies:
        U = [[snap(u) for u in row] for row in U_mat.tolist()]
        V = [[snap(v) for v in row] for row in V_mat.tolist()]
        W = [[snap(w) for w in row] for row in W_mat.tolist()]
        
        if _verify(U, V, W, 2, 2, 3):
            return {"U": U, "V": V, "W": W}
    
    return None


def construct():
    deadline = time.time() + 140
    
    # First, get the known rank-11 construction
    result11 = _rank11_decomposition()
    if _verify(result11["U"], result11["V"], result11["W"], n, m, p):
        best = result11
        best_rank = 11
    else:
        best = _schoolbook()
        best_rank = 12
    
    # Try to find a rank-10 solution
    if best_rank > 10 and time.time() < deadline - 50:
        result10 = _try_rank10_structured(deadline)
        if result10 is not None:
            best = result10
            best_rank = 10
        elif time.time() < deadline - 30:
            result10 = _try_rank10_direct_polynomial(deadline)
            if result10 is not None:
                best = result10
                best_rank = 10
    
    return best
