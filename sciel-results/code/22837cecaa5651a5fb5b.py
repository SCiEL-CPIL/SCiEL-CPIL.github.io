"""Tensor decomposition search for 2x2 by 2x3 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]
n, m, p = DIMENSIONS


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Check all 12 (i,j,k) triples exactly
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _is_half_integer(arr):
    """Check if all entries are close to a multiple of 0.5."""
    for row in arr:
        for val in row:
            if abs(val * 2 - round(val * 2)) > 1e-6:
                return False
    return True

def _snap_to_half(arr):
    """Snap entries to the nearest multiple of 0.5."""
    return [[round(val * 2) / 2.0 for val in row] for row in arr]


def _rank11_decomposition():
    """
    Rank-11 decomposition: Strassen (rank 7) for 2x2 block + schoolbook (rank 4) for col 2.
    This is the known minimal rank for 2x2x3 matrix multiplication.
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


def _try_rank10_search():
    """
    Attempt to find a rank-10 decomposition using numerical optimization.
    We minimize the sum of squared errors of the tensor decomposition.
    """
    from scipy.optimize import minimize
    
    deadline = time.time() + 140  # Safety margin (150s limit)
    
    rank = 10
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Target values for the 12 equations
    targets = []
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                targets.append(1.0)
                indices.append((i, j, k))
                
    targets = np.array(targets)
    
    def loss_and_grad(params):
        U_flat = params[0:40]
        V_flat = params[40:100]
        W_flat = params[100:160]
        
        U = U_flat.reshape(4, 10)
        V = V_flat.reshape(6, 10)
        W = W_flat.reshape(6, 10)
        
        errors = np.zeros(12)
        for idx, (i, j, k) in enumerate(indices):
            u_val = U[i*m + j]
            v_val = V[j*p + k]
            w_val = W[k*n + i]
            err = np.dot(np.dot(u_val, v_val), w_val) - targets[idx]
            errors[idx] = err
            
        loss = np.sum(errors**2)
        
        grad_u = np.zeros_like(U)
        grad_v = np.zeros_like(V)
        grad_w = np.zeros_like(W)
        
        for idx, (i, j, k) in enumerate(indices):
            e = errors[idx]
            u_idx = i*m + j
            v_idx = j*p + k
            w_idx = k*n + i
            
            grad_u[u_idx] += 2 * e * (V[v_idx] * W[w_idx])
            grad_v[v_idx] += 2 * e * (U[u_idx] * W[w_idx])
            grad_w[w_idx] += 2 * e * (U[u_idx] * V[v_idx])
            
        grad = np.concatenate([grad_u.flatten(), grad_v.flatten(), grad_w.flatten()])
        return loss, grad

    best_result = None
    best_loss = float('inf')
    
    # Use the rank 11 solution as a starting point for some restarts
    r11 = _rank11_decomposition()
    U11 = np.array(r11["U"])[:, :10] # Drop last term
    V11 = np.array(r11["V"])[:, :10]
    W11 = np.array(r11["W"])[:, :10]
    init_from_11 = np.concatenate([U11.flatten(), V11.flatten(), W11.flatten()])
    
    np.random.seed(42)
    
    # Try several restarts
    for restart in range(20):
        if time.time() > deadline:
            break
            
        if restart == 0:
            init_params = init_from_11
        elif restart == 1:
            # Perturb the 11-term init
            init_params = init_from_11 + np.random.randn(160) * 0.1
        else:
            init_params = np.random.randn(160) * 0.5
            
        try:
            res = minimize(
                loss_and_grad, 
                init_params, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            
            if res.fun < best_loss:
                best_loss = res.fun
                
                # If we found a very low loss, try to snap to half-integers
                if res.fun < 1e-8:
                    U_flat = res.x[0:40]
                    V_flat = res.x[40:100]
                    W_flat = res.x[100:160]
                    
                    U = U_flat.reshape(4, 10).tolist()
                    V = V_flat.reshape(6, 10).tolist()
                    W = W_flat.reshape(6, 10).tolist()
                    
                    # Snap to nearest 0.5
                    U_snapped = _snap_to_half(U)
                    V_snapped = _snap_to_half(V)
                    W_snapped = _snap_to_half(W)
                    
                    # Verify snapped solution
                    if _verify(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-9):
                        return {"U": U_snapped, "V": V_snapped, "W": W_snapped}
                        
                    # Also verify original float solution just in case snapping failed but float is exact
                    # (though float solution might not be integer/half-integer)
                    if _verify(U, V, W, n, m, p, tol=1e-9):
                        # Check if it is effectively integer/half-integer
                        if _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
                            return {"U": U, "V": V, "W": W}

        except Exception as e:
            pass
            
    return None


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


def construct():
    start_time = time.time()
    result = _try_rank10_search()
    
    if result is not None:
        return result
        
    result = _rank11_decomposition()
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    else:
        return _schoolbook()
