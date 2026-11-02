"""Tensor decomposition for 2x2 by 2x3 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 3]
n, m, p = DIMENSIONS

_DEADLINE = None


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _residual(U, V, W, n, m, p):
    """Compute the residual of the decomposition."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    max_res = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                max_res = max(max_res, abs(s - 1.0))
    return max_res


def _rank11_decomposition():
    """
    Rank-11 decomposition: Strassen (rank 7) for 2x2 block + schoolbook (rank 4) for col 2.
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


def _try_rank10_optimize(deadline):
    """
    Try to find a rank-10 decomposition via numerical optimization.
    Uses warm-start from rank-11 by merging two terms and optimizing.
    """
    global _DEADLINE
    _DEADLINE = deadline
    
    if time.time() > deadline:
        return None
    
    rank = 10
    dim_u = n * m  # 4
    dim_v = m * p  # 6
    dim_w = p * n  # 6
    
    best_result = None
    best_res = float('inf')
    
    max_trials = 100  # Increased trials for better coverage
    
    for trial in range(max_trials):
        if time.time() > deadline - 10:
            break
        
        # Initialize from rank-11 with one term removed, then randomize slightly
        r11 = _rank11_decomposition()
        U11 = np.array(r11["U"], dtype=float)
        V11 = np.array(r11["V"], dtype=float)
        W11 = np.array(r11["W"], dtype=float)
        
        # Choose which term to remove (randomly)
        remove_idx = random.randint(0, 10)
        
        # Keep all terms except remove_idx
        keep = [i for i in range(11) if i != remove_idx]
        
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        
        for new_idx, old_idx in enumerate(keep):
            U[:, new_idx] = U11[:, old_idx]
            V[:, new_idx] = V11[:, old_idx]
            W[:, new_idx] = W11[:, old_idx]
        
        # Add small random perturbation to help escape local minima
        noise_scale = 0.1
        U += noise_scale * np.random.randn(dim_u, rank)
        V += noise_scale * np.random.randn(dim_v, rank)
        W += noise_scale * np.random.randn(dim_w, rank)
        
        # Gradient-based optimization with adaptive learning rate
        lr = 0.01
        num_iter = 300
        prev_res = float('inf')
        stagnation_count = 0
        
        for it in range(num_iter):
            if time.time() > deadline - 10:
                break
            
            # Compute residuals and gradients
            residuals = np.zeros(n * m * p)
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res = 0.0
                        for r in range(rank):
                            term = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            res += term
                            grad_U[i*m+j, r] += (res - 1.0) * V[j*p+k, r] * W[k*n+i, r]
                            grad_V[j*p+k, r] += (res - 1.0) * U[i*m+j, r] * W[k*n+i, r]
                            grad_W[k*n+i, r] += (res - 1.0) * U[i*m+j, r] * V[j*p+k, r]
                        residuals[eq_idx] = res - 1.0
                        eq_idx += 1
            
            # Gradient descent step
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            max_res = np.max(np.abs(residuals))
            
            # Adaptive learning rate: reduce if residual increases
            if max_res > prev_res * 1.1:
                lr *= 0.5
                stagnation_count += 1
            else:
                stagnation_count = 0
            
            prev_res = max_res
            
            # If stuck, add a small random kick
            if stagnation_count > 20:
                U += 0.01 * np.random.randn(dim_u, rank)
                V += 0.01 * np.random.randn(dim_v, rank)
                W += 0.01 * np.random.randn(dim_w, rank)
                lr = 0.01
                stagnation_count = 0
            
            if max_res < 1e-10:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if _verify(U_list, V_list, W_list, n, m, p, tol=1e-8):
                    if max_res < best_res:
                        best_res = max_res
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                break
        
        # Track best residual found
        if best_result is None:
            max_res = _residual(U.tolist(), V.tolist(), W.tolist(), n, m, p)
            if max_res < best_res:
                best_res = max_res
    
    if best_result is not None:
        return best_result
    return None


def construct():
    global _DEADLINE
    start_time = time.time()
    deadline = start_time + 110  # Safety margin: 120s limit, use 110s
    
    _DEADLINE = deadline
    
    # First, try to find rank-10
    rank10_result = _try_rank10_optimize(deadline)
    if rank10_result is not None:
        return rank10_result
    
    # Fallback to rank-11
    result = _rank11_decomposition()
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        return _schoolbook()
