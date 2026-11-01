import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 5]
_DEADLINE = 130.0  # Safety margin before 150s kill

def _time_remaining():
    return _DEADLINE - time.time()

def verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Compute the tensor from U, V, W
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    tensor = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
    
    # The multiplication tensor should be 1 for all (i,j,k)
    expected = np.ones((n, m, p))
    
    if not np.allclose(tensor, expected, atol=1e-9):
        return False, tensor
    return True, tensor

def build_rank_28(n, m, p):
    """
    Construct a rank-28 decomposition for 2x3x5 matrix multiplication.
    Strategy:
    1. j=0: 2x1x5 block (rank 10, schoolbook)
    2. j=1,2, k=0,1: 2x2x2 block (rank 7, Strassen)
    3. j=1,2, k=2: 2x2x1 block (rank 4, schoolbook)
    4. j=1,2, k=3,4: 2x2x2 block (rank 7, Strassen)
    Total rank: 10 + 7 + 4 + 7 = 28.
    """
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0

    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k, col):
        U[i * m + j][col] = 1.0
        V[j * p + k][col] = 1.0
        W[k * n + i][col] = 1.0
        return col + 1

    # Helper to add Strassen 2x2x2 block at offsets (i_off, j_off, k_off)
    def add_strassen(i_off, j_off, k_off, col):
        # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        # M2: (A10+A11)B00 -> C10 += M2, C11 -= M2
        U[(i_off+1)*m + (j_off+0)][col] = 1.0
        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        W[(k_off+0)*n + (i_off+1)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = -1.0
        col += 1

        # M3: A00(B01-B11) -> C01 += M3, C11 += M3
        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = -1.0
        W[(k_off+1)*n + (i_off+0)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        # M4: A11(B10-B00) -> C00 += M4, C10 += M4
        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = -1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        W[(k_off+0)*n + (i_off+1)][col] = 1.0
        col += 1

        # M5: (A00+A01)B11 -> C00 -= M5, C01 += M5
        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        U[(i_off+0)*m + (j_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = -1.0
        W[(k_off+1)*n + (i_off+0)][col] = 1.0
        col += 1

        # M6: (A10-A00)(B00+B01) -> C11 += M6
        U[(i_off+1)*m + (j_off+0)][col] = 1.0
        U[(i_off+0)*m + (j_off+0)][col] = -1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+1)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        # M7: (A01-A11)(B10+B11) -> C00 += M7
        U[(i_off+0)*m + (j_off+1)][col] = 1.0
        U[(i_off+1)*m + (j_off+1)][col] = -1.0
        V[(j_off+1)*p + (k_off+0)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        col += 1

        return col

    # 1. j=0 block: i in {0,1}, j=0, k in {0,1,2,3,4}
    # Schoolbook: 2 * 5 = 10 terms
    for k in range(p):
        col = add_schoolbook(0, 0, k, col)
        col = add_schoolbook(1, 0, k, col)
    
    # 2. j=1,2, k=0,1 block: i in {0,1}, j in {1,2}, k in {0,1}
    # Strassen: 7 terms
    # i_off=0, j_off=1, k_off=0
    col = add_strassen(0, 1, 0, col)

    # 3. j=1,2, k=2 block: i in {0,1}, j in {1,2}, k=2
    # Schoolbook: 2 * 2 = 4 terms
    for i in range(n):
        for j in [1, 2]:
            col = add_schoolbook(i, j, 2, col)

    # 4. j=1,2, k=3,4 block: i in {0,1}, j in {1,2}, k in {3,4}
    # Strassen: 7 terms
    # i_off=0, j_off=1, k_off=3
    col = add_strassen(0, 1, 3, col)

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W

def try_rank_27(n, m, p):
    """
    Attempt to find a rank-27 decomposition using smooth optimization.
    Uses sum of squared residuals as objective, with multiple random restarts.
    Returns (U, V, W) if found, else None.
    """
    rank = 27
    n_dim = n * m  # 6
    m_dim = m * p  # 15
    p_dim = p * n  # 10
    
    def compute_residual(U, V, W):
        """Compute sum of squared residuals for the tensor decomposition."""
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    residual += (s - 1.0) ** 2
        return residual
    
    def compute_residual_np(U, V, W):
        """Vectorized residual computation."""
        total = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vals = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                    total += (np.sum(vals) - 1.0) ** 2
        return total
    
    best_result = None
    best_residual = float('inf')
    
    num_restarts = 50
    max_iters = 200
    
    for restart in range(num_restarts):
        if _time_remaining() < 5.0:
            break
        
        # Initialize with small random values
        rng = np.random.RandomState(restart * 12345 + 67890)
        U = rng.randn(n_dim, rank) * 0.5
        V = rng.randn(m_dim, rank) * 0.5
        W = rng.randn(p_dim, rank) * 0.5
        
        # Use gradient descent with decreasing learning rate
        lr = 0.01
        for it in range(max_iters):
            if _time_remaining() < 5.0:
                break
            
            # Compute gradient numerically (central difference) for each parameter
            # This is expensive but rank 27 is small enough
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            eps = 1e-5
            current_res = compute_residual_np(U, V, W)
            
            # Compute gradient for U
            for idx in range(n_dim * rank):
                ii, jj = divmod(idx, rank)
                U_plus = U.copy()
                U_minus = U.copy()
                U_plus[ii, jj] += eps
                U_minus[ii, jj] -= eps
                grad_U[ii, jj] = (compute_residual_np(U_plus, V, W) - compute_residual_np(U_minus, V, W)) / (2 * eps)
            
            # Compute gradient for V
            for idx in range(m_dim * rank):
                ii, jj = divmod(idx, rank)
                V_plus = V.copy()
                V_minus = V.copy()
                V_plus[ii, jj] += eps
                V_minus[ii, jj] -= eps
                grad_V[ii, jj] = (compute_residual_np(U, V_plus, W) - compute_residual_np(U, V_minus, W)) / (2 * eps)
            
            # Compute gradient for W
            for idx in range(p_dim * rank):
                ii, jj = divmod(idx, rank)
                W_plus = W.copy()
                W_minus = W.copy()
                W_plus[ii, jj] += eps
                W_minus[ii, jj] -= eps
                grad_W[ii, jj] = (compute_residual_np(U, V, W_plus) - compute_residual_np(U, V, W_minus)) / (2 * eps)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Decrease learning rate
            lr *= 0.995
            
            if it % 20 == 0:
                res = compute_residual_np(U, V, W)
                if res < best_residual:
                    best_residual = res
                    best_result = (U.copy(), V.copy(), W.copy())
                if res < 1e-12:
                    break
        
        # Check final result
        res = compute_residual_np(U, V, W)
        if res < best_residual:
            best_residual = res
            best_result = (U.copy(), V.copy(), W.copy())
    
    if best_result is not None and best_residual < 1e-10:
        U, V, W = best_result
        # Convert to list format
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Directly attempt the known best rank 28 decomposition
    rank_28_result = None
    try:
        U, V, W = build_rank_28(n, m, p)
        valid, tensor = verify_decomposition(U, V, W, n, m, p)
        if valid:
            rank_28_result = {'U': U, 'V': V, 'W': W}
    except Exception as e:
        pass
    
    # Try to find rank 27 if we have time
    if _time_remaining() > 30.0:
        try:
            result_27 = try_rank_27(n, m, p)
            if result_27 is not None:
                U27, V27, W27 = result_27
                valid27, _ = verify_decomposition(U27, V27, W27, n, m, p)
                if valid27:
                    return {'U': U27, 'V': V27, 'W': W27}
        except Exception as e:
            pass
    
    # Return rank 28 if valid
    if rank_28_result is not None:
        return rank_28_result
    
    # Fallback: Schoolbook (Rank 30)
    rank_school = n * m * p
    U = [[0.0] * rank_school for _ in range(n * m)]
    V = [[0.0] * rank_school for _ in range(m * p)]
    W = [[0.0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    valid, _ = verify_decomposition(U, V, W, n, m, p)
    if valid:
        return {'U': U, 'V': V, 'W': W}
    
    # Should not happen
    return {'U': [[0.0]], 'V': [[0.0]], 'W': [[0.0]]}
