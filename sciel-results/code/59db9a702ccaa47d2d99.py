"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START


def _build_trivial(n, m, p):
    """Return the rank-nmp trivial decomposition as (U, V, W) lists."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r in range(rank):
        i = r // (m * p)
        rem = r % (m * p)
        j = rem // p
        k = rem % p
        idx_u = i * m + j
        idx_v = j * p + k
        idx_w = k * n + i
        U[idx_u][r] = 1
        V[idx_v][r] = 1
        W[idx_w][r] = 1
    return U, V, W, rank


def _validate(n, m, p, U, V, W):
    """Verify that sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1 for all i,j,k."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                if not np.isclose(lhs, 1.0, atol=1e-9):
                    return False
    return True


def _build_strassen(n, m, p):
    """
    Build a rank-55 decomposition for 2x3 * 3x10 using Strassen on 2x2x2 blocks.
    """
    assert n == 2 and m == 3 and p == 10
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    strassen_u = {
        0: [(0, 1), (4, 1)],
        1: [(3, 1), (4, 1)],
        2: [(0, 1)],
        3: [(4, 1)],
        4: [(0, 1), (1, 1)],
        5: [(3, 1), (0, -1)],
        6: [(1, 1), (4, -1)],
    }
    
    strassen_v = {
        0: [(0, 1), (11, 1)],
        1: [(0, 1)],
        2: [(1, 1), (11, -1)],
        3: [(10, 1), (0, -1)],
        4: [(11, 1)],
        5: [(0, 1), (1, 1)],
        6: [(10, 1), (11, 1)],
    }
    
    strassen_w = {
        0: [(0, 1), (3, 1)],
        1: [(1, 1), (3, -1)],
        2: [(2, 1), (3, 1)],
        3: [(0, 1), (1, 1)],
        4: [(0, -1), (2, 1)],
        5: [(3, 1)],
        6: [(0, 1)],
    }
    
    for b in range(5):
        col0 = 2 * b
        col1 = 2 * b + 1
        
        for s in range(7):
            for (u_row, coeff) in strassen_u[s]:
                U[u_row][r] += coeff
            
            for (v_row_offset, coeff) in strassen_v[s]:
                if v_row_offset == 0:
                    v_row = col0
                elif v_row_offset == 1:
                    v_row = col1
                elif v_row_offset == 10:
                    v_row = 10 + col0
                elif v_row_offset == 11:
                    v_row = 10 + col1
                else:
                    raise ValueError(f"Unexpected v_row_offset {v_row_offset}")
                V[v_row][r] += coeff
            
            for (w_row_offset, coeff) in strassen_w[s]:
                if w_row_offset == 0:
                    w_row = col0 * 2 + 0
                elif w_row_offset == 1:
                    w_row = col0 * 2 + 1
                elif w_row_offset == 2:
                    w_row = col1 * 2 + 0
                elif w_row_offset == 3:
                    w_row = col1 * 2 + 1
                else:
                    raise ValueError(f"Unexpected w_row_offset {w_row_offset}")
                W[w_row][r] += coeff
            
            r += 1
    
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2][r] = 1
            V[2 * 10 + k][r] = 1
            W[k * 2 + i][r] = 1
            r += 1
    
    assert r == rank, f"Expected rank {rank}, got {r}"
    
    return U, V, W, rank


def _try_rank_als(n, m, p, target_rank, max_iters=200, n_restarts=3):
    """
    Try to find a decomposition of rank target_rank using alternating least squares.
    Returns (U, V, W, rank) if successful, else None.
    """
    U_size = n * m  # 6
    V_size = m * p  # 30
    W_size = p * n  # 20
    
    for restart in range(n_restarts):
        np.random.seed(12345 + restart * 1000)
        
        U = np.random.randn(U_size, target_rank) * 0.1
        V = np.random.randn(V_size, target_rank) * 0.1
        W = np.random.randn(W_size, target_rank) * 0.1
        
        for iteration in range(max_iters):
            # Solve for U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, target_rank))
                    for k in range(p):
                        A[k] = V[j * p + k] * W[k * n + i]
                    b = np.ones(p)
                    u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[i * m + j] = u
            
            # Solve for V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, target_rank))
                    for i in range(n):
                        A[i] = U[i * m + j] * W[k * n + i]
                    b = np.ones(n)
                    v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[j * p + k] = v
            
            # Solve for W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, target_rank))
                    for j in range(m):
                        A[j] = U[i * m + j] * V[j * p + k]
                    b = np.ones(m)
                    w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i] = w
            
            # Check convergence
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        lhs = np.dot(U[i * m + j] * V[j * p + k], W[k * n + i])
                        err = abs(lhs - 1.0)
                        if err > max_err:
                            max_err = err
            if max_err < 1e-9:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return U_list, V_list, W_list, target_rank
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Get the rank-55 decomposition as fallback
    U, V, W, rank = _build_strassen(n, m, p)
    
    # Try to find a lower rank decomposition
    for target_rank in range(54, 45, -1):
        if time.time() > deadline - 10:
            break
        result = _try_rank_als(n, m, p, target_rank, max_iters=100, n_restarts=2)
        if result is not None:
            U, V, W, rank = result
            if _validate(n, m, p, U, V, W):
                return {"U": U, "V": V, "W": W}
    
    # Fallback to rank 55
    if not _validate(n, m, p, U, V, W):
        U, V, W, rank = _build_trivial(n, m, p)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
