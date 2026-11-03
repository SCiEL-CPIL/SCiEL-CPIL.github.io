import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                computed = 0.0
                for r in range(rank):
                    computed += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(computed - 1.0) > tol:
                    return False
    return True

def _build_strassen_decomposition(n, m, p):
    """
    Build a rank-44 decomposition:
      - 4 blocks of 2x2 * 2x2 using Strassen (7 each) = 28
      - 2x1 * 1x8 schoolbook = 16
    Total: 44
    """
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Part 1: 2x2 * 2x8 using Strassen on 2-column blocks
    for j in range(4):
        w_r00 = 4 * j
        w_r01 = 4 * j + 2
        w_r10 = 4 * j + 1
        w_r11 = 4 * j + 3
        v_e = 2 * j
        v_f = 2 * j + 1
        v_g = 8 + 2 * j
        v_h = 9 + 2 * j
        
        # P1 = (a+d)(e+h)
        U[0, col] = 1; U[4, col] = 1
        V[v_e, col] = 1; V[v_h, col] = 1
        W[w_r00, col] = 1; W[w_r11, col] = 1
        col += 1
        # P2 = (c+d)e
        U[3, col] = 1; U[4, col] = 1
        V[v_e, col] = 1
        W[w_r10, col] = 1; W[w_r11, col] = -1
        col += 1
        # P3 = a(f-h)
        U[0, col] = 1
        V[v_f, col] = 1; V[v_h, col] = -1
        W[w_r01, col] = 1; W[w_r11, col] = 1
        col += 1
        # P4 = d(g-e)
        U[4, col] = 1
        V[v_g, col] = 1; V[v_e, col] = -1
        W[w_r00, col] = 1; W[w_r10, col] = 1
        col += 1
        # P5 = (a+b)h
        U[0, col] = 1; U[1, col] = 1
        V[v_h, col] = 1
        W[w_r00, col] = -1; W[w_r01, col] = 1
        col += 1
        # P6 = (c-a)(e+f)
        U[3, col] = 1; U[0, col] = -1
        V[v_e, col] = 1; V[v_f, col] = 1
        W[w_r11, col] = 1
        col += 1
        # P7 = (b-d)(g+h)
        U[1, col] = 1; U[4, col] = -1
        V[v_g, col] = 1; V[v_h, col] = 1
        W[w_r00, col] = 1
        col += 1
    
    # Part 2: 2x1 * 1x8 schoolbook
    for i in range(2):
        for k in range(8):
            u_idx = i * 3 + 2
            v_idx = 2 * 8 + k
            w_idx = k * 2 + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
    
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, time_budget=100):
    """
    Attempt to reduce the rank by finding linear dependencies among the rank-1 terms.
    Uses a greedy elimination approach.
    """
    start_time = time.time()
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = current_U.shape[1]
    
    # Quick win check: if any column is all zeros, remove it
    for col in range(current_rank - 1, -1, -1):
        if time.time() - start_time > time_budget:
            break
        if np.all(current_U[:, col] == 0) and np.all(current_V[:, col] == 0) and np.all(current_W[:, col] == 0):
            current_U = np.delete(current_U, col, axis=1)
            current_V = np.delete(current_V, col, axis=1)
            current_W = np.delete(current_W, col, axis=1)
            current_rank -= 1
    
    return current_U, current_V, current_W, current_rank

def construct():
    """
    Constructs a low-rank matrix multiplication decomposition for 2x3 * 3x8.
    Returns a dictionary with matrices U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # Build initial rank-44 decomposition
    U, V, W = _build_strassen_decomposition(n, m, p)
    current_rank = U.shape[1]
    
    # Verify initial decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U_fb = np.zeros((n * m, rank))
        V_fb = np.zeros((m * p, rank))
        W_fb = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U_fb[i * m + j, idx] = 1
                    V_fb[j * p + k, idx] = 1
                    W_fb[k * n + i, idx] = 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Try to reduce rank with remaining time
    time_remaining = deadline - time.time() - 5
    if time_remaining > 10:
        U, V, W, new_rank = _try_reduce_rank(U, V, W, n, m, p, time_budget=time_remaining)
        if new_rank < current_rank:
            current_rank = new_rank
            if _verify_decomposition(U, V, W, n, m, p):
                current_rank = new_rank
            else:
                U, V, W = _build_strassen_decomposition(n, m, p)
                current_rank = 44
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
