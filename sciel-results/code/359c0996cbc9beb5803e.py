import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 150s limit, keep 10s margin
    
    n, m, p = 2, 4, 4
    
    # Build the known rank-28 Strassen-based decomposition
    U, V, W = _build_strassen_28()
    
    # Verify correctness
    if not _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by searching for a lower-rank decomposition
    # Start with rank 27 and go down
    for target_rank in range(27, 19, -1):
        if time.time() > deadline - 20:
            break
        result = _try_reduce_rank(U, V, W, target_rank, n, m, p)
        if result is not None:
            U, V, W = result
            break
    
    return {"U": U, "V": V, "W": W}

def _build_strassen_28():
    n, m, p = 2, 4, 4
    TARGET_RANK = 28
    
    U = [[0] * TARGET_RANK for _ in range(n * m)]
    V = [[0] * TARGET_RANK for _ in range(m * p)]
    W = [[0] * TARGET_RANK for _ in range(p * n)]
    
    def get_u_idx(i, j): return i * m + j
    def get_v_idx(i, j): return i * p + j
    def get_w_idx(i, j): return j * n + i
    
    blocks = [
        (0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0),
        (0, 0, 0, 2, 2),
        (0, 2, 2, 2, 2),
    ]
    
    current_rank = 0
    
    for (a_r, a_c, b_r, b_c, c_c) in blocks:
        u11 = get_u_idx(a_r, a_c)
        u12 = get_u_idx(a_r, a_c + 1)
        u21 = get_u_idx(a_r + 1, a_c)
        u22 = get_u_idx(a_r + 1, a_c + 1)
        
        v11 = get_v_idx(b_r, b_c)
        v12 = get_v_idx(b_r, b_c + 1)
        v21 = get_v_idx(b_r + 1, b_c)
        v22 = get_v_idx(b_r + 1, b_c + 1)
        
        w11 = get_w_idx(0, c_c)
        w12 = get_w_idx(0, c_c + 1)
        w21 = get_w_idx(1, c_c)
        w22 = get_w_idx(1, c_c + 1)
        
        # M1 = (A11 + A22) * (B11 + B22)
        U[u11][current_rank] += 1; U[u22][current_rank] += 1
        V[v11][current_rank] += 1; V[v22][current_rank] += 1
        W[w11][current_rank] += 1; W[w22][current_rank] += 1
        current_rank += 1
        
        # M2 = (A21 + A22) * B11
        U[u21][current_rank] += 1; U[u22][current_rank] += 1
        V[v11][current_rank] += 1
        W[w21][current_rank] += 1; W[w22][current_rank] += -1
        current_rank += 1
        
        # M3 = A11 * (B12 - B22)
        U[u11][current_rank] += 1
        V[v12][current_rank] += 1; V[v22][current_rank] += -1
        W[w12][current_rank] += 1; W[w22][current_rank] += 1
        current_rank += 1
        
        # M4 = A22 * (B21 - B11)
        U[u22][current_rank] += 1
        V[v21][current_rank] += 1; V[v11][current_rank] += -1
        W[w11][current_rank] += 1; W[w21][current_rank] += 1
        current_rank += 1
        
        # M5 = (A11 + A12) * B22
        U[u11][current_rank] += 1; U[u12][current_rank] += 1
        V[v22][current_rank] += 1
        W[w11][current_rank] += -1; W[w12][current_rank] += 1
        current_rank += 1
        
        # M6 = (A21 - A11) * (B11 + B12)
        U[u21][current_rank] += 1; U[u11][current_rank] += -1
        V[v11][current_rank] += 1; V[v12][current_rank] += 1
        W[w22][current_rank] += 1
        current_rank += 1
        
        # M7 = (A12 - A22) * (B21 + B22)
        U[u12][current_rank] += 1; U[u22][current_rank] += -1
        V[v21][current_rank] += 1; V[v22][current_rank] += 1
        W[w11][current_rank] += 1
        current_rank += 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-10):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(len(U[i*m+j])):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_reduce_rank(U, V, W, target_rank, n, m, p):
    """Try to find a decomposition of target_rank by removing terms."""
    current_rank = len(U[0])
    if current_rank <= target_rank:
        return U, V, W
    
    # Try removing each term and checking if the tensor is still decomposable
    # with the remaining terms (this requires solving a system, which is complex)
    # Instead, try a simpler heuristic: check if any term has all-zero U, V, or W
    for r in range(current_rank):
        u_col = [U[i][r] for i in range(n*m)]
        v_col = [V[i][r] for i in range(m*p)]
        w_col = [W[i][r] for i in range(p*n)]
        
        if all(x == 0 for x in u_col) or all(x == 0 for x in v_col) or all(x == 0 for x in w_col):
            # This term is useless, remove it
            U_new = [[U[i][j] for j in range(current_rank) if j != r] for i in range(n*m)]
            V_new = [[V[i][j] for j in range(current_rank) if j != r] for i in range(m*p)]
            W_new = [[W[i][j] for j in range(current_rank) if j != r] for i in range(p*n)]
            if _verify(U_new, V_new, W_new, n, m, p):
                return U_new, V_new, W_new
    
    # Try merging two terms: if term a and term b have proportional U, V, or W columns
    for a in range(current_rank):
        for b in range(a+1, current_rank):
            u_a = [U[i][a] for i in range(n*m)]
            u_b = [U[i][b] for i in range(n*m)]
            v_a = [V[i][a] for i in range(m*p)]
            v_b = [V[i][b] for i in range(m*p)]
            w_a = [W[i][a] for i in range(p*n)]
            w_b = [W[i][b] for i in range(p*n)]
            
            # Check if u_a is proportional to u_b
            if _is_proportional(u_a, u_b) and _is_proportional(v_a, v_b) and _is_proportional(w_a, w_b):
                # These terms can be merged
                U_new = [[U[i][j] for j in range(current_rank) if j != b] for i in range(n*m)]
                V_new = [[V[i][j] for j in range(current_rank) if j != b] for i in range(m*p)]
                W_new = [[W[i][j] for j in range(current_rank) if j != b] for i in range(p*n)]
                if _verify(U_new, V_new, W_new, n, m, p):
                    return U_new, V_new, W_new
    
    return None

def _is_proportional(a, b, tol=1e-10):
    """Check if vectors a and b are proportional."""
    # Find first non-zero in a
    idx = -1
    for i in range(len(a)):
        if abs(a[i]) > tol:
            idx = i
            break
    if idx == -1:
        return all(abs(x) < tol for x in b)
    ratio = b[idx] / a[idx]
    for i in range(len(a)):
        if abs(a[i] * ratio - b[i]) > tol * max(1, abs(a[i]), abs(b[i])):
            return False
    return True
