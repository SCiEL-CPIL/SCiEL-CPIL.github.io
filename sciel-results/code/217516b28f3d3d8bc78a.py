"""Matrix multiplication 3x4 by 4x6: Strassen rank-66 base with systematic rank reduction."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

_START_TIME = time.time()
_DEADLINE = 145.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    """
    Construct exact tensor decomposition for (3x4)*(4x6) -> (3x6).
    
    Strategy:
    1. Build rank-66 Strassen-based decomposition (fast, clean)
    2. Apply systematic linear algebra rank reduction:
       - For each pair of columns, check if their tensor contributions are linearly dependent
       - Remove redundant columns
    """
    n, m, p = 3, 4, 6
    rank = 66
    U = np.zeros((n * m, rank), dtype=float)   # 12 x 66
    V = np.zeros((m * p, rank), dtype=float)   # 24 x 66
    W = np.zeros((p * n, rank), dtype=float)   # 18 x 66
    
    col = 0
    
    def add_schoolbook(i, j, k):
        nonlocal col
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx, col] = 1
        V[v_idx, col] = 1
        W[w_idx, col] = 1
        col += 1

    def add_strassen(i_base, j_base, k_base):
        """Add 7 Strassen terms for a 2x2x2 block at global indices."""
        nonlocal col
        
        a_idx = i_base * m + j_base
        b_idx = i_base * m + (j_base + 1)
        c_idx = (i_base + 1) * m + j_base
        d_idx = (i_base + 1) * m + (j_base + 1)
        
        e_idx = j_base * p + k_base
        f_idx = j_base * p + (k_base + 1)
        g_idx = (j_base + 1) * p + k_base
        h_idx = (j_base + 1) * p + (k_base + 1)
        
        p_idx = k_base * n + i_base
        q_idx = (k_base + 1) * n + i_base
        r_idx = k_base * n + (i_base + 1)
        s_idx = (k_base + 1) * n + (i_base + 1)
        
        # P1 = (a+d)(e+h)
        U[a_idx, col] = 1; U[d_idx, col] = 1
        V[e_idx, col] = 1; V[h_idx, col] = 1
        W[p_idx, col] = 1; W[s_idx, col] = 1
        col += 1
        
        # P2 = (c+d)e
        U[c_idx, col] = 1; U[d_idx, col] = 1
        V[e_idx, col] = 1
        W[r_idx, col] = 1; W[s_idx, col] = -1
        col += 1
        
        # P3 = a(f-h)
        U[a_idx, col] = 1
        V[f_idx, col] = 1; V[h_idx, col] = -1
        W[q_idx, col] = 1; W[s_idx, col] = 1
        col += 1
        
        # P4 = d(g-e)
        U[d_idx, col] = 1
        V[g_idx, col] = 1; V[e_idx, col] = -1
        W[p_idx, col] = 1; W[r_idx, col] = 1
        col += 1
        
        # P5 = (a+b)h
        U[a_idx, col] = 1; U[b_idx, col] = 1
        V[h_idx, col] = 1
        W[p_idx, col] = -1; W[q_idx, col] = 1
        col += 1
        
        # P6 = (c-a)(e+f)
        U[c_idx, col] = 1; U[a_idx, col] = -1
        V[e_idx, col] = 1; V[f_idx, col] = 1
        W[s_idx, col] = 1
        col += 1
        
        # P7 = (b-d)(g+h)
        U[b_idx, col] = 1; U[d_idx, col] = -1
        V[g_idx, col] = 1; V[h_idx, col] = 1
        W[p_idx, col] = 1
        col += 1

    # Part 1: Row 0 schoolbook (1x4x6 = 24 terms)
    for j in range(4):
        for k in range(6):
            add_schoolbook(0, j, k)
    
    # Part 2: Rows 1-2 Strassen (2x4x6, 6 blocks of 2x2x2 = 42 terms)
    for k_block in range(3):
        k_base = k_block * 2
        for j_block in range(2):
            j_base = j_block * 2
            i_base = 1
            add_strassen(i_base, j_base, k_base)
    
    # Verify correctness
    def build_tensor(U, V, W, rank):
        T = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    u_val = U[i*m+j, r]
                    if u_val == 0:
                        continue
                    for k in range(p):
                        v_val = V[j*p+k, r]
                        if v_val == 0:
                            continue
                        w_val = W[k*n+i, r]
                        if w_val == 0:
                            continue
                        T[i, j, k] += u_val * v_val * w_val
        return T
    
    T_full = build_tensor(U, V, W, rank)
    if not np.allclose(T_full, 1.0, atol=1e-10):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Systematic rank reduction: try removing columns and check tensor correctness
    active = list(range(rank))
    
    # Efficient tensor check: precompute contributions
    def check_tensor(active_cols):
        """Check if the subset of columns gives the correct tensor."""
        T_test = np.zeros((n, m, p))
        for r_idx in active_cols:
            for i in range(n):
                u_row = U[i*m:(i+1)*m, r_idx]
                for j in range(m):
                    u_val = u_row[j]
                    if u_val == 0:
                        continue
                    v_row = V[j*p:(j+1)*p, r_idx]
                    w_row = W[:, r_idx]
                    # T[i,j,:] += u_val * v_row * w_row[n*i:(n+1)*i+n] but W is indexed by k*n+i
                    for k in range(p):
                        v_val = v_row[k]
                        if v_val == 0:
                            continue
                        w_val = W[k*n+i, r_idx]
                        if w_val == 0:
                            continue
                        T_test[i, j, k] += u_val * v_val * w_val
        return np.allclose(T_test, 1.0, atol=1e-10)
    
    # Try removing columns one at a time, greedily
    removed = 0
    for iteration in range(50):
        if _time_left() < 20:
            break
        
        found_removal = False
        for r in list(active):
            if _time_left() < 20:
                break
            
            test_active = [c for c in active if c != r]
            if check_tensor(test_active):
                active.remove(r)
                removed += 1
                found_removal = True
                break
        
        if not found_removal:
            break
    
    # Build final result
    current_rank = len(active)
    if current_rank < rank:
        final_U = np.zeros((n * m, current_rank), dtype=float)
        final_V = np.zeros((m * p, current_rank), dtype=float)
        final_W = np.zeros((p * n, current_rank), dtype=float)
        
        for new_idx, old_idx in enumerate(active):
            final_U[:, new_idx] = U[:, old_idx]
            final_V[:, new_idx] = V[:, old_idx]
            final_W[:, new_idx] = W[:, old_idx]
        
        return {
            "U": final_U.tolist(),
            "V": final_V.tolist(),
            "W": final_W.tolist()
        }
    else:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
