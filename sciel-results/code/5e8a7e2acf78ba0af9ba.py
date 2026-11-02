"""Tensor decomposition for 3x5x5 matrix multiplication."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time

def _strassen_2x2x2():
    """Return the 7-term Strassen decomposition for 2x2 x 2x2 -> 2x2.
    Returns list of 7 tuples: (u_rows, v_rows, w_rows) where each is a list of
    (index, coefficient) for the respective factor.
    U index: i*2+j for row i, col j of A
    V index: j*2+k for row j, col k of B  
    W index: k*2+i for row k, col i of C (trace convention W[k*n+i,r])
    """
    terms = []
    
    # M1 = (A11+A22)(B11+B22)
    terms.append([
        (0, 0, 0, 1), (0, 0, 3, 1),
        (3, 0, 0, 1), (3, 0, 3, 1),
        (0, 3, 0, 1), (0, 3, 3, 1),
        (3, 3, 0, 1), (3, 3, 3, 1),
    ])
    
    # M2 = (A21+A22)B11
    terms.append([
        (2, 0, 1, 1), (2, 0, 3, -1),
        (3, 0, 1, 1), (3, 0, 3, -1),
    ])
    
    # M3 = A11(B21-B22)
    terms.append([
        (0, 2, 2, 1), (0, 3, 2, -1),
    ])
    
    # M4 = A22(B12-B11)
    terms.append([
        (3, 1, 0, 1), (3, 0, 0, -1),
        (3, 1, 1, 1), (3, 0, 1, -1),
    ])
    
    # M5 = (A11+A12)B22
    terms.append([
        (0, 3, 0, -1), (0, 3, 2, 1),
        (1, 3, 0, -1), (1, 3, 2, 1),
    ])
    
    # M6 = (A21-A11)(B11+B12)
    terms.append([
        (2, 0, 3, 1), (2, 1, 3, 1),
        (0, 0, 3, -1), (0, 1, 3, -1),
    ])
    
    # M7 = (A12-A22)(B21+B22)
    terms.append([
        (1, 2, 0, 1), (1, 3, 0, 1),
        (3, 2, 0, -1), (3, 3, 0, -1),
    ])
    
    return terms

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Partition: n = 2+1, m = 2+2+1, p = 2+2+1
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 1]
    
    # Compute offsets
    n_offsets = []
    off = 0
    for sz in n_parts:
        n_offsets.append(off)
        off += sz
    m_offsets = []
    off = 0
    for sz in m_parts:
        m_offsets.append(off)
        off += sz
    p_offsets = []
    off = 0
    for sz in p_parts:
        p_offsets.append(off)
        off += sz
    
    # For each combination of (n_i, m_j, p_k), determine the rank contribution
    # If all sizes are 2, use Strassen (rank 7). Otherwise schoolbook.
    
    U_entries = []  # list of (u_idx, coeff)
    V_entries = []  # list of (v_idx, coeff)
    W_entries = []  # list of (w_idx, coeff)
    
    for ni, ni_sz in enumerate(n_parts):
        for mj, mj_sz in enumerate(m_parts):
            for pk, pk_sz in enumerate(p_parts):
                n_off = n_offsets[ni]
                m_off = m_offsets[mj]
                p_off = p_offsets[pk]
                
                if ni_sz == 2 and mj_sz == 2 and pk_sz == 2:
                    # Use Strassen
                    strassen_terms = _strassen_2x2x2()
                    for term in strassen_terms:
                        for (u_rel, v_rel, w_rel, coeff) in term:
                            u_idx = (n_off + u_rel // 2) * m + (m_off + u_rel % 2)
                            v_idx = (m_off + v_rel // 2) * p + (p_off + v_rel % 2)
                            w_idx = (p_off + w_rel // 2) * n + (n_off + w_rel % 2)
                            U_entries.append((u_idx, coeff))
                            V_entries.append((v_idx, coeff))
                            W_entries.append((w_idx, coeff))
                else:
                    # Schoolbook for this sub-block
                    for i in range(ni_sz):
                        for j in range(mj_sz):
                            for k in range(pk_sz):
                                u_idx = (n_off + i) * m + (m_off + j)
                                v_idx = (m_off + j) * p + (p_off + k)
                                w_idx = (p_off + k) * n + (n_off + i)
                                U_entries.append((u_idx, 1))
                                V_entries.append((v_idx, 1))
                                W_entries.append((w_idx, 1))
    
    rank = len(U_entries)
    
    # Build matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_idx, coeff) in enumerate(U_entries):
        U[u_idx][col] = coeff
    for col, (v_idx, coeff) in enumerate(V_entries):
        V[v_idx][col] = coeff
    for col, (w_idx, coeff) in enumerate(W_entries):
        W[w_idx][col] = coeff
    
    # Verify before returning
    try:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Build the target tensor
        target = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    target[a, b, c] = 1
        
        # Compute from decomposition
        T_decomp = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        
        if not np.allclose(T_decomp, target, atol=1e-10):
            # Fallback to schoolbook
            rank_school = n * m * p
            U_s = [[0] * rank_school for _ in range(n * m)]
            V_s = [[0] * rank_school for _ in range(m * p)]
            W_s = [[0] * rank_school for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_s[i * m + j][col] = 1
                        V_s[j * p + k][col] = 1
                        W_s[k * n + i][col] = 1
                        col += 1
            return {"U": U_s, "V": V_s, "W": W_s}
    except Exception:
        pass
    
    if time.time() > deadline:
        # Fallback to schoolbook if over deadline
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col] = 1
                    V_s[j * p + k][col] = 1
                    W_s[k * n + i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
