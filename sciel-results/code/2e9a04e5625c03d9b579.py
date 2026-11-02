"""Program for 3x5x5 matrix multiplication tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 5]


def _strassen_2x2x2_terms():
    """Return the 7-term Strassen decomposition for 2x2 x 2x2 -> 2x2.
    Each term is a list of (u_rel, v_rel, w_rel, coeff) tuples.
    U index: i*2+j, V index: j*2+k, W index: k*2+i (trace convention)
    """
    terms = []
    
    # M1 = (A11+A22)(B11+B22)
    # A11: u=0, A22: u=3; B11: v=0, B22: v=3
    # C11: w=0, C22: w=3
    terms.append([
        (0, 0, 0, 1), (0, 0, 3, 1),
        (3, 0, 0, 1), (3, 0, 3, 1),
        (0, 3, 0, 1), (0, 3, 3, 1),
        (3, 3, 0, 1), (3, 3, 3, 1),
    ])
    
    # M2 = (A21+A22)B11
    # A21: u=2, A22: u=3; B11: v=0
    # C21: w=1, C22: w=3 (with -1)
    terms.append([
        (2, 0, 1, 1), (2, 0, 3, -1),
        (3, 0, 1, 1), (3, 0, 3, -1),
    ])
    
    # M3 = A11(B21-B22)
    # A11: u=0; B21: v=2, B22: v=3
    # C12: w=2, C22: w=3
    terms.append([
        (0, 2, 2, 1), (0, 3, 2, -1),
        (0, 2, 3, 1), (0, 3, 3, -1),
    ])
    
    # M4 = A22(B12-B11)
    # A22: u=3; B12: v=1, B11: v=0
    # C11: w=0, C21: w=1
    terms.append([
        (3, 1, 0, 1), (3, 0, 0, -1),
        (3, 1, 1, 1), (3, 0, 1, -1),
    ])
    
    # M5 = (A11+A12)B22
    # A11: u=0, A12: u=1; B22: v=3
    # C11: w=0 (with -1), C12: w=2
    terms.append([
        (0, 3, 0, -1), (0, 3, 2, 1),
        (1, 3, 0, -1), (1, 3, 2, 1),
    ])
    
    # M6 = (A21-A11)(B11+B12)
    # A21: u=2, A11: u=0; B11: v=0, B12: v=1
    # C22: w=3
    terms.append([
        (2, 0, 3, 1), (2, 1, 3, 1),
        (0, 0, 3, -1), (0, 1, 3, -1),
    ])
    
    # M7 = (A12-A22)(B21+B22)
    # A12: u=1, A22: u=3; B21: v=2, B22: v=3
    # C11: w=0 (with -1), C22: w=3 (with +1)
    terms.append([
        (1, 2, 0, -1), (1, 3, 0, -1),
        (3, 2, 0, 1), (3, 3, 0, 1),
        (1, 2, 3, 1), (1, 3, 3, 1),
        (3, 2, 3, -1), (3, 3, 3, -1),
    ])
    
    return terms


def _build_decomposition(n_parts, m_parts, p_parts):
    """Build a block-structured decomposition with Strassen for 2x2x2 blocks."""
    n, m, p = 3, 5, 5
    
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
    
    all_terms = []
    
    for ni, ni_sz in enumerate(n_parts):
        for mj, mj_sz in enumerate(m_parts):
            for pk, pk_sz in enumerate(p_parts):
                n_off = n_offsets[ni]
                m_off = m_offsets[mj]
                p_off = p_offsets[pk]
                
                if ni_sz == 2 and mj_sz == 2 and pk_sz == 2:
                    strassen_terms = _strassen_2x2x2_terms()
                    for term in strassen_terms:
                        for (u_rel, v_rel, w_rel, coeff) in term:
                            i_local = u_rel // 2
                            j_local = u_rel % 2
                            k_local = v_rel % 2
                            u_idx = (n_off + i_local) * m + (m_off + j_local)
                            v_idx = (m_off + j_local) * p + (p_off + k_local)
                            w_idx = (p_off + k_local) * n + (n_off + i_local)
                            all_terms.append((u_idx, v_idx, w_idx, coeff))
                else:
                    for i in range(ni_sz):
                        for j in range(mj_sz):
                            for k in range(pk_sz):
                                u_idx = (n_off + i) * m + (m_off + j)
                                v_idx = (m_off + j) * p + (p_off + k)
                                w_idx = (p_off + k) * n + (n_off + i)
                                all_terms.append((u_idx, v_idx, w_idx, 1))
    
    rank = len(all_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_idx, v_idx, w_idx, coeff) in enumerate(all_terms):
        U[u_idx][col] = coeff
        V[v_idx][col] = coeff
        W[w_idx][col] = coeff
    
    return U, V, W


def _try_3x3x3_strassen_like(n_parts, m_parts, p_parts):
    """Try a 3x3x3 decomposition using a 9-term scheme for 3x3x3 blocks."""
    # This is a placeholder; a true 3x3x3 decomposition with rank < 27
    # is not known to exist with simple patterns. We'll just do schoolbook
    # for 3x3x3 blocks for now.
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = 3, 5, 5
    
    # Strategy 1: 4 Strassen blocks (Rank 71)
    # n=[1,2], m=[2,2,1], p=[1,2,2]
    U1, V1, W1 = _build_decomposition([1, 2], [2, 2, 1], [1, 2, 2])
    
    # Verify
    try:
        U_np = np.array(U1, dtype=float)
        V_np = np.array(V1, dtype=float)
        W_np = np.array(W1, dtype=float)
        
        target = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    target[a, b, c] = 1
        
        computed = np.zeros((n*m, m*p, p*n), dtype=float)
        rank = len(U1[0])
        for r in range(rank):
            u_col = U_np[:, r]
            v_col = V_np[:, r]
            w_col = W_np[:, r]
            nz_u = np.nonzero(u_col)[0]
            nz_v = np.nonzero(v_col)[0]
            nz_w = np.nonzero(w_col)[0]
            for a in nz_u:
                for b in nz_v:
                    for c in nz_w:
                        computed[a, b, c] += u_col[a] * v_col[b] * w_col[c]
        
        if np.allclose(computed, target, atol=1e-10):
            return {"U": U1, "V": V1, "W": W1}
    except Exception:
        pass
    
    # Strategy 2: Try other partitionings with Strassen
    # n=[1,2], m=[2,1,2], p=[2,1,2]
    if time.time() < deadline:
        U2, V2, W2 = _build_decomposition([1, 2], [2, 1, 2], [2, 1, 2])
        try:
            U_np = np.array(U2, dtype=float)
            V_np = np.array(V2, dtype=float)
            W_np = np.array(W2, dtype=float)
            
            computed = np.zeros((n*m, m*p, p*n), dtype=float)
            rank = len(U2[0])
            for r in range(rank):
                u_col = U_np[:, r]
                v_col = V_np[:, r]
                w_col = W_np[:, r]
                nz_u = np.nonzero(u_col)[0]
                nz_v = np.nonzero(v_col)[0]
                nz_w = np.nonzero(w_col)[0]
                for a in nz_u:
                    for b in nz_v:
                        for c in nz_w:
                            computed[a, b, c] += u_col[a] * v_col[b] * w_col[c]
            
            if np.allclose(computed, target, atol=1e-10):
                # Only return if rank is better or equal
                if rank <= len(U1[0]):
                    return {"U": U2, "V": V2, "W": W2}
        except Exception:
            pass
    
    # Strategy 3: n=[2,1], m=[2,2,1], p=[2,1,2]
    if time.time() < deadline:
        U3, V3, W3 = _build_decomposition([2, 1], [2, 2, 1], [2, 1, 2])
        try:
            U_np = np.array(U3, dtype=float)
            V_np = np.array(V3, dtype=float)
            W_np = np.array(W3, dtype=float)
            
            computed = np.zeros((n*m, m*p, p*n), dtype=float)
            rank = len(U3[0])
            for r in range(rank):
                u_col = U_np[:, r]
                v_col = V_np[:, r]
                w_col = W_np[:, r]
                nz_u = np.nonzero(u_col)[0]
                nz_v = np.nonzero(v_col)[0]
                nz_w = np.nonzero(w_col)[0]
                for a in nz_u:
                    for b in nz_v:
                        for c in nz_w:
                            computed[a, b, c] += u_col[a] * v_col[b] * w_col[c]
            
            if np.allclose(computed, target, atol=1e-10):
                if rank <= len(U1[0]):
                    return {"U": U3, "V": V3, "W": W3}
        except Exception:
            pass
    
    # Fallback: Schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    return {"U": U_school, "V": V_school, "W": W_school}
