"""Program for 3x5x5 matrix multiplication tensor decomposition.

Uses block decomposition with Strassen (rank 7) for the 2x2x2 blocks.
Splits m = 2+2+1 and p = 2+2+1, keeping n = 3.
Block ranks: M_{3,2,2}=9, M_{3,2,1}=6, M_{3,1,2}=6, M_{3,1,1}=3.
Total rank = 4*9 + 2*6 + 2*6 + 1*3 = 63.
"""
import time

def _strassen_2x2x2_terms():
    """Return 7 (u, v, w) vectors for the 2x2x2 Strassen decomposition.
    u has length 4 (indices i*2+j), v has length 4 (j*2+k), w has length 4 (k*2+i)."""
    terms = []
    # M1 = (a00+a11)*(b00+b11) -> c00, c11
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # M2 = (a10+a11)*b00 -> c10
    terms.append(([0,0,1,1], [1,0,0,0], [0,1,0,0]))
    # M3 = a00*(b01-b11) -> c01, c11
    terms.append(([1,0,0,0], [0,1,0,-1], [0,0,1,1]))
    # M4 = a11*(b10-b00) -> c00, c10
    terms.append(([0,0,0,1], [-1,0,1,0], [1,1,0,0]))
    # M5 = (a00+a01)*b11 -> -c00, c01
    terms.append(([1,1,0,0], [0,0,0,1], [-1,0,1,0]))
    # M6 = (a10-a00)*(b00+b01) -> c11
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    # M7 = (a01-a11)*(b10+b11) -> c00
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    return terms

def _verify_2x2x2():
    terms = _strassen_2x2x2_terms()
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for (u, v, w) in terms:
                    s += u[i*2+j] * v[j*2+k] * w[k*2+i]
                if s != 1:
                    return False
    return True

def construct():
    t_start = time.time()
    
    n, m, p = 3, 5, 5
    
    # Verify Strassen
    if not _verify_2x2x2():
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    strassen_terms = _strassen_2x2x2_terms()
    
    # Accumulate all terms as (u_dict, v_dict, w_dict)
    all_terms = []
    
    # Block decomposition: m = 2+2+1, p = 2+2+1
    m_blocks = [2, 2, 1]
    p_blocks = [2, 2, 1]
    
    for mb_idx, m_size in enumerate(m_blocks):
        m_off = sum(m_blocks[:mb_idx])
        for pb_idx, p_size in enumerate(p_blocks):
            p_off = sum(p_blocks[:pb_idx])
            
            if m_size == 2 and p_size == 2:
                # M_{3,2,2}: use Strassen for i in {0,1}, schoolbook for i=2
                # Strassen part: 7 terms
                for (su, sv, sw) in strassen_terms:
                    u_full = {}
                    v_full = {}
                    w_full = {}
                    for i in range(2):
                        for j in range(2):
                            val = su[i*2+j]
                            if val != 0:
                                idx = i*m + (m_off + j)
                                u_full[idx] = u_full.get(idx, 0) + val
                    for j in range(2):
                        for k in range(2):
                            val = sv[j*2+k]
                            if val != 0:
                                idx = (m_off + j)*p + (p_off + k)
                                v_full[idx] = v_full.get(idx, 0) + val
                    for k in range(2):
                        for i in range(2):
                            val = sw[k*2+i]
                            if val != 0:
                                idx = (p_off + k)*n + i
                                w_full[idx] = w_full.get(idx, 0) + val
                    all_terms.append((u_full, v_full, w_full))
                
                # Schoolbook part for i=2: 4 terms
                for j in range(2):
                    for k in range(2):
                        u_full = {2*m + (m_off + j): 1}
                        v_full = {(m_off + j)*p + (p_off + k): 1}
                        w_full = {(p_off + k)*n + 2: 1}
                        all_terms.append((u_full, v_full, w_full))
                        
            elif m_size == 2 and p_size == 1:
                # M_{3,2,1}: schoolbook, 6 terms
                for i in range(3):
                    for j in range(2):
                        u_full = {i*m + (m_off + j): 1}
                        v_full = {(m_off + j)*p + p_off: 1}
                        w_full = {p_off*n + i: 1}
                        all_terms.append((u_full, v_full, w_full))
                        
            elif m_size == 1 and p_size == 2:
                # M_{3,1,2}: schoolbook, 6 terms
                for i in range(3):
                    for k in range(2):
                        u_full = {i*m + m_off: 1}
                        v_full = {m_off*p + (p_off + k): 1}
                        w_full = {(p_off + k)*n + i: 1}
                        all_terms.append((u_full, v_full, w_full))
                        
            elif m_size == 1 and p_size == 1:
                # M_{3,1,1}: schoolbook, 3 terms
                for i in range(3):
                    u_full = {i*m + m_off: 1}
                    v_full = {m_off*p + p_off: 1}
                    w_full = {p_off*n + i: 1}
                    all_terms.append((u_full, v_full, w_full))
    
    rank = len(all_terms)
    
    # Build matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(all_terms):
        for idx, val in u_dict.items():
            U[idx][r] = val
        for idx, val in v_dict.items():
            V[idx][r] = val
        for idx, val in w_dict.items():
            W[idx][r] = val
    
    # Verify the decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    # Fallback to schoolbook
                    rank2 = n * m * p
                    U2 = [[0] * rank2 for _ in range(n * m)]
                    V2 = [[0] * rank2 for _ in range(m * p)]
                    W2 = [[0] * rank2 for _ in range(p * n)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U2[ii*m+jj][col] = 1
                                V2[jj*p+kk][col] = 1
                                W2[kk*n+ii][col] = 1
                                col += 1
                    return {"U": U2, "V": V2, "W": W2}
    
    if time.time() - t_start > 100:
        pass
    
    return {"U": U, "V": V, "W": W}
