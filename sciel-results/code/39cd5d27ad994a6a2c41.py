import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def _get_strassen_vectors():
    """
    Returns list of 7 tuples (u_vec, v_vec, w_vec).
    u_vec, v_vec, w_vec are lists of (local_idx, coeff).
    Local indices 0..3 for 2x2 matrices.
    U local: i*2+j. 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    V local: j*2+k. 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    W local: k*2+i. 0:(0,0)->k=0,i=0, 1:(0,1)->k=0,i=1, 2:(1,0)->k=1,i=0, 3:(1,1)->k=1,i=1
    Note: W index convention in problem is W[k*n+i]. For 2x2, n=2.
    So local W index is k*2+i.
    C[0,0] -> k=0, i=0 -> idx 0.
    C[0,1] -> k=1, i=0 -> idx 2.  <-- WAIT.
    Problem says: "output factor follows the official trace convention W[k*n+i,r]".
    For a matrix C of size n x p? No, C is n x p.
    Wait, A is n x m, B is m x p, C is n x p.
    Tensor T is (n*m, m*p, p*n).
    Index 0: i*m + j.
    Index 1: j*p + k.
    Index 2: k*n + i.
    
    For a 2x2x2 block:
    n=2, m=2, p=2.
    U idx: i*2+j.
    V idx: j*2+k.
    W idx: k*2+i.
    
    C[i,k].
    i=0, k=0 -> W[0*2+0] = 0.
    i=0, k=1 -> W[1*2+0] = 2.
    i=1, k=0 -> W[0*2+1] = 1.
    i=1, k=1 -> W[1*2+1] = 3.
    
    So:
    C00 -> 0
    C10 -> 1
    C01 -> 2
    C11 -> 3
    
    Let's re-evaluate Strassen W vectors with this mapping.
    M1: C00(+1), C11(+1). W: [1, 0, 0, 1] (idx 0 and 3).
    M2: C10(+1), C11(-1). W: [0, 1, 0, -1] (idx 1 and 3).
    M3: C01(+1). W: [0, 0, 1, 0] (idx 2).
    M4: C00(+1), C10(+1). W: [1, 1, 0, 0] (idx 0 and 1).
    M5: C00(-1), C01(+1). W: [-1, 0, 1, 0] (idx 0 and 2).
    M6: C11(+1). W: [0, 0, 0, 1] (idx 3).
    M7: C00(+1). W: [1, 0, 0, 0] (idx 0).
    
    Let's check C00 (idx 0) sum: M1(1) + M4(1) + M5(-1) + M7(1) = 2?
    The term is sum_r U_r V_r W_r.
    For T[0,0,0] (A00, B00, C00):
    r=1: U[0]=1, V[0]=1, W[0]=1 -> 1
    r=2: U[0]=0, V[0]=1, W[0]=0 -> 0
    r=3: U[0]=1, V[0]=0, W[0]=0 -> 0
    r=4: U[0]=0, V[0]=-1, W[0]=1 -> 0
    r=5: U[0]=1, V[0]=0, W[0]=-1 -> 0
    r=6: U[0]=-1, V[0]=1, W[0]=0 -> 0
    r=7: U[0]=-1, V[0]=0, W[0]=1 -> 0
    Sum = 1. Correct.
    
    For T[0,0,1] (A00, B00, C10? No. C10 is i=1, k=0. W idx 1).
    Wait, T[i,j,k].
    T[0,0,0] corresponds to A00, B00, C00.
    T[1,0,0] corresponds to A10, B00, C10.
    
    Let's check T[1,0,0] (i=1, j=0, k=0).
    U idx: 1*2+0 = 2.
    V idx: 0*2+0 = 0.
    W idx: 0*2+1 = 1.
    
    r=1: U[2]=0, V[0]=1, W[1]=0 -> 0
    r=2: U[2]=1, V[0]=1, W[1]=1 -> 1
    r=3: U[2]=0, V[0]=0, W[1]=0 -> 0
    r=4: U[2]=0, V[0]=-1, W[1]=1 -> 0
    r=5: U[2]=0, V[0]=0, W[1]=0 -> 0
    r=6: U[2]=1, V[0]=1, W[1]=0 -> 0
    r=7: U[2]=0, V[0]=0, W[1]=0 -> 0
    Sum = 1. Correct.
    """
    return [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(2, 1), (3, -1)], [(2, 1)]),
        # M4
        ([(3, 1)], [(1, 1), (0, -1)], [(0, 1), (1, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (2, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
    ]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 1]
    
    n_offsets = [0, 2]
    m_offsets = [0, 2, 4]
    p_offsets = [0, 2, 4]
    
    U_entries = {} # col -> {row: coeff}
    V_entries = {}
    W_entries = {}
    
    rank = 0
    
    for ni, ni_sz in enumerate(n_parts):
        for mj, mj_sz in enumerate(m_parts):
            for pk, pk_sz in enumerate(p_parts):
                n_off = n_offsets[ni]
                m_off = m_offsets[mj]
                p_off = p_offsets[pk]
                
                if ni_sz == 2 and mj_sz == 2 and pk_sz == 2:
                    strassen_vecs = _get_strassen_vectors()
                    for u_vec, v_vec, w_vec in strassen_vecs:
                        col = rank
                        U_entries[col] = {}
                        V_entries[col] = {}
                        W_entries[col] = {}
                        
                        for u_rel, coeff in u_vec:
                            u_idx = (n_off + u_rel // 2) * m + (m_off + u_rel % 2)
                            U_entries[col][u_idx] = coeff
                            
                        for v_rel, coeff in v_vec:
                            v_idx = (m_off + v_rel // 2) * p + (p_off + v_rel % 2)
                            V_entries[col][v_idx] = coeff
                            
                        for w_rel, coeff in w_vec:
                            w_idx = (p_off + w_rel // 2) * n + (n_off + w_rel % 2)
                            W_entries[col][w_idx] = coeff
                            
                        rank += 1
                else:
                    for i in range(ni_sz):
                        for j in range(mj_sz):
                            for k in range(pk_sz):
                                col = rank
                                u_idx = (n_off + i) * m + (m_off + j)
                                v_idx = (m_off + j) * p + (p_off + k)
                                w_idx = (p_off + k) * n + (n_off + i)
                                
                                U_entries[col] = {u_idx: 1}
                                V_entries[col] = {v_idx: 1}
                                W_entries[col] = {w_idx: 1}
                                rank += 1
                                
    # Build matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for u_idx, coeff in U_entries[col].items():
            U[u_idx][col] = coeff
        for v_idx, coeff in V_entries[col].items():
            V[v_idx][col] = coeff
        for w_idx, coeff in W_entries[col].items():
            W[w_idx][col] = coeff
            
    # Verify
    try:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        target = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    target[a, b, c] = 1
        
        T_decomp = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        
        if not np.allclose(T_decomp, target, atol=1e-10):
            # Fallback
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
