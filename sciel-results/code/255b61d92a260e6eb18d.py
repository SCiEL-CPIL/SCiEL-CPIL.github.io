import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    n_m = n * m  # 20
    m_p = m * p  # 25
    p_n = p * n  # 20
    
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        u_col = np.zeros(n_m)
        v_col = np.zeros(m_p)
        w_col = np.zeros(p_n)
        for idx, val in u_vec:
            u_col[idx] = val
        for idx, val in v_vec:
            v_col[idx] = val
        for idx, val in w_vec:
            w_col[idx] = val
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
    
    def schoolbook(n_prime, m_prime, p_prime, i_off, j_off, k_off):
        for i_loc in range(n_prime):
            for j_loc in range(m_prime):
                for k_loc in range(p_prime):
                    ig = i_off + i_loc
                    jg = j_off + j_loc
                    kg = k_off + k_loc
                    I = ig * m + jg
                    J = jg * p + kg
                    K = kg * n + ig
                    add_term([(I, 1)], [(J, 1)], [(K, 1)])
    
    def strassen_2x2x2(i_off, j_off, k_off):
        def gI(i_l, j_l): return (i_off + i_l) * m + (j_off + j_l)
        def gJ(j_l, k_l): return (j_off + j_l) * p + (k_off + k_l)
        def gK(k_l, i_l): return (k_off + k_l) * n + (i_off + i_l)
        
        # M1: (A00+A11)(B00+B11) -> C00 + C11
        add_term(
            [(gI(0,0), 1), (gI(1,1), 1)],
            [(gJ(0,0), 1), (gJ(1,1), 1)],
            [(gK(0,0), 1), (gK(1,1), 1)]
        )
        # M2: (A10+A11)B00 -> C10 - C11
        add_term(
            [(gI(1,0), 1), (gI(1,1), 1)],
            [(gJ(0,0), 1)],
            [(gK(0,1), 1), (gK(1,1), -1)]
        )
        # M3: A00(B01-B11) -> C01 + C11
        add_term(
            [(gI(0,0), 1)],
            [(gJ(0,1), 1), (gJ(1,1), -1)],
            [(gK(1,0), 1), (gK(1,1), 1)]
        )
        # M4: A11(B10-B00) -> C00 + C10
        add_term(
            [(gI(1,1), 1)],
            [(gJ(1,0), 1), (gJ(0,0), -1)],
            [(gK(0,0), 1), (gK(0,1), 1)]
        )
        # M5: (A00+A01)B11 -> -C00 + C01
        add_term(
            [(gI(0,0), 1), (gI(0,1), 1)],
            [(gJ(1,1), 1)],
            [(gK(0,0), -1), (gK(1,0), 1)]
        )
        # M6: (A10-A00)(B00+B01) -> C11
        add_term(
            [(gI(1,0), 1), (gI(0,0), -1)],
            [(gJ(0,0), 1), (gJ(0,1), 1)],
            [(gK(1,1), 1)]
        )
        # M7: (A01-A11)(B10+B11) -> C00
        add_term(
            [(gI(0,1), 1), (gI(1,1), -1)],
            [(gJ(1,0), 1), (gJ(1,1), 1)],
            [(gK(0,0), 1)]
        )
    
    def handle_block(i_off, j_off, k_off, i_sz, j_sz, k_sz):
        if i_sz == 2 and j_sz == 2 and k_sz == 2:
            strassen_2x2x2(i_off, j_off, k_off)
        elif i_sz == 2 and j_sz == 2 and k_sz == 3:
            strassen_2x2x2(i_off, j_off, k_off)
            schoolbook(2, 2, 1, i_off, j_off, k_off + 2)
        elif i_sz == 2 and j_sz == 3 and k_sz == 2:
            strassen_2x2x2(i_off, j_off, k_off)
            schoolbook(2, 1, 2, i_off, j_off + 2, k_off)
        elif i_sz == 2 and j_sz == 3 and k_sz == 3:
            strassen_2x2x2(i_off, j_off, k_off)
            schoolbook(2, 2, 1, i_off, j_off, k_off + 2)
            schoolbook(2, 1, 2, i_off, j_off + 2, k_off)
            schoolbook(2, 1, 1, i_off, j_off + 2, k_off + 2)
        else:
            schoolbook(i_sz, j_sz, k_sz, i_off, j_off, k_off)
    
    # Partition: n=4 as [2,2], m=5 as [2,3], p=5 as [2,3]
    i_blocks = [(0, 2), (2, 2)]
    j_blocks = [(0, 2), (2, 3)]
    k_blocks = [(0, 2), (2, 3)]
    
    for i_off, i_sz in i_blocks:
        for j_off, j_sz in j_blocks:
            for k_off, k_sz in k_blocks:
                handle_block(i_off, j_off, k_off, i_sz, j_sz, k_sz)
    
    rank = len(U_cols)
    
    if rank == 0:
        rank = n * m * p
        U = [[0] * rank for _ in range(n_m)]
        V = [[0] * rank for _ in range(m_p)]
        W = [[0] * rank for _ in range(p_n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    # Verify using einsum
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n_m)]
        V = [[0] * rank for _ in range(m_p)]
        W = [[0] * rank for _ in range(p_n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by checking for linear dependencies
    if time.time() < deadline:
        # Attempt rank reduction: check if any column can be removed
        # by seeing if the remaining columns still span the tensor
        best_rank = rank
        best_U, best_V, best_W = U, V, W
        
        # Try removing each column one at a time
        for c_remove in range(rank):
            if time.time() > deadline:
                break
            
            # Skip this column
            mask = np.ones(rank, dtype=bool)
            mask[c_remove] = False
            
            U_sub = U[:, mask]
            V_sub = V[:, mask]
            W_sub = W[:, mask]
            
            U3_sub = U_sub.reshape(n, m, -1)
            V3_sub = V_sub.reshape(m, p, -1)
            W3_sub = W_sub.reshape(p, n, -1)
            T_sub = np.einsum('ijc,jkc,kic->ijk', U3_sub, V3_sub, W3_sub)
            
            if np.allclose(T_sub, 1.0, atol=1e-9):
                best_rank = len(mask) - 1
                best_U, best_V, best_W = U_sub, V_sub, W_sub
                # Restart search from reduced rank
                U, V, W = U_sub, V_sub, W_sub
                rank = best_rank
                c_remove = 0  # Restart
                continue
        
        if best_rank < rank:
            U, V, W = best_U, best_V, best_W
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
