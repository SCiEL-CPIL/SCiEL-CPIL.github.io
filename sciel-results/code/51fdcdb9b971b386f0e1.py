import time
import sys
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Partition dimensions
    # n = 3 -> [2, 1]
    # m = 4 -> [2, 2]
    # p = 7 -> [2, 2, 2, 1]
    
    n_splits = [2, 1]
    m_splits = [2, 2]
    p_splits = [2, 2, 2, 1]
    
    # Calculate offsets
    n_offsets = [0]
    for sz in n_splits[:-1]:
        n_offsets.append(n_offsets[-1] + sz)
        
    m_offsets = [0]
    for sz in m_splits[:-1]:
        m_offsets.append(m_offsets[-1] + sz)
        
    p_offsets = [0]
    for sz in p_splits[:-1]:
        p_offsets.append(p_offsets[-1] + sz)
        
    # Accumulate rank
    total_rank = 0
    block_data = [] # List of (n_idx, m_idx, p_idx, n_off, m_off, p_off, n_sz, m_sz, p_sz, rank, is_strassen)
    
    for i_idx, n_sz in enumerate(n_splits):
        for j_idx, m_sz in enumerate(m_splits):
            for k_idx, p_sz in enumerate(p_splits):
                if n_sz == 2 and m_sz == 2 and p_sz == 2:
                    rank = 7
                    is_strassen = True
                else:
                    rank = n_sz * m_sz * p_sz
                    is_strassen = False
                total_rank += rank
                block_data.append((i_idx, j_idx, k_idx, n_offsets[i_idx], m_offsets[j_idx], p_offsets[k_idx], n_sz, m_sz, p_sz, rank, is_strassen))
    
    # Initialize U, V, W
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    current_col = 0
    
    for (i_idx, j_idx, k_idx, n_off, m_off, p_off, n_sz, m_sz, p_sz, rank, is_strassen) in block_data:
        if is_strassen:
            # Strassen 2x2x2 decomposition
            # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
            # Global indices:
            # u_idx = (n_off + i) * m + (m_off + j)
            # v_idx = (m_off + j) * p + (p_off + k)
            # w_idx = (p_off + k) * n + (n_off + i)
            
            # Strassen multiplications M1..M7
            # We define the vectors for each multiplication term.
            # Each term t corresponds to a column in the local U, V, W matrices.
            # We need to map the Strassen formulas to the tensor decomposition.
            
            # The tensor decomposition T = sum_c U_c V_c^T W_c^T
            # For Strassen, the "multiplications" are the rank-1 terms.
            # Let's define the 7 rank-1 tensors explicitly.
            
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B10-B11)
            # M4 = A11(B01-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # We need to express C_ik in terms of these M's.
            # C00 = M1 - M2 + M4 + M7
            # C01 = M3 + M5
            # C10 = M1 + M2
            # C11 = M4 - M1 + M3 + M6
            
            # This means the coefficient of M_t in C_ik is given by the above equations.
            # The term in the bilinear form is A_ij B_jk.
            # For a specific M_t, it is a product of a linear combination of A's and a linear combination of B's.
            # M_t = (sum_i,j a_{t,ij} A_{ij}) * (sum_j,k b_{t,jk} B_{jk})
            # Then C_ik gets contribution coeff_{t,ik} * M_t.
            # This is equivalent to adding coeff_{t,ik} * a_{t,ij} * b_{t,jk} to the tensor entry T_{(i,j,k)}.
            
            # However, the interface requires U, V, W such that:
            # sum_c U[u,c] V[v,c] W[w,c] = T[u,v,w]
            # Where u = i*m+j, v = j*p+k, w = k*n+i.
            
            # For a Strassen term t:
            # U[u,c] = a_{t,ij}
            # V[v,c] = b_{t,jk}
            # W[w,c] = coeff_{t,ik}
            
            # Let's define the 7 terms.
            
            # Term 1: M1
            # a: A00, A11 -> 1
            # b: B00, B11 -> 1
            # coeffs: C00:1, C10:1
            # 
            # Term 2: M2
            # a: A10, A11 -> 1
            # b: B00 -> 1
            # coeffs: C00:-1, C10:1
            
            # Term 3: M3
            # a: A00 -> 1
            # b: B10:1, B11:-1
            # coeffs: C01:1, C11:1
            
            # Term 4: M4
            # a: A11 -> 1
            # b: B01:1, B00:-1
            # coeffs: C00:1, C11:1
            
            # Term 5: M5
            # a: A00:1, A01:1
            # b: B11 -> 1
            # coeffs: C01:1
            
            # Term 6: M6
            # a: A10:1, A00:-1
            # b: B00:1, B01:1
            # coeffs: C11:1
            
            # Term 7: M7
            # a: A01:1, A11:-1
            # b: B10:1, B11:1
            # coeffs: C00:1
            
            # Helper to set local values
            def set_local(U, V, W, n_off, m_off, p_off, n_sz, m_sz, p_sz, col_start, a_dict, b_dict, c_dict):
                # a_dict: {(i,j): val}
                # b_dict: {(j,k): val}
                # c_dict: {(i,k): val}
                
                for (i, j), val in a_dict.items():
                    if val != 0:
                        u_idx = (n_off + i) * m + (m_off + j)
                        U[u_idx, col_start] = val
                        
                for (j, k), val in b_dict.items():
                    if val != 0:
                        v_idx = (m_off + j) * p + (p_off + k)
                        V[v_idx, col_start] = val
                        
                for (i, k), val in c_dict.items():
                    if val != 0:
                        w_idx = (p_off + k) * n + (n_off + i)
                        W[w_idx, col_start] = val

            # Define the 7 terms
            # Indices are local (0,1)
            
            # M1
            a1 = {(0,0): 1, (1,1): 1}
            b1 = {(0,0): 1, (1,1): 1}
            c1 = {(0,0): 1, (1,0): 1}
            set_local(U, V, W, n_off, m_off, p_off, n_sz, m_sz, p_sz, current_col, a1, b1, c1)
            current_col += 1
            
            # M2
            a2 = {(1,0): 1, (1,1): 1}
            b2 = {(0,0): 1}
            c2 = {(0,0): -1, (1,0): 1}
            set_local(U, V, W, n_off, m_off, p_off, n_sz, m_sz, p_sz, current_col, a2, b2, c2)
            current_col += 1
            
            # M3
            a3 = {(0,0): 1}
            b3 = {(1,0): 1, (1,1): -1}
            c3 = {(0,1): 1, (1,1): 1}
            set_local(U, V, W, n_off, m_off, p_off, n_sz, m_sz, p_sz, current_col, a3, b3, c3)
            current_col += 1
            
            # M4
            a4 = {(1,1): 1}
            b4 = {(0,1): 1, (0,0): -1}
            c4 = {(0,0): 1, (1,1): 1}
            set_local(U, V, W, n_off, m_off, p_off, n_sz, m_sz, p_sz, current_col, a4, b4, c4)
            current_col += 1
            
            # M5
            a5 = {(0,0): 1, (0,1): 1}
            b5 = {(1,1): 1}
            c5 = {(0,1): 1}
            set_local(U, V, W, n_off, m_off, p_off, n_sz, m_sz, p_sz, current_col, a5, b5, c5)
            current_col += 1
            
            # M6
            a6 = {(1,0): 1, (0,0): -1}
            b6 = {(0,0): 1, (0,1): 1}
            c6 = {(1,1): 1}
            set_local(U, V, W, n_off, m_off, p_off, n_sz, m_sz, p_sz, current_col, a6, b6, c6)
            current_col += 1
            
            # M7
            a7 = {(0,1): 1, (1,1): -1}
            b7 = {(1,0): 1, (1,1): 1}
            c7 = {(0,0): 1}
            set_local(U, V, W, n_off, m_off, p_off, n_sz, m_sz, p_sz, current_col, a7, b7, c7)
            current_col += 1
            
        else:
            # Standard schoolbook for this block
            for i in range(n_sz):
                for j in range(m_sz):
                    for k in range(p_sz):
                        u_idx = (n_off + i) * m + (m_off + j)
                        v_idx = (m_off + j) * p + (p_off + k)
                        w_idx = (p_off + k) * n + (n_off + i)
                        
                        U[u_idx, current_col] = 1
                        V[v_idx, current_col] = 1
                        W[w_idx, current_col] = 1
                        current_col += 1
                        
    # Verify rank
    assert current_col == total_rank, f"Rank mismatch: {current_col} vs {total_rank}"
    
    # Convert to lists for output
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
