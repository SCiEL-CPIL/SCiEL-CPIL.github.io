"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Partition indices
    # n: 2, 1
    # m: 2, 2, 1
    # p: 2, 2, 2
    
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    # Precompute offsets
    n_offsets = [0]
    for size in n_splits:
        n_offsets.append(n_offsets[-1] + size)
        
    m_offsets = [0]
    for size in m_splits:
        m_offsets.append(m_offsets[-1] + size)
        
    p_offsets = [0]
    for size in p_splits:
        p_offsets.append(p_offsets[-1] + size)
        
    # Calculate total rank
    total_rank = 0
    for ni, ni_size in enumerate(n_splits):
        for mi, mi_size in enumerate(m_splits):
            for pi, pi_size in enumerate(p_splits):
                if ni_size == 2 and mi_size == 2 and pi_size == 2:
                    total_rank += 7
                else:
                    total_rank += ni_size * mi_size * pi_size
                    
    # Initialize global matrices
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    # Helper to get Strassen factors for 2x2x2
    def get_strassen_2x2():
        # U: 4x7, V: 4x7, W: 4x7
        # Indices: A_00, A_01, A_10, A_11
        # U rows: 0,1,2,3
        # V rows: 0,1,2,3
        # W rows: 0,1,2,3 (C_00, C_01, C_10, C_11)
        
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # M1: (A00+A11)(B00+B11)
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[3, 0] = 1 # C00 += M1, C11 += M1
        
        # M2: (A10+A11)B00
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        W_s[2, 1] = 1; W_s[3, 1] = -1 # C10 += M2, C11 -= M2
        
        # M3: A00(B01-B11)
        U_s[0, 2] = 1
        V_s[1, 2] = 1; V_s[3, 2] = -1
        W_s[1, 2] = 1; W_s[3, 2] = 1 # C01 += M3, C11 += M3
        
        # M4: A11(B10-B00)
        U_s[3, 3] = 1
        V_s[2, 3] = 1; V_s[0, 3] = -1
        W_s[0, 3] = 1; W_s[2, 3] = 1 # C00 += M4, C10 += M4
        
        # M5: (A00+A01)B11
        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[3, 4] = 1
        W_s[0, 4] = -1; W_s[1, 4] = 1 # C00 -= M5, C01 += M5
        
        # M6: (A10-A00)(B00+B01)
        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[1, 5] = 1
        W_s[3, 5] = 1 # C11 += M6
        
        # M7: (A01-A11)(B10+B11)
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[2, 6] = 1; V_s[3, 6] = 1
        W_s[0, 6] = 1 # C00 += M7
        
        return U_s, V_s, W_s

    # Helper to get schoolbook factors
    def get_schoolbook(nb, mb, pb):
        rank = nb * mb * pb
        U_b = np.zeros((nb * mb, rank))
        V_b = np.zeros((mb * pb, rank))
        W_b = np.zeros((pb * nb, rank))
        
        col = 0
        for i in range(nb):
            for j in range(mb):
                for k in range(pb):
                    U_b[i * mb + j, col] = 1
                    V_b[j * pb + k, col] = 1
                    W_b[k * nb + i, col] = 1
                    col += 1
        return U_b, V_b, W_b

    for ni, ni_size in enumerate(n_splits):
        for mi, mi_size in enumerate(m_splits):
            for pi, pi_size in enumerate(p_splits):
                if ni_size == 2 and mi_size == 2 and pi_size == 2:
                    U_b, V_b, W_b = get_strassen_2x2()
                    rank_b = 7
                else:
                    U_b, V_b, W_b = get_schoolbook(ni_size, mi_size, pi_size)
                    rank_b = ni_size * mi_size * pi_size
                
                # Map to global indices
                n_start = n_offsets[ni]
                m_start = m_offsets[mi]
                p_start = p_offsets[pi]
                
                # U indices: i * m + j
                # Local i in [0, ni_size), j in [0, mi_size)
                # Global row = (n_start + i) * m + (m_start + j)
                
                # V indices: j * p + k
                # Local j in [0, mi_size), k in [0, pi_size)
                # Global row = (m_start + j) * p + (p_start + k)
                
                # W indices: k * n + i
                # Local k in [0, pi_size), i in [0, ni_size)
                # Global row = (p_start + k) * n + (n_start + i)
                
                for r in range(rank_b):
                    c = col_offset + r
                    
                    # Fill U
                    for local_i in range(ni_size):
                        for local_j in range(mi_size):
                            val = U_b[local_i * mi_size + local_j, r]
                            if val != 0:
                                global_row = (n_start + local_i) * m + (m_start + local_j)
                                U[global_row, c] += val
                                
                    # Fill V
                    for local_j in range(mi_size):
                        for local_k in range(pi_size):
                            val = V_b[local_j * pi_size + local_k, r]
                            if val != 0:
                                global_row = (m_start + local_j) * p + (p_start + local_k)
                                V[global_row, c] += val
                                
                    # Fill W
                    for local_k in range(pi_size):
                        for local_i in range(ni_size):
                            val = W_b[local_k * ni_size + local_i, r]
                            if val != 0:
                                global_row = (p_start + local_k) * n + (n_start + local_i)
                                W[global_row, c] += val
                                                    
                col_offset += rank_b
                
    # Convert to lists of lists as per interface requirement
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
