"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Partition indices
    # n: 2, 2
    # m: 2, 2, 1
    # p: 2, 2, 2
    
    n_parts = [2, 2]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 2]
    
    # Calculate offsets
    n_offsets = [0]
    for sz in n_parts[:-1]:
        n_offsets.append(n_offsets[-1] + sz)
        
    m_offsets = [0]
    for sz in m_parts[:-1]:
        m_offsets.append(m_offsets[-1] + sz)
        
    p_offsets = [0]
    for sz in p_parts[:-1]:
        p_offsets.append(p_offsets[-1] + sz)
        
    # We will build U, V, W as lists of lists (or numpy arrays) and then combine.
    # Since the rank is small (108), we can use lists for simplicity and speed of appending.
    # U: (n*m) x Rank
    # V: (m*p) x Rank
    # W: (p*n) x Rank
    
    # Initialize U, V, W with zeros? No, we build columns.
    # Each "rank-1 tensor" adds a column to U, V, W.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_factor(u_vec, v_vec, w_vec):
        """
        u_vec: length n*m
        v_vec: length m*p
        w_vec: length p*n
        """
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Helper to create a zero vector
    def zero_u(): return [0.0] * (n * m)
    def zero_v(): return [0.0] * (m * p)
    def zero_w(): return [0.0] * (p * n)
    
    # Strassen factors for 2x2x2
    # Indices i,j,k in {0,1}
    # U_r[i*2+j], V_r[j*2+k], W_r[k*2+i]
    
    # r=1: M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # U1: A00(1), A11(1) -> idx 0, 3
    # V1: B00(1), B11(1) -> idx 0, 3
    # W1: C00(1), C11(1) -> idx 0, 3 (k=0,i=0 -> 0; k=1,i=1 -> 3)
    
    # r=2: M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # U2: A10(1), A11(1) -> idx 2, 3
    # V2: B00(1) -> idx 0
    # W2: C10(1), C11(-1) -> C10: k=0,i=1 -> idx 1; C11: k=1,i=1 -> idx 3
    
    # r=3: M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    # U3: A00(1) -> idx 0
    # V3: B01(1), B11(-1) -> idx 1, 3
    # W3: C01(1), C11(1) -> C01: k=1,i=0 -> idx 2; C11: k=1,i=1 -> idx 3
    
    # r=4: M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    # U4: A11(1) -> idx 3
    # V4: B10(1), B00(-1) -> idx 2, 0
    # W4: C00(1), C10(1) -> C00: k=0,i=0 -> idx 0; C10: k=0,i=1 -> idx 1
    
    # r=5: M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # U5: A00(1), A01(1) -> idx 0, 1
    # V5: B11(1) -> idx 3
    # W5: C00(-1), C01(1) -> C00: idx 0 (-1); C01: idx 2 (1)
    
    # r=6: M6 = (A10-A00)(B00+B01) -> C11 -= M6
    # U6: A10(1), A00(-1) -> idx 2, 0
    # V6: B00(1), B01(1) -> idx 0, 1
    # W6: C11(-1) -> idx 3 (-1)
    
    # r=7: M7 = (A01-A11)(B10+B11) -> C00 += M7
    # U7: A01(1), A11(-1) -> idx 1, 3
    # V7: B10(1), B11(1) -> idx 2, 3
    # W7: C00(1) -> idx 0 (1)
    
    strassen_factors = [
        # (U_indices, U_vals, V_indices, V_vals, W_indices, W_vals)
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        ([2, 3], [1, 1], [0], [1], [1, 3], [1, -1]),
        ([0], [1], [1, 3], [1, -1], [2, 3], [1, 1]),
        ([3], [1], [2, 0], [1, -1], [0, 1], [1, 1]),
        ([0, 1], [1, 1], [3], [1], [0, 2], [-1, 1]),
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [-1]),
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),
    ]
    
    # Iterate over blocks
    for i in range(len(n_parts)):
        for j in range(len(m_parts)):
            for k in range(len(p_parts)):
                ni = n_parts[i]
                mj = m_parts[j]
                pk = p_parts[k]
                
                i0 = n_offsets[i]
                j0 = m_offsets[j]
                k0 = p_offsets[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    # Use Strassen
                    for f in strassen_factors:
                        u_idx, u_val, v_idx, v_val, w_idx, w_val = f
                        u_vec = zero_u()
                        v_vec = zero_v()
                        w_vec = zero_w()
                        
                        for idx, val in zip(u_idx, u_val):
                            # idx is local 0..3 for 2x2
                            # global index: (i0 + li) * m + (j0 + lj)
                            li, lj = divmod(idx, 2)
                            gi = i0 + li
                            gj = j0 + lj
                            u_vec[gi * m + gj] = val
                            
                        for idx, val in zip(v_idx, v_val):
                            # idx is local 0..3 for 2x2 (j,k)
                            # global index: (j0 + lj) * p + (k0 + lk)
                            lj, lk = divmod(idx, 2)
                            gj = j0 + lj
                            gk = k0 + lk
                            v_vec[gj * p + gk] = val
                            
                        for idx, val in zip(w_idx, w_val):
                            # idx is local 0..3 for 2x2 (k,i)
                            # global index: (k0 + lk) * n + (i0 + li)
                            lk, li = divmod(idx, 2)
                            gk = k0 + lk
                            gi = i0 + li
                            w_vec[gk * n + gi] = val
                            
                        add_factor(u_vec, v_vec, w_vec)
                else:
                    # Schoolbook
                    # Rank ni * mj * pk
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                u_vec = zero_u()
                                v_vec = zero_v()
                                w_vec = zero_w()
                                
                                gi = i0 + li
                                gj = j0 + lj
                                gk = k0 + lk
                                
                                u_vec[gi * m + gj] = 1.0
                                v_vec[gj * p + gk] = 1.0
                                w_vec[gk * n + gi] = 1.0
                                
                                add_factor(u_vec, v_vec, w_vec)
                                
    # Convert to numpy arrays
    rank = len(U_cols)
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
