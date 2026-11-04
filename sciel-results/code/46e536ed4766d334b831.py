"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_dict, v_dict, w_dict):
        """Add a rank-1 term. u_dict: {index: value}, etc."""
        u = np.zeros(n*m)
        v = np.zeros(m*p)
        w = np.zeros(p*n)
        for idx, val in u_dict.items():
            u[idx] += val
        for idx, val in v_dict.items():
            v[idx] += val
        for idx, val in w_dict.items():
            w[idx] += val
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    # Strassen for 2x2 x 2x2
    # A indices: a00=0, a01=1, a10=3, a11=4 (in U: i*m+j)
    # B indices: b00=0, b01=1, b10=10, b11=11 (in V: j*p+k)
    # C indices: c00=0, c01=2, c10=1, c11=3 (in W: k*n+i)
    
    def strassen_2x2(A_base, B_base, C_base):
        """A_base: (a00_idx, a01_idx, a10_idx, a11_idx)
        B_base: (b00_idx, b01_idx, b10_idx, b11_idx)
        C_base: (c00_idx, c01_idx, c10_idx, c11_idx)
        """
        a00, a01, a10, a11 = A_base
        b00, b01, b10, b11 = B_base
        c00, c01, c10, c11 = C_base
        
        # M1 = (a00+a11)(b00+b11)
        add_term({a00: 1, a11: 1}, {b00: 1, b11: 1}, {c00: 1, c11: 1})
        # M2 = (a10+a11)b00
        add_term({a10: 1, a11: 1}, {b00: 1}, {c10: 1})
        # M3 = a00(b01-b11)
        add_term({a00: 1}, {b01: 1, b11: -1}, {c00: 1, c01: 1})
        # M4 = a11(b10-b00)
        add_term({a11: 1}, {b10: 1, b00: -1}, {c00: 1, c10: 1})
        # M5 = (a00+a01)b11
        add_term({a00: 1, a01: 1}, {b11: 1}, {c00: -1, c01: 1})
        # M6 = (a10-a00)(b00+b01)
        add_term({a10: 1, a00: -1}, {b00: 1, b01: 1}, {c11: 1})
        # M7 = (a01-a11)(b10+b11)
        add_term({a01: 1, a11: -1}, {b10: 1, b11: 1}, {c00: 1, c11: 1})
    
    def schoolbook_2x2_2x1(A_base, B_base, C_base):
        """2x2 x 2x1 -> 2x1. Rank 2.
        A_base: (a00, a01, a10, a11)
        B_base: (b0, b1)
        C_base: (c0, c1)
        """
        a00, a01, a10, a11 = A_base
        b0, b1 = B_base
        c0, c1 = C_base
        add_term({a00: 1, a10: 1}, {b0: 1}, {c0: 1, c1: 1})
        add_term({a01: 1, a11: 1}, {b1: 1}, {c0: 1, c1: 1})
    
    def schoolbook_2x1_1xk(A_idx, B_base, C_base):
        """2x1 x 1xk -> 2xk. Rank 2k.
        A_idx: (a0, a1)
        B_base: (b0, b1, ..., b_{k-1})
        C_base: (c0, c1, ..., c_{k-1})
        """
        a0, a1 = A_idx
        for j in range(len(B_base)):
            add_term({a0: 1}, {B_base[j]: 1}, {C_base[j]: 1})
            add_term({a1: 1}, {B_base[j]: 1}, {C_base[len(B_base)+j]: 1})
    
    # A indices: A[i,j] at U[i*m+j]
    # A1 (2x2): a00=0, a01=1, a10=3, a11=4
    # A2 (2x1): a02=2, a12=5
    A1 = (0, 1, 3, 4)
    A2 = (2, 5)
    
    # B indices: B[j,k] at V[j*p+k]
    # B11 (2x3): j=0,1; k=0,1,2 -> (0,1,2,10,11,12)
    # B12 (2x3): j=0,1; k=3,4,5 -> (3,4,5,13,14,15)
    # B13 (2x3): j=0,1; k=6,7,8 -> (6,7,8,16,17,18)
    # B14 (2x1): j=0,1; k=9 -> (9,19)
    # B21 (1x3): j=2; k=0,1,2 -> (20,21,22)
    # B22 (1x3): j=2; k=3,4,5 -> (23,24,25)
    # B23 (1x3): j=2; k=6,7,8 -> (26,27,28)
    # B24 (1x1): j=2; k=9 -> (29,)
    
    # C indices: C[i,k] at W[k*n+i]
    # C11 (2x3): i=0,1; k=0,1,2 -> (0,2,1,3,4,5)
    # C12 (2x3): i=0,1; k=3,4,5 -> (6,8,7,9,10,11)
    # C13 (2x3): i=0,1; k=6,7,8 -> (12,14,13,15,16,17)
    # C14 (2x1): i=0,1; k=9 -> (18,19)
    # C21 (2x3): same as C11
    # C22 (2x3): same as C12
    # C23 (2x3): same as C13
    # C24 (2x1): same as C14
    
    # Block 1: A1 * B11 -> C11
    # B11 split: B11a (2x2): (0,1,10,11), B11b (2x1): (2,12)
    # C11 split: C11a (2x2): (0,2,1,3), C11b (2x1): (4,5)
    strassen_2x2(A1, (0,1,10,11), (0,2,1,3))
    schoolbook_2x2_2x1(A1, (2,12), (4,5))
    
    # Block 2: A1 * B12 -> C12
    # B12a (2x2): (3,4,13,14), B12b (2x1): (5,15)
    # C12a (2x2): (6,8,7,9), C12b (2x1): (10,11)
    strassen_2x2(A1, (3,4,13,14), (6,8,7,9))
    schoolbook_2x2_2x1(A1, (5,15), (10,11))
    
    # Block 3: A1 * B13 -> C13
    # B13a (2x2): (6,7,16,17), B13b (2x1): (8,18)
    # C13a (2x2): (12,14,13,15), C13b (2x1): (16,17)
    strassen_2x2(A1, (6,7,16,17), (12,14,13,15))
    schoolbook_2x2_2x1(A1, (8,18), (16,17))
    
    # Block 4: A1 * B14 -> C14
    # B14 (2x1): (9,19)
    # C14 (2x1): (18,19)
    schoolbook_2x2_2x1(A1, (9,19), (18,19))
    
    # Block 5: A2 * B21 -> C21
    # B21 (1x3): (20,21,22)
    # C21 (2x3): (0,2,1,3,4,5)
    schoolbook_2x1_1xk(A2, (20,21,22), (0,2,1,3,4,5))
    
    # Block 6: A2 * B22 -> C22
    # B22 (1x3): (23,24,25)
    # C22 (2x3): (6,8,7,9,10,11)
    schoolbook_2x1_1xk(A2, (23,24,25), (6,8,7,9,10,11))
    
    # Block 7: A2 * B23 -> C23
    # B23 (1x3): (26,27,28)
    # C23 (2x3): (12,14,13,15,16,17)
    schoolbook_2x1_1xk(A2, (26,27,28), (12,14,13,15,16,17))
    
    # Block 8: A2 * B24 -> C24
    # B24 (1x1): (29,)
    # C24 (2x1): (18,19)
    schoolbook_2x1_1xk(A2, (29,), (18,19))
    
    rank = len(U_cols)
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
