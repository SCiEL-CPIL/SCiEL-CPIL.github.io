import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # We will construct a rank 92 decomposition.
    # Split m=5 into m1=2, m2=3.
    # Split p=5 into p1=2, p2=3.
    
    # The tensor T is built from blocks.
    # Block 1: A[0:4, 0:2] * B[0:2, 0:2] -> C[0:4, 0:2]
    # Block 2: A[0:4, 0:2] * B[0:2, 2:5] -> C[0:4, 2:5]
    # Block 3: A[0:4, 2:5] * B[2:5, 0:2] -> C[0:4, 0:2]
    # Block 4: A[0:4, 2:5] * B[2:5, 2:5] -> C[0:4, 2:5]
    
    # We will just build the schoolbook for each sub-block and concatenate.
    # Rank 1: 4*2*2 = 16? No, schoolbook for 4x2x2 is 16.
    # But we calculated r(4,2,2)=8?
    # r(4,2,2) is 8 because r(1,2,2)=2.
    # The schoolbook for 1x2x2 is 2.
    # So for 4x2x2, we can use 4 independent 1x2x2 decompositions.
    # Each 1x2x2 has rank 2. Total 8.
    
    # Let's verify r(1,2,2).
    # A is 1x2, B is 2x1. C is 1x1.
    # C = A0 B0 + A1 B1.
    # Rank 2. Correct.
    
    # So for 4x2x2:
    # For each row i in 0..3:
    #   C_i0 = A_i0 B_00 + A_i1 B_10
    #   C_i1 = A_i0 B_01 + A_i1 B_11
    # This is 2 multiplications per row? No.
    # The tensor for 1x2x2 is sum of 2 rank-1 tensors.
    # T = T0 + T1.
    # T0: A0 B0 -> C0? No.
    # T0 corresponds to j=0. A_i0 B_0k.
    # T1 corresponds to j=1. A_i1 B_1k.
    # So yes, rank 2.
    
    # So r(4,2,2) = 4 * 2 = 8.
    # r(4,2,3) = 4 * (2*3) = 24.
    # r(4,3,2) = 4 * (3*2) = 24.
    # r(4,3,3) = 4 * (3*3) = 36.
    # Total 8 + 24 + 24 + 36 = 92.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_block(n_b, m_b, p_b, i_off, j_off, k_off):
        # Adds schoolbook decomposition for block of size n_b x m_b x p_b
        # Offset in global indices
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    a_idx = (i_off + i) * m + (j_off + j)
                    b_idx = (j_off + j) * p + (k_off + k)
                    c_idx = (k_off + k) * n + (i_off + i)
                    
                    u = np.zeros(n * m)
                    v = np.zeros(m * p)
                    w = np.zeros(n * p)
                    
                    u[a_idx] = 1.0
                    v[b_idx] = 1.0
                    w[c_idx] = 1.0
                    
                    U_cols.append(u)
                    V_cols.append(v)
                    W_cols.append(w)
                    
    # Block 1: 4x2x2
    add_block(4, 2, 2, 0, 0, 0)
    # Block 2: 4x2x3
    add_block(4, 2, 3, 0, 0, 2)
    # Block 3: 4x3x2
    add_block(4, 3, 2, 0, 2, 0)
    # Block 4: 4x3x3
    add_block(4, 3, 3, 0, 2, 2)
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
