import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 38 decomposition
    rank = 38
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen(a_base, b_base, c_base, offset):
        # a_base: (i, j) -> U index
        # b_base: (j, k) -> V index
        # c_base: (i, k) -> W index
        # offset: rank offset
        
        # M1: (A00 + A11)(B00 + B11)
        U[a_base(0,0)][offset] += 1
        U[a_base(1,1)][offset] += 1
        V[b_base(0,0)][offset] += 1
        V[b_base(1,1)][offset] += 1
        W[c_base(0,0)][offset] += 1
        W[c_base(1,1)][offset] += 1
        
        # M2: (A10 + A11)B00
        U[a_base(1,0)][offset+1] += 1
        U[a_base(1,1)][offset+1] += 1
        V[b_base(0,0)][offset+1] += 1
        W[c_base(1,0)][offset+1] += 1
        W[c_base(1,1)][offset+1] -= 1
        
        # M3: A00(B01 - B11)
        U[a_base(0,0)][offset+2] += 1
        V[b_base(0,1)][offset+2] += 1
        V[b_base(1,1)][offset+2] -= 1
        W[c_base(0,1)][offset+2] += 1
        W[c_base(1,1)][offset+2] += 1
        
        # M4: A11(B10 - B00)
        U[a_base(1,1)][offset+3] += 1
        V[b_base(1,0)][offset+3] += 1
        V[b_base(0,0)][offset+3] -= 1
        W[c_base(0,0)][offset+3] += 1
        W[c_base(1,0)][offset+3] += 1
        
        # M5: (A00 + A01)B11
        U[a_base(0,0)][offset+4] += 1
        U[a_base(0,1)][offset+4] += 1
        V[b_base(1,1)][offset+4] += 1
        W[c_base(0,0)][offset+4] -= 1
        W[c_base(0,1)][offset+4] += 1
        
        # M6: (A10 - A00)(B00 + B01)
        U[a_base(1,0)][offset+5] += 1
        U[a_base(0,0)][offset+5] -= 1
        V[b_base(0,0)][offset+5] += 1
        V[b_base(0,1)][offset+5] += 1
        W[c_base(1,1)][offset+5] += 1
        
        # M7: (A01 - A11)(B10 + B11)
        U[a_base(0,1)][offset+6] += 1
        U[a_base(1,1)][offset+6] -= 1
        V[b_base(1,0)][offset+6] += 1
        V[b_base(1,1)][offset+6] += 1
        W[c_base(0,0)][offset+6] += 1
        W[c_base(1,1)][offset+6] += 1

    def add_schoolbook_223(a_base, b_base, c_base, offset):
        # A is 2x2, B is 2x3, C is 2x3
        # C[i][k] = A[i][0]B[0][k] + A[i][1]B[1][k]
        for k in range(3):
            for i in range(2):
                idx = offset + k * 2 + i
                U[a_base(i,0)][idx] += 1
                V[b_base(0,k)][idx] += 1
                W[c_base(i,k)][idx] += 1
                U[a_base(i,1)][idx] += 1
                V[b_base(1,k)][idx] += 1
                W[c_base(i,k)][idx] += 1 # Wait, W is shared? No, W is separate for each term?
                # Correction: Each term is a rank-1 tensor.
                # Term 1: A[i][0] * B[0][k] * C[i][k]
                # Term 2: A[i][1] * B[1][k] * C[i][k]
                # So we need 2 rank-1 tensors per (i,k) pair.
                # My loop above adds to the same idx for both terms?
                # No, I should use separate indices.
                
    # Let's re-implement add_schoolbook_223 correctly
    def add_schoolbook_223_correct(a_base, b_base, c_base, offset):
        idx = offset
        for k in range(3):
            for i in range(2):
                # Term A[i][0] B[0][k]
                U[a_base(i,0)][idx] += 1
                V[b_base(0,k)][idx] += 1
                W[c_base(i,k)][idx] += 1
                idx += 1
                # Term A[i][1] B[1][k]
                U[a_base(i,1)][idx] += 1
                V[b_base(1,k)][idx] += 1
                W[c_base(i,k)][idx] += 1
                idx += 1

    # Define base index functions
    # Block 1: A cols 0,1. B rows 0,1.
    a1 = lambda i, j: i * m + j
    b1l = lambda j, k: j * p + k # k 0..1
    b1r = lambda j, k: j * p + (k+2) # k 0..2 -> 2..4
    c1l = lambda i, k: k * n + i # k 0..1
    c1r = lambda i, k: (k+2) * n + i # k 0..2 -> 2..4

    # Block 2: A cols 2,3. B rows 2,3.
    a2 = lambda i, j: i * m + (j+2)
    b2l = lambda j, k: (j+2) * p + k # k 0..1
    b2r = lambda j, k: (j+2) * p + (k+2) # k 0..2 -> 2..4
    c2l = lambda i, k: k * n + i # k 0..1
    c2r = lambda i, k: (k+2) * n + i # k 0..2 -> 2..4

    # Add Block 1
    add_strassen(a1, b1l, c1l, 0)
    add_schoolbook_223_correct(a1, b1r, c1r, 7)
    
    # Add Block 2
    add_strassen(a2, b2l, c2l, 19)
    add_schoolbook_223_correct(a2, b2r, c2r, 26)

    return {"U": U, "V": V, "W": W}
