import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen's algorithm decomposition for 2x2 matrix multiplication
    # Rank 7
    
    n, m, p = DIMENSIONS
    rank = 7
    
    # U shape: (n*m, rank) = (4, 7)
    # V shape: (m*p, rank) = (4, 7)
    # W shape: (p*n, rank) = (4, 7)
    
    # Indices for U (A): 0:00, 1:01, 2:10, 3:11
    # Indices for V (B): 0:00, 1:01, 2:10, 3:11
    # Indices for W (C): 0:00, 1:01, 2:10, 3:11 (using k*n+i convention: 0*2+0=0, 1*2+0=2? No.)
    
    # Wait, let's re-verify W index convention.
    # "W[k*n+i,r]"
    # n=2.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # My previous mapping:
    # Index 0: C[0,0]
    # Index 1: C[0,1] -> k=1, i=0? No. C[i,k].
    # Usually C is n x p. Rows i, cols k.
    # If W index is k*n + i:
    # k=0, i=0 -> 0 (C[0,0])
    # k=0, i=1 -> 1 (C[1,0])
    # k=1, i=0 -> 2 (C[0,1])
    # k=1, i=1 -> 3 (C[1,1])
    
    # Let's check the standard trace convention.
    # Often W is flattened such that W[k*n+i] corresponds to C[i,k].
    # If the task says "W[k*n+i,r]", then:
    # r=0: k=0, i=0 -> C[0,0]
    # r=1: k=0, i=1 -> C[1,0]
    # r=2: k=1, i=0 -> C[0,1]
    # r=3: k=1, i=1 -> C[1,1]
    
    # Let's re-map my W vectors with this index order:
    # 0: C[0,0]
    # 1: C[1,0]
    # 2: C[0,1]
    # 3: C[1,1]
    
    # My previous W vectors were mapped to:
    # 0: C[0,0]
    # 1: C[0,1]
    # 2: C[1,0]
    # 3: C[1,1]
    
    # So I need to swap indices 1 and 2 in my W vectors.
    
    # Let's re-derive W vectors with indices:
    # w[0] = coeff for C[0,0]
    # w[1] = coeff for C[1,0]
    # w[2] = coeff for C[0,1]
    # w[3] = coeff for C[1,1]
    
    # M1: C[0,0] + C[1,1]
    # w1 = [1, 0, 0, 1]
    
    # M2: C[1,0] - C[1,1]
    # w2 = [0, 1, 0, -1]
    
    # M3: C[0,1] + C[1,1]
    # w3 = [0, 0, 1, 1]
    
    # M4: C[0,0] + C[1,0]
    # w4 = [1, 1, 0, 0]
    
    # M5: -C[0,0] + C[0,1]
    # w5 = [-1, 0, 1, 0]
    
    # M6: C[1,1]
    # w6 = [0, 0, 0, 1]
    
    # M7: C[0,0]
    # w7 = [1, 0, 0, 0]
    
    # Let's verify C[0,1] = M3 + M5.
    # w3[2] = 1, w5[2] = 1. Sum 2?
    # No, C[0,1] = M3 + M5.
    # M3 = a00(b01 - b11).
    # M5 = (a00 + a01)b11.
    # Sum = a00b01 - a00b11 + a00b11 + a01b11 = a00b01 + a01b11.
    # Correct.
    
    # Let's verify C[1,0] = M2 + M4.
    # M2 = (a10 + a11)b00.
    # M4 = a11(b10 - b00).
    # Sum = a10b00 + a11b00 + a11b10 - a11b00 = a10b00 + a11b10.
    # Correct.
    
    # Let's verify C[0,0] = M1 + M4 - M5 + M7.
    # w1[0]=1, w4[0]=1, w5[0]=-1, w7[0]=1.
    # Sum coeffs: 1 + 1 - 1 + 1 = 2?
    # Wait.
    # M1 = (a00+a11)(b00+b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M7 = (a01-a11)(b10+b11)
    
    # Coeff of a00b00:
    # M1: 1
    # M4: 0
    # M5: 0
    # M7: 0
    # Sum: 1. Correct.
    
    # Coeff of a01b10:
    # M1: 0
    # M4: 0
    # M5: 0
    # M7: 1 (from a01*b10)
    # Sum: 1. Correct.
    
    # Coeff of a11b11:
    # M1: 1
    # M4: 0
    # M5: 0
    # M7: -1 (from -a11*b11)
    # Sum: 0. Correct.
    
    # So the W vectors are correct.
    
    U = [
        [1, 0, 1, 0, 1, -1, 0], # u1..u7 for A[0,0]
        [0, 0, 0, 0, 1, 0, 1],  # for A[0,1]
        [0, 1, 0, 0, 0, 1, 0],  # for A[1,0]
        [1, 1, 0, 1, 0, 0, -1]  # for A[1,1]
    ]
    
    V = [
        [1, 1, 0, -1, 0, 1, 0], # v1..v7 for B[0,0]
        [0, 0, 1, 0, 0, 1, 0],  # for B[0,1]
        [0, 0, 0, 1, 0, 0, 1],  # for B[1,0]
        [1, 0, -1, 0, 1, 0, 1]  # for B[1,1]
    ]
    
    W = [
        [1, 0, 0, 1, -1, 0, 1], # w1..w7 for C[0,0] (idx 0)
        [0, 1, 0, 1, 0, 0, 0],  # for C[1,0] (idx 1)
        [0, 0, 1, 0, 1, 0, 0],  # for C[0,1] (idx 2)
        [1, -1, 1, 0, 0, 1, 0]  # for C[1,1] (idx 3)
    ]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
