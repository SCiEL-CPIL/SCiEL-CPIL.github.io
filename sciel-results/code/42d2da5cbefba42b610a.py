import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115  # safety margin before 150s limit
    
    # We are solving matrix multiplication: C = A @ B
    # A is (n x m), B is (m x p), C is (n x p)
    # n=2, m=2, p=11
    
    # The tensor T_{i,j,k} = delta_{i,j} delta_{j,k} ... wait, the standard tensor for matrix mult is:
    # T_{(i,k), j} = A_{i,j} B_{j,k} -> C_{i,k}
    # The task says U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # This matches:
    # U rows indexed by (i, j) where i in [0,n), j in [0,m) -> A indices
    # V rows indexed by (j, k) where j in [0,m), k in [0,p) -> B indices
    # W rows indexed by (k, i) where k in [0,p), i in [0,n) -> C indices (note the order p*n, so index is k*n + i)
    # The decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = A[i,j] B[j,k]
    # This is exactly the standard matrix multiplication tensor.
    
    # Strategy: Use Strassen-like decomposition for blocks of 2 columns in B (and C), and standard for the last column.
    # Since p=11, we can have 5 blocks of 2 columns and 1 block of 1 column.
    # Block size 2: rank 7 (Strassen)
    # Block size 1: rank 4 (standard)
    # Total rank: 5*7 + 4 = 39
    
    n, m, p = 2, 2, 11
    rank = 39
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2:
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = A @ B
    # m1 = (a11 + a22)(b11 + b22)
    # m2 = (a21 + a22)b11
    # m3 = a11(b21 - b22)
    # m4 = a22(b11 - b12)
    # m5 = (a11 + a12)b22
    # m6 = (a21 - a11)(b11 + b12)
    # m7 = (a12 - a22)(b21 + b22)
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # Indices for U (A indices): i*m + j
    # i=0, j=0 -> 0 (a11)
    # i=0, j=1 -> 1 (a12)
    # i=1, j=0 -> 2 (a21)
    # i=1, j=1 -> 3 (a22)
    
    # Indices for V (B indices): j*p + k
    # For a 2-column block starting at column k0 (0-indexed), the columns are k0, k0+1.
    # j=0, k=k0 -> 0*p + k0 = k0
    # j=0, k=k0+1 -> k0+1
    # j=1, k=k0 -> 1*p + k0 = p + k0
    # j=1, k=k0+1 -> p + k0 + 1
    
    # Indices for W (C indices): k*n + i
    # k=k0, i=0 -> k0*2 + 0 = 2*k0
    # k=k0, i=1 -> 2*k0 + 1
    # k=k0+1, i=0 -> (k0+1)*2 + 0 = 2*k0 + 2
    # k=k0+1, i=1 -> 2*k0 + 3
    
    def add_strassen_block(k0, rank_offset):
        # m1
        U[0][rank_offset] += 1; U[3][rank_offset] += 1
        V[k0][rank_offset] += 1; V[p+k0+1][rank_offset] += 1
        W[2*k0][rank_offset] += 1; W[2*k0+1][rank_offset] += 1; W[2*k0+2][rank_offset] += 1; W[2*k0+3][rank_offset] -= 1
        
        # m2
        U[2][rank_offset] += 1; U[3][rank_offset] += 1
        V[k0][rank_offset] += 1
        W[2*k0+1][rank_offset] += 1; W[2*k0+3][rank_offset] += 1
        
        # m3
        U[0][rank_offset] += 1
        V[p+k0][rank_offset] += 1; V[p+k0+1][rank_offset] -= 1
        W[2*k0+2][rank_offset] += 1; W[2*k0+3][rank_offset] += 1
        
        # m4
        U[3][rank_offset] += 1
        V[k0][rank_offset] += 1; V[k0+1][rank_offset] -= 1
        W[2*k0][rank_offset] += 1; W[2*k0+1][rank_offset] += 1
        
        # m5
        U[0][rank_offset] += 1; U[1][rank_offset] += 1
        V[p+k0+1][rank_offset] += 1
        W[2*k0][rank_offset] -= 1; W[2*k0+2][rank_offset] += 1
        
        # m6
        U[2][rank_offset] += 1; U[0][rank_offset] -= 1
        V[k0][rank_offset] += 1; V[k0+1][rank_offset] += 1
        W[2*k0+3][rank_offset] += 1
        
        # m7
        U[1][rank_offset] += 1; U[3][rank_offset] -= 1
        V[p+k0][rank_offset] += 1; V[p+k0+1][rank_offset] += 1
        W[2*k0][rank_offset] += 1; W[2*k0+3][rank_offset] += 1
    
    def add_standard_block(k0, rank_offset):
        # Single column k0
        # c[i, k0] = sum_j a[i,j] b[j, k0]
        # m1 = a11 * b1k
        # m2 = a12 * b2k
        # m3 = a21 * b1k
        # m4 = a22 * b2k
        # c1k = m1 + m2
        # c2k = m3 + m4
        
        # m1
        U[0][rank_offset] += 1
        V[k0][rank_offset] += 1
        W[2*k0][rank_offset] += 1
        
        # m2
        U[1][rank_offset] += 1
        V[p+k0][rank_offset] += 1
        W[2*k0][rank_offset] += 1
        
        # m3
        U[2][rank_offset] += 1
        V[k0][rank_offset] += 1
        W[2*k0+1][rank_offset] += 1
        
        # m4
        U[3][rank_offset] += 1
        V[p+k0][rank_offset] += 1
        W[2*k0+1][rank_offset] += 1
    
    rank_offset = 0
    # 5 blocks of 2 columns
    for b in range(5):
        if time.time() > deadline:
            # If we ran out of time, return what we have so far (though it might be incomplete/invalid)
            # But we are building it in memory, so it's better to just finish if it's fast.
            # 39 iterations is extremely fast, so this won't happen.
            pass
        add_strassen_block(2*b, rank_offset)
        rank_offset += 7
    
    # 1 block of 1 column (column 10)
    add_standard_block(10, rank_offset)
    rank_offset += 4
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
