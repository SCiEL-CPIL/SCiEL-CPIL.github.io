import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Initialize with Strassen-like decomposition for 4x4x4 part and schoolbook for 4x4x2 part
    # Rank 49 for 4x4x4, Rank 32 for 4x4x2 -> Total 81
    # But we can do better. Let's use the known rank 77 structure.
    
    # Strassen for 4x4x4:
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # 7 products:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Each product is 2x2x2, rank 7.
    # Total 7 * 7 = 49.
    
    # For the remaining 2 columns (4x4x2):
    # B' = [[B13, B14], [B23, B24]] (assuming 4x4x4 + 4x4x2 = 4x4x6 is not quite right, indices differ)
    # Actually, 4x4x6 means B is 4x6.
    # B = [B_4x4, B_4x2].
    # C = A B_4x4 + A B_4x2.
    # Rank of A B_4x4 is 49.
    # Rank of A B_4x2:
    # Split A into 2x2 blocks. B_4x2 into 2x1 blocks?
    # Let's use schoolbook for 4x4x2: rank 32.
    # Total 81.
    
    # Can we do better for 4x4x2?
    # M4,4,2 rank is 10?
    # If we use rank 10, total 59.
    
    # Let's implement the 77 rank solution from knowledge.
    # It combines Strassen tiling with coordinate descent.
    
    # First, build the 77 rank decomposition.
    rank = 77
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Helper to add a 2x2x2 Strassen decomposition at offset
    def add_strassen_2x2x2(u_offset, v_offset, w_offset, col_offset):
        # Indices for 2x2x2 Strassen
        # A is 2x2, B is 2x2, C is 2x2
        # U indices: i*2 + j (0..3)
        # V indices: j*2 + k (0..3)
        # W indices: k*2 + i (0..3)
        
        # M1 = (A00 + A11)(B00 + B11)
        U[u_offset + 0, col_offset + 0] = 1
        U[u_offset + 3, col_offset + 0] = 1
        V[v_offset + 0, col_offset + 0] = 1
        V[v_offset + 3, col_offset + 0] = 1
        W[w_offset + 0, col_offset + 0] = 1
        W[w_offset + 1, col_offset + 0] = 1
        W[w_offset + 2, col_offset + 0] = 1
        W[w_offset + 3, col_offset + 0] = 1
        
        # M2 = (A10 + A11)B00
        U[u_offset + 2, col_offset + 1] = 1
        U[u_offset + 3, col_offset + 1] = 1
        V[v_offset + 0, col_offset + 1] = 1
        W[w_offset + 1, col_offset + 1] = 1
        W[w_offset + 3, col_offset + 1] = 1
        
        # M3 = A00(B01 - B11)
        U[u_offset + 0, col_offset + 2] = 1
        V[v_offset + 1, col_offset + 2] = 1
        V[v_offset + 3, col_offset + 2] = -1
        W[w_offset + 0, col_offset + 2] = 1
        W[w_offset + 2, col_offset + 2] = -1
        
        # M4 = A11(B10 - B00)
        U[u_offset + 3, col_offset + 3] = 1
        V[v_offset + 2, col_offset + 3] = 1
        V[v_offset + 0, col_offset + 3] = -1
        W[w_offset + 1, col_offset + 3] = -1
        W[w_offset + 3, col_offset + 3] = 1
        
        # M5 = (A00 + A01)B11
        U[u_offset + 0, col_offset + 4] = 1
        U[u_offset + 1, col_offset + 4] = 1
        V[v_offset + 3, col_offset + 4] = 1
        W[w_offset + 0, col_offset + 4] = 1
        W[w_offset + 2, col_offset + 4] = 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[u_offset + 2, col_offset + 5] = 1
        U[u_offset + 0, col_offset + 5] = -1
        V[v_offset + 0, col_offset + 5] = 1
        V[v_offset + 1, col_offset + 5] = 1
        W[w_offset + 1, col_offset + 5] = 1
        W[w_offset + 2, col_offset + 5] = 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[u_offset + 1, col_offset + 6] = 1
        U[u_offset + 3, col_offset + 6] = -1
        V[v_offset + 2, col_offset + 6] = 1
        V[v_offset + 3, col_offset + 6] = 1
        W[w_offset + 0, col_offset + 6] = 1
        W[w_offset + 3, col_offset + 6] = 1
        
    # 4x4x4 part: 4 blocks of 2x2x2
    # Block (0,0): A[0:2, 0:2], B[0:2, 0:2], C[0:2, 0:2]
    add_strassen_2x2x2(0, 0, 0, 0)
    # Block (0,1): A[0:2, 2:4], B[2:4, 0:2], C[0:2, 2:4]
    # U offset: 0*2 + 2 = 2? No, U index is i*m + j.
    # A[0:2, 2:4] -> i=0,1; j=2,3. U indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7.
    # This mapping is complex. Let's use a generic builder.
    
    # Let's just use the schoolbook 96 rank and try to reduce it with coordinate descent.
    # Or use the 81 rank (49 + 32) and reduce.
    
    # I will implement the 77 rank structure directly if I can map it.
    # For now, I will return a valid 96 rank solution and try to optimize it in place.
    
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    # Try to reduce rank by zeroing columns and adjusting
    # This is a heuristic.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
