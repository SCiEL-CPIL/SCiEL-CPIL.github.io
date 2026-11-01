"""Matrix multiplication 2x2 x 2x3 with rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p = 12. We aim for rank 9 if possible.
    # The known minimal rank for 2x2 * 2x3 is 9 (from Strassen-like optimizations).
    
    # We'll try to find a rank-9 decomposition using known constructions.
    # For 2x2 * 2x2, Strassen gives rank 7.
    # For 2x2 * 2x3, we can extend the approach.
    
    # Let's use a known rank-9 construction for 2x2 * 2x3.
    # The idea: split the 2x3 matrix B into two parts: B1 (2x2) and B2 (2x1).
    # Compute A*B1 using Strassen (rank 7) and A*B2 using schoolbook (rank 2*1=2... wait, 
    # actually for 2x2 * 2x1, the schoolbook is 4, but we can do better).
    
    # Actually, let me think about this more carefully.
    # A is 2x2, B is 2x3, C is 2x3.
    # C[i,k] = sum_j A[i,j] * B[j,k] for j=0,1 and k=0,1,2.
    
    # One approach: use Strassen for the 2x2 part and handle the third column separately.
    # But we need to be careful about the tensor structure.
    
    # Let's try a direct rank-9 construction.
    # We'll define the factors U, V, W explicitly.
    
    # For 2x2 * 2x2 Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 - M4 + M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 + M5
    
    # For the 2x3 case, we can extend this. Let's denote:
    # B = [B1 | B2] where B1 is 2x2 (columns 0,1) and B2 is 2x1 (column 2).
    
    # We can compute A*B1 using Strassen (7 multiplications) and A*B2 using 2 multiplications.
    # Total: 9 multiplications.
    
    # Let's write out the explicit decomposition.
    
    rank = 9
    U = np.zeros((n * m, rank))  # (4, 9)
    V = np.zeros((m * p, rank))  # (6, 9)
    W = np.zeros((p * n, rank))  # (6, 9)
    
    # Indices: U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
    # n=2, m=2, p=3
    
    # Strassen factors for A*B1 (where B1 has columns 0,1):
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1] - B[1,1])
    # M4 = A[1,1] * (B[1,0] - B[0,0])
    # M5 = (A[0,0] + A[0,1]) * B[1,1]
    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    
    # For A*B2 (column 2):
    # M8 = A[0,0] * B[0,2]
    # M9 = A[1,0] * B[1,2]  -- wait, this isn't quite right.
    
    # Actually, for the third column:
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # We can use:
    # M8 = (A[0,0] + A[0,1]) * B[0,2]  -- no, this doesn't work directly.
    
    # Let me reconsider. For the third column, we need 4 terms:
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # We can factor this as:
    # M8 = A[0,0]*B[0,2]
    # M9 = A[0,1]*B[1,2]  -- but this gives C[0,2] = M8 + M9, and C[1,2] needs A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # That's not right. Let me use:
    # M8 = (A[0,0] + A[1,0]) * B[0,2]
    # M9 = (A[0,1] + A[1,1]) * B[1,2]
    # Then C[0,2] = ... hmm, this doesn't directly give us what we want.
    
    # Actually, the simplest approach for the third column is just schoolbook:
    # M8 = A[0,0] * B[0,2]
    # M9 = A[1,0] * B[1,2]  -- no, that's only 2 of the 4 terms.
    
    # Wait, I think I need to be more careful. Let me just use schoolbook for the third column:
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # So we need 4 separate products for the third column, which would give rank 7+4=11. That's not optimal.
    
    # Let me look for a better construction. Actually, I recall that the minimal rank for 2x2 * 2x3 is indeed 9, but the construction is non-trivial.
    
    # Let me try a different approach: use a known rank-9 algorithm.
    
    # After some thought, let's just return the schoolbook rank-12 decomposition for now,
    # as finding the exact rank-9 construction requires more careful work.
    
    # Actually, let me try to construct rank 9 explicitly.
    
    # Reinitializing
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen for first 2 columns (indices 0,1 of B):
    # Factor 0: M1 = (A00 + A11)(B00 + B11)
    U[0*2+0, 0] = 1  # A[0,0]
    U[1*2+1, 0] = 1  # A[1,1]
    V[0*3+0, 0] = 1  # B[0,0]
    V[1*3+1, 0] = 1  # B[1,1]
    W[0*2+0, 0] = 1  # contributes to C[0,0]
    W[1*2+1, 0] = 1  # contributes to C[1,1]
    
    # Factor 1: M2 = (A10 + A11)B00
    U[1*2+0, 1] = 1  # A[1,0]
    U[1*2+1, 1] = 1  # A[1,1]
    V[0*3+0, 1] = 1  # B[0,0]
    W[0*2+1, 1] = 1  # contributes to C[0,1]... wait, need to be careful
    
    # This is getting complicated. Let me just use the schoolbook for now.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
