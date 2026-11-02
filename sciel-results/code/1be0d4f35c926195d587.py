"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # Rank-45 decomposition for 3x3x6 matrix multiplication.
    # Strategy: treat as three parallel 3x3x2 multiplications.
    # Each 3x3x2 uses the optimal rank-15 decomposition.
    #
    # For a 3x3x2 multiplication (A is 3x3, B is 3x2, C is 3x2):
    # We use the known optimal rank-15 decomposition.
    #
    # The 3x3x2 tensor can be decomposed using the fact that it's equivalent
    # to three 1x3x2 multiplications (one per row of A), but more efficiently
    # we can use structure.
    #
    # Actually, let me think about this more carefully.
    #
    # For 3x3x2: A (3x3) * B (3x2) = C (3x2)
    # C[i][k] = sum_j A[i][j] * B[j][k]
    #
    # The schoolbook rank is 3*3*2 = 18.
    #
    # Known optimal rank for 3x3x2: Let me recall.
    # For 2x2x2, optimal rank is 7 (Strassen).
    # For 3x3x2, I need to look up or derive.
    #
    # Actually, let me use a different approach. The 3x3x6 problem can be seen
    # as computing C = A*B where A is 3x3 and B is 3x6.
    # C[i][k] = sum_j A[i][j] * B[j][k]
    #
    # We can partition B into three 3x2 blocks: B = [B1, B2, B3] where each Bt is 3x2.
    # Then C = [A*B1, A*B2, A*B3] = [C1, C2, C3] where Ct = A*Bt.
    #
    # So we need rank for three independent 3x3x2 multiplications.
    # If the optimal rank for 3x3x2 is R, then total rank is 3*R.
    #
    # What is the optimal rank for 3x3x2?
    # The tensor T_{i,j,k} where i in {0,1,2}, j in {0,1,2}, k in {0,1}.
    # This is the matrix multiplication tensor for 3x3 * 3x2 -> 3x2.
    #
    # I recall that the rank of the 3x3x2 multiplication tensor is 15.
    # This can be achieved by embedding the 2x2x2 Strassen decomposition.
    #
    # Let me construct the rank-15 decomposition for 3x3x2.
    #
    # One way: Use the fact that 3x3x2 can be built from 2x2x2 blocks.
    # Actually, let me just use a known construction.
    #
    # For 3x3x2, one known rank-15 decomposition:
    # We can think of it as: for each of the 3 output rows, we compute a 1x3x2 product.
    # But that's not efficient.
    #
    # Alternative: The 3x3x2 tensor has rank 15. Here's a construction:
    # Consider the 2x2x2 Strassen decomposition (rank 7).
    # We can embed this into 3x3x2 by adding extra terms.
    #
    # Actually, let me just use a direct construction.
    #
    # For 3x3x2: C[i][k] = sum_j A[i][j]*B[j][k], i=0,1,2; j=0,1,2; k=0,1
    #
    # Let me define 15 rank-1 terms.
    #
    # Here's a known rank-15 decomposition for 3x3x2:
    # Using the approach from the literature, we can get rank 15.
    #
    # Let me try a different approach: use the fact that we can write
    # the 3x3x2 tensor as a combination of simpler tensors.
    #
    # Actually, I'll use a constructive approach. The key insight is:
    # 3x3x2 can be decomposed by noting that it's like computing 3 independent
    # 1x3x2 products, but with shared structure in the first index.
    #
    # Let me just hard-code a known rank-15 decomposition for 3x3x2.
    #
    # For 3x3x2, the rank-15 decomposition can be constructed as follows:
    # We use 15 terms indexed by (a, b, c) where the structure comes from
    # combining Strassen-like ideas.
    #
    # Actually, let me use a simpler known result:
    # The rank of [3,3,2] matrix multiplication is 15.
    #
    # Here's a construction using the "embedding" method:
    # Start with 2x2x2 Strassen (rank 7) and extend to 3x3x2.
    #
    # Let me define the 3x3x2 decomposition directly.
    #
    # For i in {0,1,2}, j in {0,1,2}, k in {0,1}:
    # T[i,j,k] = 1
    #
    # Rank-15 decomposition:
    # I'll use the following 15 rank-1 terms:
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Build rank-15 decomposition for 3x3x2
    # Then use it three times for the three 3x2 blocks of B
    
    # For 3x3x2: A (3x3), B (3x2), C (3x2)
    # U has shape (3*3, 15) = (9, 15)
    # V has shape (3*2, 15) = (6, 15)  
    # W has shape (2*3, 15) = (6, 15)
    
    # Let me construct the rank-15 decomposition for 3x3x2.
    # 
    # Using the known optimal decomposition:
    # The 3x3x2 tensor can be decomposed with rank 15.
    #
    # Here's one such decomposition. I'll define it explicitly.
    
    # For 3x3x2 multiplication:
    # A is 3x3, B is 3x2, C = A*B is 3x2
    #
    # I'll use the following approach:
    # Decompose into:
    # - 7 terms from a 2x2x2 Strassen on rows 0,1 and cols 0,1 of A, all of B
    #   Wait, that doesn't quite work directly.
    #
    # Let me use a different known construction.
    #
    # Actually, here's a clean way: The 3x3x2 tensor has the same structure as
    # three copies of 1x3x2 with shared B indices.
    #
    # Let me just use the fact that rank([3,3,2]) = 15 and construct it.
    #
    # One known rank-15 decomposition for 3x3x2:
    #
    # Term 1: U has 1 at (0,0), V has 1 at (0,0), W has 1 at (0,0)  -> A[0,0]*B[0,0] -> C[0,0]
    # ... this is getting complicated. Let me use a systematic approach.
    
    # I'll use the following known rank-15 decomposition for 3x3x2:
    # Based on the work of Landsberg and others, the rank is 15.
    
    # Let me construct it using the "border rank" / "rank" decomposition.
    #
    # Here's a concrete rank-15 decomposition for 3x3x2:
    
    # Define U3, V3, W3 for 3x3x2 (shapes 9x15, 6x15, 6x15)
    U3 = np.zeros((9, 15))
    V3 = np.zeros((6, 15))
    W3 = np.zeros((6, 15))
    
    # I'll use a known construction. Let me think about this differently.
    #
    # For 3x3x2, we can use the following 15 rank-1 terms:
    #
    # The idea: partition the 3x3 matrix A into a 2x2 block plus extra row/col.
    # 
    # Actually, let me just use a verified construction.
    #
    # Here's a rank-15 decomposition for 3x3x2:
    #
    # Terms 1-7: Strassen-like on the top-left 2x2 of A, with both columns of B
    #   This gives us the contributions to C[0,0], C[0,1], C[1,0], C[1,1]
    #   from A[0,0], A[0,1], A[1,0], A[1,1] times B[0,*], B[1,*]
    #
    # But we also need contributions from A[2,*] and B[2,*].
    #
    # This is getting complex. Let me use a simpler but valid approach.
    #
    # Alternative: Use rank 18 (schoolbook) for each 3x3x2, giving total rank 54.
    # But that's not an improvement.
    #
    # Let me try to get rank 15 for 3x3x2 properly.
    #
    # Known: rank([3,3,2]) = 15 (this is a well-known result).
    #
    # Here's a concrete decomposition. I'll define it term by term.
    #
    # For 3x3x2: indices i (0-2), j (0-2), k (0-1)
    # U index: i*m+j = i*3+j (0-8)
    # V index: j*p+k = j*2+k (0-5)
    # W index: k*n+i = k*3+i (0-5)
    
    # Rank-15 decomposition for 3x3x2:
    #
    # I'll use the following construction based on the 2x2x2 Strassen decomposition
    # extended to handle the third row and column.
    
    # Strassen for 2x2x2 (A is 2x2, B is 2x2, C is 2x2):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4 + M7
    # C10 = M2 + M6 - M1 + M5
    # C11 = M1 - M3 + M6 - M2
    #
    # For 3x3x2, we need to handle the third row/col of A and the fact that B is 3x2.
    #
    # Let me use a different approach. I'll construct the rank-15 decomposition
    # by combining:
    # 1. A 2x2x2 Strassen on A[0:2,0:2] * B[0:2,:] -> contributes to C[0:2,0:2]
    #    But B is 3x2, so B[0:2,:] is 2x2. This gives rank 7.
    # 2. Handle the remaining terms: A[2,:]*B[:,:] and A[:2,2]*B[2,:]
    #    A[2,:]*B[:,:] gives C[2,:] = A[2,0]*B[0,:] + A[2,1]*B[1,:] + A[2,2]*B[2,:]
    #    A[:2,2]*B[2,:] adds to C[0:2,:]
    #
    # This is still complex. Let me just use a verified rank-15 construction.
    
    # After research, here's a rank-15 decomposition for 3x3x2:
    #
    # I'll define it explicitly. The 15 terms are:
    
    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    U3[0, 0] = 1; V3[0, 0] = 1; W3[0, 0] = 1
    # Term 1: A[0,0]*B[0,1] -> C[0,1]  
    U3[0, 1] = 1; V3[1, 1] = 1; W3[3, 1] = 1
    # Term 2: A[0,1]*B[0,0] -> C[0,0]
    U3[1, 2] = 1; V3[0, 2] = 1; W3[0, 2] = 1
    # Term 3: A[0,1]*B[0,1] -> C[0,1]
    U3[1, 3] = 1; V3[1, 3] = 1; W3[3, 3] = 1
    # Term 4: A[1,0]*B[1,0] -> C[1,0]
    U3[3, 4] = 1; V3[2, 4] = 1; W3[1, 4] = 1
    # Term 5: A[1,0]*B[1,1] -> C[1,1]
    U3[3, 5] = 1; V3[3, 5] = 1; W3[4, 5] = 1
    # Term 6: A[1,1]*B[1,0] -> C[1,0]
    U3[4, 6] = 1; V3[2, 6] = 1; W3[1, 6] = 1
    # Term 7: A[1,1]*B[1,1] -> C[1,1]
    U3[4, 7] = 1; V3[3, 7] = 1; W3[4, 7] = 1
    # Term 8: A[0,2]*B[2,0] -> C[0,0]
    U3[2, 8] = 1; V3[4, 8] = 1; W3[0, 8] = 1
    # Term 9: A[0,2]*B[2,1] -> C[0,1]
    U3[2, 9] = 1; V3[5, 9] = 1; W3[3, 9] = 1
    # Term 10: A[1,2]*B[2,0] -> C[1,0]
    U3[5, 10] = 1; V3[4, 10] = 1; W3[1, 10] = 1
    # Term 11: A[1,2]*B[2,1] -> C[1,1]
    U3[5, 11] = 1; V3[5, 11] = 1; W3[4, 11] = 1
    # Term 12: A[2,0]*B[0,0] -> C[2,0]
    U3[6, 12] = 1; V3[0, 12] = 1; W3[2, 12] = 1
    # Term 13: A[2,0]*B[0,1] -> C[2,1]
    U3[6, 13] = 1; V3[1, 13] = 1; W3[5, 13] = 1
    # Term 14: A[2,1]*B[1,0] -> C[2,0]
    U3[7, 14] = 1; V3[2, 14] = 1; W3[2, 14] = 1
    
    # Wait, this is only 15 terms but I haven't covered A[2,2]*B[2,0] and A[2,2]*B[2,1].
    # This is just the schoolbook for a subset, not a proper rank-15 decomposition.
    #
    # Let me restart with a proper construction.
    
    # The correct rank-15 decomposition for 3x3x2 uses Strassen-like tricks.
    # Let me use the known optimal decomposition.
    
    # After careful consideration, here's a valid rank-15 decomposition for 3x3x2:
    #
    # I'll use the following 15 rank-1 terms based on the structure:
    
    U3 = np.zeros((9, 15))
    V3 = np.zeros((6, 15))
    W3 = np.zeros((6, 15))
    
    # Use Strassen for the 2x2x2 part (rows 0,1 of A; cols 0,1 of B)
    # M1 = (A00+A11)(B00+B11)
    U3[0, 0] = 1; U3[4, 0] = 1  # A[0,0] + A[1,1]
    V3[0, 0] = 1; V3[2, 0] = 1  # B[0,0] + B[1,0]
    W3[0, 0] = 1; W3[1, 0] = 1  # contributes to C[0,0] and C[1,0]... 
    # This isn't quite right for the 3x3x2 case.
    
    # Let me use a different, verified approach.
    #
    # I'll use the fact that rank([3,3,2]) = 15 and construct it using
    # the "Landsberg" construction or similar.
    
    # Actually, let me just use the schoolbook rank 18 for each 3x3x2,
    # giving total rank 54, which is the same as the original.
    # But the task asks for improvement.
    
    # Let me try a different strategy: use rank 15 for 3x3x2.
    #
    # Here's a proper rank-15 decomposition for 3x3x2:
    
    U3 = np.zeros((9, 15))
    V3 = np.zeros((6, 15))
    W3 = np.zeros((6, 15))
    
    # Terms 0-6: Strassen on A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # But B is 3x2, so we need to handle B[2,:] separately.
    
    # Let me use a cleaner construction. I'll define the 15 terms:
    
    # For 3x3x2, here's a rank-15 decomposition:
    #
    # t=0: (A00+A11)*(B00+B11) -> C00 += 1, C10 += 1, C01 += 1, C11 += 1
    # This doesn't work directly for 3x3x2.
    
    # I'll use a known valid construction from the literature.
    # The rank-15 decomposition for [3,3,2]:
    
    # After much deliberation, let me use a constructive approach:
    # 
    # The 3x3x2 tensor can be written as:
    # sum over 15 terms
    #
    # Here's one such decomposition:
    
    U3[0,0]=1; V3[0,0]=1; W3[0,0]=1   # A00*B00 -> C00
    U3[0,1]=1; V3[1,1]=1; W3[3,1]=1   # A00*B01 -> C01
    U3[1,2]=1; V3[0,2]=1; W3[0,2]=1   # A01*B00 -> C00
    U3[1,3]=1; V3[1,3]=1; W3[3,3]=1   # A01*B01 -> C01
    U3[3,4]=1; V3[2,4]=1; W3[1,4]=1   # A10*B10 -> C10
    U3[3,5]=1; V3[3,5]=1; W3[4,5]=1   # A10*B11 -> C11
    U3[4,6]=1; V3[2,6]=1; W3[1,6]=1   # A11*B10 -> C10
    U3[4,7]=1; V3[3,7]=1; W3[4,7]=1   # A11*B11 -> C11
    U3[2,8]=1; V3[4,8]=1; W3[0,8]=1   # A02*B20 -> C00
    U3[2,9]=1; V3[5,9]=1; W3[3,9]=1   # A02*B21 -> C01
    U3[5,10]=1; V3[4,10]=1; W3[1,10]=1 # A12*B20 -> C10
    U3[5,11]=1; V3[5,11]=1; W3[4,11]=1 # A12*B21 -> C11
    U3[6,12]=1; V3[0,12]=1; W3[2,12]=1 # A20*B00 -> C20
    U3[6,13]=1; V3[1,13]=1; W3[5,13]=1 # A20*B01 -> C21
    U3[7,14]=1; V3[2,14]=1; W3[2,14]=1 # A21*B10 -> C20
    
    # Missing: A21*B11 -> C21, A22*B20 -> C20, A22*B21 -> C21
    # That's 3 more terms needed, giving 18 total (schoolbook).
    
    # So this isn't rank 15. Let me use Strassen tricks to reduce.
    
    # For the 2x2x2 subproblem (A[0:2,0:2] * B[0:2,:]), I can use Strassen (rank 7).
    # Then handle the remaining terms.
    
    # Let me construct it properly:
    
    U3 = np.zeros((9, 15))
    V3 = np.zeros((6, 15))
    W3 = np.zeros((6, 15))
    
    # Strassen for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] (7 terms)
    # M1 = (A00+A11)(B00+B11)
    U3[0,0]=1; U3[4,0]=1
    V3[0,0]=1; V3[2,0]=1
    W3[0,0]=1; W3[1,0]=1; W3[3,0]=1; W3[4,0]=1
    # M2 = (A10+A11)B00
    U3[3,1]=1; U3[4,1]=1
    V3[0,1]=1
    W3[1,1]=1
    # M3 = A00(B10-B11)
    U3[0,2]=1
    V3[2,2]=1; V3[3,2]=-1
    W3[0,2]=1
    # M4 = A11(B01-B00)
    U3[4,3]=1
    V3[1,3]=1; V3[0,3]=-1
    W3[1,3]=1
    # M5 = (A00+A01)B11
    U3[0,4]=1; U3[1,4]=1
    V3[3,4]=1
    W3[0,4]=1; W3[1,4]=1
    # M6 = (A10-A00)(B00+B01)
    U3[3,5]=1; U3[0,5]=-1
    V3[0,5]=1; V3[1,5]=1
    W3[1,5]=1
    # M7 = (A01-A11)(B10+B11)
    U3[1,6]=1; U3[4,6]=-1
    V3[2,6]=1; V3[3,6]=1
    W3[0,6]=1; W3[1,6]=1
    
    # Now handle remaining terms:
    # A02*B20 -> C00, A02*B21 -> C01
    # A12*B20 -> C10, A12*B21 -> C11
    # A20*B00 -> C20, A20*B01 -> C21
    # A21*B10 -> C20, A21*B11 -> C21
    # A22*B20 -> C20, A22*B21 -> C21
    
    # That's 10 more terms, total 17. Still not 15.
    
    # I need to use Strassen-like tricks for the remaining terms too.
    
    # This is getting very complex. Let me just use the schoolbook rank 54
    # which is known to be valid, and accept that I can't easily find the
    # rank-45 decomposition in this context.
    
    # Actually, let me try a different approach: use rank 18 for each 3x3x2
    # (schoolbook), giving total rank 54. This is the same as the original.
    
    # Since I can't easily construct the rank-45 decomposition, let me
    # just return the schoolbook decomposition which is known to be valid.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
