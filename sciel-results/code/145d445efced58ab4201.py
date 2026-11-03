import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 16
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Index mapping:
    # A: i in 0..1, j in 0..2 -> idx = i*3 + j
    # B: j in 0..2, k in 0..2 -> idx = j*3 + k
    # C: k in 0..2, i in 0..1 -> idx = k*2 + i (Trace convention W[k*n+i, r])
    
    # Part 1: Strassen for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # Covers (i,j,k) for i in {0,1}, j in {0,1}, k in {0,1}
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10, -C11
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3: A00(B01-B11) -> C01, C11
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4: A11(B10-B00) -> C00, C10
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5: (A00+A01)B11 -> -C00, C01
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C11
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Part 2: j=2, k in {0,1} (4 terms)
    # r=7: A02*B20 -> C00
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    # r=8: A02*B21 -> C01
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    # r=9: A12*B20 -> C10
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    # r=10: A12*B21 -> C11
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    # Part 3: k=2 terms (5 terms with sharing)
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C12 = A10*B02 + A11*B12 + A12*B22
    
    # r=11: (A00+A10)*B02 -> C02, C12
    U[0, 11] = 1; U[3, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1; W[5, 11] = 1
    
    # r=12: (A01+A11)*B12 -> C02, C12
    U[1, 12] = 1; U[4, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1; W[5, 12] = 1
    
    # r=13: (A02+A12)*B22 -> C02, C12
    U[2, 13] = 1; U[5, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1; W[5, 13] = 1
    
    # That's only 13 terms... I need 3 more for the cross terms.
    # Wait, let me reconsider. The above only covers:
    # (0,0,2),(1,0,2) from r=11
    # (0,1,2),(1,1,2) from r=12
    # (0,2,2),(1,2,2) from r=13
    # That's all 6 k=2 terms! So total is 7+4+3 = 14?
    # No, that can't be right. Let me verify by checking the full tensor.
    
    # Actually, I realize the issue: in a valid tensor decomposition,
    # each rank-1 term contributes to specific (i,j,k) entries.
    # The Strassen terms M1-M7 each contribute to multiple (i,j,k) in the 2x2x2 block.
    # But they ONLY involve A[0:2,0:2] and B[0:2,0:2] entries.
    # The j=2 terms involve A[*,2] and B[2,*].
    # The k=2 terms involve B[*,2].
    # These are disjoint in terms of which A and B entries they use,
    # EXCEPT that Strassen uses A00,A01,A10,A11 and B00,B01,B10,B11,
    # while k=2 terms use B02,B12,B22.
    # And j=2 terms use A02,A12 and B20,B21.
    # So there's no overlap in the A,B entries between the three parts!
    # This means the decomposition is valid and rank is 7+4+3=14?
    
    # That seems too low. Let me double-check by computing the tensor.
    
    # Actually, I think I made an error. Let me recount the Strassen terms.
    # Standard Strassen for 2x2x2 uses 7 products to compute 8 outputs.
    # Each product contributes to 2 output entries.
    # So 7 products cover 8 (i,j,k) triples where i,j,k in {0,1}.
    
    # The 8 triples are: (0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)
    # Strassen covers all 8 with 7 products. ✓
    
    # j=2, k in {0,1}: (0,2,0),(0,2,1),(1,2,0),(1,2,1) = 4 triples, 4 products. ✓
    # k=2: (0,0,2),(0,1,2),(0,2,2),(1,0,2),(1,1,2),(1,2,2) = 6 triples, 3 products. ✓
    
    # Total: 8+4+6 = 18 triples covered by 7+4+3 = 14 products.
    # This would mean rank ≤ 14, but the known rank of 2x3x3 is 16.
    # So something is wrong.
    
    # The issue: in a tensor decomposition, the rank-1 terms must be linearly
    # independent as tensors. My "sharing" in the k=2 part might not work
    # because the resulting tensor terms might not be linearly independent
    # from the Strassen terms.
    
    # Let me just use the safe rank-17 and verify it works.
    
    # Reset and use rank 17
    rank = 17
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: Strassen (7 terms)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Part 2: j=2, k in {0,1} (4 terms)
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    # Part 3: k=2 (6 terms, no sharing)
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
