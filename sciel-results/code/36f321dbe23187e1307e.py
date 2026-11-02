"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    # Use Strassen on 2x2x2 blocks to reduce rank.
    # The 2x3x8 multiplication can be decomposed into 4 blocks of 2x2x2 along the p dimension (k=0..7).
    # For each block k in {0,2,4,6} (start indices), we have a 2x2x2 sub-problem.
    # Strassen's algorithm for 2x2x2 has rank 7.
    # Total rank = 4 * 7 = 28.
    
    # Verify: is the 2x3x8 tensor separable into 4 disjoint 2x2x2 blocks?
    # The tensor T[i, j, k] for i in {0,1}, j in {0,1,2}, k in {0..7}.
    # A 2x2x2 block uses i in {0,1}, j in {j0, j0+1}, k in {k0, k0+1}.
    # We have j in {0,1,2}. We can partition j into pairs: (0,1) and (2,?) - but 2 is left over.
    # Wait, the middle dimension is 3. We cannot partition 3 into pairs of 2 cleanly without overlap or leftover.
    # So we can't just tile 2x2x2 blocks over j=0..2.
    
    # Let's re-evaluate.
    # Dimensions: n=2, m=3, p=8.
    # We can group the k-dimension into 4 pairs: (0,1), (2,3), (4,5), (6,7).
    # For each pair, we have a 2x3x2 sub-tensor.
    # We need to decompose 2x3x2.
    # 2x3x2 has rank? 
    # It is equivalent to 3 separate 2x1x2 multiplications? No.
    # C[i, k] = sum_j A[i,j] B[j,k].
    # For a fixed k-pair (k0, k0+1), we compute C[i, k0] and C[i, k0+1].
    # This is a 2x3 matrix times a 3x2 matrix.
    # Rank of 2x3x2 tensor?
    # Standard rank of 2x2x2 is 7. 
    # 2x3x2 can be viewed as two 2x3x1 tensors? No.
    # It is a 2x3x2 tensor. The rank is 7? Or less?
    # Actually, 2x3x2 is isomorphic to 3x2x2? No.
    # Let's check known ranks.
    # Rank(2,2,2) = 7.
    # Rank(2,3,2) = ?
    # We can decompose 2x3x2 by treating it as 2x2x2 + 2x1x2.
    # Rank(2,1,2) = 2.
    # So Rank(2,3,2) <= 7 + 2 = 9.
    # Is it 8?
    # Actually, 2x3x2 can be decomposed with rank 7? No, 2x2x2 is 7.
    # If we split j=0,1,2 into j=0,1 (2x2x2 part) and j=2 (2x1x2 part).
    # The j=2 part contributes to all k.
    # So we have 4 blocks of k.
    # For each k-block, we have a 2x3x2 tensor.
    # If Rank(2,3,2) = 9, then total rank = 4 * 9 = 36.
    # If Rank(2,3,2) = 8, total = 32.
    # If Rank(2,3,2) = 7, total = 28.
    
    # Let's try to construct a rank 7 decomposition for 2x3x2.
    # If not possible, we use rank 8 or 9.
    # Known result: Rank(2,3,2) = 8? Or 7?
    # Actually, I recall that Rank(2,3,2) is 7? No, 2x2x2 is 7.
    # Let's assume we can do better than schoolbook (48).
    # Schoolbook for 2x3x2 is 12.
    # 4 blocks * 12 = 48.
    
    # Let's try a rank 8 decomposition for 2x3x2.
    # Or maybe we can do a global decomposition.
    
    # Alternative: Use the known rank of 2x3x8 directly if available.
    # But we don't have it.
    
    # Let's stick to a safe improvement:
    # Decompose 2x3x8 into two 2x3x4 blocks?
    # Rank(2,3,4)?
    # 2x3x4 is isomorphic to 4x2x3?
    
    # Let's just implement the 2x3x2 block decomposition with rank 9 (7 for 2x2x2 + 2 for 2x1x2).
    # Total rank = 4 * 9 = 36.
    # This is an improvement over 48.
    
    # We need to construct the U, V, W matrices for 2x3x2 with rank 9.
    # Indices for 2x3x2:
    # U shape: (2*3, 9) = (6, 9)
    # V shape: (3*2, 9) = (6, 9)
    # W shape: (2*2, 9) = (4, 9)
    
    # Let's define the 2x2x2 Strassen part (j=0,1; k=0,1) and 2x1x2 part (j=2; k=0,1).
    
    # Strassen for 2x2x2 (A[2,2], B[2,2], C[2,2]):
    # U, V, W for 2x2x2 with rank 7.
    # U indices: i*m+j. m=2. i in 0,1; j in 0,1.
    # V indices: j*p+k. p=2. j in 0,1; k in 0,1.
    # W indices: k*n+i. n=2. k in 0,1; i in 0,1.
    
    # 2x1x2 part (A[2,1], B[1,2], C[2,2]):
    # U indices: i*m+j. Here m=1? No, in the 2x3x2 context, m=3.
    # The j=2 component is the third row of A.
    # U index for A[i, 2] is i*3 + 2.
    # V index for B[2, k] is 2*2 + k.
    # W index for C[i, k] is k*2 + i.
    
    # Let's build the 9-term decomposition for 2x3x2.
    # Terms 0-6: Strassen for j=0,1.
    # Terms 7-8: Schoolbook for j=2.
    
    # Strassen terms for 2x2x2 (local j=0,1, k=0,1):
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b11
    # M3 = a00(b00-b10)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b10
    # M6 = (a10+a11)(b00+b01)
    # M7 = (a00+a01-a10-a11)b11
    
    # Local to Global mapping for 2x3x2:
    # A[i, j] -> U[i*3 + j]
    # B[j, k] -> V[j*2 + k]
    # C[i, k] -> W[k*2 + i]
    
    # Strassen Local A:
    # a00 -> A[0,0] -> U[0]
    # a01 -> A[0,1] -> U[1]
    # a10 -> A[1,0] -> U[3]
    # a11 -> A[1,1] -> U[4]
    
    # Strassen Local B:
    # b00 -> B[0,0] -> V[0]
    # b01 -> B[0,1] -> V[1]
    # b10 -> B[1,0] -> V[2]
    # b11 -> B[1,1] -> V[3]
    
    # Strassen Local C:
    # c00 -> C[0,0] -> W[0]
    # c01 -> C[0,1] -> W[1]
    # c10 -> C[1,0] -> W[2]
    # c11 -> C[1,1] -> W[3]
    
    # Term 0: M1
    # U: a00, a11 -> U[0]=1, U[4]=1
    # V: b00, b11 -> V[0]=1, V[3]=1
    # W: c00, c11 -> W[0]=1, W[3]=1
    
    # Term 1: M2
    # U: a10, a11 -> U[3]=1, U[4]=1
    # V: b11 -> V[3]=1
    # W: c10, -c11 -> W[2]=1, W[3]=-1
    
    # Term 2: M3
    # U: a00 -> U[0]=1
    # V: b00, -b10 -> V[0]=1, V[2]=-1
    # W: c01, -c11 -> W[1]=1, W[3]=-1
    
    # Term 3: M4
    # U: a11 -> U[4]=1
    # V: b01, -b00 -> V[1]=1, V[0]=-1
    # W: c10, -c00 -> W[2]=1, W[0]=-1
    
    # Term 4: M5
    # U: a00, a01 -> U[0]=1, U[1]=1
    # V: b10 -> V[2]=1
    # W: -c00, c01 -> W[0]=-1, W[1]=1
    
    # Term 5: M6
    # U: a10, a11 -> U[3]=1, U[4]=1
    # V: b00, b01 -> V[0]=1, V[1]=1
    # W: -c11 -> W[3]=-1
    
    # Term 6: M7
    # U: a00, a01, -a10, -a11 -> U[0]=1, U[1]=1, U[3]=-1, U[4]=-1
    # V: b11 -> V[3]=1
    # W: c00 -> W[0]=1
    
    # Terms 7-8: j=2 part.
    # C[i, k] += A[i, 2] * B[2, k]
    # Term 7: k=0
    # U: A[0,2], A[1,2] -> U[2]=1, U[5]=1? No, each term is a rank-1 tensor.
    # The term is A[i,2] * B[2,0] * C[i,0].
    # U: A[i,2] is non-zero for i=0,1. But rank-1 means we can't sum over i.
    # Wait, the term is a single product.
    # For k=0:
    # We need to compute C[0,0] += A[0,2]B[2,0] and C[1,0] += A[1,2]B[2,0].
    # This is two separate rank-1 terms?
    # Term 7: A[0,2] * B[2,0] * C[0,0].
    # U: U[2]=1 (A[0,2])
    # V: V[4]=1 (B[2,0])
    # W: W[0]=1 (C[0,0])
    
    # Term 8: A[1,2] * B[2,0] * C[1,0].
    # U: U[5]=1 (A[1,2])
    # V: V[4]=1 (B[2,0])
    # W: W[2]=1 (C[1,0])
    
    # Wait, we also have k=1.
    # C[0,1] += A[0,2]B[2,1]
    # C[1,1] += A[1,2]B[2,1]
    # These require 2 more terms.
    # So j=2 part requires 4 terms (2x1x2 schoolbook rank is 4? No, 2*1*2=4).
    # So total rank = 7 + 4 = 11.
    # Total for 2x3x8 = 4 * 11 = 44.
    # Still better than 48.
    
    # Can we do better for 2x1x2?
    # 2x1x2 rank is 4.
    # So 11 is minimal for this split.
    
    # Let's try a different split.
    # Maybe we can use a rank 8 decomposition for 2x3x2 if it exists.
    # If Rank(2,3,2) = 8, total = 32.
    # I will check if I can find a rank 8 decomp.
    # If not, I will implement rank 11 per block -> total 44.
    
    # Let's implement rank 11 per block.
    
    U_block = np.zeros((6, 11))
    V_block = np.zeros((6, 11))
    W_block = np.zeros((4, 11))
    
    # Strassen terms (0-6)
    # T0
    U_block[0, 0] = 1; U_block[4, 0] = 1
    V_block[0, 0] = 1; V_block[3, 0] = 1
    W_block[0, 0] = 1; W_block[3, 0] = 1
    
    # T1
    U_block[3, 1] = 1; U_block[4, 1] = 1
    V_block[3, 1] = 1
    W_block[2, 1] = 1; W_block[3, 1] = -1
    
    # T2
    U_block[0, 2] = 1
    V_block[0, 2] = 1; V_block[2, 2] = -1
    W_block[1, 2] = 1; W_block[3, 2] = -1
    
    # T3
    U_block[4, 3] = 1
    V_block[1, 3] = 1; V_block[0, 3] = -1
    W_block[2, 3] = 1; W_block[0, 3] = -1
    
    # T4
    U_block[0, 4] = 1; U_block[1, 4] = 1
    V_block[2, 4] = 1
    W_block[0, 4] = -1; W_block[1, 4] = 1
    
    # T5
    U_block[3, 5] = 1; U_block[4, 5] = 1
    V_block[0, 5] = 1; V_block[1, 5] = 1
    W_block[3, 5] = -1
    
    # T6
    U_block[0, 6] = 1; U_block[1, 6] = 1; U_block[3, 6] = -1; U_block[4, 6] = -1
    V_block[3, 6] = 1
    W_block[0, 6] = 1
    
    # j=2 terms (7-10)
    # T7: A[0,2] B[2,0] C[0,0]
    U_block[2, 7] = 1
    V_block[4, 7] = 1
    W_block[0, 7] = 1
    
    # T8: A[1,2] B[2,0] C[1,0]
    U_block[5, 8] = 1
    V_block[4, 8] = 1
    W_block[2, 8] = 1
    
    # T9: A[0,2] B[2,1] C[0,1]
    U_block[2, 9] = 1
    V_block[5, 9] = 1
    W_block[1, 9] = 1
    
    # T10: A[1,2] B[2,1] C[1,1]
    U_block[5, 10] = 1
    V_block[5, 10] = 1
    W_block[3, 10] = 1
    
    # Now assemble 4 blocks for k=0..7.
    # Block b corresponds to k in {2b, 2b+1}.
    # U is independent of k, so we just repeat the U_block columns?
    # No, U is (n*m, rank). The rank is the number of terms.
    # Each block has 11 terms. Total rank = 44.
    # U[i*m+j, c] where c is the global term index.
    # For block b, term t (0..10), global index = b*11 + t.
    # U[i*m+j, b*11+t] = U_block[i*m+j, t].
    # V[j*p+k, b*11+t] = V_block[j*2+k, t] (with k local).
    # W[k*n+i, b*11+t] = W_block[k*2+i, t] (with k local).
    
    rank = 44
    U = np.zeros((6, rank))
    V = np.zeros((6, rank)) # m*p = 3*8 = 24? No, m=3, p=8. V is (3*8, 44) = (24, 44).
    # Wait, V shape is (m*p, rank) = (24, 44).
    # In my block construction, V_block is (6, 11) corresponding to j in {0,1,2} and k in {0,1} (local).
    # Global V index is j*p + k_global.
    # For block b, k_global = 2b + k_local.
    # So V[j*8 + 2b + k_local, b*11 + t] = V_block[j*2 + k_local, t].
    
    # W shape is (p*n, rank) = (16, 44).
    # W index is k_global*n + i.
    # W[(2b+k_local)*2 + i, b*11 + t] = W_block[k_local*2 + i, t].
    
    V = np.zeros((24, rank))
    W = np.zeros((16, rank))
    
    for b in range(4):
        for t in range(11):
            col = b * 11 + t
            # U
            U[:, col] = U_block[:, t]
            # V
            for j in range(3):
                for k_local in range(2):
                    row_v = j * 8 + (2 * b + k_local)
                    V[row_v, col] = V_block[j * 2 + k_local, t]
            # W
            for k_local in range(2):
                for i in range(2):
                    row_w = (2 * b + k_local) * 2 + i
                    W[row_w, col] = W_block[k_local * 2 + i, t]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
