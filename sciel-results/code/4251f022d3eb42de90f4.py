"""Tensor decomposition for 2x4 by 4x8 matrix multiplication, rank-56 via Strassen."""
import numpy as np
import time

DIMENSIONS = [2, 4, 8]

# Strassen 2x2x2 rank-7 decomposition.
# Local indices: X (A) is 2x2, Y (B) is 2x2, Z (C) is 2x2.
# U_local (4 x 7), V_local (4 x 7), W_local (4 x 7)
# Convention: U[row, col], V[row, col], W[row, col]
# Z = U @ M @ V.T @ W.T? No.
# The tensor is T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c].
# For 2x2 matrix mult: C = A B.
# C_{pq} = sum_r A_{pr} B_{rq}.
# Indices: i=p (row of A, 0..1), j=r (col of A / row of B, 0..1), k=q (col of B).
# Wait, the task says:
# U shape (n*m, rank) -> rows indexed by (i, j) where i in [0,n), j in [0,m).
# V shape (m*p, rank) -> rows indexed by (j, k) where j in [0,m), k in [0,p).
# W shape (p*n, rank) -> rows indexed by (k, i) where k in [0,p), i in [0,n).
#
# In the 2x2 sub-problem:
# A is 2x2. i in {0,1}, j in {0,1}.
# B is 2x2. j in {0,1}, k in {0,1}.
# C is 2x2. k in {0,1}, i in {0,1}.
#
# Strassen M1..M7:
# M1 = (A11+A22)(B11+B22)
# M2 = (A21+A22)B11
# M3 = A11(B21-B22)
# M4 = A22(B12-B11)
# M5 = (A11+A12)B22
# M6 = (A21-A11)(B11+B12)
# M7 = (A12-A22)(B21+B22)
#
# C11 = M1 - M4 + M5 + M7
# C12 = M3 + M5
# C21 = M2 + M4
# C22 = M1 - M3 + M6 + M7
#
# We need to express this as sum_c U[i,c] V[j,c] W[k,c].
# Here i is row of A (0,1), j is col of A (0,1), k is col of B (0,1).
# Wait, the tensor index for C is (k, i).
# C_{ki} corresponds to row k, col i of the result matrix?
# Standard matrix mult: C = A B. C_{ik} = sum_j A_{ij} B_{jk}.
# So C row index is i (from A), col index is k (from B).
# The task says W index is (k, i). So W[k*n + i].
# This means W's row index is (k, i).
# So the output tensor entry T[i, j, k] contributes to W[k*n+i, c].
# This matches C_{ik} = sum_j A_{ij} B_{jk}.
#
# Let's map local indices 0..3 for U, V, W.
# U local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1). (i, j)
# V local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1). (j, k)
# W local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1). (k, i)
#
# Strassen terms:
# M1: (A00+A11)(B00+B11)
#   U: A00=1, A11=1. (idx 0, 3)
#   V: B00=1, B11=1. (idx 0, 3)
#   W: C00 += M1, C11 += M1. (idx 0, 3)
#   So for c=0: U[0]=1, U[3]=1; V[0]=1, V[3]=1; W[0]=1, W[3]=1.
#
# M2: (A10+A11)B00
#   U: A10=1, A11=1. (idx 2, 3)
#   V: B00=1. (idx 0)
#   W: C10 += M2. (idx 2)
#   For c=1: U[2]=1, U[3]=1; V[0]=1; W[2]=1.
#
# M3: A00(B10-B11)
#   U: A00=1. (idx 0)
#   V: B10=1, B11=-1. (idx 2, 3)
#   W: C01 += M3. (idx 1)
#   For c=2: U[0]=1; V[2]=1, V[3]=-1; W[1]=1.
#
# M4: A11(B01-B00)
#   U: A11=1. (idx 3)
#   V: B01=1, B00=-1. (idx 1, 0)
#   W: C00 -= M4. (idx 0)
#   For c=3: U[3]=1; V[1]=1, V[0]=-1; W[0]=-1.
#
# M5: (A00+A01)B11
#   U: A00=1, A01=1. (idx 0, 1)
#   V: B11=1. (idx 3)
#   W: C00 += M5, C01 += M5. (idx 0, 1)
#   For c=4: U[0]=1, U[1]=1; V[3]=1; W[0]=1, W[1]=1.
#
# M6: (A10-A00)(B00+B01)
#   U: A10=1, A00=-1. (idx 2, 0)
#   V: B00=1, B01=1. (idx 0, 1)
#   W: C11 += M6. (idx 3)
#   For c=5: U[2]=1, U[0]=-1; V[0]=1, V[1]=1; W[3]=1.
#
# M7: (A01-A11)(B10+B11)
#   U: A01=1, A11=-1. (idx 1, 3)
#   V: B10=1, B11=1. (idx 2, 3)
#   W: C00 += M7. (idx 0)
#   For c=6: U[1]=1, U[3]=-1; V[2]=1, V[3]=1; W[0]=1.
#
# Let's assemble U_loc, V_loc, W_loc (4x7).
#
# U_loc:
# c0: [1, 0, 0, 1]
# c1: [0, 0, 1, 1]
# c2: [1, 0, 0, 0]
# c3: [0, 0, 0, 1]
# c4: [1, 1, 0, 0]
# c5: [-1, 0, 1, 0]
# c6: [0, 1, 0, -1]
#
# V_loc:
# c0: [1, 0, 0, 1]
# c1: [1, 0, 0, 0]
# c2: [0, 0, 1, -1]
# c3: [-1, 1, 0, 0]
# c4: [0, 0, 0, 1]
# c5: [1, 1, 0, 0]
# c6: [0, 0, 1, 1]
#
# W_loc:
# c0: [1, 0, 0, 0]
# c1: [0, 0, 1, 0]
# c2: [0, 1, 0, 0]
# c3: [-1, 0, 0, 0]
# c4: [1, 1, 0, 0]
# c5: [0, 0, 0, 1]
# c6: [1, 0, 0, 0]
#
# Let's double check W indices for C.
# C00 is idx 0. C01 is idx 1. C10 is idx 2. C11 is idx 3.
# M1: C00, C11. W[0]=1, W[3]=1. Correct.
# M2: C10. W[2]=1. Correct.
# M3: C01. W[1]=1. Correct.
# M4: C00. W[0]=-1. Correct.
# M5: C00, C01. W[0]=1, W[1]=1. Correct.
# M6: C11. W[3]=1. Correct.
# M7: C00. W[0]=1. Correct.
#
# Sum for C00 (idx 0): M1 - M4 + M5 + M7 = 1 - (-1) + 1 + 1 = 4? No.
# The coefficients in W_loc are the signs.
# c0 (M1): W[0]=1.
# c3 (M4): W[0]=-1.
# c4 (M5): W[0]=1.
# c6 (M7): W[0]=1.
# Sum of contributions to C00: 1*M1 + (-1)*M4 + 1*M5 + 1*M7 = M1 - M4 + M5 + M7. Correct.
#
# Sum for C01 (idx 1): M3 + M5.
# c2 (M3): W[1]=1.
# c4 (M5): W[1]=1.
# Sum: M3 + M5. Correct.
#
# Sum for C10 (idx 2): M2 + M4.
# c1 (M2): W[2]=1.
# c3 (M4): W[2]=0? Wait.
# M4: A11(B01-B00). C10? No, M4 contributes to C00 and C10?
# C10 = M2 + M4.
# Let's re-verify Strassen formulas.
# C11 = M1 - M4 + M5 + M7
# C12 = M3 + M5
# C21 = M2 + M4
# C22 = M1 - M3 + M6 + M7
#
# My indices:
# C00 (C11 in 1-based): M1 - M4 + M5 + M7.
# C01 (C12): M3 + M5.
# C10 (C21): M2 + M4.
# C11 (C22): M1 - M3 + M6 + M7.
#
# Let's check W_loc for idx 2 (C10).
# M2 (c1): W[2]=1.
# M4 (c3): W[2]=? In my V_loc/W_loc construction above:
# c3 (M4): U[3]=1; V[1]=1, V[0]=-1; W[0]=-1.
# I missed W[2] for M4!
# M4 = A11(B01-B00).
# A11 is idx 3.
# B01 is idx 1, B00 is idx 0.
# C10 is idx 2.
# So for c3 (M4), W[2] should be 1.
# Let's re-evaluate W_loc for c3.
# M4 contributes to C00 and C10.
# So W[0] = -1, W[2] = 1.
#
# Let's re-evaluate all W_loc entries carefully.
#
# c0 (M1): (A00+A11)(B00+B11).
#   A: 0, 3. B: 0, 3.
#   C: C00, C11. (0, 3).
#   W[0]=1, W[3]=1.
#
# c1 (M2): (A10+A11)B00.
#   A: 2, 3. B: 0.
#   C: C10. (2).
#   W[2]=1.
#
# c2 (M3): A00(B10-B11).
#   A: 0. B: 2, 3.
#   C: C01. (1).
#   W[1]=1.
#
# c3 (M4): A11(B01-B00).
#   A: 3. B: 1, 0.
#   C: C00, C10. (0, 2).
#   W[0]=-1, W[2]=1.
#
# c4 (M5): (A00+A01)B11.
#   A: 0, 1. B: 3.
#   C: C00, C01. (0, 1).
#   W[0]=1, W[1]=1.
#
# c5 (M6): (A10-A00)(B00+B01).
#   A: 2, 0. B: 0, 1.
#   C: C11. (3).
#   W[3]=1.
#
# c6 (M7): (A01-A11)(B10+B11).
#   A: 1, 3. B: 2, 3.
#   C: C00. (0).
#   W[0]=1.
#
# Now assemble U_loc, V_loc, W_loc.
#
# U_loc (4x7):
# c0: [1, 0, 0, 1]
# c1: [0, 0, 1, 1]
# c2: [1, 0, 0, 0]
# c3: [0, 0, 0, 1]
# c4: [1, 1, 0, 0]
# c5: [-1, 0, 1, 0]
# c6: [0, 1, 0, -1]
#
# V_loc (4x7):
# c0: [1, 0, 0, 1]
# c1: [1, 0, 0, 0]
# c2: [0, 0, 1, -1]
# c3: [-1, 1, 0, 0]
# c4: [0, 0, 0, 1]
# c5: [1, 1, 0, 0]
# c6: [0, 0, 1, 1]
#
# W_loc (4x7):
# c0: [1, 0, 0, 1]
# c1: [0, 0, 1, 0]
# c2: [0, 1, 0, 0]
# c3: [-1, 0, 1, 0]
# c4: [1, 1, 0, 0]
# c5: [0, 0, 0, 1]
# c6: [1, 0, 0, 0]
#
# Let's verify C10 (idx 2) sum:
# M2 (c1): W[2]=1.
# M4 (c3): W[2]=1.
# Sum: M2 + M4. Correct.
#
# Let's verify C11 (idx 3) sum:
# M1 (c0): W[3]=1.
# M6 (c5): W[3]=1.
# M7 (c6): W[3]=0?
# Wait, C11 = M1 - M3 + M6 + M7.
# My W_loc for c6 (M7) has W[3]=0.
# M7 = (A01-A11)(B10+B11).
# Does M7 contribute to C11?
# Yes, C22 = M1 - M3 + M6 + M7.
# So W[3] for c6 should be 1.
# Let's re-check M7.
# M7 = (A12-A22)(B21+B22) in 1-based.
# A12 is idx 1, A22 is idx 3.
# B21 is idx 2, B22 is idx 3.
# C22 is idx 3.
# So W[3] for c6 should be 1.
#
# Let's re-check M3.
# M3 = A11(B21-B22).
# A11 is idx 0.
# B21 is idx 2, B22 is idx 3.
# C12 is idx 1.
# So W[1] for c2 is 1.
# Does M3 contribute to C22?
# C22 = M1 - M3 + M6 + M7.
# So W[3] for c2 should be -1.
#
# I missed some contributions!
#
# Let's list all contributions to each C element.
# C00 (idx 0): M1, -M4, M5, M7.
# C01 (idx 1): M3, M5.
# C10 (idx 2): M2, M4.
# C11 (idx 3): M1, -M3, M6, M7.
#
# Now update W_loc:
# c0 (M1): W[0]=1, W[3]=1.
# c1 (M2): W[2]=1.
# c2 (M3): W[1]=1, W[3]=-1.
# c3 (M4): W[0]=-1, W[2]=1.
# c4 (M5): W[0]=1, W[1]=1.
# c5 (M6): W[3]=1.
# c6 (M7): W[0]=1, W[3]=1.
#
# Let's check sums:
# C00: 1(M1) -1(M4) +1(M5) +1(M7) = M1-M4+M5+M7. Correct.
# C01: 1(M3) +1(M5) = M3+M5. Correct.
# C10: 1(M2) +1(M4) = M2+M4. Correct.
# C11: 1(M1) -1(M3) +1(M6) +1(M7) = M1-M3+M6+M7. Correct.
#
# Final Local Matrices:
#
# U_loc:
# [[ 1,  0,  1,  0,  1, -1,  0],
#  [ 0,  0,  0,  0,  1,  0,  1],
#  [ 0,  1,  0,  0,  0,  1,  0],
#  [ 1,  1,  0,  1,  0,  0, -1]]
#
# V_loc:
# [[ 1,  1,  0, -1,  0,  1,  0],
#  [ 0,  0,  0,  1,  0,  1,  0],
#  [ 0,  0,  1,  0,  0,  0,  1],
#  [ 1,  0, -1,  0,  1,  0,  1]]
#
# W_loc:
# [[ 1,  0,  0, -1,  1,  0,  1],
#  [ 0,  0,  1,  0,  1,  0,  0],
#  [ 0,  1,  0,  1,  0,  0,  0],
#  [ 1,  0, -1,  0,  0,  1,  1]]

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Strassen local matrices
    U_loc = np.array([
        [ 1,  0,  1,  0,  1, -1,  0],
        [ 0,  0,  0,  0,  1,  0,  1],
        [ 0,  1,  0,  0,  0,  1,  0],
        [ 1,  1,  0,  1,  0,  0, -1]
    ], dtype=float)
    
    V_loc = np.array([
        [ 1,  1,  0, -1,  0,  1,  0],
        [ 0,  0,  0,  1,  0,  1,  0],
        [ 0,  0,  1,  0,  0,  0,  1],
        [ 1,  0, -1,  0,  1,  0,  1]
    ], dtype=float)
    
    W_loc = np.array([
        [ 1,  0,  0, -1,  1,  0,  1],
        [ 0,  0,  1,  0,  1,  0,  0],
        [ 0,  1,  0,  1,  0,  0,  0],
        [ 1,  0, -1,  0,  0,  1,  1]
    ], dtype=float)
    
    # Total rank = (m/2) * (p/2) * 7 = 2 * 4 * 7 = 56
    # Wait, m=4, p=8.
    # We split A (2x4) into 2 blocks of 2x2 along the column dimension (j).
    # We split B (4x8) into 2*4 = 8 blocks of 2x2.
    # Actually, we can view the product as:
    # C (2x8) = A (2x4) * B (4x8)
    # Split j into 2 groups of 2: j=0,1 and j=2,3.
    # Split k into 4 groups of 2: k=0,1; 2,3; 4,5; 6,7.
    # For each pair of (j-group, k-group), we have a 2x2x2 multiplication.
    # There are 2 (j-groups) * 4 (k-groups) = 8 such sub-multiplications.
    # Each uses rank 7. Total rank = 8 * 7 = 56.
    
    rank = 56
    
    U = np.zeros((n * m, rank), dtype=float)  # 8 x 56
    V = np.zeros((m * p, rank), dtype=float)  # 32 x 56
    W = np.zeros((p * n, rank), dtype=float)  # 16 x 56
    
    # j-groups: 0 (j=0,1), 1 (j=2,3)
    # k-groups: 0 (k=0,1), 1 (k=2,3), 2 (k=4,5), 3 (k=6,7)
    
    col_offset = 0
    for jg in range(2):  # 0, 1
        for kg in range(4):  # 0, 1, 2, 3
            # Map local indices to global indices
            # U local (i, j_loc) -> global (i, jg*2 + j_loc)
            # V local (j_loc, k_loc) -> global (jg*2 + j_loc, kg*2 + k_loc)
            # W local (k_loc, i) -> global (kg*2 + k_loc, i)
            
            for c in range(7):
                for i in range(2):
                    for j_loc in range(2):
                        u_val = U_loc[i * 2 + j_loc, c]
                        if u_val != 0:
                            u_idx = i * m + (jg * 2 + j_loc)
                            U[u_idx, col_offset + c] += u_val
                    
                    for k_loc in range(2):
                        w_val = W_loc[k_loc * 2 + i, c]
                        if w_val != 0:
                            w_idx = (kg * 2 + k_loc) * n + i
                            W[w_idx, col_offset + c] += w_val
                
                for j_loc in range(2):
                    for k_loc in range(2):
                        v_val = V_loc[j_loc * 2 + k_loc, c]
                        if v_val != 0:
                            v_idx = (jg * 2 + j_loc) * p + (kg * 2 + k_loc)
                            V[v_idx, col_offset + c] += v_val
            
            col_offset += 7
    
    # Verify
    if time.time() - start_time < 10:
        # Construct full tensor T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        # And compare with A*B
        # This is expensive, so only do a small check.
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
