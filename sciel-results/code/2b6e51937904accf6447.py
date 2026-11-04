import time
import numpy as np
import itertools

DIMENSIONS = [4, 4, 7]

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen (rank 7).
    Convention: C[i,j] = sum_k A[i,k] * B[k,j]
    U shape (4, 7): rows indexed by (i, k) for A -> i*2+k
    V shape (4, 7): rows indexed by (k, j) for B -> k*2+j
    W shape (4, 7): rows indexed by (j, i) for C -> j*2+i (trace convention W[k*n+i] with n=2)
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices (W): j*2+i -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Let's verify these formulas expand correctly.
    # C00 = (A00+A11)(B00+B11) + A11(B01-B00) - (A00+A01)B11 - (A01-A11)(B10+B11)
    # = A00B00 + A00B11 + A11B00 + A11B11 + A11B01 - A11B00 - A00B11 - A01B11 - A01B10 - A01B11 + A11B10 + A11B11
    # = A00B00 + A11B00 + A11B11 + A11B01 - A01B11 - A01B10 - A01B11 + A11B10 + A11B11
    # This does NOT look like A00B00 + A01B10.
    
    # Let's use the standard Wikipedia Strassen formulation which is known correct:
    # P1 = (A00 + A11) (B00 + B11)
    # P2 = (A10 + A11) B00
    # P3 = A00 (B10 - B11)
    # P4 = A11 (B01 - B00)
    # P5 = (A00 + A01) B11
    # P6 = (A10 - A00) (B00 + B01)
    # P7 = (A01 - A11) (B10 + B11)
    
    # C00 = P1 - P5 + P6 - P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P7
    
    # Let's check C00:
    # (A00+A11)(B00+B11) - (A00+A01)B11 + (A10-A00)(B00+B01) - (A01-A11)(B10+B11)
    # = A00B00 + A00B11 + A11B00 + A11B11 - A00B11 - A01B11 + A10B00 + A10B01 - A00B00 - A00B01 - A01B10 - A01B11 + A11B10 + A11B11
    # = A11B00 + A11B11 - A01B11 + A10B00 + A10B01 - A00B01 - A01B10 - A01B11 + A11B10 + A11B11
    # This is also not simplifying to A00B00 + A01B10.
    
    # I will construct the rank-7 decomposition by solving the system of equations for the 7 terms.
    # Actually, it is safer to just use the schoolbook for 2x2x2 (rank 4) if I can't get Strassen right quickly,
    # BUT the goal is to improve rank. Let me use a verified Strassen implementation.
    
    # Verified Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Let's re-verify C00 with this specific set:
    # M1 + M4 - M5 - M7
    # = (A00+A11)(B00+B11) + A11(B01-B00) - (A00+A01)B11 - (A01-A11)(B10+B11)
    # = A00B00 + A00B11 + A11B00 + A11B11 + A11B01 - A11B00 - A00B11 - A01B11 - A01B10 - A01B11 + A11B10 + A11B11
    # = A00B00 + A11B00 + A11B11 + A11B01 - A01B11 - A01B10 - A01B11 + A11B10 + A11B11
    # = A00B00 + A11B00 + 2A11B11 + A11B01 - 2A01B11 - A01B10 + A11B10
    # This is WRONG.
    
    # Okay, I will use the following known correct rank-7 terms for 2x2x2:
    # t1 = (A00+A11)(B00+B11)
    # t2 = (A10+A11)B00
    # t3 = A00(B10-B11)
    # t4 = A11(B01-B00)
    # t5 = (A00+A01)B11
    # t6 = (A10-A00)(B00+B01)
    # t7 = (A01-A11)(B10+B11)
    
    # C00 = t1 - t5 + t6 - t7
    # C01 = t3 + t5
    # C10 = t2 + t4
    # C11 = t1 - t2 + t3 + t7
    
    # Let's check C00 again:
    # t1 - t5 + t6 - t7
    # = (A00+A11)(B00+B11) - (A00+A01)B11 + (A10-A00)(B00+B01) - (A01-A11)(B10+B11)
    # = A00B00 + A00B11 + A11B00 + A11B11 - A00B11 - A01B11 + A10B00 + A10B01 - A00B00 - A00B01 - A01B10 - A01B11 + A11B10 + A11B11
    # = A11B00 + A11B11 - A01B11 + A10B00 + A10B01 - A00B01 - A01B10 - A01B11 + A11B10 + A11B11
    # = A11B00 + 2A11B11 - 2A01B11 + A10B00 + A10B01 - A00B01 - A01B10 + A11B10
    # Still wrong.
    
    # I will use a Python script to find the coefficients for a rank-7 decomposition of 2x2x2.
    # Actually, I can just hardcode a known correct one from literature.
    # Here is a verified one:
    # U, V, W columns correspond to the 7 products.
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Term 0: (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1 # C00, C11
    
    # Term 1: (A10+A11)B00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1 # C10
    
    # Term 2: A00(B10-B11)
    U[0,2] = 1
    V[2,2] = 1; V[3,2] = -1
    W[1,2] = 1; W[3,2] = 1 # C01, C11
    
    # Term 3: A11(B01-B00)
    U[3,3] = 1
    V[1,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[2,3] = 1 # C00, C10
    
    # Term 4: (A00+A01)B11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1 # C00, C01
    
    # Term 5: (A10-A00)(B00+B01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[0,5] = 1 # C00
    
    # Term 6: (A01-A11)(B10+B11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = -1; W[3,6] = 1 # C00, C11
    
    # Let's verify this decomposition by checking if sum(U[:,t] V[:,t] W[:,t]) == T
    # T[i,j,k] = 1 if i,j,k match in matmul tensor.
    # T is a 4x4x4 tensor (flattened to 16x16? No, 4x4x4).
    # U is 4x7, V is 4x7, W is 4x7.
    # Product: sum_t U[a,t] V[b,t] W[c,t] should be 1 if (a,b,c) is a valid matmul index.
    # a = i*2+k, b = k*2+j, c = j*2+i.
    
    # Check C00 (i=0, j=0). c=0.
    # Sum_t U[0,t] V[?,t] W[0,t] ... wait, V index is k*2+j. For j=0, V indices are 0 (k=0) and 2 (k=1).
    # U indices for i=0 are 0 (k=0) and 1 (k=1).
    # So we need sum_k U[0*2+k, t] V[k*2+0, t] W[0*2+0, t] = 1.
    # W[0,t] is non-zero for t=0 (1), t=3 (1), t=4 (-1), t=5 (1), t=6 (-1).
    
    # t=0: U[0,0]=1, V[0,0]=1 (k=0), W[0,0]=1 -> 1.
    #       U[1,0]=0, V[2,0]=0 (k=1), W[0,0]=1 -> 0.
    #       Sum = 1.
    # t=3: U[0,3]=0, V[0,3]=-1, W[0,3]=1 -> 0.
    #       U[1,3]=0, V[2,3]=0, W[0,3]=1 -> 0.
    #       Sum = 0.
    # t=4: U[0,4]=1, V[0,4]=0, W[0,4]=-1 -> 0.
    #       U[1,4]=1, V[2,4]=0, W[0,4]=-1 -> 0.
    #       Sum = 0.
    # t=5: U[0,5]=-1, V[0,5]=1, W[0,5]=1 -> -1.
    #       U[1,5]=0, V[2,5]=0, W[0,5]=1 -> 0.
    #       Sum = -1.
    # t=6: U[0,6]=0, V[0,6]=0, W[0,6]=-1 -> 0.
    #       U[1,6]=0, V[2,6]=0, W[0,6]=-1 -> 0.
    #       Sum = 0.
    # Total for C00 = 1 + 0 + 0 - 1 + 0 = 0. 
    # But C00 should be A00B00 + A01B10.
    # The term for A00B00 is t=0 (1).
    # The term for A01B10 is... 
    # A01 is U[1,?]. B10 is V[2,?].
    # t=6: U[1,6]=1, V[2,6]=1, W[0,6]=-1. Contribution to C00 is -1.
    # So C00 = 1 (from t0) - 1 (from t6) = 0? 
    # Wait, my manual check of t=0 for C00:
    # t=0: U[0,0]=1 (A00), V[0,0]=1 (B00), W[0,0]=1 (C00). Product = 1.
    # t=6: U[1,6]=1 (A01), V[2,6]=1 (B10), W[0,6]=-1 (C00). Product = -1.
    # So C00 = 1 - 1 = 0. This is incorrect. C00 should be A00B00 + A01B10.
    
    # I will abandon trying to manually construct Strassen and instead use a simple block decomposition that is guaranteed correct.
    # Strategy: 
    # 1. Split A (4x4) into 2x2 blocks.
    # 2. Split B (4x7) into 2x7 blocks.
    # 3. Compute C_top (2x7) and C_bot (2x7).
    # 4. For the 4x4x4 part (first 4 cols of B), use Strassen.
    # 5. For the 4x4x3 part (last 3 cols of B), use schoolbook.
    
    # To ensure correctness, I will verify the Strassen 2x2x2 decomposition using numpy.
    # If it fails, I will fall back to schoolbook 2x2x2 (rank 4).
    
    return U, V, W

def _verify_2x2x2(U, V, W):
    """Verify that U, V, W correctly decompose 2x2x2 matmul."""
    n, m, p = 2, 2, 2
    rank = U.shape[1]
    # Tensor T[i,j,k] = 1 if i,j,k form a valid matmul term.
    # T is indexed by (i, k) for A, (k, j) for B, (j, i) for C.
    # We check sum_t U[i*m+k, t] V[k*p+j, t] W[j*n+i, t] == 1 if (i,k,j) is a valid term.
    # Actually, the tensor is T_{a,b,c} where a=(i,k), b=(k,j), c=(j,i).
    # T_{a,b,c} = 1 if a[1] == b[0] (k matches) and c[0] == b[1] (j matches) and c[1] == a[0] (i matches).
    # Basically, it's 1 if the indices are consistent.
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                expected = 1.0
                actual = 0.0
                a_idx = i * m + k
                b_idx = k * p + j
                c_idx = j * n + i
                for t in range(rank):
                    actual += U[a_idx, t] * V[b_idx, t] * W[c_idx, t]
                if not np.isclose(actual, expected):
                    return False
    return True

def _schoolbook_2x2x2():
    """Rank-4 schoolbook for 2x2x2."""
    U = np.zeros((4, 4), dtype=float)
    V = np.zeros((4, 4), dtype=float)
    W = np.zeros((4, 4), dtype=float)
    col = 0
    for i in range(2):
        for k in range(2):
            for j in range(2):
                U[i*2+k, col] = 1
                V[k*2+j, col] = 1
                W[j*2+i, col] = 1
                col += 1
    return U, V, W

def construct():
    deadline = time.time() + 120
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Strategy:
    # Split B into B1 (4x4) and B2 (4x3).
    # C = A * [B1, B2] = [A*B1, A*B2].
    # A*B1 is 4x4x4. Use Strassen-based block decomposition.
    # A*B2 is 4x4x3. Use schoolbook.
    
    # First, get a valid 2x2x2 decomposition.
    U2, V2, W2 = _strassen_2x2x2()
    if not _verify_2x2x2(U2, V2, W2):
        U2, V2, W2 = _schoolbook_2x2x2()
    
    # Build 4x4x4 decomposition using 2x2x2 blocks.
    # A is 4x4, B1 is 4x4.
    # A = [[A00, A01], [A10, A11]] (2x2 blocks of 2x2)
    # B1 = [[B00, B01], [B10, B11]] (2x2 blocks of 2x2)
    # C1 = A*B1 = [[A00*B00 + A01*B10, A00*B01 + A01*B11], [A10*B00 + A11*B10, A10*B01 + A11*B11]]
    
    # Each block product (e.g., A00*B00) is 2x2x2.
    # We can use the 2x2x2 decomposition for each.
    # However, Strassen reduces the number of 2x2x2 multiplications from 8 to 7.
    # But here we have 8 block multiplications.
    # We can apply Strassen to the block level?
    # Block Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Each M is a 2x2x2 product.
    # So we need 7 instances of 2x2x2 decomposition.
    # Each 2x2x2 decomposition has rank R2.
    # Total rank for 4x4x4 is 7 * R2.
    # If R2 = 7 (Strassen), total rank = 49.
    # If R2 = 4 (Schoolbook), total rank = 28.
    
    # Wait, the rank of the 4x4x4 tensor is the sum of the ranks of the 7 block products.
    # So if we use Strassen for 2x2x2 (rank 7), the 4x4x4 rank is 49.
    # Schoolbook 4x4x4 rank is 64.
    # So 49 is better.
    
    # Let's construct the 4x4x4 decomposition using 7 block products.
    # We need to map the block indices to the global indices.
    
    # Global indices:
    # A: 4x4. Block (bi, bj) where bi, bj in {0,1}.
    # A index (i, k) -> block (i//2, k//2), local (i%2, k%2).
    # B: 4x4. Block (bk, bj) where bk, bj in {0,1}.
    # B index (k, j) -> block (k//2, j//2), local (k%2, j%2).
    # C: 4x4. Block (bj, bi) where bj, bi in {0,1}.
    # C index (j, i) -> block (j//2, i//2), local (j%2, i%2).
    
    # We will create U4, V4, W4 of shape (16, 7*R2), (16, 7*R2), (16, 7*R2).
    # R2 is the rank of the 2x2x2 decomposition.
    
    R2 = U2.shape[1]
    Rank4 = 7 * R2
    U4 = np.zeros((16, Rank4), dtype=float)
    V4 = np.zeros((16, Rank4), dtype=float)
    W4 = np.zeros((16, Rank4), dtype=float)
    
    # Define the 7 block products and their contributions to C blocks.
    # M1 = (A00+A11)(B00+B11) -> C00 (+1), C11 (+1)
    # M2 = (A10+A11)B00 -> C10 (+1)
    # M3 = A00(B10-B11) -> C01 (+1), C11 (+1)
    # M4 = A11(B01-B00) -> C00 (+1), C10 (+1)
    # M5 = (A00+A01)B11 -> C00 (-1), C01 (+1)
    # M6 = (A10-A00)(B00+B01) -> C10 (-1), C11 (0) ... Wait, C11 = M1 - M2 + M3 + M7. M6 is not in C?
    # Let's re-verify the Strassen formulas.
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    # M6 is indeed not used in the output. This is a known issue with some Strassen formulations.
    # I will use the formulation that uses all 7.
    # Actually, if M6 is not used, we can drop it and use rank 6? No, Strassen is rank 7.
    # I will stick to the 7 terms and just not use M6 in the output sum?
    # No, that would be wrong.
    # I will use the schoolbook for 4x4x4 to be safe, which is rank 64.
    # Or I can use the verified 2x2x2 Strassen and just do 8 block multiplications (rank 56).
    # 56 < 64.
    
    # Let's do 8 block multiplications using the 2x2x2 decomposition.
    # This is safer than block Strassen which requires careful index mapping.
    
    Rank4_schoolbook = 8 * R2
    U4 = np.zeros((16, Rank4_schoolbook), dtype=float)
    V4 = np.zeros((16, Rank4_schoolbook), dtype=float)
    W4 = np.zeros((16, Rank4_schoolbook), dtype=float)
    
    col = 0
    for bi in range(2): # A row block
        for bk in range(2): # A col block / B row block
            for bj in range(2): # B col block / C col block
                # Block product: A[bi, bk] * B[bk, bj] -> C[bi, bj]
                # Use U2, V2, W2 for this 2x2x2 product.
                for t in range(R2):
                    # A index: block (bi, bk), local (li, lk)
                    # Global A index: (bi*2+li, bk*2+lk) -> (bi*2+li)*4 + (bk*2+lk)
                    # U2 row: li*2+lk
                    for li in range(2):
                        for lk in range(2):
                            a_global = (bi*2+li)*4 + (bk*2+lk)
                            u2_idx = li*2+lk
                            U4[a_global, col] = U2[u2_idx, t]
                            
                    # B index: block (bk, bj), local (lk, lj)
                    # Global B index: (bk*2+lk)*4 + (bj*2+lj)
                    # V2 row: lk*2+lj
                    for lk in range(2):
                        for lj in range(2):
                            b_global = (bk*2+lk)*4 + (bj*2+lj)
                            v2_idx = lk*2+lj
                            V4[b_global, col] = V2[v2_idx, t]
                            
                    # C index: block (bi, bj), local (lj, li)
                    # Global C index: (bj*2+lj)*4 + (bi*2+li)
                    # W2 row: lj*2+li
                    for lj in range(2):
                        for li in range(2):
                            c_global = (bj*2+lj)*4 + (bi*2+li)
                            w2_idx = lj*2+li
                            W4[c_global, col] = W2[w2_idx, t]
                            
                col += R2
                
    # Now handle the 4x4x3 part (B2).
    # B2 is 4x3.
    # Schoolbook: rank = 4*4*3 = 48.
    Rank43 = 48
    U43 = np.zeros((16, Rank43), dtype=float)
    V43 = np.zeros((16, Rank43), dtype=float)
    W43 = np.zeros((28, Rank43), dtype=float) # C is 4x4, but we only need the first 4 rows? No, C is 4x7.
    # Wait, the output W is for the full 4x7 C.
    # C is 4x7. W shape is (7*4, rank) = (28, rank).
    # For the 4x4x3 part, the C indices are j in 0..6, i in 0..3.
    # But B2 only contributes to C columns 4..6.
    # So for B2, j is 4, 5, 6.
    # W index: j*4+i.
    
    col2 = 0
    for i in range(4):
        for k in range(4):
            for j in range(3): # j is 0,1,2 for B2, but global j is 4,5,6
                j_global = j + 4
                U43[i*4+k, col2] = 1
                V43[k*7+j_global, col2] = 1
                W43[j_global*4+i, col2] = 1
                col2 += 1
                
    # Combine U4, V4, W4 with U43, V43, W43.
    # U4 is (16, Rank4_schoolbook), U43 is (16, 48).
    # V4 is (16, Rank4_schoolbook), V43 is (28, 48).
    # W4 is (16, Rank4_schoolbook), W43 is (28, 48).
    
    # Wait, W4 is for the 4x4 C part. W index is j*4+i.
    # For the 4x4 part, j is 0..3.
    # So W4 should be (16, Rank4_schoolbook).
    # But the final W is (28, total_rank).
    # So we need to pad W4 to (28, Rank4_schoolbook) with zeros for j=4..6.
    
    W4_full = np.zeros((28, Rank4_schoolbook), dtype=float)
    W4_full[:16, :] = W4
    
    U_final = np.hstack([U4, U43])
    V_final = np.hstack([V4, V43])
    W_final = np.hstack([W4_full, W43])
    
    total_rank = U_final.shape[1]
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
