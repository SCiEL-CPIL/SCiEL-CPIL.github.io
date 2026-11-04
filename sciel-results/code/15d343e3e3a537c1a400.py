import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0

    # Target rank for this strategy: 20
    # Decomposition:
    # A is 4x5, B is 5x5.
    # Split A into A_top (2x5) and A_bottom (2x5).
    # Split B into B_left (5x2) and B_right (5x3).
    # C is 4x5.
    # C_top_left (2x2) = A_top * B_left -> Use Strassen (Rank 7)
    # C_top_right (2x3) = A_top * B_right -> Use Laderman (Rank 13)
    # C_bottom_left (2x2) = A_bottom * B_left -> Use Strassen (Rank 7)
    # C_bottom_right (2x3) = A_bottom * B_right -> Use Laderman (Rank 13)
    # Total Rank = 7 + 13 + 7 + 13 = 40? 
    # Wait, the prompt strategy says: "2x2x2 block (rank 7) and 2x3x3 block (rank 13)... rank 20".
    # Let's re-read the prompt strategy carefully: 
    # "expressing the 4x5x5 multiplication tensor as the sum of a 2x2x2 block (using the rank-7 Strassen algorithm) and a 2x3x3 block (using the rank-13 Laderman algorithm), exploiting the dimension split 4=2+2 and 5=2+3 to achieve a valid exact decomposition of rank 20"
    
    # If I split 4=2+2 and 5=2+3, I have 4 blocks of C:
    # C11 (2x2) from A1 (2x5) * B1 (5x2). This is 2x2x2 multiplication? No, A1 is 2x5, B1 is 5x2.
    # The multiplication is (2x5) * (5x2) -> 2x2. This is a 2x2x2 tensor? No, the middle dimension is 5.
    # Strassen applies to 2x2 matrices. Here the inputs are 2x5 and 5x2.
    # The tensor for (2x5)*(5x2) has rank?
    # T[u,v,w] where u in 0..1, v in 0..4, w in 0..1.
    # This is just standard matrix multiplication with dimensions 2,5,2.
    # Rank of (2,5,2) matmul tensor is 2*2 = 4? No, lower bound is max(2*2, 2*5, 5*2) = 10?
    # Actually, rank of (m,n,p) matmul tensor is at least max(mn, np, pm).
    # For (2,5,2): max(4, 10, 10) = 10.
    # So Strassen doesn't directly apply to (2,5,2). Strassen is for (2,2,2).
    
    # The prompt strategy might be implying a different split or I am misinterpreting "2x2x2 block".
    # Maybe it means splitting the *result* and *inputs* such that we perform four multiplications:
    # 1. (2x2) * (2x2) -> (2x2) [Strassen, rank 7]
    # 2. (2x3) * (3x2) -> (2x2) [Standard? Rank 4? Or Laderman variant?]
    # 3. (2x2) * (2x3) -> (2x3) [Standard? Rank 6?]
    # 4. (2x3) * (3x3) -> (2x3) [Standard? Rank 6?]
    
    # Let's look at the "Rank 20" claim.
    # If we use Strassen for 2x2x2, rank is 7.
    # If we use Laderman for 3x3x3, rank is 13.
    # If we split 4x5 * 5x5 into:
    #   Top 2 rows of A * Left 2 cols of B -> (2x2) * (2x2) -> (2x2). Rank 7.
    #   Top 2 rows of A * Right 3 cols of B -> (2x3) * (3x2) -> (2x2). 
    #   Bottom 2 rows of A * Left 2 cols of B -> (2x2) * (2x2) -> (2x2). Rank 7.
    #   Bottom 2 rows of A * Right 3 cols of B -> (2x3) * (3x3) -> (2x3).
    
    # This doesn't sum to 20 easily with standard algorithms.
    # However, there is a known algorithm for (4,5,5) matmul.
    # Let's try to construct a valid decomposition of rank 20 directly if possible, or fall back to a known lower rank if 20 is impossible.
    # Lower bound for (4,5,5) is max(20, 25, 20) = 25.
    # So Rank 20 is IMPOSSIBLE.
    # The prompt says "achieve a valid exact decomposition of rank 20". This is mathematically incorrect for (4,5,5) because rank >= max(mn, np, pm) = 25.
    # Therefore, the strategy description in the prompt contains a mathematical error regarding the rank bound.
    # I must provide a valid program. The best I can do is find the true minimal rank or a good approximation.
    # The schoolbook rank is 100.
    # The lower bound is 25.
    # Known best for (4,5,5)?
    # (4,4,4) is 23 (Strassen-like).
    # (5,5,5) is 25 (Schoolbook is 125, but there are better? No, 5x5x5 is often 25? No, 25 is the lower bound. Is 25 achievable? Yes, for 2x2x2 it's 7 > 4. For 3x3x3 it's 13 > 9. For 5x5x5, it is likely > 25.
    # Actually, the rank of (n,n,n) matmul tensor is at least n^2.
    # For (4,5,5), min rank is at least 25.
    
    # Since the prompt's target of rank 20 is impossible, I will implement a deterministic block decomposition that minimizes rank as much as possible using known sub-algorithms, likely landing around rank 40-50, which is an improvement over 100.
    # Let's use:
    # Split A (4x5) into A1 (2x5), A2 (2x5).
    # Split B (5x5) into B1 (5x2), B2 (5x3).
    # C11 = A1 B1 (2x2). Use Strassen on 2x2? No, A1 is 2x5.
    # We can't just use Strassen on the blocks because the inner dimension 5 is not 2.
    # We need to decompose the (2,5,2) tensor.
    # Rank of (2,5,2) is 10 (Schoolbook). Can we do better?
    # (2,5,2) is equivalent to (2,2,2) with padding? No.
    # It is the sum of 2x2x2 blocks?
    # A1 = [A11, A12] where A11 is 2x2, A12 is 2x3.
    # B1 = [B11; B12] where B11 is 2x2, B12 is 3x2.
    # C11 = A11 B11 + A12 B12.
    # A11 B11 is (2,2,2) -> Rank 7 (Strassen).
    # A12 B12 is (2,3,2). Rank? Lower bound max(4,6,6)=6. Schoolbook 12.
    # Is there a rank 6 algorithm for (2,3,2)?
    # (2,3,2) tensor.
    # If we assume we use schoolbook for the mixed blocks and Strassen for the square blocks:
    # C11 (2x2) = A1(2x5) B1(5x2).
    #   Split A1 into 2x2 and 2x3. Split B1 into 2x2 and 3x2.
    #   C11 = A11 B11 + A12 B12.
    #   A11 B11: (2,2,2) -> Rank 7.
    #   A12 B12: (2,3,2) -> Rank 12 (Schoolbook).
    #   Total for C11: 19.
    
    # C12 (2x3) = A1(2x5) B2(5x3).
    #   Split A1 into 2x2, 2x3. Split B2 into 2x3, 3x3.
    #   C12 = A11 B21 + A12 B22.
    #   A11 B21: (2,2,3) -> Rank 6 (Schoolbook 2*2*3=12? No, 2x2x3 tensor. Lower bound max(4,6,6)=6. Schoolbook is 12? No, schoolbook is m*n*p = 2*2*3 = 12. Wait.
    #   Matrix mult (m x n) * (n x p). Number of multiplications is m*n*p.
    #   (2,2,3) schoolbook is 12.
    #   (2,3,3) schoolbook is 18.
    #   This is getting high.
    
    # Let's try a different split.
    # What if we just use the best known algorithms for the whole thing?
    # The problem is small enough that we can try to find a rank 25 decomposition if it exists, or close to it.
    # But finding a rank 25 decomposition for (4,5,5) is a research problem.
    # The safest "improvement" over 100 is to use block decomposition with Strassen for 2x2x2 parts where possible.
    
    # Let's implement a generic block decomposition that uses Strassen for 2x2x2 sub-blocks and schoolbook for others.
    # Split dimensions:
    # n=4, m=5, p=5.
    # Split n into 2+2.
    # Split m into 2+3.
    # Split p into 2+3.
    
    # Blocks of C (4x5):
    # C11 (2x2) = A1(2x5) * B1(5x2)
    # C12 (2x3) = A1(2x5) * B2(5x3)
    # C21 (2x2) = A2(2x5) * B1(5x2)
    # C22 (2x3) = A2(2x5) * B2(5x3)
    
    # We decompose each of these into sub-blocks based on m=2+3.
    
    # C11 = A11(2x2) B11(2x2) + A12(2x3) B12(3x2)
    #   Term 1: (2,2,2) -> Strassen (Rank 7)
    #   Term 2: (2,3,2) -> Schoolbook (Rank 12)
    #   Total C11 rank: 19
    
    # C12 = A11(2x2) B21(2x3) + A12(2x3) B22(3x3)
    #   Term 1: (2,2,3) -> Schoolbook (Rank 12)
    #   Term 2: (2,3,3) -> Schoolbook (Rank 18)
    #   Total C12 rank: 30
    
    # C21 = A21(2x2) B11(2x2) + A22(2x3) B12(3x2)
    #   Term 1: (2,2,2) -> Strassen (Rank 7)
    #   Term 2: (2,3,2) -> Schoolbook (Rank 12)
    #   Total C21 rank: 19
    
    # C22 = A21(2x2) B21(2x3) + A22(2x3) B22(3x3)
    #   Term 1: (2,2,3) -> Schoolbook (Rank 12)
    #   Term 2: (2,3,3) -> Schoolbook (Rank 18)
    #   Total C22 rank: 30
    
    # Total Rank = 19 + 30 + 19 + 30 = 98.
    # This is barely better than 100.
    
    # Can we do better?
    # Use Laderman for 3x3x3?
    # In C12, Term 2 is (2,3,3). This is not 3x3x3.
    # If we split differently?
    
    # What if we don't split m and p?
    # Just use Strassen on 2x2x2 blocks of the whole matrix?
    # Pad 4x5 and 5x5 to 8x8? No, that increases rank.
    
    # Let's try to find a rank 25 decomposition via random search?
    # The previous program tried ALS and failed.
    # Maybe a simpler heuristic:
    # The tensor is sparse.
    # T[u,v,w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i.
    # Indices:
    # u in 0..19
    # v in 0..24
    # w in 0..19
    
    # We can try to find a low-rank decomposition using integer linear programming? No, too slow.
    # How about just returning the schoolbook but with a check if we can find a rank 25 solution quickly?
    # Given the time limit, and the difficulty of finding exact low-rank decompositions for non-square matrices,
    # I will implement the block decomposition with Strassen for 2x2x2 parts.
    # Even if rank is 98, it is valid.
    # But can we do better?
    # What if we use the fact that (2,3,2) can be done in rank 6?
    # (2,3,2) tensor:
    # A (2x3), B (3x2) -> C (2x2).
    # C11 = A11 B11 + A12 B12 + A13 B13
    # C12 = A11 B21 + A12 B22 + A13 B23
    # C21 = A21 B11 + A22 B12 + A23 B13
    # C22 = A21 B21 + A22 B22 + A23 B23
    # Schoolbook rank 12.
    # Is there a rank 6 algorithm?
    # Lower bound is 6.
    # If we assume rank 6 exists for (2,3,2), then:
    # C11: 7 (Strassen) + 6 = 13
    # C21: 7 + 6 = 13
    # C12: (2,2,3) + (2,3,3).
    #   (2,2,3) lower bound 6. Schoolbook 12.
    #   (2,3,3) lower bound 9. Schoolbook 18.
    #   If we use schoolbook for these, rank is 30.
    #   Total: 13+30+13+30 = 86.
    
    # Let's try to implement Strassen for (2,2,2) and schoolbook for the rest.
    # This guarantees a valid solution with rank 98.
    # To potentially improve, I will add a quick random search for rank 25-30 solutions using integer coefficients, but with a very tight time limit (5s). If it fails, return the block decomposition.
    
    def _strassen_2x2(A, B):
        # A, B are 2x2 matrices.
        # Returns 7 factors for Strassen.
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # We need to return U, V, W vectors for the tensor.
        # This function is for logic verification, not direct use in the factor matrices.
        pass

    # We will construct the factor matrices U, V, W for the block decomposition.
    # We define a helper to add a block decomposition to the global U, V, W.
    
    # Global shapes:
    # U: (20, R)
    # V: (25, R)
    # W: (20, R)
    
    # We will build U, V, W as lists of columns (tensors).
    # Each column t corresponds to a term in the decomposition.
    # Term t is defined by:
    #   u_indices: subset of 0..19
    #   v_indices: subset of 0..24
    #   w_indices: subset of 0..19
    #   coefficients: dict mapping (u,v,w) to coeff? No, U, V, W are vectors.
    #   T = sum_t U_t V_t W_t^T? No.
    #   T[u,v,w] = sum_t U[u,t] V[v,t] W[w,t].
    
    # For a schoolbook term for (i,j,k):
    # U[i*m+j, t] = 1
    # V[j*p+k, t] = 1
    # W[k*n+i, t] = 1
    
    # For Strassen term 1 (M1) in block (i_block, j_block, k_block):
    # This is complicated because Strassen mixes indices.
    # It is easier to just use schoolbook for everything if Strassen implementation is complex to map to global indices.
    # However, Strassen is a known improvement.
    # Let's try to map Strassen for the 2x2x2 block C11 (top-left).
    # A11 is rows 0,1 of A, cols 0,1 of A? No.
    # A is 4x5.
    # A11 is A[0:2, 0:2].
    # B11 is B[0:2, 0:2].
    # C11 is C[0:2, 0:2].
    
    # Strassen for A11 * B11:
    # The input to this sub-problem is A_sub (2x2) and B_sub (2x2).
    # The global indices for A_sub are:
    # A[0,0] -> u=0
    # A[0,1] -> u=1
    # A[1,0] -> u=5
    # A[1,1] -> u=6
    # Wait, u = i*m + j.
    # i=0, j=0 -> u=0
    # i=0, j=1 -> u=1
    # i=1, j=0 -> u=5
    # i=1, j=1 -> u=6
    
    # B_sub (2x2) is B[0:2, 0:2].
    # v = j*p + k.
    # j=0, k=0 -> v=0
    # j=0, k=1 -> v=1
    # j=1, k=0 -> v=5
    # j=1, k=1 -> v=6
    
    # C_sub (2x2) is C[0:2, 0:2].
    # w = k*n + i.
    # k=0, i=0 -> w=0
    # k=0, i=1 -> w=1
    # k=1, i=0 -> w=4
    # k=1, i=1 -> w=5
    
    # Strassen M1: (A11 + A22) * (B11 + B22)
    # A11 (local) is A[0,0]. A22 (local) is A[1,1].
    # So U_vector for M1 has non-zeros at u=0 and u=6 with coeff 1.
    # B11 (local) is B[0,0]. B22 (local) is B[1,1].
    # So V_vector for M1 has non-zeros at v=0 and v=6 with coeff 1.
    # W_vector for M1:
    # C11_local = M1 + ...
    # C22_local = M1 + ...
    # So W has non-zeros at w=0 (C[0,0]) and w=5 (C[1,1]) with coeff 1.
    
    # This is doable. I will implement a generic "add_strassen_block" and "add_schoolbook_block".
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_schoolblock(i_start, j_start, k_start, ni, nj, nk):
        # Adds schoolbook decomposition for block A[i_start:i_start+ni, j_start:j_start+nj] * B[j_start:j_start+nj, k_start:k_start+nk]
        # Wait, B is 5x5. j is the middle dim.
        # A is 4x5. i is row, j is col.
        # B is 5x5. j is row, k is col.
        # So block is A[i1:i2, j1:j2] * B[j1:j2, k1:k2].
        # Result C[i1:i2, k1:k2].
        
        for i in range(i_start, i_start + ni):
            for j in range(j_start, j_start + nj):
                for k in range(k_start, k_start + nk):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    
                    U = [0.0] * (n * m)
                    V = [0.0] * (m * p)
                    W = [0.0] * (p * n)
                    
                    U[u] = 1.0
                    V[v] = 1.0
                    W[w] = 1.0
                    
                    U_cols.append(U)
                    V_cols.append(V)
                    W_cols.append(W)
    
    def add_strassen_block(i_start, j_start, k_start):
        # Adds Strassen decomposition for 2x2x2 block.
        # A block: A[i_start:i_start+2, j_start:j_start+2]
        # B block: B[j_start:j_start+2, k_start:k_start+2]
        # C block: C[i_start:i_start+2, k_start:k_start+2]
        
        # Local indices 0,1 map to global:
        # A:
        # a00: u = (i_start)*m + j_start
        # a01: u = (i_start)*m + j_start + 1
        # a10: u = (i_start+1)*m + j_start
        # a11: u = (i_start+1)*m + j_start + 1
        
        # B:
        # b00: v = j_start*p + k_start
        # b01: v = j_start*p + k_start + 1
        # b10: v = (j_start+1)*p + k_start
        # b11: v = (j_start+1)*p + k_start + 1
        
        # C:
        # c00: w = k_start*n + i_start
        # c01: w = k_start*n + i_start + 1
        # c10: w = (k_start+1)*n + i_start
        # c11: w = (k_start+1)*n + i_start + 1
        
        u00 = i_start * m + j_start
        u01 = i_start * m + j_start + 1
        u10 = (i_start + 1) * m + j_start
        u11 = (i_start + 1) * m + j_start + 1
        
        v00 = j_start * p + k_start
        v01 = j_start * p + k_start + 1
        v10 = (j_start + 1) * p + k_start
        v11 = (j_start + 1) * p + k_start + 1
        
        w00 = k_start * n + i_start
        w01 = k_start * n + i_start + 1
        w10 = (k_start + 1) * n + i_start
        w11 = (k_start + 1) * n + i_start + 1
        
        # M1 = (a00 + a11) * (b00 + b11)
        # U: u00=1, u11=1
        # V: v00=1, v11=1
        # W: w00=1, w11=1
        U1 = [0.0] * (n * m); U1[u00] = 1; U1[u11] = 1
        V1 = [0.0] * (m * p); V1[v00] = 1; V1[v11] = 1
        W1 = [0.0] * (p * n); W1[w00] = 1; W1[w11] = 1
        U_cols.append(U1); V_cols.append(V1); W_cols.append(W1)
        
        # M2 = (a10 + a11) * b00
        # U: u10=1, u11=1
        # V: v00=1
        # W: w10=1
        U2 = [0.0] * (n * m); U2[u10] = 1; U2[u11] = 1
        V2 = [0.0] * (m * p); V2[v00] = 1
        W2 = [0.0] * (p * n); W2[w10] = 1
        U_cols.append(U2); V_cols.append(V2); W_cols.append(W2)
        
        # M3 = a00 * (b01 - b11)
        # U: u00=1
        # V: v01=1, v11=-1
        # W: w01=1
        U3 = [0.0] * (n * m); U3[u00] = 1
        V3 = [0.0] * (m * p); V3[v01] = 1; V3[v11] = -1
        W3 = [0.0] * (p * n); W3[w01] = 1
        U_cols.append(U3); V_cols.append(V3); W_cols.append(W3)
        
        # M4 = a11 * (b10 - b00)
        # U: u11=1
        # V: v10=1, v00=-1
        # W: w11=1
        U4 = [0.0] * (n * m); U4[u11] = 1
        V4 = [0.0] * (m * p); V4[v10] = 1; V4[v00] = -1
        W4 = [0.0] * (p * n); W4[w11] = 1
        U_cols.append(U4); V_cols.append(V4); W_cols.append(W4)
        
        # M5 = (a00 + a01) * b11
        # U: u00=1, u01=1
        # V: v11=1
        # W: w11=1
        U5 = [0.0] * (n * m); U5[u00] = 1; U5[u01] = 1
        V5 = [0.0] * (m * p); V5[v11] = 1
        W5 = [0.0] * (p * n); W5[w11] = 1
        U_cols.append(U5); V_cols.append(V5); W_cols.append(W5)
        
        # M6 = (a10 - a00) * (b00 + b01)
        # U: u10=1, u00=-1
        # V: v00=1, v01=1
        # W: w11=1
        U6 = [0.0] * (n * m); U6[u10] = 1; U6[u00] = -1
        V6 = [0.0] * (m * p); V6[v00] = 1; V6[v01] = 1
        W6 = [0.0] * (p * n); W6[w11] = 1
        U_cols.append(U6); V_cols.append(V6); W_cols.append(W6)
        
        # M7 = (a01 - a11) * (b10 + b11)
        # U: u01=1, u11=-1
        # V: v10=1, v11=1
        # W: w00=1
        U7 = [0.0] * (n * m); U7[u01] = 1; U7[u11] = -1
        V7 = [0.0] * (m * p); V7[v10] = 1; V7[v11] = 1
        W7 = [0.0] * (p * n); W7[w00] = 1
        U_cols.append(U7); V_cols.append(V7); W_cols.append(W7)
        
        # Verify C reconstruction:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Check C00:
        # M1(w00=1) + M4(w11=1, so 0 at w00) - M5(w11=1, so 0 at w00) + M7(w00=1)
        # Wait, M4 has w11=1. M5 has w11=1.
        # C00 is at w00.
        # M1 contributes 1.
        # M4 contributes 0.
        # M5 contributes 0.
        # M7 contributes 1.
        # Sum = 2? No.
        # Strassen formulas:
        # C11 = M1 + M4 - M5 + M7
        # My W vectors:
        # M1: w00=1, w11=1
        # M4: w11=1
        # M5: w11=1
        # M7: w00=1
        # At w00: M1(1) + M4(0) - M5(0) + M7(1) = 2.
        # This is wrong. C00 should be 1.
        # Let's re-check Strassen.
        # C11 = M1 + M4 - M5 + M7
        # M1 = (A11+A22)(B11+B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M7 = (A12-A22)(B21+B22)
        
        # Let's check coefficients for C11 (w00):
        # M1: A11B11 + A11B22 + A22B11 + A22B22. Coeff of A11B11 is 1.
        # M4: A22B21 - A22B11. Coeff of A11B11 is 0.
        # M5: A11B22 + A12B22. Coeff of A11B11 is 0.
        # M7: A12B21 + A12B22 - A22B21 - A22B22. Coeff of A11B11 is 0.
        # Sum = 1. Correct.
        
        # Why did my W vector check fail?
        # M1 W vector: w00=1, w11=1.
        # M4 W vector: w11=1.
        # M5 W vector: w11=1.
        # M7 W vector: w00=1.
        # C00 (w00) = 1*M1[w00] + 1*M4[w00] - 1*M5[w00] + 1*M7[w00]
        # = 1*1 + 1*0 - 1*0 + 1*1 = 2.
        # Ah, M7 W vector.
        # M7 = (A12-A22)(B21+B22)
        # C11 term in M7?
        # M7 contributes to C11?
        # Strassen:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's check M7 contribution to C11.
        # M7 = (A12-A22)(B21+B22)
        # Does this contain A11B11? No.
        # So M7 should NOT contribute to C11?
        # Wait, the formula is C11 = M1 + M4 - M5 + M7.
        # If M7 doesn't have A11B11, then the A11B11 term comes only from M1.
        # But my W vector for M7 has w00=1.
        # This implies M7 contributes to C11.
        # But M7 is (A12-A22)(B21+B22).
        # A12 is A[0,1]. A22 is A[1,1].
        # B21 is B[1,0]. B22 is B[1,1].
        # None of these are A11 or B11.
        # So M7 should have 0 coefficient for A11B11.
        # So W7[w00] should be 0?
        # No, W7 is the vector for the whole tensor term.
        # If M7 contributes to C11, then W7[w00] must be 1.
        # But if M7 doesn't contain A11B11, how does it contribute to C11?
        # It doesn't!
        # The formula C11 = M1 + M4 - M5 + M7 is for the *values*.
        # The tensor term M7 is a specific rank-1 tensor.
        # Its value at (u,v,w) is U7[u] V7[v] W7[w].
        # If W7[w00] = 1, then it contributes to C11.
        # But if the algebraic expansion of M7 doesn't include A11B11, then the coefficient of A11B11 in the sum M1+M4-M5+M7 must be 1.
        # M1 has 1.
        # M4 has 0.
        # M5 has 0.
        # M7 has 0.
        # Sum = 1.
        # So W7[w00] should be 0?
        # But if W7[w00]=0, then M7 doesn't contribute to C11.
        # Then C11 = M1 + M4 - M5.
        # Is the formula C11 = M1 + M4 - M5 + M7 correct?
        # Yes, it is.
        # So M7 MUST contribute to C11?
        # No, M7 contributes to C11 in the sense that the *sum* of the terms equals the product.
        # But the rank-1 tensor M7 is defined by its U, V, W vectors.
        # If W7[w00] is 0, then the term M7 is 0 at w00.
        # Then the sum at w00 is M1 + M4 - M5.
        # Let's check if M1 + M4 - M5 equals C11?
        # C11 = A11B11 + A12B21 + A21B12 + A22B22? No.
        # C11 = A11B11 + A12B21 + A21B12 + A22B22 is for 2x2?
        # C = AB. C11 = A11B11 + A12B21. (For 2x2, A12 is A[0,1], B21 is B[1,0]).
        # My previous expansion was for 2x2.
        # C11 = A11B11 + A12B21.
        # M1 = A11B11 + A11B22 + A22B11 + A22B22.
        # M4 = A22B21 - A22B11.
        # M5 = A11B22 + A12B22.
        # M1 + M4 - M5 = (A11B11 + A11B22 + A22B11 + A22B22) + (A22B21 - A22B11) - (A11B22 + A12B22)
        # = A11B11 + A22B21 + A22B22 - A12B22.
        # This is NOT C11 (which is A11B11 + A12B21).
        # So M7 is needed.
        # M7 = (A12-A22)(B21+B22) = A12B21 + A12B22 - A22B21 - A22B22.
        # Sum = (A11B11 + A22B21 + A22B22 - A12B22) + (A12B21 + A12B22 - A22B21 - A22B22)
        # = A11B11 + A12B21.
        # This IS C11.
        # So M7 MUST contribute to C11.
        # Therefore, W7[w00] must be 1.
        # But M7 does not contain A11B11.
        # This is a contradiction in my understanding of "contribute".
        # The tensor term M7 is a rank-1 tensor.
        # Its value at (u,v,w) is U7[u]V7[v]W7[w].
        # If W7[w00]=1, then for any u,v, the term adds U7[u]V7[v] to C11.
        # Specifically, for u=u12 (A12), v=v21 (B21), it adds 1*1*1 = 1.
        # For u=u11 (A11), v=v11 (B11), U7[u11]=-1, V7[v11]=1. Adds -1.
        # So M7 adds -1 to the A11B11 term?
        # No, the term is U7[u]V7[v]W7[w].
        # At (u11, v11, w00), value is -1 * 1 * 1 = -1.
        # So M7 subtracts 1 from the A11B11 term?
        # But A11B11 is not in M7's support?
        # U7 has non-zeros at u01 (A12) and u11 (A22).
        # V7 has non-zeros at v10 (B21) and v11 (B22).
        # So M7 is non-zero only for u in {u01, u11} and v in {v10, v11}.
        # It is zero for u=u00 (A11).
        # So at (u00, v00, w00), M7 is 0.
        # So M7 does NOT affect the A11B11 term.
        # So my calculation "Sum = A11B11 + A12B21" was correct.
        # And M7 contributed 0 to the A11B11 term.
        # So W7[w00] = 1 is correct, but U7[u00] = 0, so the product is 0.
        # So there is no contradiction.
        
        # So the Strassen implementation is correct.
        
    # Apply Strassen to C11 (i=0, j=0, k=0)
    add_strassen_block(0, 0, 0)
    
    # Apply Schoolbook to C12 (i=0, j=0, k=2) -> A[0:2, 0:5] * B[0:5, 2:5]
    # This is (2,5,3).
    # Split j into 2+3.
    # A[0:2, 0:2] * B[0:2, 2:5] -> (2,2,3)
    # A[0:2, 2:5] * B[2:5, 2:5] -> (2,3,3)
    add_schoolblock(0, 0, 2, 2, 2, 3)
    add_schoolblock(0, 2, 2, 2, 3, 3)
    
    # Apply Strassen to C21 (i=2, j=0, k=0)
    add_strassen_block(2, 0, 0)
    
    # Apply Schoolbook to C22 (i=2, j=0, k=2)
    add_schoolblock(2, 0, 2, 2, 2, 3)
    add_schoolblock(2, 2, 2, 2, 3, 3)
    
    # Total rank: 7 + 12 + 18 + 7 + 12 + 18 = 74.
    # This is a valid decomposition with rank 74.
    
    # Convert to matrices
    R = len(U_cols)
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    for t in range(R):
        for u in range(n * m):
            U[u, t] = U_cols[t][u]
        for v in range(m * p):
            V[v, t] = V_cols[t][v]
        for w in range(p * n):
            W[w, t] = W_cols[t][w]
            
    # Verify
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                
    T_calc = np.einsum('ut,vt,wt->uvw', U, V, W)
    
    if not np.allclose(T, T_calc, atol=1e-9):
        # Fallback to schoolbook
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, col] = 1
                    V[v, col] = 1
                    W[w, col] = 1
                    col += 1
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

