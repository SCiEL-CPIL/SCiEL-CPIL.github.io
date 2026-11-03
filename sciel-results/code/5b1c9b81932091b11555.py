import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
import sys

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p
    
    # 1. Generate the target tensor T
    # T[i, j, k] = 1 if the (i,k) entry of C = A*B depends on A[i,j] * B[j,k]
    # In the trivial decomposition, every (i,j,k) triplet is a rank-1 term.
    # We represent the tensor as a vector of length n*m*p.
    T_vec = np.zeros((n, m, p), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_vec[i, j, k] = 1
                
    T_flat = T_vec.flatten()

    # 2. Generate candidate rank-1 tensors
    # A rank-1 tensor is defined by u (size n*m), v (size m*p), w (size p*n).
    # To keep the search space manageable, we restrict candidates to "sparse" terms.
    # Specifically, we consider terms where u and v have small support.
    # Let's start with terms where u has exactly 1 non-zero (standard basis) or 2 non-zeros.
    # And v has exactly 1 or 2 non-zeros.
    # Actually, for the trivial decomposition, u=e_i, v=e_j, w=e_k.
    # Strassen-like terms involve sums.
    
    # Let's generate a dictionary of candidate terms.
    # A term is a tuple (u_list, v_list, w_list) where lists are sparse (index, value).
    # We will build a matrix M where columns are candidate terms and rows are tensor entries (i,j,k).
    # M[row, col] = coefficient of tensor entry (i,j,k) in the candidate term.
    
    candidates = []
    
    # Helper to add a candidate term
    def add_candidate(u_sparse, v_sparse, w_sparse):
        # u_sparse: list of (idx, val)
        # v_sparse: list of (idx, val)
        # w_sparse: list of (idx, val)
        # Check if this term is linearly independent or new? 
        # For ILP, we just need the vector representation.
        # Vector representation: for each (i, j, k), coeff = sum_{i', j', k'} u[i'] v[j'] w[k'] * delta(i, i') delta(j, j') delta(k, k')?
        # No, the tensor entry T[i,j,k] is reconstructed as sum_r U[i,j,r] V[j,k,r] W[k,i,r].
        # So the contribution of term r to T[i,j,k] is U[i,j,r] * V[j,k,r] * W[k,i,r].
        
        # Let's map indices:
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        term_vec = np.zeros(n*m*p, dtype=int)
        
        # Iterate over non-zeros of u, v, w
        # To be efficient, we can just compute the product for all i,j,k? 
        # n*m*p is 48. It's small.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = 0
                    for idx, val in u_sparse:
                        if idx == i*m + j:
                            u_val = val
                            break
                    if u_val == 0: continue
                    
                    v_val = 0
                    for idx, val in v_sparse:
                        if idx == j*p + k:
                            v_val = val
                            break
                    if v_val == 0: continue
                    
                    w_val = 0
                    for idx, val in w_sparse:
                        if idx == k*n + i:
                            w_val = val
                            break
                    if w_val == 0: continue
                    
                    term_vec[i*m*p + j*p + k] = u_val * v_val * w_val
                    
        # Only add if not all zeros
        if np.any(term_vec != 0):
            # Check if this term is already in candidates (to reduce width)
            # Simple hash check
            key = tuple(term_vec.tolist())
            if key not in [tuple(c[0].tolist()) for c in candidates]:
                candidates.append((term_vec, u_sparse, v_sparse, w_sparse))

    # Generate candidates
    # 1. Standard basis terms (trivial decomposition)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [(i*m + j, 1)]
                v = [(j*p + k, 1)]
                w = [(k*n + i, 1)]
                add_candidate(u, v, w)
                
    # 2. Try some "Strassen-like" combinations
    # We look for terms where u, v, w have 2 non-zeros each.
    # This is combinatorially large, so we sample or use specific patterns.
    # Let's try all pairs of indices for u, v, w?
    # Number of pairs in u: C(8,2)=28. v: C(24,2)=276. w: C(12,2)=66.
    # 28*276*66 is ~500k candidates. Too many for a single ILP?
    # Let's restrict to "block" aligned pairs or random sampling.
    
    # Let's add candidates where u has 2 non-zeros, v has 2 non-zeros, w has 2 non-zeros.
    # But only if they are "compatible" (i.e., the product isn't zero everywhere).
    # Actually, let's just generate a reasonable set.
    
    # To ensure we don't miss the trivial solution, we already have it.
    # Can we find a lower rank?
    # Let's add candidates where u, v, w are sums of 2 basis vectors.
    
    # For efficiency, let's limit the number of candidates to ~500-1000.
    # We already have 48.
    
    # Let's try to add some random combinations to see if ILP finds a sparser solution.
    # But random might miss the structure.
    
    # Alternative: Use the known Strassen structure for 2x2x2 blocks?
    # The current program failed to correctly implement the block reduction to rank 42 due to index mapping errors or complexity.
    # If we can correctly implement the rank-42 decomposition, that's better than 48.
    # The inspiration says: "Uses Strassen's algorithm ... reducing the rank from the schoolbook 48 to 42. Measured: 1 -> 1.143."
    # Wait, the score is seed/value. If value is rank, lower is better.
    # Seed value (rank 48) gives score 1.0?
    # "score is the rank" -> No, "score is seed/value".
    # If seed is 48, and value is 48, score is 1.
    # If value is 42, score is 48/42 = 1.142.
    # So we want to minimize rank.
    
    # The ILP approach is a "black box" search. It might find 42 if the candidates are good.
    # But generating good candidates for 2x4x6 is hard.
    
    # Let's try a different structural approach: **Explicit Strassen Construction**.
    # The previous attempt failed because it tried to map 1x2 * 2x3 blocks to scalar multiplications incorrectly or the rank count was wrong.
    # Let's carefully construct the rank-42 decomposition.
    
    # Block partition:
    # A (2x4) -> A11(1x2), A12(1x2); A21(1x2), A22(1x2)
    # B (4x6) -> B11(2x3), B12(2x3); B13(2x3), B14(2x3)
    # C (2x6) -> C11(1x3), C12(1x3); C21(1x3), C22(1x3)
    
    # Strassen terms:
    # M1 = (A11 + A22) * (B11 + B14)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B14)
    # M4 = A22 * (B13 - B11)
    # M5 = (A11 + A12) * B14
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B13 + B14)
    
    # Each Mi is a matrix product of size (1x2) * (2x3) = 1x3.
    # The rank of the tensor for 1x2 * 2x3 multiplication is 6 (trivial).
    # So total rank = 7 * 6 = 42.
    
    # We need to construct U, V, W for this.
    # U has shape (8, 42). V (24, 42). W (12, 42).
    # Indices:
    # U[i, j] where i is row of A (0,1), j is col of A (0..3).
    # V[j, k] where j is row of B (0..3), k is col of B (0..5).
    # W[k, i] where k is col of B (0..5), i is row of A (0,1). Note: W[k*n+i, r].
    
    # For each Strassen term t (0..6), we have a sub-problem.
    # Let's define the coefficients for A and B for each term.
    # A_t is a 2x4 matrix. B_t is a 4x6 matrix.
    # The product C_t = A_t * B_t is a 2x6 matrix.
    # We decompose C_t into 6 rank-1 tensors?
    # No, the "rank 6" refers to the tensor rank of the *operation* 1x2 * 2x3.
    # The operation is: given a (1x2) vector a and (2x3) matrix b, compute a*b (1x3).
    # The tensor T_op[a_i, b_jk, c_k] = delta(i, j) ? No.
    # T_op[i, j, k] = 1 if i==j? No, a is 1x2, so index i is 0 or 1.
    # b is 2x3, index j is 0 or 1, k is 0,1,2.
    # c is 1x3, index k is 0,1,2.
    # c_k = sum_j a_j b_jk.
    # Tensor T[j, k, l] ? No.
    # Inputs: a (size 2), b (size 6). Output: c (size 3).
    # T is a tensor of size 2 x 6 x 3.
    # T[a_idx, b_idx, c_idx] = 1 if b_idx corresponds to (a_idx, c_idx).
    # b_idx = j*p + k = j*3 + k.
    # So T[j, j*3+k, k] = 1.
    # Rank of this tensor?
    # Trivial rank is 6 (for each j, k pair).
    # Can we do better?
    # This is the rank of the "vec" multiplication.
    # For 1x2 * 2x3, the rank is indeed 6. It's just dot product for each output element.
    # c_0 = a_0 b_00 + a_1 b_10
    # c_1 = a_0 b_01 + a_1 b_11
    # c_2 = a_0 b_02 + a_1 b_12
    # These are 3 independent dot products of size 2.
    # Rank of dot product of size 2 is 2 (a0*b0 + a1*b1).
    # So total rank 3 * 2 = 6.
    
    # So, for each Strassen term t, we add 6 rank-1 terms to the global U, V, W.
    # Total 42 terms.
    
    # Let's build the U, V, W matrices.
    
    rank_total = 42
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # Define A and B coefficients for the 7 terms
    # A_coeffs[t] is 2x4
    # B_coeffs[t] is 4x6
    
    def zero_A(): return np.zeros((2, 4))
    def zero_B(): return np.zeros((4, 6))
    
    A_c = []
    B_c = []
    
    # T1: (A11 + A22) * (B11 + B14)
    A1 = zero_A(); A1[0,0]=1; A1[0,1]=1; A1[1,2]=1; A1[1,3]=1
    B1 = zero_B(); B1[0,0:3]=1; B1[1,0:3]=1; B1[2,3:6]=1; B1[3,3:6]=1
    A_c.append(A1); B_c.append(B1)
    
    # T2: (A21 + A22) * B11
    A2 = zero_A(); A2[1,0]=1; A2[1,1]=1; A2[1,2]=1; A2[1,3]=1
    B2 = zero_B(); B2[0,0:3]=1; B2[1,0:3]=1
    A_c.append(A2); B_c.append(B2)
    
    # T3: A11 * (B12 - B14)
    A3 = zero_A(); A3[0,0]=1; A3[0,1]=1
    B3 = zero_B(); B3[0,3:6]=1; B3[1,3:6]=1; B3[2,3:6]=-1; B3[3,3:6]=-1
    A_c.append(A3); B_c.append(B3)
    
    # T4: A22 * (B13 - B11)
    A4 = zero_A(); A4[1,2]=1; A4[1,3]=1
    B4 = zero_B(); B4[2,0:3]=1; B4[3,0:3]=1; B4[0,0:3]=-1; B4[1,0:3]=-1
    A_c.append(A4); B_c.append(B4)
    
    # T5: (A11 + A12) * B14
    A5 = zero_A(); A5[0,0]=1; A5[0,1]=1; A5[0,2]=1; A5[0,3]=1
    B5 = zero_B(); B5[2,3:6]=1; B5[3,3:6]=1
    A_c.append(A5); B_c.append(B5)
    
    # T6: (A21 - A11) * (B11 + B12)
    A6 = zero_A(); A6[1,0]=1; A6[1,1]=1; A6[0,0]=-1; A6[0,1]=-1
    B6 = zero_B(); B6[0,0:6]=1; B6[1,0:6]=1
    A_c.append(A6); B_c.append(B6)
    
    # T7: (A12 - A22) * (B13 + B14)
    A7 = zero_A(); A7[0,2]=1; A7[0,3]=1; A7[1,2]=-1; A7[1,3]=-1
    B7 = zero_B(); B7[2,0:6]=1; B7[3,0:6]=1
    A_c.append(A7); B_c.append(B7)
    
    # Now, for each term t, we decompose the product A_t * B_t into rank-1 terms.
    # The product is a 2x6 matrix.
    # We want to express the *operation* defined by A_t and B_t as a sum of rank-1 tensors.
    # The "inputs" to the sub-problem are the *original* A and B matrices?
    # No, the inputs to the sub-problem are the *linear combinations* A_t and B_t.
    # But A_t and B_t are themselves linear combinations of the original A and B.
    # So we can "push" the coefficients down to the original A and B indices.
    
    # For term t, we have A_t[i, j] and B_t[j, k].
    # The product is C_t[i, k] = sum_j A_t[i, j] B_t[j, k].
    # We decompose this sum into rank-1 terms.
    # The trivial decomposition of this specific sum uses 6 terms?
    # Wait, A_t is 2x4, B_t is 4x6.
    # The support of A_t and B_t determines the number of terms.
    # In the previous analysis, I counted the non-zero triplets (i, j, k) where A_t[i,j]!=0 and B_t[j,k]!=0.
    # For T1, I counted 12.
    # For T2, I counted 6.
    # Total was 60.
    
    # Why did I think Strassen gives 42?
    # Because I assumed each "block multiplication" has rank 6.
    # But the "block multiplication" is not a scalar multiplication.
    # It is a matrix multiplication of rectangular blocks.
    # The rank of the tensor for 1x2 * 2x3 is 6.
    # BUT, the blocks in Strassen are not 1x2 and 2x3 in the sense of independent sub-problems.
    # They are *parts* of the larger matrix.
    # The Strassen algorithm works by reducing the number of *scalar* multiplications in a 2x2 block multiplication where blocks are scalars.
    # When blocks are matrices, the rank is multiplicative?
    # Rank(M_{n,m,p}) <= Rank(M_{r,s,t}) * Rank(M_{n/r, m/s, p/t})?
    # If we use Strassen (rank 7) for 2x2x2 blocks, and trivial (rank 6) for 1x2x3 blocks.
    # Total rank 7 * 6 = 42.
    
    # How does this mapping work?
    # The 2x2x2 tensor T_strassen has entries T_strassen[I, J, K] for I,J,K in {0,1}.
    # T_strassen[I, J, K] = sum_{L=0}^1 S_I[J, L] * M_L * S_K[L, I]? No.
    # T_strassen is the tensor for C = A B.
    # C_IK = sum_J A_IJ B_JK.
    # The rank-7 decomposition of this tensor provides 7 terms.
    # Each term is a product of a linear combo of A blocks, a linear combo of B blocks, and a linear combo of C blocks.
    # Let the terms be (u_t, v_t, w_t) where u_t is a 2x2 matrix (coeffs for A blocks), v_t is 2x2 (coeffs for B blocks), w_t is 2x2 (coeffs for C blocks).
    # Then C = sum_t w_t * (u_t * A_blocks) * (v_t * B_blocks)?
    # No, the decomposition is:
    # T = sum_t u_t \otimes v_t \otimes w_t.
    # This means C[I, K] = sum_t w_t[I, K] * (sum_J u_t[I, J] A[J, ?]) ...
    
    # Actually, the standard Kronecker product property says:
    # If we have a decomposition of the 2x2x2 tensor with rank 7, and a decomposition of the 1x2x3 tensor with rank 6,
    # we can construct a decomposition of the 2x4x6 tensor with rank 42.
    
    # Let's implement this Kronecker product construction.
    
    # 1. Get the rank-7 decomposition of 2x2x2 (Strassen).
    # Strassen terms for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need the u, v, w matrices for each term.
    # u_t is 2x2 (coeffs for A blocks A00, A01, A10, A11)
    # v_t is 2x2 (coeffs for B blocks B00, B01, B10, B11)
    # w_t is 2x2 (coeffs for C blocks C00, C01, C10, C11)
    
    # Let's define them.
    # Indexing: A[I, J] where I is row, J is col.
    
    U_strassen = []
    V_strassen = []
    W_strassen = []
    
    # Term 1: (A00+A11)(B00+B11)
    u1 = np.array([[1, 0], [0, 1]]) # A00 and A11
    v1 = np.array([[1, 0], [0, 1]]) # B00 and B11
    w1 = np.array([[1, 0], [0, 1]]) # C00 and C11 (Wait, M1 contributes to C00 and C11)
    # C00 gets +M1. C11 gets +M1.
    # So w1 should have 1 at (0,0) and (1,1).
    U_strassen.append(u1)
    V_strassen.append(v1)
    W_strassen.append(w1)
    
    # Term 2: (A10+A11)B00
    u2 = np.array([[0, 0], [1, 1]])
    v2 = np.array([[1, 0], [0, 0]])
    w2 = np.array([[0, 0], [1, 0]]) # C10 gets +M2
    U_strassen.append(u2)
    V_strassen.append(v2)
    W_strassen.append(w2)
    
    # Term 3: A00(B01-B11)
    u3 = np.array([[1, 0], [0, 0]])
    v3 = np.array([[0, 1], [0, -1]])
    w3 = np.array([[0, 1], [0, 0]]) # C01 gets +M3
    U_strassen.append(u3)
    V_strassen.append(v3)
    W_strassen.append(w3)
    
    # Term 4: A11(B10-B00)
    u4 = np.array([[0, 0], [0, 1]])
    v4 = np.array([[-1, 0], [1, 0]])
    w4 = np.array([[1, 0], [1, 0]]) # C00 gets +M4, C10 gets +M4
    U_strassen.append(u4)
    V_strassen.append(v4)
    W_strassen.append(w4)
    
    # Term 5: (A00+A01)B11
    u5 = np.array([[1, 1], [0, 0]])
    v5 = np.array([[0, 0], [0, 1]])
    w5 = np.array([[-1, 1], [0, 0]]) # C00 gets -M5, C01 gets +M5
    U_strassen.append(u5)
    V_strassen.append(v5)
    W_strassen.append(w5)
    
    # Term 6: (A10-A00)(B00+B01)
    u6 = np.array([[-1, 0], [1, 0]])
    v6 = np.array([[1, 1], [0, 0]])
    w6 = np.array([[0, 0], [0, 1]]) # C11 gets +M6
    U_strassen.append(u6)
    V_strassen.append(v6)
    W_strassen.append(w6)
    
    # Term 7: (A01-A11)(B10+B11)
    u7 = np.array([[0, 1], [0, -1]])
    v7 = np.array([[0, 0], [1, 1]])
    w7 = np.array([[1, 0], [0, -1]]) # C00 gets +M7, C11 gets -M7
    U_strassen.append(u7)
    V_strassen.append(v7)
    W_strassen.append(w7)
    
    # 2. Get the rank-6 decomposition of 1x2x3 (Trivial).
    # Tensor for 1x2 * 2x3.
    # Inputs: A_block (1x2), B_block (2x3). Output: C_block (1x3).
    # Trivial rank 6:
    # For j in 0,1 and k in 0,1,2:
    # Term (j, k):
    # u_jk is 1x2: 1 at index j.
    # v_jk is 2x3: 1 at (j, k).
    # w_jk is 1x3: 1 at index k.
    
    U_sub = []
    V_sub = []
    W_sub = []
    
    for j in range(2):
        for k in range(3):
            u_sk = np.zeros((1, 2)); u_sk[0, j] = 1
            v_sk = np.zeros((2, 3)); v_sk[j, k] = 1
            w_sk = np.zeros((1, 3)); w_sk[0, k] = 1
            U_sub.append(u_sk)
            V_sub.append(v_sk)
            W_sub.append(w_sk)
            
    # 3. Construct the full 2x4x6 decomposition using Kronecker products.
    # Total rank = 7 * 6 = 42.
    
    U_full = np.zeros((n * m, 42))
    V_full = np.zeros((m * p, 42))
    W_full = np.zeros((p * n, 42))
    
    # Mapping indices:
    # A is 2x4. Blocks A00, A01, A10, A11 are 1x2.
    # A[I, J] where I in {0,1}, J in {0,1,2,3}.
    # A00: I=0, J=0,1.
    # A01: I=0, J=2,3.
    # A10: I=1, J=0,1.
    # A11: I=1, J=2,3.
    
    # B is 4x6. Blocks B00, B01, B10, B11 are 2x3.
    # B[I, J] where I in {0,1,2,3}, J in {0,1,2,3,4,5}.
    # B00: I=0,1, J=0,1,2.
    # B01: I=0,1, J=3,4,5.
    # B10: I=2,3, J=0,1,2.
    # B11: I=2,3, J=3,4,5.
    
    # C is 2x6. Blocks C00, C01, C10, C11 are 1x3.
    # C[I, J] where I in {0,1}, J in {0,1,2,3,4,5}.
    # C00: I=0, J=0,1,2.
    # C01: I=0, J=3,4,5.
    # C10: I=1, J=0,1,2.
    # C11: I=1, J=3,4,5.
    
    # For each Strassen term s (0..6) and Sub term t (0..5):
    # Rank index r = s * 6 + t.
    
    for s in range(7):
        for t in range(6):
            r = s * 6 + t
            
            u_s = U_strassen[s] # 2x2
            v_s = V_strassen[s] # 2x2
            w_s = W_strassen[s] # 2x2
            
            u_t = U_sub[t] # 1x2
            v_t = V_sub[t] # 2x3
            w_t = W_sub[t] # 1x3
            
            # Construct U_full[:, r]
            # U_full index: i*m + j = i*4 + j.
            # u_s[I, J] is coeff for block A[I, J].
            # u_t[0, j_local] is coeff for element in block.
            
            for I in range(2):
                for J in range(2):
                    coeff_s = u_s[I, J]
                    if coeff_s == 0: continue
                    
                    # Block A[I, J] corresponds to rows I, cols J*2, J*2+1.
                    for j_local in range(2):
                        coeff_t = u_t[0, j_local]
                        if coeff_t == 0: continue
                        
                        j_global = J * 2 + j_local
                        idx_u = I * 4 + j_global
                        U_full[idx_u, r] += coeff_s * coeff_t
                        
            # Construct V_full[:, r]
            # V_full index: j*p + k = j*6 + k.
            # v_s[I, J] is coeff for block B[I, J].
            # v_t[j_local, k_local] is coeff for element in block.
            
            for I in range(2):
                for J in range(2):
                    coeff_s = v_s[I, J]
                    if coeff_s == 0: continue
                    
                    # Block B[I, J] corresponds to rows I*2, I*2+1, cols J*3, J*3+1, J*3+2.
                    for i_local in range(2):
                        for k_local in range(3):
                            coeff_t = v_t[i_local, k_local]
                            if coeff_t == 0: continue
                            
                            i_global = I * 2 + i_local
                            k_global = J * 3 + k_local
                            idx_v = i_global * 6 + k_global
                            V_full[idx_v, r] += coeff_s * coeff_t
                            
            # Construct W_full[:, r]
            # W_full index: k*n + i = k*2 + i.
            # w_s[I, J] is coeff for block C[I, J].
            # w_t[0, k_local] is coeff for element in block.
            
            for I in range(2):
                for J in range(2):
                    coeff_s = w_s[I, J]
                    if coeff_s == 0: continue
                    
                    # Block C[I, J] corresponds to row I, cols J*3, J*3+1, J*3+2.
                    for k_local in range(3):
                        coeff_t = w_t[0, k_local]
                        if coeff_t == 0: continue
                        
                        k_global = J * 3 + k_local
                        idx_w = k_global * 2 + I
                        W_full[idx_w, r] += coeff_s * coeff_t
                        
    # Verify the decomposition
    # T_reconstructed[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # We want T_reconstructed[i, j, k] == 1 for all i,j,k.
    
    # Check a few entries
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(42):
                    val += U_full[i*m+j, r] * V_full[j*p+k, r] * W_full[k*n+i, r]
                if val != 1:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if not valid:
        # Fallback to trivial
        U = np.zeros((n * m, 48))
        V = np.zeros((m * p, 48))
        W = np.zeros((p * n, 48))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * (m * p) + j * p + k
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
    else:
        U = U_full
        V = V_full
        W = W_full
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
