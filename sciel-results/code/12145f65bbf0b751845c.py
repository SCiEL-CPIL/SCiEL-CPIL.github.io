"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strategy: 3x3 block partition where each 2x2 block uses Strassen (rank 7)
    # Total rank = 3*3*7 = 63? No, wait.
    # The matrix is 6x6. If we partition into 3x3 blocks of size 2x2, that's 9 blocks.
    # Each block multiplication C_ij = sum_k A_ik B_kj.
    # This is a standard block matrix multiplication.
    # The rank of standard block multiplication for 6x6 is 6^3 = 216.
    # Using Strassen on 2x2 blocks doesn't reduce the overall rank unless we treat the whole 6x6 as a larger structure or use a specific decomposition.
    
    # Actually, the prompt says: "Apply a 3x3 block partition where each 2x2 block multiplication is computed using the 7-term Strassen algorithm, yielding a total rank of 3^3 * 7 = 189"
    # Wait, 3^3 * 7 = 27 * 7 = 189.
    # How does that work?
    # If we view the 6x6 multiplication as a tensor, can we decompose it into 27 independent 2x2 multiplications?
    # No, standard block multiplication has cross-terms.
    # Unless... the "3x3 block partition" implies a Kronecker product structure or similar?
    # Let's re-read carefully: "3x3 block partition where each 2x2 block multiplication is computed using the 7-term Strassen algorithm"
    # This phrasing is slightly ambiguous. Does it mean we compute each of the 36 output blocks (2x2) independently? No, that's not how matrix mult works.
    
    # Let's look at the rank calculation: 189.
    # Current schoolbook rank is 216.
    # 189 < 216. So this is an improvement.
    # How to achieve rank 189?
    # Perhaps it refers to a specific known decomposition for 6x6?
    # Or maybe it's a misinterpretation of "3x3 block partition".
    # Let's try to construct a decomposition of rank 189.
    
    # One possibility: The 6x6 matrix multiplication tensor can be viewed as a composition.
    # But let's stick to the prompt's specific claim: "yielding a total rank of 3^3 * 7 = 189".
    # This suggests there are 27 "units" of computation, each of rank 7.
    # This looks like a tensor product of a 3x3 matrix multiplication (rank 27) with a 2x2 Strassen (rank 7)?
    # If we have a 3x3 matrix mult (rank 27) and we "inflate" each scalar multiplication by a 2x2 Strassen multiplication (rank 7), the total rank would be 27 * 7 = 189.
    # This corresponds to multiplying two block matrices where each block is 2x2, but the block structure is treated as a 3x3 matrix of 2x2 blocks, and the multiplication of the "block scalars" is done via standard 3x3 matrix mult, but each scalar product is replaced by a 2x2 matrix product?
    # No, that would be the Kronecker product of the two multiplication tensors.
    # The Kronecker product of the 3x3 mult tensor (rank 27) and the 2x2 mult tensor (rank 8, or 7 for Strassen) gives a tensor for multiplying 6x6 matrices where the entries are 2x2 blocks?
    # Actually, if we identify a 6x6 matrix with a 3x3 matrix of 2x2 blocks, the multiplication is:
    # C_{ij} = sum_{k} A_{ik} B_{kj}
    # Here A_{ik} and B_{kj} are 2x2 matrices. Their product is a 2x2 matrix product.
    # So the overall operation is:
    # 1. For each i,j,k (3x3x3 = 27 terms), compute the 2x2 matrix product A_{ik} B_{kj}.
    # 2. Sum the results for each i,j.
    # The rank of this operation is the sum of the ranks of the individual 2x2 products, because the summation is linear and the terms are independent in the tensor sense?
    # Actually, the tensor for 6x6 multiplication decomposes as the sum over k of (Tensor for 3x3 index mult) ⊗ (Tensor for 2x2 matrix mult).
    # The rank of a sum of tensors is at most the sum of the ranks.
    # The 3x3 index multiplication has rank 27 (schoolbook).
    # The 2x2 matrix multiplication has rank 7 (Strassen).
    # So the total rank is 27 * 7 = 189.
    
    # Let's construct this.
    # U, V, W will have dimensions:
    # U: (n*m, rank) = (36, 189)
    # V: (m*p, rank) = (36, 189)
    # W: (p*n, rank) = (36, 189)
    
    # The indices for U are (i, j) where i in 0..5, j in 0..5.
    # We can map (i, j) to (I, i_mod, J, j_mod) where I, J in 0..2 are block indices, i_mod, j_mod in 0..1 are within-block indices.
    # i = 2*I + i_mod
    # j = 2*J + j_mod
    
    # Similarly for V: (j, k) -> (J, j_mod, K, k_mod)
    # And W: (k, i) -> (K, k_mod, I, i_mod)  (Note: W index is k*n + i)
    
    # The decomposition is:
    # For each (I, J, K) in 0..2, we have a 2x2 matrix multiplication between block (I,K) of A and block (K,J) of B, contributing to block (I,J) of C.
    # Wait, the summation is over K.
    # C_{I,J} = sum_{K} A_{I,K} B_{K,J}
    # Each term A_{I,K} B_{K,J} is a 2x2 matrix product.
    # Let Strassen decomposition of 2x2 mult be defined by matrices U_s, V_s, W_s of size (4, 7) for s in 0..6 (7 terms).
    # Actually, standard Strassen uses 7 multiplications.
    # Let's define the Strassen matrices for 2x2.
    # Input A: [[a, b], [c, d]], Input B: [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-g)
    # m4 = d(h-g)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    # We need to express each multiplication m_t as a product of a row of U_t, a column of V_t, and a column of W_t?
    # No, the rank-1 decomposition of the 2x2 multiplication tensor T_2x2.
    # T_2x2 is a tensor of shape (4, 4, 4) corresponding to A, B, C.
    # We want T_2x2 = sum_{t=1}^7 u_t v_t^T w_t^T (in some index arrangement).
    # Let's use the standard Strassen decomposition to build U, V, W for the 2x2 case.
    
    # Indices for 2x2:
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Strassen terms:
    # m1 = (a+d)(e+h) -> A: a,d; B: e,h; C: c11, c22 (with signs)
    # This is getting complicated to map directly to the global U, V, W.
    
    # Alternative approach:
    # The global tensor T_6x6 is the sum over K=0..2 of T_block(K) where T_block(K) is the tensor for multiplying block A_{I,K} by B_{K,J} to contribute to C_{I,J}.
    # But actually, the structure is:
    # T_6x6[i, j, k] (where i is row of A, j is col of A/row of B, k is col of B)
    # Wait, standard matrix mult tensor T[i, j, k] contributes to C[i, k] via A[i, j] * B[j, k].
    # So the indices are (i, j, k) -> (i, k).
    # T_6x6 = sum_{j=0}^5 T_j where T_j is the tensor for fixing j.
    # This doesn't seem to lead to 189 directly.
    
    # Let's go back to the block interpretation.
    # i = 2*I + r, k = 2*J + s, j = 2*K + t.
    # C[i, k] = sum_{j} A[i, j] B[j, k]
    # C[2I+r, 2J+s] = sum_{K=0}^2 sum_{t=0}^1 A[2I+r, 2K+t] B[2K+t, 2J+s]
    # This is sum_{K=0}^2 (A_{I,K} B_{K,J})_{r,s}
    # So the total tensor is sum_{K=0}^2 T_{2x2}^{(K)} where T_{2x2}^{(K)} is the tensor for the 2x2 multiplication of blocks A_{I,K} and B_{K,J} contributing to C_{I,J}, but with the specific index mappings.
    # Since the 2x2 multiplication is the same for all K, and the index mappings are just permutations/shuffles, the rank is 3 * rank(2x2 mult).
    # rank(2x2 mult) = 7 (Strassen).
    # So total rank = 3 * 7 = 21? No, that's too low. 6x6 rank is at least 6^2 = 36? No, lower bound is max(n^2, m^2, p^2) = 36. Upper bound 216.
    # Why did the prompt say 189?
    # "3^3 * 7 = 189".
    # 3^3 = 27.
    # This implies there are 27 independent 2x2 multiplications.
    # But there are only 3 K values.
    # Unless... the "3x3 block partition" means something else.
    # Maybe it means we partition the 6x6 into 2x2 blocks, but we don't just do standard block mult.
    # What if we use a decomposition of the 3x3 matrix multiplication tensor (for the block indices) of rank 27, and for each of those 27 "scalar" multiplications, we replace it with a 2x2 matrix multiplication (rank 7)?
    # This is the Kronecker product of the 3x3 mult tensor and the 2x2 mult tensor.
    # The 3x3 mult tensor T_3 has rank 27 (schoolbook).
    # The 2x2 mult tensor T_2 has rank 7 (Strassen).
    # The Kronecker product T_3 ⊗ T_2 has rank 27 * 7 = 189.
    # Does T_3 ⊗ T_2 correspond to 6x6 matrix multiplication?
    # Yes, if we view a 6x6 matrix as a 3x3 matrix of 2x2 blocks, the multiplication is exactly the Kronecker product of the block index multiplication and the within-block multiplication.
    # C_{I,J}^{r,s} = sum_{K=0}^2 sum_{t=0}^1 (sum_{u=0}^1 A_{I,K}^{r,u} B_{K,J}^{u,s})? No.
    # C_{I,J} = sum_K A_{I,K} B_{K,J}
    # (A_{I,K} B_{K,J})_{r,s} = sum_u A_{I,K}^{r,u} B_{K,J}^{u,s}
    # So C_{I,J}^{r,s} = sum_K sum_u A_{I,K}^{r,u} B_{K,J}^{u,s}
    # This is exactly the Kronecker product structure.
    # The tensor for this operation is T_3 ⊗ T_2.
    # And its rank is rank(T_3) * rank(T_2) = 27 * 7 = 189.
    
    # So the plan is:
    # 1. Construct the schoolbook decomposition for 3x3 matrix multiplication (rank 27).
    #    U_3, V_3, W_3 of shape (9, 27), (9, 27), (9, 27).
    #    Indices for 3x3: i, j, k in 0..2.
    #    U_3[i, c] = 1 if c corresponds to (i, j, k) with i fixed?
    #    Standard schoolbook for 3x3:
    #    For each i, j, k in 0..2:
    #       c = i*9 + j*3 + k
    #       U_3[i, c] = 1
    #       V_3[j, c] = 1
    #       W_3[k, c] = 1  (Wait, W index for 3x3 mult is (k, i)? No, W is for output C[i, k]. So W index is i*3+k? Or k*3+i?
    #       Let's check the convention.
    #       For n x m * m x p -> n x p.
    #       U: (n*m, rank), V: (m*p, rank), W: (p*n, rank).
    #       For 3x3: n=3, m=3, p=3.
    #       U: (9, 27), V: (9, 27), W: (9, 27).
    #       U index: i*3 + j.
    #       V index: j*3 + k.
    #       W index: k*3 + i. (Because output is C[i, k], and W is indexed by (output_col, output_row) or similar?
    #       The prompt says: "output factor follows the official trace convention W[k*n+i,r]".
    #       So W index is k*n + i. For 3x3, W index is k*3 + i.
    
    # 2. Construct the Strassen decomposition for 2x2 matrix multiplication (rank 7).
    #    U_2, V_2, W_2 of shape (4, 7), (4, 7), (4, 7).
    #    Indices for 2x2:
    #    U index: r*2 + u (r is row of A, u is col of A).
    #    V index: u*2 + s (u is row of B, s is col of B).
    #    W index: s*2 + r (s is col of C, r is row of C). Convention: W[k*n+i] -> W[s*2+r].
    
    # 3. Combine them into U, V, W for 6x6.
    #    Rank R = 27 * 7 = 189.
    #    U: (36, 189).
    #    Global U index: i*6 + j.
    #    i = 2*I + r, j = 2*K + u.
    #    So global U index = (2*I + r)*6 + (2*K + u) = 12*I + 6*r + 4*K + 2*u.
    #    The column index c in the 189-dim space corresponds to (c_3, c_2) where c_3 in 0..26, c_2 in 0..6.
    #    c = c_3 * 7 + c_2.
    #    U[global_i, global_j, c] = U_3[I, c_3] * U_2[r, u, c_2]?
    #    Wait, U_3 is (9, 27). U_3[I, j_block, c_3]?
    #    In the 3x3 schoolbook, U_3[i, c] is non-zero only for one j, k.
    #    Specifically, for column c_3 corresponding to (I, K, ?), U_3[I, c_3] = 1.
    #    Actually, let's define the 3x3 schoolbook columns by (I, K, J_block)?
    #    Standard: For each I, J, K:
    #       U_3[I, c] = 1 (where c is index for I, J, K)
    #       V_3[J, c] = 1
    #       W_3[K, c] = 1 (with W index K*3+I)
    #    Here, the "scalar" multiplication is A[I, J] * B[J, K] -> C[I, K].
    #    In our 6x6 case, the "scalar" A[I, J] is actually a 2x2 block A_{I,J}.
    #    So the term A_{I,J} * B_{J,K} is a 2x2 matrix product.
    #    So for each (I, J, K), we have a 2x2 matrix product.
    #    There are 3*3*3 = 27 such terms.
    #    Each term has rank 7 (Strassen).
    #    Total rank 27 * 7 = 189.
    
    #    So, for each (I, J, K) in 0..2:
    #       Let c_3 be the index for this (I, J, K) in the 3x3 schoolbook.
    #       For each c_2 in 0..6 (Strassen term):
    #          c = c_3 * 7 + c_2
    #          U[global_idx, c] = U_2[r, u, c_2] * (1 if I matches, else 0)?
    #          Wait, U_3[I, c_3] is 1 for the specific I, J, K.
    #          So U[global_idx, c] = U_2[r, u, c_2] * U_3[I, c_3]
    #          But U_3[I, c_3] is 1 only for the I, J, K associated with c_3.
    #          So effectively, for a fixed c_3 (which fixes I, J, K), and for all r, u:
    #          U[(2*I+r)*6 + (2*K+u), c] = U_2[r, u, c_2]
    #          
    #          Similarly for V:
    #          V[global_idx, c] = V_2[u, s, c_2] * V_3[J, c_3]
    #          V_3[J, c_3] is 1 for the J associated with c_3.
    #          So for fixed c_3 (fixing I, J, K), and for all u, s:
    #          V[(2*K+u)*6 + (2*J+s), c] = V_2[u, s, c_2]
    #          
    #          Similarly for W:
    #          W[global_idx, c] = W_2[s, r, c_2] * W_3[K, c_3]
    #          W_3[K, c_3] is 1 for the K associated with c_3.
    #          W index is s*2 + r? No, W index is k*n + i.
    #          In 6x6, W index is k*6 + i.
    #          k = 2*J + s, i = 2*I + r.
    #          So W index = (2*J+s)*6 + (2*I+r).
    #          W_2 index is s*2 + r (since W_2 is for 2x2, index is col*2+row).
    #          So W[(2*J+s)*6 + (2*I+r), c] = W_2[s, r, c_2]
    
    # Let's implement this.
    
    # Step 1: Build Strassen matrices for 2x2.
    # U_2, V_2, W_2 are 4x7 matrices.
    # Indices:
    # U_2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V_2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W_2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> Wait, W index is col*2+row.
    # W_2 index 0: (0,0) -> col 0, row 0.
    # W_2 index 1: (0,1) -> col 0, row 1.
    # W_2 index 2: (1,0) -> col 1, row 0.
    # W_2 index 3: (1,1) -> col 1, row 1.
    
    U_2 = [[0]*7 for _ in range(4)]
    V_2 = [[0]*7 for _ in range(4)]
    W_2 = [[0]*7 for _ in range(4)]
    
    # Strassen formulas:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-g)
    # m4 = d(h-g)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # Let's fill U_2, V_2, W_2.
    # Column 0 (m1): (a+d)(e+h)
    # U_2[0,0]=1 (a), U_2[3,0]=1 (d)
    # V_2[0,0]=1 (e), V_2[3,0]=1 (h)
    # W_2[0,0]=1 (C11), W_2[3,0]=1 (C22)
    
    U_2[0][0] = 1; U_2[3][0] = 1
    V_2[0][0] = 1; V_2[3][0] = 1
    W_2[0][0] = 1; W_2[3][0] = 1
    
    # Column 1 (m2): (c+d)e
    # U_2[2,1]=1 (c), U_2[3,1]=1 (d)
    # V_2[0,1]=1 (e)
    # W_2[2,1]=1 (C21)
    
    U_2[2][1] = 1; U_2[3][1] = 1
    V_2[0][1] = 1
    W_2[2][1] = 1
    
    # Column 2 (m3): a(f-g)
    # U_2[0,2]=1 (a)
    # V_2[1,2]=1 (f), V_2[2,2]=-1 (g)
    # W_2[1,2]=1 (C12), W_2[3,2]=1 (C22)
    
    U_2[0][2] = 1
    V_2[1][2] = 1; V_2[2][2] = -1
    W_2[1][2] = 1; W_2[3][2] = 1
    
    # Column 3 (m4): d(h-g)
    # U_2[3,3]=1 (d)
    # V_2[3,3]=1 (h), V_2[2,3]=-1 (g)
    # W_2[0,3]=1 (C11), W_2[2,3]=1 (C21)
    
    U_2[3][3] = 1
    V_2[3][3] = 1; V_2[2][3] = -1
    W_2[0][3] = 1; W_2[2][3] = 1
    
    # Column 4 (m5): (a+b)h
    # U_2[0,4]=1 (a), U_2[1,4]=1 (b)
    # V_2[3,4]=1 (h)
    # W_2[0,4]=-1 (C11), W_2[1,4]=1 (C12)
    
    U_2[0][4] = 1; U_2[1][4] = 1
    V_2[3][4] = 1
    W_2[0][4] = -1; W_2[1][4] = 1
    
    # Column 5 (m6): (c-a)(e+f)
    # U_2[2,5]=1 (c), U_2[0,5]=-1 (a)
    # V_2[0,5]=1 (e), V_2[1,5]=1 (f)
    # W_2[3,5]=1 (C22)
    
    U_2[2][5] = 1; U_2[0][5] = -1
    V_2[0][5] = 1; V_2[1][5] = 1
    W_2[3][5] = 1
    
    # Column 6 (m7): (b-d)(g+h)
    # U_2[1,6]=1 (b), U_2[3,6]=-1 (d)
    # V_2[2,6]=1 (g), V_2[3,6]=1 (h)
    # W_2[0,6]=1 (C11)
    
    U_2[1][6] = 1; U_2[3][6] = -1
    V_2[2][6] = 1; V_2[3][6] = 1
    W_2[0][6] = 1
    
    # Step 2: Build 3x3 schoolbook indices.
    # For I, J, K in 0..2:
    # c_3 = I*9 + J*3 + K
    # U_3[I, c_3] = 1
    # V_3[J, c_3] = 1
    # W_3[K, c_3] = 1 (W index is K*3+I)
    
    # Step 3: Construct global U, V, W.
    rank = 189
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    for I in range(3):
        for J in range(3):
            for K in range(3):
                c_3 = I*9 + J*3 + K
                for c_2 in range(7):
                    c = c_3 * 7 + c_2
                    # Fill U
                    for r in range(2):
                        for u in range(2):
                            global_i = 2*I + r
                            global_j = 2*K + u
                            idx_u = global_i * 6 + global_j
                            U[idx_u][c] = U_2[r*2+u][c_2]
                    
                    # Fill V
                    for u in range(2):
                        for s in range(2):
                            global_j = 2*K + u
                            global_k = 2*J + s
                            idx_v = global_j * 6 + global_k
                            V[idx_v][c] = V_2[u*2+s][c_2]
                    
                    # Fill W
                    for s in range(2):
                        for r in range(2):
                            global_k = 2*J + s
                            global_i = 2*I + r
                            idx_w = global_k * 6 + global_i
                            W[idx_w][c] = W_2[s*2+r][c_2]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
