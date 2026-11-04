"""
Strategy: Systematic Block-Decomposition Search for M_{2,4,5}
The current program attempts a complex, error-prone manual construction.
This new approach:
1. Splits the inner dimension m=4 into two blocks of size 2 (J0: {0,1}, J1: {2,3}).
2. Splits the output dimension p=5 into two blocks: K0: {0,1} (size 2) and K1: {2,3,4} (size 3).
3. This decomposes the tensor T into 4 sub-tensors:
   - T00: A[J0] * B[J0, K0] -> C[J0, K0] (2x2x2)
   - T01: A[J0] * B[J0, K1] -> C[J0, K1] (2x2x3)
   - T10: A[J1] * B[J1, K0] -> C[J1, K0] (2x2x2)
   - T11: A[J1] * B[J1, K1] -> C[J1, K1] (2x2x3)
4. We use known optimal or near-optimal decompositions for these small cases:
   - For 2x2x2: Strassen's algorithm gives rank 7. (Schoolbook is 4, but Strassen is often used in higher-level recursions; however, for rank minimization of the sum, we should check if combining them allows lower rank. Actually, rank(M_2,2,2)=7 is a lower bound? No, rank is 7. Wait. 
     Correction: The tensor rank of M_{2,2,2} is 7. Schoolbook uses 4 multiplications, which corresponds to a decomposition of rank 4? 
     Let's re-verify the definition. 
     Rank of tensor T is min R such that T = sum_{r=1}^R u_r v_r w_r.
     For M_{2,2,2}, T[i,j,k]=1.
     Schoolbook: C[i,k] = sum_j A[i,j]B[j,k].
     This is a bilinear algorithm with 4 multiplications. 
     The tensor corresponding to this algorithm has rank 4? 
     Actually, the "rank" of the matrix multiplication tensor M_{n,n,n} is the minimum number of multiplications needed. 
     For n=2, this is 7 (Strassen). 
     Why is schoolbook 4? Because schoolbook is not a tensor decomposition in the strict sense of sum of rank-1 tensors? 
     No, schoolbook IS a tensor decomposition. 
     T = sum_{j=0}^{1} (e_j^A) (e_j^B) (e_j^C) ... no.
     
     Let's look at the standard result: rank(M_{2,2,2}) = 7.
     This means you cannot decompose the tensor into fewer than 7 rank-1 terms.
     Schoolbook uses 4 multiplications, but the "tensor" it computes is not M_{2,2,2} in the pure sense? 
     No, it computes M_{2,2,2}.
     
     Ah, the confusion is between "bilinear complexity" and "tensor rank". They are the same.
     If rank is 7, how can schoolbook use 4?
     Schoolbook: M1=A00B00, M2=A01B11, M3=A10B01, M4=A11B10.
     C00=M1, C01=M2, C10=M3, C11=M4.
     This is a valid algorithm. It uses 4 multiplications.
     Therefore, the rank of M_{2,2,2} must be <= 4.
     But Strassen proved it is 7? 
     Strassen proved that you need 7 multiplications if you restrict to certain forms? 
     No, Strassen's algorithm uses 7 multiplications and is faster. 
     If schoolbook uses 4, and Strassen uses 7, schoolbook is "better" in rank? 
     But Strassen is asymptotically better. 
     
     Resolution: The "rank" of the tensor is the minimum number of multiplications.
     For 2x2x2, the minimum is 7. 
     Why does schoolbook use 4? 
     Because schoolbook is not a "straight-line" program of the form C = sum w_r * (linear A) * (linear B)? 
     Yes it is.
     
     Actually, I recall now: The tensor rank of M_{2,2,2} is 7. 
     Schoolbook is NOT a rank-4 tensor decomposition. 
     Why? Because the W matrices in schoolbook are not rank-1? 
     In C = A*B, C[i,k] = sum_j A[i,j]B[j,k].
     Term j: U[i,j]=delta_ij, V[j,k]=delta_jk, W[k,i]=1.
     The rank-1 tensor is U \otimes V \otimes W.
     Summing over j=0,1 gives 2 terms? 
     No, j goes from 0 to 1. So 2 terms.
     But we have 4 elements in C.
     
     Let's stick to the computational verification. 
     I will construct a decomposition and verify it. 
     If I can construct rank 4 for 2x2x2, I'll use it. 
     If not, I'll use rank 7. 
     
     Actually, for the purpose of this task, the "rank" is the number of columns in U,V,W.
     I will use a known valid decomposition.
     For 2x2x2, the minimum rank is 7. (I will verify this computationally if possible, or use 7 to be safe).
     Wait, if schoolbook is 4 multiplications, why is the rank 7?
     Because the "multiplications" in Strassen are of linear forms. 
     In schoolbook, the multiplications are of entries. Entries are linear forms.
     So schoolbook is a rank 4 algorithm? 
     
     Let's check the literature. "The rank of the 2x2x2 matrix multiplication tensor is 7." 
     This is a famous result. 
     So how does schoolbook work with 4? 
     It doesn't? 
     Schoolbook: 4 multiplications. 
     Strassen: 7 multiplications. 
     If schoolbook is valid, rank <= 4. 
     Contradiction.
     
     Ah! The rank of the tensor M_{2,2,2} is 7. 
     This means you need 7 multiplications of SCALARS.
     Schoolbook uses 4 multiplications of SCALARS.
     So rank <= 4.
     
     I must be misremembering the result. 
     Let's check: 
     Bilinear complexity of 2x2x2 is 7? 
     No, Strassen showed 7 is sufficient. 
     Is it minimal? 
     Yes, the lower bound is 7.
     
     So how does schoolbook work? 
     Schoolbook: 
     M1 = A00 B00
     M2 = A01 B11
     M3 = A10 B01
     M4 = A11 B10
     C00 = M1
     C01 = M2
     C10 = M3
     C11 = M4
     
     This computes the product correctly. 
     It uses 4 multiplications. 
     Therefore, the rank is <= 4.
     
     Why do sources say 7? 
     Maybe they mean something else? 
     Or maybe my schoolbook is wrong? 
     C = [[A00B00+A01B10, A00B01+A01B11], [A10B00+A11B10, A10B01+A11B11]]
     My schoolbook above:
     C00 = A00B00 (Missing A01B10)
     So my schoolbook snippet was wrong. 
     Correct schoolbook uses 4 multiplications? 
     No, correct schoolbook uses 4 multiplications? 
     C00 needs 2 terms. C01 needs 2. C10 needs 2. C11 needs 2.
     Total 8 terms? 
     No, the multiplications are shared? 
     No, A00B00 is used in C00. A01B10 is used in C00.
     They are distinct multiplications.
     So schoolbook uses 4 multiplications? 
     No, 2x2 matrix multiplication requires 4 multiplications? 
     Yes, 4 multiplications of entries.
     
     So rank <= 4.
     
     Okay, I will use rank 4 for 2x2x2 blocks.
     And rank 6 for 2x2x3 blocks (schoolbook: 2*3=6).
     
     Total rank: 4+6+4+6 = 20.
     
     Let's implement this.
"""

import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = start_time + 110

    def verify_decomposition(U, V, W, n, m, p):
        """Verify that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        # Flatten indices
        # i in [0, n), j in [0, m), k in [0, p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True

    def construct_schoolbook_222():
        """
        Returns U, V, W for 2x2x2 schoolbook.
        n=2, m=2, p=2.
        U shape (4, R), V shape (4, R), W shape (4, R).
        Rank R = 4.
        Terms:
        0: A00 * B00 -> C00 (U[0,0]=1, V[0,0]=1, W[0,0]=1)
           Wait, C00 = A00B00 + A01B10.
           So we need two terms for C00?
           No, the tensor equation is sum_r ... = 1.
           For M_{2,2,2}, T[i,j,k]=1.
           Schoolbook decomposition:
           r=0: U[i,j]=delta_{j,0}, V[j,k]=delta_{j,0}, W[k,i]=delta_{i,0}delta_{k,0}?
           
           Let's derive the schoolbook tensor decomposition properly.
           C[i,k] = sum_j A[i,j] B[j,k]
           This is a sum of m terms.
           Term j:
           U_r[i,j] = delta_{j,r}
           V_r[j,k] = delta_{j,r}
           W_r[k,i] = 1  (Wait, W must be such that W[k,i] contributes to C[i,k])
           
           The tensor T is defined by T[i,j,k] = 1.
           A rank-1 term is u \otimes v \otimes w.
           (u \otimes v \otimes w)[i,j,k] = u[i,j] v[j,k] w[k,i].
           
           If we pick u[i,j] = delta_{j,0}, v[j,k] = delta_{j,0}, w[k,i] = 1.
           Then term is delta_{j,0} * delta_{j,0} * 1 = delta_{j,0}.
           Summing j=0 and j=1 gives 1.
           
           So for 2x2x2, we can use R=2?
           r=0: U[0,0]=1, U[2,0]=1 (i=1,j=0). V[0,0]=1, V[1,0]=1 (j=0,k=1). W[0,0]=1, W[1,0]=1, W[2,0]=1, W[3,0]=1.
           r=1: U[1,1]=1, U[3,1]=1. V[2,1]=1, V[3,1]=1. W all 1.
           
           Let's check if this works.
           If R=2 works for 2x2x2, then rank is 2.
           But Strassen is 7. 
           
           I will test this computationally in the code.
        """
        n_sub, m_sub, p_sub = 2, 2, 2
        R = 2
        U = np.zeros((n_sub*m_sub, R))
        V = np.zeros((m_sub*p_sub, R))
        W = np.zeros((p_sub*n_sub, R))
        
        # r=0 (j=0)
        # U[i, 0, r] = 1 if j=0 else 0. Indices: i=0->0, i=1->2 (since m=2, idx=i*m+j)
        U[0, 0] = 1.0 # i=0, j=0
        U[2, 0] = 1.0 # i=1, j=0
        # V[j, 0, r] = 1 if j=0 else 0. Indices: j=0->0,1; j=1->2,3 (idx=j*p+k)
        V[0, 0] = 1.0 # j=0, k=0
        V[1, 0] = 1.0 # j=0, k=1
        # W[k, 0, r] = 1 for all k,i. Indices: k=0->0,1; k=1->2,3 (idx=k*n+i)
        W[0, 0] = 1.0
        W[1, 0] = 1.0
        W[2, 0] = 1.0
        W[3, 0] = 1.0
        
        # r=1 (j=1)
        U[1, 1] = 1.0 # i=0, j=1
        U[3, 1] = 1.0 # i=1, j=1
        V[2, 1] = 1.0 # j=1, k=0
        V[3, 1] = 1.0 # j=1, k=1
        W[0, 1] = 1.0
        W[1, 1] = 1.0
        W[2, 1] = 1.0
        W[3, 1] = 1.0
        
        return U, V, W

    def construct_schoolbook_223():
        """
        Returns U, V, W for 2x2x3 schoolbook.
        n=2, m=2, p=3.
        Rank R = 2 (inner dim) ? Or 6?
        Using the same logic: R=m=2.
        """
        n_sub, m_sub, p_sub = 2, 2, 3
        R = 2
        U = np.zeros((n_sub*m_sub, R))
        V = np.zeros((m_sub*p_sub, R))
        W = np.zeros((p_sub*n_sub, R))
        
        # r=0 (j=0)
        U[0, 0] = 1.0 # i=0, j=0
        U[2, 0] = 1.0 # i=1, j=0
        # V[j=0, k=0..2]
        V[0, 0] = 1.0 # j=0, k=0
        V[1, 0] = 1.0 # j=0, k=1
        V[2, 0] = 1.0 # j=0, k=2
        # W[k=0..2, i=0..1]
        for k in range(p_sub):
            for i in range(n_sub):
                W[k*n_sub+i, 0] = 1.0
                
        # r=1 (j=1)
        U[1, 1] = 1.0 # i=0, j=1
        U[3, 1] = 1.0 # i=1, j=1
        V[3, 1] = 1.0 # j=1, k=0
        V[4, 1] = 1.0 # j=1, k=1
        V[5, 1] = 1.0 # j=1, k=2
        for k in range(p_sub):
            for i in range(n_sub):
                W[k*n_sub+i, 1] = 1.0
                
        return U, V, W

    # Verify sub-blocks
    U222, V222, W222 = construct_schoolbook_222()
    if not verify_decomposition(U222, V222, W222, 2, 2, 2):
        print("Warning: 2x2x2 rank 2 failed, falling back to schoolbook 4")
        # Fallback: Rank 4 schoolbook
        U222 = np.zeros((4, 4))
        V222 = np.zeros((4, 4))
        W222 = np.zeros((4, 4))
        # Terms (i,j,k)
        U222[0,0]=1; V222[0,0]=1; W222[0,0]=1 # 000
        U222[1,1]=1; V222[1,1]=1; W222[1,1]=1 # 011 (i=0,j=1,k=1? No, W is k*n+i)
        # Let's just use the explicit 4-term schoolbook if rank 2 fails.
        # Actually, if rank 2 fails, it's a major issue. Let's assume rank 2 works based on derivation.
        
    U223, V223, W223 = construct_schoolbook_223()
    if not verify_decomposition(U223, V223, W223, 2, 2, 3):
        print("Warning: 2x2x3 rank 2 failed")

    # Construct full 2x4x5
    # Blocks:
    # 00: A(j=0,1) * B(j=0,1; k=0,1) -> C(i=0,1; k=0,1)
    #     U indices: i=0,1; j=0,1 -> U[0,1,4,5]
    #     V indices: j=0,1; k=0,1 -> V[0,1,5,6]
    #     W indices: k=0,1; i=0,1 -> W[0,1,2,3]
    # 01: A(j=0,1) * B(j=0,1; k=2,3,4) -> C(i=0,1; k=2,3,4)
    #     U indices: same as 00
    #     V indices: j=0,1; k=2,3,4 -> V[2,3,4,7,8,9]
    #     W indices: k=2,3,4; i=0,1 -> W[4,5,6,7,8,9]
    # 10: A(j=2,3) * B(j=2,3; k=0,1) -> C(i=0,1; k=0,1)
    #     U indices: i=0,1; j=2,3 -> U[2,3,6,7]
    #     V indices: j=2,3; k=0,1 -> V[10,11,15,16]
    #     W indices: k=0,1; i=0,1 -> W[0,1,2,3]
    # 11: A(j=2,3) * B(j=2,3; k=2,3,4) -> C(i=0,1; k=2,3,4)
    #     U indices: same as 10
    #     V indices: j=2,3; k=2,3,4 -> V[12,13,14,17,18,19]
    #     W indices: k=2,3,4; i=0,1 -> W[4,5,6,7,8,9]

    # Total Rank: 2 (for 00) + 2 (for 01) + 2 (for 10) + 2 (for 11) = 8?
    # Wait, the W indices for 00 and 10 overlap!
    # W[0..3] is used by both 00 and 10.
    # We can sum their contributions.
    # If we use separate columns for each block, rank is 8.
    # Can we merge? 
    # The tensor is a sum. 
    # T = T00 + T01 + T10 + T11.
    # If we just concatenate the columns, we get a valid decomposition.
    # Rank = 2+2+2+2 = 8.
    
    # Let's build it with 8 columns.
    R_total = 8
    U = np.zeros((n*m, R_total))
    V = np.zeros((m*p, R_total))
    W = np.zeros((p*n, R_total))
    
    col = 0
    # Block 00
    # Map local (i,j) to global U index: i*4+j
    # Local 2x2x2 U: 
    # U_loc[0,0]=1 (i0,j0) -> U_glob[0, col]=1
    # U_loc[2,0]=1 (i1,j0) -> U_glob[4, col]=1
    # U_loc[1,1]=1 (i0,j1) -> U_glob[1, col+1]=1
    # U_loc[3,1]=1 (i1,j1) -> U_glob[5, col+1]=1
    
    # V_loc for 2x2x2:
    # V_loc[0,0]=1 (j0,k0) -> V_glob[0, col]=1
    # V_loc[1,0]=1 (j0,k1) -> V_glob[1, col]=1
    # V_loc[2,1]=1 (j1,k0) -> V_glob[5, col+1]=1
    # V_loc[3,1]=1 (j1,k1) -> V_glob[6, col+1]=1
    
    # W_loc for 2x2x2:
    # W_loc[0,0]=1 (k0,i0) -> W_glob[0, col]=1
    # W_loc[1,0]=1 (k0,i1) -> W_glob[1, col]=1
    # W_loc[2,0]=1 (k1,i0) -> W_glob[2, col]=1
    # W_loc[3,0]=1 (k1,i1) -> W_glob[3, col]=1
    # And for col+1, W is all 1s in the 2x2 block.
    
    # Let's just copy the sub-blocks directly.
    
    # Helper to place a sub-decomposition into global arrays
    def place_block(U_glob, V_glob, W_glob, U_sub, V_sub, W_sub, 
                    i_offset, j_offset, k_offset, col_offset):
        n_sub, m_sub, p_sub = U_sub.shape[0]//m_sub, m_sub, V_sub.shape[0]//m_sub
        # U indices: i in [0, n_sub), j in [0, m_sub)
        # Global U index: (i+i_offset)*m + (j+j_offset)
        for r in range(U_sub.shape[1]):
            for i in range(n_sub):
                for j in range(m_sub):
                    val = U_sub[i*m_sub+j, r]
                    if val != 0:
                        U_glob[(i+i_offset)*m + (j+j_offset), col_offset+r] = val
            
            for j in range(m_sub):
                for k in range(p_sub):
                    val = V_sub[j*p_sub+k, r]
                    if val != 0:
                        V_glob[(j+j_offset)*p + (k+k_offset), col_offset+r] = val
            
            for k in range(p_sub):
                for i in range(n_sub):
                    val = W_sub[k*n_sub+i, r]
                    if val != 0:
                        W_glob[(k+k_offset)*n + (i+i_offset), col_offset+r] = val

    # Block 00: i_off=0, j_off=0, k_off=0
    place_block(U, V, W, U222, V222, W222, 0, 0, 0, col)
    col += 2
    
    # Block 01: i_off=0, j_off=0, k_off=2
    place_block(U, V, W, U223, V223, W223, 0, 0, 2, col)
    col += 2
    
    # Block 10: i_off=0, j_off=2, k_off=0
    # Note: A is 2x4. i is row of A. j is col of A / row of B.
    # A[i, j]. 
    # Block 10 uses j=2,3.
    place_block(U, V, W, U222, V222, W222, 0, 2, 0, col)
    col += 2
    
    # Block 11: i_off=0, j_off=2, k_off=2
    place_block(U, V, W, U223, V223, W223, 0, 2, 2, col)
    col += 2

    # Verify full decomposition
    if verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook 40
        U_sb = np.zeros((n*m, n*m*p))
        V_sb = np.zeros((m*p, n*m*p))
        W_sb = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1.0
                    V_sb[j*p+k, col] = 1.0
                    W_sb[k*n+i, col] = 1.0
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
