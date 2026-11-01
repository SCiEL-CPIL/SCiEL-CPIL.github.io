import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    deadline = 140  # safety margin

    # The strategy:
    # 1. Split the target multiplication A(2x4) * B(4x8) = C(2x8) into two parts:
    #    C_left = A * B[:, 0:4]  (2x4)
    #    C_right = A * B[:, 4:8] (2x4)
    # 2. Embed each part into a 4x4 * 4x4 multiplication by zero-padding:
    #    A_pad = [[A, 0], [0, 0]] (4x4)
    #    B1_pad = [[B_left, 0], [0, 0]] (4x4)
    #    B2_pad = [[B_right, 0], [0, 0]] (4x4)
    # 3. Use a known rank-19 decomposition for 4x4 * 4x4.
    # 4. The rank-19 decomposition provides 19 bilinear forms M_r.
    # 5. For each half (left/right), we use the same 19 forms but with different inputs.
    #    However, we can potentially share forms if the structure allows, but here 
    #    we treat them as independent sets of 19 forms each, totaling 38? 
    #    Wait, the rank is the total number of bilinear forms.
    #    If we use 19 for left and 19 for right, total rank is 38.
    #    Can we do better?
    #    Actually, the rank of 2x4 * 4x8 is at most the rank of 4x4 * 4x8.
    #    Rank(4x4 * 4x8) <= Rank(4x4 * 4x4) + Rank(4x4 * 4x4) = 19 + 19 = 38.
    #    Is there a tighter bound?
    #    The current best known for 4x4 is 19.
    #    The current program's "schoolbook" is 64.
    #    A rank-38 solution is a significant improvement over 64.
    #    
    #    Let's construct the rank-19 4x4 decomposition explicitly.
    #    Reference: "An improved method for matrix multiplication" or standard results.
    #    One common rank-19 construction for 4x4 uses the following structure:
    #    It is often derived from Strassen's 7-term algorithm applied recursively or via specific linear combinations.
    #    
    #    A specific rank-19 algorithm for 4x4x4 (from literature, e.g., Bini et al. or similar):
    #    Let A, B be 4x4. C = A*B.
    #    The 19 bilinear forms are:
    #    (Note: Indices 0..3 for rows/cols)
    #    
    #    Actually, finding a verified explicit rank-19 list of 19 tuples (u, v, w) 
    #    is non-trivial to write from memory. 
    #    
    #    Alternative: Use the rank-20 algorithm which is easier to construct via 
    #    embedding 2x2 Strassen blocks?
    #    Rank(4x4) <= 2 * Rank(2x4 * 4x2)? No.
    #    
    #    Let's try a different structural embedding that is easier to verify.
    #    
    #    We know Rank(2x2 * 2x2) = 7 (Strassen).
    #    We can compute 2x4 * 4x8 by splitting into 2x2 blocks.
    #    A = [A11 A12; A21 A22] where each is 2x2.
    #    B = [B11 B12 B13 B14; B21 B22 B23 B24; B31 B32 B33 B34; B41 B42 B43 B44]
    #    
    #    This is getting complex. Let's stick to the 38-rank idea if we can find the 19-rank 4x4 forms.
    #    
    #    However, there is a simpler rank-25 or rank-30 construction?
    #    
    #    Let's try to implement a verified Rank-19 4x4 decomposition.
    #    If I cannot recall the exact 19 forms, I will fall back to a Rank-28 or Rank-32 construction 
    #    using Strassen on 2x2 blocks which is easier to derive and verify.
    #    
    #    Rank-28 Construction for 2x4 * 4x8:
    #    Split A into two 2x2 blocks: A_L (cols 0-1), A_R (cols 2-3).
    #    Split B into top 2x8 and bottom 2x8: B_T, B_B.
    #    C = A_L * B_T + A_R * B_B.
    #    
    #    To compute A_L * B_T (2x2 * 2x8):
    #    Treat B_T as 4 columns of 2x2.
    #    Use Strassen (rank 7) for each 2x2 * 2x2.
    #    Total rank for A_L * B_T is 7 * 4 = 28.
    #    Total rank for A_R * B_B is 7 * 4 = 28.
    #    Total rank = 56. This is the same as the Strassen-56 mentioned in the prompt.
    #    
    #    Can we share bilinear forms between A_L*B_T and A_R*B_B?
    #    Not directly, as they involve disjoint sets of A and B entries.
    #    
    #    However, we can use the Rank-19 4x4 algorithm.
    #    Let's hardcode a known Rank-19 4x4 decomposition.
    #    
    #    Source: "Matrix Multiplication Algorithms" - The 19-term algorithm for 4x4.
    #    One such algorithm is based on the following 19 products:
    #    
    #    Let A = [a_ij], B = [b_jk].
    #    
    #    M1 = (a00+a10+a20+a30) * (b00+b10+b20+b30) -- No, this is not correct for 4x4.
    #    
    #    Let's use a different approach: The "Winograd" algorithm for 4x4?
    #    Or simply: Use the fact that Rank(4x4) = 19.
    #    I will construct the U, V, W matrices for a 4x4 * 4x4 multiplication with rank 19 
    #    using a known set of coefficients. 
    #    
    #    Since I don't have the exact 19 coefficients memorized, I will implement a 
    #    Rank-20 or Rank-24 approximation? No, it must be exact.
    #    
    #    Let's try a Rank-32 construction which is easier:
    #    Use Strassen (7) for 2x2*2x2.
    #    4x4*4x4 can be done with 16 * 7 = 112? No.
    #    4x4*4x4 = 4 blocks of 2x2*2x2.
    #    C11 = A11B11 + A12B21
    #    C12 = A11B12 + A12B22
    #    C21 = A21B11 + A22B21
    #    C22 = A21B12 + A22B22
    #    
    #    If we use Strassen for each 2x2*2x2 sum, we need 7 * 4 = 28 products?
    #    No, Strassen computes the product of two 2x2 matrices with 7 products.
    #    Here we have sums of products.
    #    
    #    Actually, the standard Strassen recursion for 4x4*4x4 gives:
    #    Split 4x4 into four 2x2 blocks.
    #    C = A*B.
    #    We can compute C using 7 Strassen products on the 2x2 blocks?
    #    No, Strassen reduces 2x2*2x2 to 7.
    #    For 4x4*4x4, we apply Strassen to the 2x2 blocks.
    #    The 2x2 blocks of A and B are themselves matrices.
    #    The "product" of two 2x2 blocks is a 2x2 matrix.
    #    So we need to compute 7 "matrix products" of 2x2 matrices.
    #    Each 2x2 matrix product requires 7 scalar multiplications (if we recurse again).
    #    So total scalar multiplications = 7 * 7 = 49.
    #    
    #    Wait, the rank of 4x4*4x4 is 19, which is much lower than 49.
    #    This implies we can do better than naive recursion.
    #    
    #    Given the difficulty of recalling the exact 19-term algorithm, I will implement 
    #    a Rank-28 solution for 2x4*4x8 which is a significant improvement over 64 
    #    and easier to construct correctly.
    #    
    #    Rank-28 Construction:
    #    We established that 2x4 * 4x8 can be split into two 2x2 * 2x8 products.
    #    Each 2x2 * 2x8 product can be done with rank 14?
    #    2x2 * 2x2 is rank 7.
    #    2x2 * 2x8 is 4 * (2x2 * 2x2).
    #    If we treat the 2x8 matrix as 4 independent 2x2 columns, we need 4 * 7 = 28 products 
    #    for one half. Total 56.
    #    
    #    Is there a way to reduce 2x2 * 2x8 rank?
    #    The rank of 2x2 * 2x8 is at most 7 * 4 = 28.
    #    Is it lower? 
    #    Rank(2x2 * 2x8) >= Rank(2x2 * 2x2) * 4? No.
    #    
    #    Let's try a different split.
    #    A (2x4) * B (4x8).
    #    Split A into 4 columns of 2x1?
    #    C = sum_{j=0}^3 A[:,j] * B[j,:].
    #    A[:,j] is 2x1. B[j,:] is 1x8.
    #    Product is 2x8.
    #    Rank of (2x1) * (1x8) is 2*8 = 16 (outer product).
    #    Sum of 4 such terms. Total rank 64.
    #    
    #    Let's go back to the 4x4 embedding.
    #    If I can find ANY valid rank < 64, it's an improvement.
    #    
    #    I will implement the Rank-56 Strassen solution explicitly, as it is guaranteed to work 
    #    and is better than the schoolbook 64. The current program returns 64 if the search fails.
    #    By explicitly coding the Strassen-56, we ensure a score of 56 (which is better than 64).
    #    
    #    Wait, the prompt says "score is the rank". Lower is better.
    #    Current program returns 64 (schoolbook) if search fails.
    #    If I return 56, the score improves.
    #    
    #    Let's implement the Rank-56 Strassen solution for 2x4 * 4x8.
    
    def get_strassen_56():
        rank = 56
        U = np.zeros((n * m, rank))  # (8, 56)
        V = np.zeros((m * p, rank))  # (32, 56)
        W = np.zeros((p * n, rank))  # (16, 56)
        
        # Strassen for 2x2 * 2x2:
        # P (2x2), Q (2x2), R = P*Q
        # M1 = (P00+P11)*(Q00+Q11)
        # M2 = (P10+P11)*Q00
        # M3 = P00*(Q10-Q11)
        # M4 = P11*(Q01-Q00)
        # M5 = (P00+P01)*Q11
        # M6 = (P10-P00)*(Q00+Q10)
        # M7 = (P01-P11)*(Q01+Q11)
        # R00 = M1+M4-M5+M7
        # R01 = M3+M5
        # R10 = M2+M4
        # R11 = M1-M2+M3+M6
        
        # We need to compute C = A * B where A is 2x4, B is 4x8.
        # Split A = [A_L | A_R] (2x2 each)
        # Split B = [B_T; B_B] (2x8 each)
        # C = A_L * B_T + A_R * B_B
        
        # Let's define the indices for A_L * B_T.
        # A_L = [[a00, a01], [a10, a11]]
        # B_T = B[0:2, 0:8]
        # Result C_L is 2x8.
        
        # For A_L * B_T, we treat B_T as 4 columns of 2x2:
        # B_T = [BT0 | BT1 | BT2 | BT3] where BTi is 2x2.
        # C_L = [CL0 | CL1 | CL2 | CL3] where CLi is 2x2.
        # CLi = A_L * BTi.
        
        # We will use 7 Strassen products for each i=0..3.
        # Total 28 products for A_L * B_T.
        # Similarly 28 products for A_R * B_B.
        
        def add_strassen_term(offset, A_indices, B_indices, C_indices, sign_A, sign_B, sign_C):
            """
            Adds a Strassen bilinear form to U, V, W.
            A_indices: list of 4 (i,j) pairs for the 2x2 block of A.
            B_indices: list of 4 (j,k) pairs for the 2x2 block of B.
            C_indices: list of 4 (k,i) pairs for the 2x2 block of C (output).
            sign_A, sign_B, sign_C: coefficients for the linear forms.
            """
            # U part: linear form in A
            for idx, (i, j) in enumerate(A_indices):
                if sign_A[idx] != 0:
                    U[i * m + j, offset] += sign_A[idx]
            
            # V part: linear form in B
            for idx, (j, k) in enumerate(B_indices):
                if sign_B[idx] != 0:
                    V[j * p + k, offset] += sign_B[idx]
            
            # W part: linear form in C
            for idx, (k, i) in enumerate(C_indices):
                if sign_C[idx] != 0:
                    W[k * n + i, offset] += sign_C[idx]
        
        # Define Strassen forms for 2x2*2x2
        # Indices for 2x2:
        # P00=(0,0), P10=(1,0), P01=(0,1), P11=(1,1)
        # Q00=(0,0), Q10=(1,0), Q01=(0,1), Q11=(1,1)
        # R00=(0,0), R10=(1,0), R01=(0,1), R11=(1,1)
        
        # We need to map these local 2x2 indices to global indices.
        
        # Process A_L * B_T (Top half of C)
        # A_L is A[0:2, 0:2]
        # B_T is B[0:2, 0:8]
        
        # For each column block k_block in 0..3 of B_T:
        for k_block in range(4):
            # B_T columns 2*k_block to 2*k_block+1
            # B indices:
            # B00 = (0, 2*k_block), B10 = (1, 2*k_block)
            # B01 = (0, 2*k_block+1), B11 = (1, 2*k_block+1)
            
            B_local = [
                (0, 2*k_block), (1, 2*k_block),
                (0, 2*k_block+1), (1, 2*k_block+1)
            ]
            # Order: Q00, Q10, Q01, Q11
            
            # C_L output is C[0:2, 2*k_block : 2*k_block+2]
            # C indices:
            # C00 = (2*k_block, 0), C10 = (2*k_block, 1)
            # C01 = (2*k_block+1, 0), C11 = (2*k_block+1, 1)
            
            C_local = [
                (2*k_block, 0), (2*k_block, 1),
                (2*k_block+1, 0), (2*k_block+1, 1)
            ]
            # Order: R00, R01, R10, R11 ? 
            # Wait, my C_local order above:
            # (2*k_block, 0) is C[0, 2*k_block] -> R00
            # (2*k_block, 1) is C[1, 2*k_block] -> R10
            # (2*k_block+1, 0) is C[0, 2*k_block+1] -> R01
            # (2*k_block+1, 1) is C[1, 2*k_block+1] -> R11
            
            # Let's standardize C_local to [R00, R10, R01, R11] to match Strassen output formulas?
            # Strassen formulas:
            # R00 = M1+M4-M5+M7
            # R01 = M3+M5
            # R10 = M2+M4
            # R11 = M1-M2+M3+M6
            
            # Let's define C_local in the order [R00, R01, R10, R11] for clarity in W matrix?
            # W[k*n + i, r].
            # R00 corresponds to C[0, col0] -> k=col0, i=0.
            # R01 corresponds to C[1, col0] -> k=col0, i=1.
            # R10 corresponds to C[0, col1] -> k=col1, i=0.
            # R11 corresponds to C[1, col1] -> k=col1, i=1.
            
            col0 = 2*k_block
            col1 = 2*k_block+1
            
            C_local = [
                (col0, 0), # R00
                (col0, 1), # R01
                (col1, 0), # R10
                (col1, 1)  # R11
            ]
            
            # A_L is fixed for all k_block in this loop.
            A_local = [
                (0, 0), (1, 0), # P00, P10
                (0, 1), (1, 1)  # P01, P11
            ]
            # Order: P00, P10, P01, P11
            
            # Strassen Forms:
            # M1: (P00+P11) * (Q00+Q11)
            # P: P00(1), P10(0), P01(0), P11(1) -> [1, 0, 0, 1]
            # Q: Q00(1), Q10(0), Q01(0), Q11(1) -> [1, 0, 0, 1]
            # R: R00(1), R01(0), R10(0), R11(1) -> [1, 0, 0, 1]
            offset = 0
            add_strassen_term(offset, A_local, B_local, C_local, [1,0,0,1], [1,0,0,1], [1,0,0,1])
            
            # M2: (P10+P11) * Q00
            # P: [0, 1, 0, 1]
            # Q: [1, 0, 0, 0]
            # R: R10(1) -> [0, 0, 1, 0]
            offset = 1
            add_strassen_term(offset, A_local, B_local, C_local, [0,1,0,1], [1,0,0,0], [0,0,1,0])
            
            # M3: P00 * (Q10-Q11)
            # P: [1, 0, 0, 0]
            # Q: [0, 1, 0, -1]
            # R: R01(1) -> [0, 1, 0, 0]
            offset = 2
            add_strassen_term(offset, A_local, B_local, C_local, [1,0,0,0], [0,1,0,-1], [0,1,0,0])
            
            # M4: P11 * (Q01-Q00)
            # P: [0, 0, 0, 1]
            # Q: [-1, 0, 1, 0]
            # R: R00(1), R10(1) -> [1, 0, 1, 0]
            offset = 3
            add_strassen_term(offset, A_local, B_local, C_local, [0,0,0,1], [-1,0,1,0], [1,0,1,0])
            
            # M5: (P00+P01) * Q11
            # P: [1, 0, 1, 0]
            # Q: [0, 0, 0, 1]
            # R: R00(-1), R01(1) -> [-1, 1, 0, 0]
            offset = 4
            add_strassen_term(offset, A_local, B_local, C_local, [1,0,1,0], [0,0,0,1], [-1,1,0,0])
            
            # M6: (P10-P00) * (Q00+Q10)
            # P: [-1, 1, 0, 0]
            # Q: [1, 1, 0, 0]
            # R: R11(1) -> [0, 0, 0, 1]
            offset = 5
            add_strassen_term(offset, A_local, B_local, C_local, [-1,1,0,0], [1,1,0,0], [0,0,0,1])
            
            # M7: (P01-P11) * (Q01+Q11)
            # P: [0, 0, 1, -1]
            # Q: [0, 0, 1, 1]
            # R: R00(1), R11(-1) -> [1, 0, 0, -1]
            offset = 6
            add_strassen_term(offset, A_local, B_local, C_local, [0,0,1,-1], [0,0,1,1], [1,0,0,-1])
            
            # Update offsets for the next k_block
            # We used offsets 0-6 for the first k_block.
            # But wait, the offsets are local to this k_block?
            # No, the rank is global. We need to increment the base offset.
            # Let's refactor to use a base offset.
        
        # The above loop overwrote the same offsets. 
        # Let's rewrite the construction with a proper base offset.
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        def add_strassen_form(base_offset, A_local, B_local, C_local, sA, sB, sC):
            for idx in range(4):
                if sA[idx] != 0:
                    i, j = A_local[idx]
                    U[i * m + j, base_offset] += sA[idx]
                if sB[idx] != 0:
                    j, k = B_local[idx]
                    V[j * p + k, base_offset] += sB[idx]
                if sC[idx] != 0:
                    k, i = C_local[idx]
                    W[k * n + i, base_offset] += sC[idx]
        
        # A_L * B_T
        A_L_local = [
            (0, 0), (1, 0),
            (0, 1), (1, 1)
        ]
        
        for k_block in range(4):
            base = k_block * 7
            col0 = 2*k_block
            col1 = 2*k_block+1
            B_T_local = [
                (0, col0), (1, col0),
                (0, col1), (1, col1)
            ]
            C_T_local = [
                (col0, 0), (col0, 1),
                (col1, 0), (col1, 1)
            ]
            
            add_strassen_form(base+0, A_L_local, B_T_local, C_T_local, [1,0,0,1], [1,0,0,1], [1,0,0,1])
            add_strassen_form(base+1, A_L_local, B_T_local, C_T_local, [0,1,0,1], [1,0,0,0], [0,0,1,0])
            add_strassen_form(base+2, A_L_local, B_T_local, C_T_local, [1,0,0,0], [0,1,0,-1], [0,1,0,0])
            add_strassen_form(base+3, A_L_local, B_T_local, C_T_local, [0,0,0,1], [-1,0,1,0], [1,0,1,0])
            add_strassen_form(base+4, A_L_local, B_T_local, C_T_local, [1,0,1,0], [0,0,0,1], [-1,1,0,0])
            add_strassen_form(base+5, A_L_local, B_T_local, C_T_local, [-1,1,0,0], [1,1,0,0], [0,0,0,1])
            add_strassen_form(base+6, A_L_local, B_T_local, C_T_local, [0,0,1,-1], [0,0,1,1], [1,0,0,-1])
            
        # A_R * B_B
        # A_R is A[0:2, 2:4]
        A_R_local = [
            (0, 2), (1, 2),
            (0, 3), (1, 3)
        ]
        
        for k_block in range(4):
            base = 28 + k_block * 7
            col0 = 2*k_block
            col1 = 2*k_block+1
            # B_B is B[2:4, 0:8]
            B_B_local = [
                (2, col0), (3, col0),
                (2, col1), (3, col1)
            ]
            # C output is the same C[0:2, 0:8]
            C_B_local = [
                (col0, 0), (col0, 1),
                (col1, 0), (col1, 1)
            ]
            
            add_strassen_form(base+0, A_R_local, B_B_local, C_B_local, [1,0,0,1], [1,0,0,1], [1,0,0,1])
            add_strassen_form(base+1, A_R_local, B_B_local, C_B_local, [0,1,0,1], [1,0,0,0], [0,0,1,0])
            add_strassen_form(base+2, A_R_local, B_B_local, C_B_local, [1,0,0,0], [0,1,0,-1], [0,1,0,0])
            add_strassen_form(base+3, A_R_local, B_B_local, C_B_local, [0,0,0,1], [-1,0,1,0], [1,0,1,0])
            add_strassen_form(base+4, A_R_local, B_B_local, C_B_local, [1,0,1,0], [0,0,0,1], [-1,1,0,0])
            add_strassen_form(base+5, A_R_local, B_B_local, C_B_local, [-1,1,0,0], [1,1,0,0], [0,0,0,1])
            add_strassen_form(base+6, A_R_local, B_B_local, C_B_local, [0,0,1,-1], [0,0,1,1], [1,0,0,-1])
        
        return U, V, W

    U, V, W = get_strassen_56()
    
    # Verify the decomposition
    def verify_decomposition(U, V, W, num_tests=10, seed=42):
        rng = np.random.RandomState(seed)
        max_err = 0
        for _ in range(num_tests):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            U_reshaped = U.reshape(n, m, -1)
            V_reshaped = V.reshape(m, p, -1)
            W_reshaped = W.reshape(p, n, -1)
            
            X = np.einsum('ijr,ij->ir', U_reshaped, A)
            Y = np.einsum('jkr,jk->kr', V_reshaped, B)
            C_recon = np.einsum('ir,kr,ki->ik', X, Y, W_reshaped)
            C_true = A @ B
            err = np.sum((C_recon - C_true) ** 2)
            max_err = max(max_err, err)
        return max_err

    err = verify_decomposition(U, V, W)
    if err < 1e-10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, r] = 1.0
                    V[j * p + k, r] = 1.0
                    W[k * n + i, r] = 1.0
                    r += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
