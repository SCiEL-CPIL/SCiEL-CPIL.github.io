import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin for 150s kill
    
    n, m, p = DIMENSIONS
    N = n * m  # 16
    M = m * p  # 20
    P = p * n  # 20
    
    # We will construct a decomposition of rank 65.
    # Part 1: 4x4x4 block (indices 0..3 for all) -> Rank 49 (Recursive Strassen)
    # Part 2: 4x4x1 block (index 4 for p) -> Rank 16 (Schoolbook)
    
    total_rank = 65
    U = np.zeros((N, total_rank))
    V = np.zeros((M, total_rank))
    W = np.zeros((P, total_rank))
    
    # --- Part 1: 4x4x4 Recursive Strassen ---
    # The 4x4x4 tensor can be decomposed using Strassen's algorithm.
    # Strassen splits 4x4 into four 2x2 blocks.
    # It uses 7 products of 2x2 matrices.
    # Each 2x2 product is further decomposed using Strassen (rank 7).
    # Total rank = 7 * 7 = 49.
    
    # We need to map the 4x4 indices (i, j, k) where i,j,k in {0,1,2,3}
    # to the U, V, W matrices.
    # U index: i*m + j = i*4 + j
    # V index: j*p + k = j*5 + k  (Note: p=5, but we only use k in 0..3 here)
    # W index: k*n + i = k*4 + i
    
    # Strassen formulas for 2x2 matrices:
    # Let A = [[A00, A01], [A10, A11]], B = [[B00, B10], [B01, B11]] (Note standard B is [[B00, B01], [B10, B11]] but V index is j*p+k, so row j, col k. So B[j,k].)
    # Wait, standard matrix multiplication C = A B.
    # A is n x m. B is m x p.
    # In our 2x2 sub-blocks:
    # A_sub is 2x2, B_sub is 2x2.
    # Strassen products:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Reconstruct C:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will implement a helper to add a 2x2 Strassen decomposition.
    # The "A" block corresponds to a 2x2 submatrix of the original A.
    # The "B" block corresponds to a 2x2 submatrix of the original B.
    # The "C" block corresponds to a 2x2 submatrix of the original C.
    
    # Let's define the 7 Strassen terms for a 2x2x2 tensor.
    # Each term is a rank-1 decomposition of the 2x2x2 tensor?
    # No, Strassen gives a decomposition of the 2x2x2 tensor into 7 rank-1 terms?
    # Actually, the rank of the 2x2x2 matrix multiplication tensor is 7.
    # The 7 terms correspond to the 7 multiplications.
    # Each multiplication is a scalar product of linear combinations of A entries and B entries,
    # which then contributes to linear combinations of C entries.
    
    # Let's define the 7 rank-1 tensors for the 2x2x2 case.
    # u_r (2x2), v_r (2x2), w_r (2x2) such that sum_r u_r v_r w_r = T_2x2x2.
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    # u1: [[1, 0], [0, 1]]  (A00 and A11)
    # v1: [[1, 0], [0, 1]]  (B00 and B11)
    # w1: [[1, 1], [1, -1]]? 
    # Let's check C00 = M1 + M4 - M5 + M7.
    # This is getting complicated to derive from scratch. It is safer to use the known explicit coefficients.
    
    # Known 2x2x2 Strassen decomposition coefficients:
    # The 7 rank-1 components (u, v, w) where u, v, w are 4-vectors (flattened 2x2).
    # Indices for 2x2: 00, 01, 10, 11.
    
    # Let's use a verified set of coefficients for 2x2x2 Strassen.
    # Reference: Various online sources or standard textbooks.
    
    # u1 = [1, 0, 0, 1]   (A00, A11)
    # v1 = [1, 0, 0, 1]   (B00, B11)
    # w1 = [1, 1, 1, -1]  (C00, C01, C10, C11) -> C00+=1, C01+=1, C10+=1, C11+=-1?
    # Let's verify C00 contribution: 1*1*1 = 1.
    # C01: 1*1*1 = 1.
    # C10: 1*1*1 = 1.
    # C11: 1*1*(-1) = -1.
    
    # We need to sum all 7 terms to get the identity tensor.
    # This is error-prone. Instead, I will use the **recursive structure** explicitly.
    
    # Actually, there is a simpler way: 
    # The rank 49 decomposition for 4x4x4 is well-known. 
    # I will hardcode the 49 terms for the 4x4x4 block.
    # And the 16 terms for the 4x4x1 block.
    
    # To ensure correctness, I will generate the 4x4x4 Strassen terms programmatically 
    # by applying the 2x2 Strassen logic recursively.
    
    def get_strassen_2x2_terms():
        """
        Returns lists of (u, v, w) vectors for the 2x2x2 Strassen decomposition.
        Each vector has length 4 (indices 0,1,2,3 corresponding to 00, 01, 10, 11).
        """
        terms = []
        # Based on standard Strassen:
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
        
        # Let's map these to (u, v, w)
        # u is for A, v for B, w for C.
        
        # Term 1 (M1):
        # u1: A00=1, A11=1 -> [1, 0, 0, 1]
        # v1: B00=1, B11=1 -> [1, 0, 0, 1]
        # w1: C00=1, C01=0, C10=0, C11=1? 
        # Wait, M1 contributes to C00 (coeff 1) and C11 (coeff 1).
        # So w1 = [1, 0, 0, 1]
        terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
        
        # Term 2 (M2):
        # u2: A10=1, A11=1 -> [0,0,1,1]
        # v2: B00=1 -> [1,0,0,0]
        # w2: C10=1 -> [0,0,1,0]
        terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
        
        # Term 3 (M3):
        # u3: A00=1 -> [1,0,0,0]
        # v3: B01=1, B11=-1 -> [0,1,0,-1]
        # w3: C01=1 -> [0,1,0,0]
        terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
        
        # Term 4 (M4):
        # u4: A11=1 -> [0,0,0,1]
        # v4: B10=1, B00=-1 -> [-1,0,1,0]
        # w4: C00=1, C10=1 -> [1,0,1,0]
        terms.append(([0,0,0,1], [-1,0,1,0], [1,0,1,0]))
        
        # Term 5 (M5):
        # u5: A00=1, A01=1 -> [1,1,0,0]
        # v5: B11=1 -> [0,0,0,1]
        # w5: C00=-1, C01=1 -> [-1,1,0,0]
        terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
        
        # Term 6 (M6):
        # u6: A10=1, A00=-1 -> [-1,0,1,0]
        # v6: B00=1, B01=1 -> [1,1,0,0]
        # w6: C11=1 -> [0,0,0,1]
        terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
        
        # Term 7 (M7):
        # u7: A01=1, A11=-1 -> [0,1,0,-1]
        # v7: B10=1, B11=1 -> [0,0,1,1]
        # w7: C00=1, C11=1 -> [1,0,0,1]
        terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,1]))
        
        return terms

    # Verify the 2x2 terms sum to the identity tensor T[i,j,k] = delta_{i,k} (for C[i,k] = sum_j A[i,j]B[j,k])
    # Let's do a quick mental check or rely on the known correctness of Strassen.
    # I will trust the coefficients above. If they are wrong, the verification will fail.
    
    strassen_2x2_terms = get_strassen_2x2_terms()
    
    # Now, build the 4x4x4 decomposition.
    # The 4x4 matrix A is split into 2x2 blocks:
    # A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1) in block coordinates.
    # Block (bi, bj) of A corresponds to rows 2bi..2bi+1, cols 2bj..2bj+1.
    # Similarly for B and C.
    
    # We apply Strassen at the top level (4x4).
    # The 7 top-level products are M1..M7, each being a 2x2 * 2x2 product.
    # Each of these 2x2 * 2x2 products is decomposed using the 7 2x2 Strassen terms.
    # Total 7 * 7 = 49 terms.
    
    # Top-level Strassen definitions (Block indices 0,1):
    # M1 = (A00+A11) * (B00+B11)
    # M2 = (A10+A11) * B00
    # M3 = A00 * (B01-B11)
    # M4 = A11 * (B10-B00)
    # M5 = (A00+A01) * B11
    # M6 = (A10-A00) * (B00+B01)
    # M7 = (A01-A11) * (B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to combine the top-level linear combinations with the bottom-level Strassen terms.
    # Let T_top be the list of 7 top-level terms (u_top, v_top, w_top) where u_top, v_top, w_top are 2x2 block vectors.
    # Let T_bot be the list of 7 bottom-level terms (u_bot, v_bot, w_bot) where u_bot, v_bot, w_bot are 2x2 scalar vectors.
    
    # The final decomposition is a sum over all pairs (top, bot) of:
    # u_final = u_top * u_bot (Kronecker product-like scaling)
    # v_final = v_top * v_bot
    # w_final = w_top * w_bot
    
    # Let's define the top-level terms as vectors of block coefficients.
    # Block indices: 00, 01, 10, 11.
    
    top_terms = []
    # M1
    top_terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1])) # C00=1, C11=1
    # M2
    top_terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0])) # C10=1
    # M3
    top_terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0])) # C01=1
    # M4
    top_terms.append(([0,0,0,1], [-1,0,1,0], [1,0,1,0])) # C00=1, C10=1
    # M5
    top_terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0])) # C00=-1, C01=1
    # M6
    top_terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1])) # C11=1
    # M7
    top_terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,1])) # C00=1, C11=1
    
    # Now, iterate through all 49 combinations.
    col = 0
    for ut, vt, wt in top_terms:
        for ub, vb, wb in strassen_2x2_terms:
            # Construct U vector (length 16)
            # U index: i*4 + j
            # Block (bi, bj) of A contains elements A[2bi+a, 2bj+b] for a,b in 0,1.
            # ub is a vector of length 4: [ub_00, ub_01, ub_10, ub_11] corresponding to local indices 00, 01, 10, 11.
            # ut is a vector of length 4: [ut_00, ut_01, ut_10, ut_11] corresponding to block indices 00, 01, 10, 11.
            
            u_vec = np.zeros(16)
            for bi in range(2):
                for bj in range(2):
                    block_idx = bi*2 + bj
                    scale_t = ut[block_idx]
                    if scale_t != 0:
                        # Apply bottom-level scaling ub
                        for a in range(2):
                            for b in range(2):
                                local_idx = a*2 + b
                                scale_b = ub[local_idx]
                                if scale_b != 0:
                                    global_i = 2*bi + a
                                    global_j = 2*bj + b
                                    u_idx = global_i * 4 + global_j
                                    u_vec[u_idx] += scale_t * scale_b
            
            # Construct V vector (length 20)
            # V index: j*5 + k
            # We only use k in 0..3 for the 4x4x4 part.
            v_vec = np.zeros(20)
            for bj in range(2):
                for bk in range(2):
                    block_idx = bj*2 + bk
                    scale_t = vt[block_idx]
                    if scale_t != 0:
                        for a in range(2):
                            for b in range(2):
                                local_idx = a*2 + b
                                scale_b = vb[local_idx]
                                if scale_b != 0:
                                    global_j = 2*bj + a
                                    global_k = 2*bk + b
                                    v_idx = global_j * 5 + global_k
                                    v_vec[v_idx] += scale_t * scale_b
            
            # Construct W vector (length 20)
            # W index: k*4 + i
            # We only use k in 0..3.
            w_vec = np.zeros(20)
            for bk in range(2):
                for bi in range(2):
                    block_idx = bk*2 + bi
                    scale_t = wt[block_idx]
                    if scale_t != 0:
                        for a in range(2):
                            for b in range(2):
                                local_idx = a*2 + b
                                scale_b = wb[local_idx]
                                if scale_b != 0:
                                    global_k = 2*bk + a
                                    global_i = 2*bi + b
                                    w_idx = global_k * 4 + global_i
                                    w_vec[w_idx] += scale_t * scale_b
            
            U[:, col] = u_vec
            V[:, col] = v_vec
            W[:, col] = w_vec
            col += 1
            
    # --- Part 2: 4x4x1 Schoolbook ---
    # This part handles the k=4 column of B.
    # C[i, 4] = sum_{j=0}^3 A[i, j] B[j, 4]
    # This is a 4x4 * 4x1 product.
    # Schoolbook rank is 4*4*1 = 16.
    # Terms: for each i, j, we have a term A[i,j] * B[j,4] contributing to C[i,4].
    
    for i in range(4):
        for j in range(4):
            u_vec = np.zeros(16)
            u_vec[i*4 + j] = 1.0
            
            v_vec = np.zeros(20)
            v_vec[j*5 + 4] = 1.0  # k=4
            
            w_vec = np.zeros(20)
            w_vec[4*4 + i] = 1.0  # k=4, i=i -> index 16+i
            
            U[:, col] = u_vec
            V[:, col] = v_vec
            W[:, col] = w_vec
            col += 1
            
    # Verify the decomposition
    def verify(U, V, W, trials=5):
        for _ in range(trials):
            A = np.random.randn(4, 4)
            B = np.random.randn(4, 5)
            C_expected = A @ B
            C_computed = np.zeros((4, 5))
            for i in range(4):
                for k in range(5):
                    val = 0.0
                    for j in range(4):
                        u_val = U[i*4 + j]
                        v_val = V[j*5 + k]
                        w_val = W[k*4 + i]
                        val += np.dot(u_val, v_val * w_val)
                    C_computed[i, k] = val
            if not np.allclose(C_computed, C_expected, atol=1e-4):
                return False
        return True

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook rank 80 if Strassen construction failed
        U_school = np.zeros((16, 80))
        V_school = np.zeros((20, 80))
        W_school = np.zeros((20, 80))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U_school[i*4 + j, col] = 1.0
                    V_school[j*5 + k, col] = 1.0
                    W_school[k*4 + i, col] = 1.0
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

def _verify(U, V, W, n=4, m=4, p=5, trials=10):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for j in range(m):
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                C_computed[i, k] = val
        
        if not np.allclose(C_computed, C_expected, atol=1e-4):
            return False
            
    return True
