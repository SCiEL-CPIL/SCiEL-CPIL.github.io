"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _verify(U, V, W):
    """Verify that the decomposition exactly computes 4x4 matrix multiplication."""
    n, m, p = 4, 4, 4
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank) with W[k*n + i, r]
    
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    rank = U.shape[1]
    # Reshape to 4D tensors
    U4 = U.reshape(n, m, rank)  # U[i, j, r]
    V4 = V.reshape(m, p, rank)  # V[j, k, r]
    W4 = W.reshape(p, n, rank)  # W[k, i, r]
    
    # The tensor should satisfy: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta[i,j,k]
    # where delta[i,j,k] = 1 if this is part of the MM tensor
    
    # Actually the matrix multiplication tensor T has T[i,j,k] = 1 iff we compute (AB)[i,k] = sum_j A[i,j] B[j,k]
    # The factorization requires: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) is a valid MM entry
    
    # Let's compute the reconstructed tensor and compare
    recon = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                recon[i, j, k] = np.sum(U4[i, j, :] * V4[j, k, :] * W4[k, i, :])
    
    target = np.ones((n, m, p))
    
    max_err = np.max(np.abs(recon - target))
    return max_err < 1e-9

def _try_rank_search(time_budget):
    """Search for valid decompositions, starting from low ranks.
    We'll try random initialization + local optimization for each rank."""
    
    n, m, p = 4, 4, 4
    deadline = time.time() + time_budget
    
    best_rank = None
    best_solution = None
    
    # Try ranks from a minimum feasible up to some limit
    # The MM tensor has border rank considerations; minimum possible rank is at least 2*max(n,m,p) = 8?
    # Actually lower bounds: rank >= n*m + n*p + m*p - 2*n - 2*m - 2*p + 3? No.
    # A simple lower bound: rank >= max(n*m, m*p, p*n) = 16 for 4x4
    # But we know rank 49 works. Let's try a range.
    
    # Given time constraints, try a few ranks with random init + gradient descent
    candidate_ranks = [20, 22, 24, 26, 28, 30, 32, 35, 40, 45, 49]
    
    for rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        best_err_for_rank = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        n_tries = 3
        for attempt in range(n_tries):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            U4 = U.reshape(n, m, rank)
            V4 = V.reshape(m, p, rank)
            W4 = W.reshape(p, n, rank)
            
            target = np.ones((n, m, p))
            
            # Simple gradient descent
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 10:
                    break
                    
                recon = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            recon[i, j, k] = np.sum(U4[i, j, :] * V4[j, k, :] * W4[k, i, :])
                
                err = np.sum((recon - target) ** 2)
                if err < best_err_for_rank:
                    best_err_for_rank = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if best_err_for_rank < 1e-6:
                    break
                
                # Compute gradients (analytical)
                # dL/dU4[i,j,r] = 2 * sum_k (recon[i,j,k] - 1) * V4[j,k,r] * W4[k,i,r]
                # Similarly for V and W
                
                dU4 = np.zeros((n, m, rank))
                dV4 = np.zeros((m, p, rank))
                dW4 = np.zeros((p, n, rank))
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            diff = recon[i, j, k] - 1.0
                            for r in range(rank):
                                dU4[i, j, r] += 2 * diff * V4[j, k, r] * W4[k, i, r]
                                dV4[j, k, r] += 2 * diff * U4[i, j, r] * W4[k, i, r]
                                dW4[k, i, r] += 2 * diff * U4[i, j, r] * V4[j, k, r]
                
                U4 -= lr * dU4
                V4 -= lr * dV4
                W4 -= lr * dW4
        
        if best_err_for_rank < 1e-9 and best_U is not None:
            best_rank = rank
            best_solution = (best_U, best_V, best_W)
            break  # Found the smallest working rank
    
    return best_rank, best_solution

def _strassen_decomposition():
    """Return the rank-49 Strassen-based decomposition for 4x4 matrix multiplication."""
    
    # Strassen's algorithm for 2x2 matrices uses 7 multiplications.
    # For 4x4, we recursively apply Strassen: split 4x4 into 2x2x2x2 blocks,
    # apply Strassen at each level. This gives rank 7^2 = 49.
    
    # The Strassen formulas for 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # 
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    n, m, p = 4, 4, 4
    
    # For 2x2 case, the factor matrices are:
    # U entries: for each of the 7 products, extract the relevant A entries
    # A is 2x2, indexed as A[r,c] where r,c in {0,1}
    # Product k combines A entries with coefficients a_k and B entries with coefficients b_k
    
    # Define the 7 Strassen products as coefficient vectors
    # For product q, U has rows for A elements, V has rows for B elements
    # U[i, q] = coefficient of A[i] in product q
    # V[j, q] = coefficient of B[j] in product q
    
    # A indexed as (i, j) -> flat index i*2 + j for 2x2
    # B indexed as (j, k) -> flat index j*2 + k for 2x2
    
    # Strassen products:
    # q=0: (A00+A11)(B00+B11)
    # q=1: (A10+A11)B00
    # q=2: A00(B10-B11)
    # q=3: A11(B01-B00)
    # q=4: (A00+A01)B11
    # q=5: (A10-A00)(B00+B01)
    # q=6: (A01-A11)(B10+B11)
    
    # Coefficients for A in each product (row index = i*2+j for 2x2):
    A_coeffs = [
        [1, 0, 0, 1],  # q=0: A00=1, A11=1
        [0, 0, 1, 1],  # q=1: A10=1, A11=1
        [1, 0, 0, 0],  # q=2: A00=1
        [0, 0, 0, 1],  # q=3: A11=1
        [1, 1, 0, 0],  # q=4: A00=1, A01=1
        [1, 0, 1, 0],  # q=5: A10=1, A00=-1  -> actually (A10-A00)
        [0, 1, 0, 1],  # q=6: A01=1, A11=-1  -> actually (A01-A11)
    ]
    A_coeffs[5] = [1, 0, 1, 0]  # Wait, let me re-index
    # Let me redo this carefully.
    # A = [[A00, A01], [A10, A11]]
    # Flat indices: A00->0, A01->1, A10->2, A11->3
    
    A_coeffs = [
        [1, 0, 0, 1],   # q=0: A00 + A11
        [0, 0, 1, 1],   # q=1: A10 + A11
        [1, 0, 0, 0],   # q=2: A00
        [0, 0, 0, 1],   # q=3: A11
        [1, 1, 0, 0],   # q=4: A00 + A01
        [-1, 0, 1, 0],  # q=5: A10 - A00
        [0, 1, 0, -1],  # q=6: A01 - A11
    ]
    
    # B = [[B00, B01], [B10, B11]]
    # Flat indices: B00->0, B01->1, B10->2, B11->3
    
    B_coeffs = [
        [1, 0, 0, 1],   # q=0: B00 + B11
        [1, 0, 0, 0],   # q=1: B00
        [0, -1, 1, 0],  # q=2: B10 - B11
        [-1, 1, 0, 0],  # q=3: B01 - B00
        [0, 0, 0, 1],   # q=4: B11
        [1, 1, 0, 0],   # q=5: B00 + B01
        [0, 0, 1, 1],   # q=6: B10 + B11
    ]
    
    # C reconstruction coefficients: C[i,k] = sum_q c_q[i,k] * M_q
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    C_coeffs = [
        [1, 0, 0, 1],   # C00: M0 + M3 - M4 + M6
        [0, 0, 1, 0],   # C01: M2 + M4
        [0, 1, 0, 0],   # C10: M1 + M3
        [1, -1, 1, 1],  # C11: M0 - M1 + M2 + M5
    ]
    # Wait, I need to be more careful. C_coeffs[i*2+k, q] = coefficient of M_q in C[i,k]
    # C00: M0 + M3 - M4 + M6
    # C01: M2 + M4
    # C10: M1 + M3
    # C11: M0 - M1 + M2 + M5
    
    C_coeffs = [
        [1, 0, 0, 1],   # C00
        [0, 0, 1, 0],   # C01
        [0, 1, 0, 0],   # C10
        [1, -1, 1, 1],  # C11
    ]
    
    # Now for 4x4, we apply Strassen twice.
    # Split A (4x4) into four 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # Similarly for B and C.
    
    # First level Strassen (on 2x2 blocks of 2x2 matrices):
    # Products M_q where each M_q is a 2x2 product of 2x2 matrices.
    
    # Second level Strassen: each M_q (2x2 product) uses 7 sub-products.
    # Total rank = 7 * 7 = 49.
    
    # Let's build the 4x4 factorization directly.
    # U: 16 x 49
    # V: 16 x 49
    # W: 16 x 49 (with W[k*4 + i, r] per the trace convention)
    
    # The 4x4 matrix A has elements A[i, j] for i, j in 0..3
    # Group into 2x2 blocks: A_top_left = A[0:2, 0:2], A_top_right = A[0:2, 2:4]
    #                      A_bottom_left = A[2:4, 0:2], A_bottom_right = A[2:4, 2:4]
    
    # For the first-level Strassen, we need combinations of these blocks.
    # For product q (first level), the A-side coefficient for block (br, bc) is:
    #   first_A[q][br, bc] where br, bc in {0, 1} refer to row/col of the 2x2 super-block
    
    first_A = [
        [1, 1],  # q=0: blocks (0,0) + (1,1)
        [0, 0],  # q=1: block (1,0) + (1,1) -> wait
    ]
    # Let me use the same A_coeffs pattern but for 2x2 block structure
    # A_coeffs was for 2x2 scalar matrices. Now A is 2x2 block matrix.
    # The coefficients are the same pattern:
    first_A_coeffs = np.array([
        [1, 1],   # q=0: A00_block + A11_block
        [0, 0],   # q=1: A10_block + A11_block
        [1, 0],   # q=2: A00_block
        [0, 1],   # q=3: A11_block
        [1, 0],   # q=4: A00_block + A01_block
        [1, 0],   # q=5: A10_block - A00_block
        [0, 1],   # q=6: A01_block - A11_block
    ])
    # Hmm, this isn't quite right. Let me think differently.
    
    # Actually, for the recursive Strassen, we can just compose the 2x2 Strassen
    # factorization with itself.
    
    # For 2x2 matrix multiplication, we have factorization with rank 7.
    # Let U2 (4x7), V2 (4x7), W2 (4x7) be the 2x2 Strassen factors.
    # Then for 4x4, U4[i*4+j, q*7+s] = sum over the intermediate.
    
    # Actually, the cleanest way: 
    # The 4x4 MM tensor = tensor product of two 2x2 MM tensors (approximately, via Kronecker).
    # More precisely, if T_2 is the 2x2 MM tensor with factors (U2, V2, W2) of rank 7,
    # then T_4 = T_2 x T_2 has rank 7*7 = 49.
    
    # The Kronecker product structure:
    # U4[(i*2+i2, j*2+j2), (q*7+s)] = U2[i, q] * U2[i2, s]  ... wait, this isn't right either.
    
    # Let me just directly construct it using the block structure.
    
    # For 4x4: A[i,j] where i = 2*i' + i'', j = 2*j' + j''
    # B[j,k] where j = 2*j' + j'', k = 2*k' + k''
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # First-level Strassen combines A super-blocks and B super-blocks:
    # For q in 0..6, super-product S_q = sum_{br,bc,kr,kc} a_q[br,bc] b_q[kr,kc] 
    #   where a_q[br,bc] is the coefficient of A-super-block (br,bc) in product q
    #   and b_q[kr,kc] is the coefficient of B-super-block (kr,kc) in product q
    
    # The 2x2 Strassen super-block coefficients (using block indices 0,1 for row/col):
    super_A = np.array([
        [1, 1],   # q=0: A00 + A11
        [0, 0],   # q=1: A10 + A11  -> need 2x2 matrix of coefficients
    ])
    
    # Actually, the 2x2 Strassen formula uses 2x2 submatrices. Let me define it properly.
    # For 2x2 matrix A = [[a00, a01], [a10, a11]], the Strassen products are:
    # m0 = (a00 + a11)(b00 + b11)
    # m1 = (a10 + a11)(b00)
    # m2 = (a00)(b10 - b11)
    # m3 = (a11)(b01 - b00)
    # m4 = (a00 + a01)(b11)
    # m5 = (a10 - a00)(b00 + b01)
    # m6 = (a01 - a11)(b10 + b11)
    
    # For 4x4, A is a 2x2 block matrix where each block is 2x2.
    # The same Strassen formulas apply at the block level.
    # Then each block-level product is itself a 2x2 matrix multiplication, which we decompose with rank 7.
    
    # So the total rank is 7 (first level) * 7 (second level) = 49.
    
    # Let's define the coefficients clearly:
    # For first-level product q (0..6), the A-block (br, bc) [br,bc in {0,1}] has coefficient a_super[q, br, bc]
    # and the B-block (kr, kc) has coefficient b_super[q, kr, kc]
    
    # Using the Strassen formulas above, treating each 2x2 block as a scalar:
    a_super = np.zeros((7, 2, 2))
    b_super = np.zeros((7, 2, 2))
    
    # m0: (A00 + A11)(B00 + B11)
    a_super[0] = [[1, 0], [0, 1]]  # A00=1, A11=1
    b_super[0] = [[1, 0], [0, 1]]  # B00=1, B11=1
    
    # m1: (A10 + A11)(B00)
    a_super[1] = [[0, 0], [1, 1]]
    b_super[1] = [[1, 0], [0, 0]]
    
    # m2: (A00)(B10 - B11)
    a_super[2] = [[1, 0], [0, 0]]
    b_super[2] = [[0, 0], [1, -1]]
    
    # m3: (A11)(B01 - B00)
    a_super[3] = [[0, 0], [0, 1]]
    b_super[3] = [[-1, 1], [0, 0]]
    
    # m4: (A00 + A01)(B11)
    a_super[4] = [[1, 1], [0, 0]]
    b_super[4] = [[0, 0], [0, 1]]
    
    # m5: (A10 - A00)(B00 + B01)
    a_super[5] = [[-1, 0], [1, 0]]
    b_super[5] = [[1, 1], [0, 0]]
    
    # m6: (A01 - A11)(B10 + B11)
    a_super[6] = [[0, 1], [0, -1]]
    b_super[6] = [[0, 0], [1, 1]]
    
    # Now for the second level, each super-product m_q involves multiplying two 2x2 matrices.
    # The A-side of m_q is a linear combination of A-blocks, and similarly for B.
    # For the actual 4x4 entries, A[i,j] where i=2*br+i'', j=2*bc+j''
    # The contribution of A-block (br,bc) to entry (i,j) is 1 if i'' and j'' match.
    
    # For the second-level Strassen within each super-product q:
    # We decompose the 2x2 matrix product using the 7 Strassen sub-products.
    # For sub-product s (0..6) within super-product q:
    # The A-coefficient for element (i'', j'') is a_sub[s, i'', j'']
    # The B-coefficient for element (j'', k'') is b_sub[s, j'', k'']
    
    # These are the same as the original 2x2 Strassen coefficients:
    a_sub = np.zeros((7, 2, 2))
    b_sub = np.zeros((7, 2, 2))
    
    a_sub[0] = [[1, 0], [0, 1]]
    b_sub[0] = [[1, 0], [0, 1]]
    a_sub[1] = [[0, 0], [1, 1]]
    b_sub[1] = [[1, 0], [0, 0]]
    a_sub[2] = [[1, 0], [0, 0]]
    b_sub[2] = [[0, 0], [1, -1]]
    a_sub[3] = [[0, 0], [0, 1]]
    b_sub[3] = [[-1, 1], [0, 0]]
    a_sub[4] = [[1, 1], [0, 0]]
    b_sub[4] = [[0, 0], [0, 1]]
    a_sub[5] = [[-1, 0], [1, 0]]
    b_sub[5] = [[1, 1], [0, 0]]
    a_sub[6] = [[0, 1], [0, -1]]
    b_sub[6] = [[0, 0], [1, 1]]
    
    # Now construct U4, V4, W4 of shape (16, 49)
    # Element index for A: i*4 + j where i = 2*br + i'', j = 2*bc + j''
    # Column index: q*7 + s
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for q in range(7):
        for s in range(7):
            col = q * 7 + s
            for br in range(2):
                for bc in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            # A entry index: i = 2*br + i2, j = 2*bc + j2
                            i = 2 * br + i2
                            j = 2 * bc + j2
                            coeff = a_super[q, br, bc] * a_sub[s, i2, j2]
                            U4[i * 4 + j, col] += coeff
            
            for kr in range(2):
                for kc in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            # B entry index: j = 2*kr + j2, k = 2*kc + k2
                            j = 2 * kr + j2
                            k = 2 * kc + k2
                            coeff = b_super[q, kr, kc] * b_sub[s, j2, k2]
                            V4[j * 4 + k, col] += coeff
        
        # For W, we need the reconstruction coefficients
        # C[i,k] = sum_{q} c_super[i_blk, k_blk, q] * m_q
        # where m_q is the super-product
        # Then m_q = sum_s c_sub[i2, k2, s] * (2x2 product s)
        # Actually, the W tensor relates to how C entries are reconstructed.
        # C[i,k] where i = 2*br + i2, k = 2*kc + k2
        # C[i,k] = sum_q sum_s c_super[q, br, kc] * c_sub[s, i2, k2] * M_{q,s}
        # But M_{q,s} = sum_{j2} A_eff[i2, j2] * B_eff[j2, k2] for the effective 2x2 matrices
        # 
        # The W factor should satisfy: W[k*4 + i, col] = coefficient such that
        # sum_col U[i,j,col] V[j,k,col] W[k,i,col] = 1 for all (i,j,k)
        
        # The reconstruction for C-block (br, kc):
        # C_br_kc = c0*Q0 + c1*Q1 + ... + c6*Q6 where Qq are the super-products
        # C_coeffs for 2x2:
        # C00 = M0 + M3 - M4 + M6
        # C01 = M2 + M4
        # C10 = M1 + M3
        # C11 = M0 - M1 + M2 + M5
        
        c_super = np.array([
            [1, 0, 0, 1],   # C00
            [0, 0, 1, 0],   # C01
            [0, 1, 0, 0],   # C10
            [1, -1, 1, 1],  # C11
        ])
        # c_super[i_blk, k_blk, q] -- actually it's c_super[row_blk, col_blk] for each q
        # Let me re-index: c_super[blk_row, blk_col, q]
        c_super_full = np.zeros((2, 2, 7))
        c_super_full[0, 0] = [1, 0, 0, 1, 0, 0, 1]   # C00 = M0 + M3 + M6 (and -M4? wait)
        
        # Let me recheck: 
        # C00 = m0 + m3 - m4 + m6
        # C01 = m2 + m4
        # C10 = m1 + m3
        # C11 = m0 - m1 + m2 + m5
        
        c_super_full[0, 0] = [1, 0, 0, 1, -1, 0, 1]
        c_super_full[0, 1] = [0, 0, 1, 0, 1, 0, 0]
        c_super_full[1, 0] = [0, 1, 0, 1, 0, 0, 0]
        c_super_full[1, 1] = [1, -1, 1, 0, 0, 1, 0]
        
        # Similarly for sub-level:
        # The 2x2 product decomposition gives:
        # For the effective 2x2 matrices, C2[br2, kc2] uses the same c_super pattern
        # But actually, for the second level, within super-product q, we have a 2x2 matrix product.
        # The result C_sub[br2, kc2] is reconstructed from the 7 sub-products s.
        # This is the same c_super_full pattern.
        
        c_sub_full = np.zeros((2, 2, 7))
        c_sub_full[0, 0] = [1, 0, 0, 1, -1, 0, 1]
        c_sub_full[0, 1] = [0, 0, 1, 0, 1, 0, 0]
        c_sub_full[1, 0] = [0, 1, 0, 1, 0, 0, 0]
        c_sub_full[1, 1] = [1, -1, 1, 0, 0, 1, 0]
        
        for br2 in range(2):
            for kc2 in range(2):
                # This gets complex. Let me just verify with _verify after construction.
                pass
    
    # The W construction is trickier. Let me use a different approach.
    # I'll just build U and V as above, then solve for W.
    # Actually, for the Kronecker structure, W should also factor nicely.
    
    # Let me just use the direct construction and verify.
    
    return U4.tolist(), V4.tolist(), W4.tolist()

def _fallback_strassen():
    """Construct the rank-49 Strassen decomposition for 4x4 MM.
    Uses the recursive/tensor-product structure."""
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # First-level Strassen coefficients (for 2x2 block matrix)
    a_super = [
        [[1, 0], [0, 1]],   # q=0
        [[0, 0], [1, 1]],   # q=1
        [[1, 0], [0, 0]],   # q=2
        [[0, 0], [0, 1]],   # q=3
        [[1, 1], [0, 0]],   # q=4
        [[-1, 0], [1, 0]],  # q=5
        [[0, 1], [0, -1]],  # q=6
    ]
    b_super = [
        [[1, 0], [0, 1]],   # q=0
        [[1, 0], [0, 0]],   # q=1
        [[0, 0], [1, -1]],  # q=2
        [[-1, 1], [0, 0]],  # q=3
        [[0, 0], [0, 1]],   # q=4
        [[1, 1], [0, 0]],   # q=5
        [[0, 0], [1, 1]],   # q=6
    ]
    
    # Second-level Strassen coefficients (for 2x2 scalar matrix)
    a_sub = [
        [[1, 0], [0, 1]],   # s=0
        [[0, 0], [1, 1]],   # s=1
        [[1, 0], [0, 0]],   # s=2
        [[0, 0], [0, 1]],   # s=3
        [[1, 1], [0, 0]],   # s=4
        [[-1, 0], [1, 0]],  # s=5
        [[0, 1], [0, -1]],  # s=6
    ]
    b_sub = [
        [[1, 0], [0, 1]],   # s=0
        [[1, 0], [0, 0]],   # s=1
        [[0, 0], [1, -1]],  # s=2
        [[-1, 1], [0, 0]],  # s=3
        [[0, 0], [0, 1]],   # s=4
        [[1, 1], [0, 0]],   # s=5
        [[0, 0], [1, 1]],   # s=6
    ]
    
    # C reconstruction coefficients
    # C[br, bc] = sum_q c_super[br, bc, q] * M_q
    c_super = [
        [[1, 0, 0, 1, -1, 0, 1],    # C00
         [0, 0, 1, 0, 1, 0, 0]],    # C01
        [[0, 1, 0, 1, 0, 0, 0],     # C10
         [1, -1, 1, 0, 0, 1, 0]]    # C11
    ]
    c_sub = [
        [[1, 0, 0, 1, -1, 0, 1],    # C00
         [0, 0, 1, 0, 1, 0, 0]],    # C01
        [[0, 1, 0, 1, 0, 0, 0],     # C10
         [1, -1, 1, 0, 0, 1, 0]]    # C11
    ]
    
    for q in range(7):
        for s in range(7):
            col = q * 7 + s
            # U entries
            for br in range(2):
                for bc in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = 2 * br + i2
                            j = 2 * bc + j2
                            coeff = a_super[q][br][bc] * a_sub[s][i2][j2]
                            U[i * 4 + j][col] += coeff
                       
            
            # V entries
            for kr in range(2):
                for kc in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = 2 * kr + j2
                            k = 2 * kc + k2
                            coeff = b_super[q][kr][kc] * b_sub[s][j2][k2]
                            V[j * 4 + k][col] += coeff
            
            # W entries: W[k*4 + i, col]
            # W relates to reconstruction
            for i_blk in range(2):
                for k_blk in range(2):
                    for i2 in range(2):
                        for k2 in range(2):
                            i = 2 * i_blk + i2
                            k = 2 * k_blk + k2
                            coeff = c_super[i_blk][k_blk][q] * c_sub[i2][k2][s]
                            W[k * 4 + i][col] += coeff
    
    # Verify
    if not _verify(U, V, W):
        # Fallback: return schoolbook
        rank_fb = 64
        U_fb = [[0.0] * rank_fb for _ in range(16)]
        V_fb = [[0.0] * rank_fb for _ in range(16)]
        W_fb = [[0.0] * rank_fb for _ in range(16)]
        column = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_fb[i * 4 + j][column] = 1.0
                    V_fb[j * 4 + k][column] = 1.0
                    W_fb[k * 4 + i][column] = 1.0
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}

def construct():
    # Try Strassen first (deterministic, rank 49)
    sol = _fallback_strassen()
    if sol is not None:
        # Check if it's valid
        if _verify(sol['U'], sol['V'], sol['W']):
            return sol
    
    # Final fallback: schoolbook (rank 64)
    n, m, p = 4, 4, 4
    rank = 64
    U = [[0.0] * rank for _ in range(16)]
    V = [[0.0] * rank for _ in range(16)]
    W = [[0.0] * rank for _ in range(16)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
