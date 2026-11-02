"""Matrix multiplication 4x4x5 tensor decomposition using block Strassen + naive."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Block decomposition:
    # A is n x m = 4x4, B is m x p = 4x5, C is n x p = 4x5
    # Split: rows of A: [0:2], [2:4]  (i-blocks)
    #        cols of A: [0:2], [2:4]  (j-blocks)  
    #        cols of B: [0:2], [2:5]  (k-blocks)
    #
    # Sub-problems:
    #   (i_blk, j_blk, k_blk) -> A[i_blk, j_blk] @ B[j_blk, k_blk] -> C[i_blk, k_blk]
    #
    # For 2x2x2 blocks: use Strassen (rank 7)
    # For 2x2x3 blocks: use naive (rank 12)
    
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (20, rank)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_row, v_row, w_row, u_val, v_val, w_val):
        """Add one tensor term (rank-1 factor)."""
        U_cols.append((u_row, u_val))
        V_cols.append((v_row, v_val))
        W_cols.append((w_row, w_val))
    
    def add_naive_block(i0, j0, k0, ni, nj, nk):
        """Add naive matmul terms for A[i0:i0+ni, j0:j0+nj] @ B[j0:j0+nj, k0:k0+nk] -> C[i0:i0+ni, k0:k0+nk]"""
        for ii in range(ni):
            for jj in range(nj):
                for kk in range(nk):
                    # U index: (i0+ii) * m + (j0+jj)
                    # V index: (j0+jj) * p + (k0+kk)
                    # W index: (k0+kk) * n + (i0+ii)
                    u_idx = (i0 + ii) * m + (j0 + jj)
                    v_idx = (j0 + jj) * p + (k0 + kk)
                    w_idx = (k0 + kk) * n + (i0 + ii)
                    add_term(u_idx, v_idx, w_idx, 1.0, 1.0, 1.0)
    
    def add_strassen_block(i0, j0, k0):
        """Add Strassen matmul terms for 2x2x2 block.
        A is 2x2 at rows i0,i0+1 and cols j0,j0+1
        B is 2x2 at rows j0,j0+1 and cols k0,k0+1
        C is 2x2 at rows i0,i0+1 and cols k0,k0+1
        
        Strassen:
        M1 = (A00+A11)*(B00+B11)
        M2 = (A10+A11)*B00
        M3 = A00*(B10-B11)
        M4 = A11*(B01-B10)
        M5 = (A00+A01)*B11
        M6 = (A10-A00)*(B00+B10)
        M7 = (A01-A11)*(B01+B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        # A00 = A[i0, j0], A01 = A[i0, j0+1], A10 = A[i0+1, j0], A11 = A[i0+1, j0+1]
        # B00 = B[j0, k0], B01 = B[j0, k0+1], B10 = B[j0+1, k0], B11 = B[j0+1, k0+1]
        # C00 = C[i0, k0], C01 = C[i0, k0+1], C10 = C[i0+1, k0], C11 = C[i0+1, k0+1]
        
        # U indices for A elements
        u00 = i0 * m + j0
        u01 = i0 * m + (j0 + 1)
        u10 = (i0 + 1) * m + j0
        u11 = (i0 + 1) * m + (j0 + 1)
        
        # V indices for B elements
        v00 = j0 * p + k0
        v01 = j0 * p + (k0 + 1)
        v10 = (j0 + 1) * p + k0
        v11 = (j0 + 1) * p + (k0 + 1)
        
        # W indices for C elements
        w00 = k0 * n + i0
        w01 = k0 * n + (i0 + 1)  # WRONG: W index is (k*n + i)
        # Actually W[k*n + i]: 
        # C[i0, k0] -> W[k0*n + i0]
        # C[i0, k0+1] -> W[(k0+1)*n + i0]
        # C[i0+1, k0] -> W[k0*n + (i0+1)]
        # C[i0+1, k0+1] -> W[(k0+1)*n + (i0+1)]
        w00 = k0 * n + i0
        w01 = (k0 + 1) * n + i0
        w10 = k0 * n + (i0 + 1)
        w11 = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (A00+A11)*(B00+B11)
        # We need: A00*B00 + A00*B11 + A11*B00 + A11*B11
        # Each contributes to C00 and C11 with appropriate signs
        # C00 += M1, C11 += M1
        # So: 4 terms for M1, each contributes to both C00 and C11
        
        # Term: A00*B00 -> contributes to C00 (+1) and C11 (+1)
        # We can't have one term contribute to two W positions. So we need separate terms.
        # Actually in tensor decomposition, each rank-1 term has ONE W row.
        # So we need to split: A00*B00 appears in C00 and in C11.
        # That means we need two separate rank-1 terms, OR we combine them.
        
        # The standard way: each M_i is a product of a linear combo of A and linear combo of B.
        # Then M_i contributes to multiple C entries with coefficients.
        # But in our tensor form, each rank-1 term is u_row * v_row * w_row with a single W entry.
        # So we need to expand each M_i into its constituent A*B products, and for each product,
        # it may contribute to multiple C entries.
        
        # Let me think differently. The tensor T[i,j,k] = 1 for matmul.
        # We want T = sum_r U[:,r] V[:,r] W[:,r] (as a 3-tensor)
        # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # and the reconstruction is: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # For Strassen, M1 = (A00+A11)*(B00+B11)
        # This means: the product (A00+A11)*(B00+B11) contributes to C00 and C11.
        # But in our tensor, each term is a single (i,j,k) triple.
        # 
        # Actually, let me reconsider. The Strassen decomposition works as:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # where M1 = (A00+A11)(B00+B11), etc.
        #
        # To express this in tensor form, I need to expand each M_i.
        # M1 = A00*B00 + A00*B11 + A11*B00 + A11*B11
        # Each of these 4 products is a rank-1 term in the matmul tensor for the 2x2x2 block.
        # But M1 contributes to C00 with coefficient +1 and C11 with coefficient +1.
        #
        # The issue: in the tensor decomposition, a single rank-1 term U[:,r]⊗V[:,r]⊗W[:,r] 
        # has W with exactly one nonzero entry (at position k*n+i). So each rank-1 term 
        # contributes to exactly one (i,k) pair of C.
        #
        # Therefore, to have M1 contribute to both C00 and C11, I need TWO rank-1 terms 
        # (or more if M1 has multiple A*B components).
        #
        # This means the "rank 7" for Strassen 2x2x2 is NOT directly applicable in this 
        # tensor factorization form where W has a single nonzero per column.
        #
        # Let me reconsider. In the standard tensor rank of matmul(2,2,2), the rank is 7,
        # but that's for the tensor T[i,j,k] where T[i,j,k]=1 for all i,j,k in {0,1}.
        # The factorization is T = sum_{r=1}^{7} u_r ⊗ v_r ⊗ w_r where u_r ∈ R^4, v_r ∈ R^4, w_r ∈ R^4.
        # Here w_r can have multiple nonzero entries!
        #
        # In our problem, W has shape (p*n, rank) = (20, rank), and W[k*n+i, r] is the entry.
        # So W[:,r] is a vector in R^20, and it CAN have multiple nonzero entries.
        # 
        # So the Strassen decomposition DOES work! Each M_i corresponds to one rank-1 term
        # where:
        #   U[:,r] has entries for the A linear combo (nonzero at the relevant A positions)
        #   V[:,r] has entries for the B linear combo (nonzero at the relevant B positions)
        #   W[:,r] has entries for the C linear combo (nonzero at the relevant C positions with coefficients)
        
        # Great, so let me implement Strassen properly.
        
        # For a 2x2x2 block at (i0, j0, k0):
        # A positions: (i0,j0), (i0,j0+1), (i0+1,j0), (i0+1,j0+1)
        # B positions: (j0,k0), (j0,k0+1), (j0+1,k0), (j0+1,k0+1)
        # C positions: (i0,k0), (i0,k0+1), (i0+1,k0), (i0+1,k0+1)
        
        # U indices:
        uA00 = i0 * m + j0
        uA01 = i0 * m + (j0 + 1)
        uA10 = (i0 + 1) * m + j0
        uA11 = (i0 + 1) * m + (j0 + 1)
        
        # V indices:
        vB00 = j0 * p + k0
        vB01 = j0 * p + (k0 + 1)
        vB10 = (j0 + 1) * p + k0
        vB11 = (j0 + 1) * p + (k0 + 1)
        
        # W indices (W[k*n + i]):
        wC00 = k0 * n + i0
        wC01 = (k0 + 1) * n + i0
        wC10 = k0 * n + (i0 + 1)
        wC11 = (k0 + 1) * n + (i0 + 1)
        
        # Strassen terms:
        # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        add_strassen_term(uA00, uA11, vB00, vB11, wC00, wC11, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        
        # M2 = (A10+A11)(B00) -> C10 += 1, C11 += -1
        add_strassen_term(uA10, uA11, vB00, None, wC10, wC11, 1.0, 1.0, 1.0, 0.0, 1.0, -1.0)
        
        # M3 = A00(B10-B11) -> C01 += 1, C11 += 1
        add_strassen_term(uA00, None, vB10, vB11, wC01, wC11, 1.0, 0.0, 1.0, -1.0, 1.0, 1.0)
        
        # M4 = A11(B01-B10) -> C00 += 1, C10 += 1
        add_strassen_term(uA11, None, vB01, vB10, wC00, wC10, 1.0, 0.0, 1.0, -1.0, 1.0, 1.0)
        
        # M5 = (A00+A01)(B11) -> C00 += -1, C01 += 1
        add_strassen_term(uA00, uA01, vB11, None, wC00, wC01, 1.0, 1.0, 1.0, 0.0, -1.0, 1.0)
        
        # M6 = (A10-A00)(B00+B10) -> C11 += 1
        add_strassen_term(uA10, uA00, vB00, vB10, wC11, None, 1.0, -1.0, 1.0, 1.0, 1.0, 0.0)
        
        # M7 = (A01-A11)(B01+B11) -> C00 += 1
        add_strassen_term(uA01, uA11, vB01, vB11, wC00, None, 1.0, -1.0, 1.0, 1.0, 1.0, 0.0)
    
    def add_strassen_term(u1, u2, v1, v2, w1, w2, cu1, cu2, cv1, cv2, cw1, cw2):
        """Add a Strassen term.
        U has cu1 at u1, cu2 at u2 (if u2 is not None)
        V has cv1 at v1, cv2 at v2 (if v2 is not None)
        W has cw1 at w1, cw2 at w2 (if w2 is not None)
        """
        U_cols.append([(u1, cu1)] + ([(u2, cu2)] if u2 is not None else []))
        V_cols.append([(v1, cv1)] + ([(v2, cv2)] if v2 is not None else []))
        W_cols.append([(w1, cw1)] + ([(w2, cw2)] if w2 is not None else []))
    
    # Now build the blocks
    # i-blocks: [0:2], [2:4]
    # j-blocks: [0:2], [2:4]
    # k-blocks: [0:2], [2:5]
    
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            for k0, nk in [(0, 2), (2, 3)]:
                if nk == 2:
                    # 2x2x2 block: use Strassen
                    add_strassen_block(i0, j0, k0)
                else:
                    # 2x2x3 block: use naive
                    add_naive_block(i0, j0, k0, 2, 2, 3)
    
    # Convert to matrices
    rank = len(U_cols)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        for (idx, val) in U_cols[r]:
            U[idx, r] = val
        for (idx, val) in V_cols[r]:
            V[idx, r] = val
        for (idx, val) in W_cols[r]:
            W[idx, r] = val
    
    # Verify correctness
    if time.time() < deadline:
        # Check: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For random matrices
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                total = 0.0
                for j in range(m):
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                C_computed[i, k] = total
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fallback to schoolbook
            U = np.zeros((n * m, n * m * p))
            V = np.zeros((m * p, n * m * p))
            W = np.zeros((p * n, n * m * p))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1.0
                        V[j * p + k, col] = 1.0
                        W[k * n + i, col] = 1.0
                        col += 1
            rank = n * m * p
    
    # Convert to lists
    U_list = [[int(U[i, j]) if U[i, j] == int(U[i, j]) else float(U[i, j]) for j in range(rank)] for i in range(n * m)]
    V_list = [[int(V[i, j]) if V[i, j] == int(V[i, j]) else float(V[i, j]) for j in range(rank)] for i in range(m * p)]
    W_list = [[int(W[i, j]) if W[i, j] == int(W[i, j]) else float(W[i, j]) for j in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
