"""
Strategy: Clean recursive Strassen decomposition for 4x4 matrix multiplication.
This uses the standard Strassen recursion: 4x4 = 2x2 blocks of 2x2 matrices.
Each level uses 7 multiplications instead of 8, giving rank 7^2 = 49.
The decomposition is built programmatically and verified before returning.
"""
import time
import numpy as np

def _strassen_2x2():
    """
    Returns the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 products for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B10-B11)
    # M4 = (A11)(B01-B10)
    # M5 = (A00+A01)(B11)
    # M6 = (A01-A10)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1       # A00, A11
    V[0, 0] = 1; V[3, 0] = 1       # B00, B11
    W[0, 0] = 1; W[3, 0] = 1       # C00, C11
    
    # Term 1: M2 = (A10+A11)(B00)
    U[2, 1] = 1; U[3, 1] = 1       # A10, A11
    V[0, 1] = 1                    # B00
    W[2, 1] = 1; W[3, 1] = -1      # C10, C11
    
    # Term 2: M3 = (A00)(B10-B11)
    U[0, 2] = 1                    # A00
    V[2, 2] = 1; V[3, 2] = -1      # B10, B11
    W[1, 2] = 1; W[3, 2] = 1       # C01, C11
    
    # Term 3: M4 = (A11)(B01-B10)
    U[3, 3] = 1                    # A11
    V[1, 3] = 1; V[2, 3] = -1      # B01, B10
    W[0, 3] = 1; W[2, 3] = 1       # C00, C10
    
    # Term 4: M5 = (A00+A01)(B11)
    U[0, 4] = 1; U[1, 4] = 1       # A00, A01
    V[3, 4] = 1                    # B11
    W[0, 4] = -1; W[1, 4] = 1      # C00, C01
    
    # Term 5: M6 = (A01-A10)(B00+B10)
    U[1, 5] = 1; U[2, 5] = -1      # A01, A10
    V[0, 5] = 1; V[2, 5] = 1       # B00, B10
    W[3, 5] = 1                    # C11
    
    # Term 6: M7 = (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1      # A01, A11
    V[1, 6] = 1; V[3, 6] = 1       # B01, B11
    W[0, 6] = 1                    # C00
    
    return U, V, W

def _build_strassen_4x4():
    """
    Build the rank-49 Strassen decomposition for 4x4 matrix multiplication.
    Uses 2x2 blocks: A = [[A00, A01], [A10, A11]] where each block is 2x2.
    The Strassen decomposition for the block multiplication gives 7 products,
    each of which is a 2x2 matrix multiplication (rank 7), so total rank = 7*7 = 49.
    
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Get the 2x2 Strassen decomposition
    U2, V2, W2 = _strassen_2x2()
    
    # Block indices: block (bi, bj) corresponds to rows/cols 2*bi..2*bi+1, 2*bj..2*bj+1
    # For 4x4, blocks are indexed (0,0), (0,1), (1,0), (1,1)
    # A block (bi, bj): rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    # In flat index: row*4+col
    
    # Strassen block multiplication:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B10-B11)
    # M4 = (A11)(B01-B10)
    # M5 = (A00+A01)(B11)
    # M6 = (A01-A10)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M_i is a 2x2 matrix product, computed via the rank-7 Strassen decomposition.
    # For each of the 7 block products, we use 7 scalar products (from the 2x2 Strassen).
    # Total: 7 * 7 = 49 scalar products.
    
    # Block A indices:
    # A00: rows 0,1; cols 0,1 -> flat: 0,1,4,5
    # A01: rows 0,1; cols 2,3 -> flat: 2,3,6,7
    # A10: rows 2,3; cols 0,1 -> flat: 8,9,12,13
    # A11: rows 2,3; cols 2,3 -> flat: 10,11,14,15
    
    # Block B indices:
    # B00: rows 0,1; cols 0,1 -> flat: 0,1,4,5
    # B01: rows 0,1; cols 2,3 -> flat: 2,3,6,7
    # B10: rows 2,3; cols 0,1 -> flat: 8,9,12,13
    # B11: rows 2,3; cols 2,3 -> flat: 10,11,14,15
    
    # Block C indices (output):
    # C00: rows 0,1; cols 0,1 -> flat: 0,1,4,5
    # C01: rows 0,1; cols 2,3 -> flat: 2,3,6,7
    # C10: rows 2,3; cols 0,1 -> flat: 8,9,12,13
    # C11: rows 2,3; cols 2,3 -> flat: 10,11,14,15
    
    def block_flat_rows(block_row):
        """Return the 4 flat row indices for a 2x2 block starting at row 2*block_row."""
        base = 2 * block_row
        return [base * 4 + c for c in range(4)]  # This is wrong, let me fix
    
    # Actually, let's define block indices properly.
    # For a 4x4 matrix, element (i, j) has flat index i*4 + j.
    # Block (bi, bj) has elements (i, j) where i in {2*bi, 2*bi+1}, j in {2*bj, 2*bj+1}.
    # Flat indices: (2*bi)*4 + 2*bj, (2*bi)*4 + 2*bj + 1, (2*bi+1)*4 + 2*bj, (2*bi+1)*4 + 2*bj + 1
    
    def get_block_indices(bi, bj):
        """Return the 4 flat indices for block (bi, bj) in a 4x4 matrix."""
        return [
            (2*bi) * 4 + (2*bj),
            (2*bi) * 4 + (2*bj + 1),
            (2*bi + 1) * 4 + (2*bj),
            (2*bi + 1) * 4 + (2*bj + 1)
        ]
    
    # For the 2x2 Strassen, the local indices are 0,1,2,3 corresponding to:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    # Now, for each of the 7 block products, we need to map the local 2x2 indices
    # to the global 4x4 flat indices.
    
    # Block product definitions:
    # (block_A_indices, block_B_indices, coefficient)
    # where block_A_indices is a list of (bi, bj, coeff) for A blocks
    # and block_B_indices is a list of (bi, bj, coeff) for B blocks
    
    block_products = [
        # M1 = (A00+A11)(B00+B11)
        {
            'A': [(0, 0, 1), (1, 1, 1)],
            'B': [(0, 0, 1), (1, 1, 1)],
        },
        # M2 = (A10+A11)(B00)
        {
            'A': [(1, 0, 1), (1, 1, 1)],
            'B': [(0, 0, 1)],
        },
        # M3 = (A00)(B10-B11)
        {
            'A': [(0, 0, 1)],
            'B': [(1, 0, 1), (1, 1, -1)],
        },
        # M4 = (A11)(B01-B10)
        {
            'A': [(1, 1, 1)],
            'B': [(0, 1, 1), (1, 0, -1)],
        },
        # M5 = (A00+A01)(B11)
        {
            'A': [(0, 0, 1), (0, 1, 1)],
            'B': [(1, 1, 1)],
        },
        # M6 = (A01-A10)(B00+B10)
        {
            'A': [(0, 1, 1), (1, 0, -1)],
            'B': [(0, 0, 1), (1, 0, 1)],
        },
        # M7 = (A01-A11)(B01+B11)
        {
            'A': [(0, 1, 1), (1, 1, -1)],
            'B': [(0, 1, 1), (1, 1, 1)],
        },
    ]
    
    # Output block contributions:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    output_contributions = [
        # (output_block, list of (product_index, coefficient))
        ((0, 0), [(0, 1), (3, 1), (4, -1), (6, 1)]),
        ((0, 1), [(2, 1), (4, 1)]),
        ((1, 0), [(1, 1), (3, 1)]),
        ((1, 1), [(0, 1), (1, -1), (2, 1), (5, 1)]),
    ]
    
    # Now build U, V, W
    # For each block product p (0..6), and for each local Strassen term r (0..6):
    # The global term index is p*7 + r
    # U: for each A block (bi, bj, coeff_a) in product p, and for each local U2 index (local_i, r) with value v:
    #   global_flat = get_block_indices(bi, bj)[local_i]
    #   U[global_flat, p*7+r] += coeff_a * v
    # Similarly for V and W.
    
    for p_idx, prod in enumerate(block_products):
        for r in range(7):
            global_term = p_idx * 7 + r
            
            # U contributions
            for (bi, bj, coeff_a) in prod['A']:
                block_indices = get_block_indices(bi, bj)
                for local_i in range(4):
                    val = U2[local_i, r]
                    if val != 0:
                        global_flat = block_indices[local_i]
                        U[global_flat, global_term] += coeff_a * val
            
            # V contributions
            for (bi, bj, coeff_b) in prod['B']:
                block_indices = get_block_indices(bi, bj)
                for local_j in range(4):
                    val = V2[local_j, r]
                    if val != 0:
                        global_flat = block_indices[local_j]
                        V[global_flat, global_term] += coeff_b * val
            
            # W contributions: we need to add to each output block
            for (out_bi, out_bj), contributions in output_contributions:
                for (prod_idx, coeff_w) in contributions:
                    if prod_idx == p_idx:
                        block_indices = get_block_indices(out_bi, out_bj)
                        for local_k in range(4):
                            val = W2[local_k, r]
                            if val != 0:
                                global_flat = block_indices[local_k]
                                W[global_flat, global_term] += coeff_w * val
    
    return U, V, W

def _verify_matrix_mult(U, V, W, n_tests=20):
    """
    Verify that the decomposition exactly computes 4x4 matrix multiplication.
    """
    n, m, p = 4, 4, 4
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    rank = U.shape[1]
    
    U4 = U.reshape(n, m, rank)
    V4 = V.reshape(m, p, rank)
    W4 = W.reshape(p, n, rank)
    
    rng = np.random.default_rng(12345)
    for _ in range(n_tests):
        A = rng.integers(-10, 10, size=(n, m)).astype(float)
        B = rng.integers(-10, 10, size=(m, p)).astype(float)
        C_target = A @ B
        
        term_u = np.einsum('ijr,ij->r', U4, A)
        term_v = np.einsum('jkr,jk->r', V4, B)
        C_recon = np.einsum('kir,r,r->ik', W4, term_u, term_v)
        
        if not np.allclose(C_recon, C_target, atol=1e-9):
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    # Build the rank-49 Strassen decomposition
    U, V, W = _build_strassen_4x4()
    
    if _verify_matrix_mult(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook rank-64
        n, m, p = 4, 4, 4
        rank = 64
        U_fb = np.zeros((n * m, rank))
        V_fb = np.zeros((m * p, rank))
        W_fb = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1.0
                    V_fb[j * p + k, col] = 1.0
                    W_fb[k * n + i, col] = 1.0
                    col += 1
        
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
