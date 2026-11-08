"""Tensor decomposition for 4x4 * 4x6 matrix multiplication using Strassen-style block decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    Indexing: A is 2x2 with A[i,j] at position i*2+j in U rows.
              B is 2x2 with B[j,k] at position j*2+k in V rows.
              C is 2x2 with C[i,k] at position i*2+k in W rows.
    
    Strassen formulas:
      M1 = (A11+A22)(B11+B22)
      M2 = (A21+A22)B11
      M3 = A11(B12-B22)
      M4 = A22(B21-B11)
      M5 = (A11+A12)B22
      M6 = (A21-A11)(B11+B12)
      M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    n = 2
    m = 2
    p = 2
    
    # Each term t has: U contribution (dict from (i,j) -> coeff), V contribution (dict from (j,k) -> coeff), W contribution (dict from (i,k) -> coeff)
    terms = []
    
    # M1 = (A11+A22)(B11+B22)
    # U: (0,0)->1, (1,1)->1
    # V: (0,0)->1, (1,1)->1
    # W: (0,0)->1 (C11), (1,1)->1 (C22)
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    
    # M2 = (A21+A22)B11
    # U: (1,0)->1, (1,1)->1
    # V: (0,0)->1
    # W: (1,0)->1 (C21)
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1}
    })
    
    # M3 = A11(B12-B22)
    # U: (0,0)->1
    # V: (0,1)->1, (1,1)->-1
    # W: (0,1)->1 (C12), (1,1)->1 (C22)
    terms.append({
        'U': {(0,0): 1},
        'V': {(0,1): 1, (1,1): -1},
        'W': {(0,1): 1, (1,1): 1}
    })
    
    # M4 = A22(B21-B11)
    # U: (1,1)->1
    # V: (1,0)->1, (0,0)->-1
    # W: (0,0)->1 (C11), (1,0)->1 (C21)
    terms.append({
        'U': {(1,1): 1},
        'V': {(1,0): 1, (0,0): -1},
        'W': {(0,0): 1, (1,0): 1}
    })
    
    # M5 = (A11+A12)B22
    # U: (0,0)->1, (0,1)->1
    # V: (1,1)->1
    # W: (0,0)->-1 (C11), (0,1)->1 (C12)
    terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,0): -1, (0,1): 1}
    })
    
    # M6 = (A21-A11)(B11+B12)
    # U: (1,0)->1, (0,0)->-1
    # V: (0,0)->1, (0,1)->1
    # W: (1,1)->1 (C22)
    terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(1,1): 1}
    })
    
    # M7 = (A12-A22)(B21+B22)
    # U: (0,1)->1, (1,1)->-1
    # V: (1,0)->1, (1,1)->1
    # W: (0,0)->1 (C11)
    terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1}
    })
    
    return terms


def _build_block_decomp(block_size_a, block_size_b, block_size_c, base_terms):
    """
    Build a block decomposition.
    block_size_a: size of blocks in A (n x m)
    block_size_b: size of blocks in B (m x p)  
    block_size_c: size of blocks in C (n x p)
    base_terms: list of dicts with 'U', 'V', 'W' keys containing coefficient dicts
    """
    n, m, p = block_size_a[0] * block_size_a[1], block_size_b[0] * block_size_b[1], block_size_c[0] * block_size_c[1]
    
    # Actually, let me reconsider. The block structure is:
    # A is n x m, divided into blocks of size (n_b x m_b)
    # B is m x p, divided into blocks of size (m_b x p_b)
    # C is n x p, divided into blocks of size (n_b x p_b)
    # Number of block rows: nr = n / n_b, block cols for A: nc_a = m / m_b
    # For B: block rows: nc_a = m / m_b, block cols: nc_b = p / p_b
    # For C: block rows: nr = n / n_b, block cols: nc_b = p / p_b
    
    # Each base term corresponds to a block multiplication.
    # The base terms use indices (i_block, j_block) for U, (j_block, k_block) for V, (i_block, k_block) for W
    
    return base_terms


def construct():
    deadline = time.time() + 140
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try Strassen-based decomposition
    # A is 4x4, B is 4x6, C is 4x6
    # Use 2x2 blocks: 
    #   A: 2x2 blocks of 2x2 => 2 block rows, 2 block cols
    #   B: 2x3 blocks? No, B is 4x6. If we use 2x2 blocks: 2 block rows, 3 block cols
    #   C: 4x6. If we use 2x2 blocks: 2 block rows, 3 block cols
    
    # This doesn't work cleanly because B has 3 block columns but A has 2 block columns.
    
    # Alternative: use block structure where:
    #   A: split into 2x2 blocks (2 block rows x 2 block cols)
    #   B: split into 2x3? No.
    
    # Let me think differently. 
    # For 4x4 * 4x6:
    #   Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]] each 2x2
    #   Split B into 2x2 blocks: B = [[B11, B12, B13], [B21, B22, B23]] where each is 2x2
    #   C = [[C11, C12, C13], [C21, C22, C23]] where each is 2x2
    #   C11 = A11*B11 + A12*B21
    #   C12 = A11*B12 + A12*B22
    #   C13 = A11*B13 + A12*B23
    #   C21 = A21*B11 + A22*B21
    #   C22 = A21*B12 + A22*B22
    #   C23 = A21*B13 + A22*B23
    
    # Each Ci,j is a sum of two 2x2 * 2x2 products.
    # If we use Strassen for each pair, we can compute each Ci,j with 7 multiplications
    # but Strassen is for a single product, not a sum.
    
    # Actually, for C11 = A11*B11 + A12*B21, we can't directly use Strassen.
    # We'd need to compute A11*B11 and A12*B21 separately.
    
    # Better approach: treat the whole thing as blocks where we can apply Strassen at a higher level.
    
    # Let's try: 
    #   A = [[A1, A2], [A3, A4]] where each is 2x2
    #   B = [[B1, B2, B3], [B4, B5, B6]] where each is 2x2
    #   C = [[C1, C2, C3], [C4, C5, C6]] where each is 2x2
    #   
    #   C1 = A1*B1 + A2*B4
    #   C2 = A1*B2 + A2*B5
    #   C3 = A1*B3 + A2*B6
    #   C4 = A3*B1 + A4*B4
    #   C5 = A3*B2 + A4*B5
    #   C6 = A3*B3 + A4*B6
    
    # We can view this as:
    #   [C1 C2 C3] = A1 * [B1 B2 B3] + A2 * [B4 B5 B6]
    #   [C4 C5 C6] = A3 * [B1 B2 B3] + A4 * [B4 B5 B6]
    
    # Let P = [B1 B2 B3] (2x6) and Q = [B4 B5 B6] (2x6)
    # Let R = [C1 C2 C3] (2x6) and S = [C4 C5 C6] (2x6)
    # R = A1*P + A2*Q
    # S = A3*P + A4*Q
    
    # We can compute A1*P, A2*Q, A3*P, A4*Q each using Strassen-style if P and Q are structured.
    # But P is 2x6, not 2x2.
    
    # Alternative: just compute each 2x2 * 2x2 product using Strassen (7 terms each)
    # and add them up.
    # Total: 6 C-blocks * 2 products each * 7 terms = 84 terms
    # Plus we need to handle the additions.
    
    # Actually, let me just build it explicitly.
    
    # For each output block C_ij (2x2), it's a sum of two 2x2*2x2 products.
    # I'll use Strassen for each product and combine.
    
    strassen_terms = _strassen_2x2x2()
    
    # Build the full decomposition
    # Each term in the full decomposition will be a scaled/shifted version of a Strassen term
    # applied to specific blocks.
    
    full_U = []
    full_V = []
    full_W = []
    
    # Map from (block_row, block_col) to actual indices
    # For A (4x4): block (br, bc) corresponds to rows 2*br to 2*br+1, cols 2*bc to 2*bc+1
    # For B (4x6): block (br, bc) corresponds to rows 2*br to 2*br+1, cols 3*bc to 3*bc+1... wait, 6/2=3 so cols 2*bc to 2*bc+1
    # For C (4x6): block (br, bc) corresponds to rows 2*br to 2*br+1, cols 2*bc to 2*bc+1
    
    # Wait, B is 4x6. If blocks are 2x2, then B has 2 block rows and 3 block cols.
    # Block (br, bc) of B: rows 2*br to 2*br+1, cols 2*bc to 2*bc+1
    
    # A block (ai, aj): rows 2*ai to 2*ai+1, cols 2*aj to 2*aj+1
    # B block (bj, bk): rows 2*bj to 2*bj+1, cols 2*bk to 2*bk+1
    # C block (ci, ck): rows 2*ci to 2*ci+1, cols 2*ck to 2*ck+1
    
    # C[ci, ck] = A[ci, 0] * B[0, ck] + A[ci, 1] * B[1, ck]
    
    # So for each ci in {0,1}, ck in {0,1,2}:
    #   C_block(ci, ck) = A_block(ci, 0) * B_block(0, ck) + A_block(ci, 1) * B_block(1, ck)
    
    # For each such pair, use Strassen to compute the product, then add.
    
    for ci in range(2):
        for ck in range(3):
            # First product: A_block(ci, 0) * B_block(0, ck)
            for st in strassen_terms:
                U_row = [0] * (n * m)
                V_row = [0] * (m * p)
                W_row = [0] * (p * n)
                
                for (si, sj), coeff in st['U'].items():
                    # A_block(ci, 0): local (si, sj) maps to global row 2*ci + si, col 2*0 + sj
                    gi = 2 * ci + si
                    gj = 2 * 0 + sj
                    U_row[gi * m + gj] = coeff
                
                for (sj, sk), coeff in st['V'].items():
                    # B_block(0, ck): local (sj, sk) maps to global row 2*0 + sj, col 2*ck + sk
                    gj = 2 * 0 + sj
                    gk = 2 * ck + sk
                    V_row[gj * p + gk] = coeff
                
                for (si, sk), coeff in st['W'].items():
                    # C_block(ci, ck): local (si, sk) maps to global row 2*ci + si, col 2*ck + sk
                    gi = 2 * ci + si
                    gk = 2 * ck + sk
                    W_row[gk * n + gi] = coeff
                
                full_U.append(U_row)
                full_V.append(V_row)
                full_W.append(W_row)
            
            # Second product: A_block(ci, 1) * B_block(1, ck)
            for st in strassen_terms:
                U_row = [0] * (n * m)
                V_row = [0] * (m * p)
                W_row = [0] * (p * n)
                
                for (si, sj), coeff in st['U'].items():
                    # A_block(ci, 1): local (si, sj) maps to global row 2*ci + si, col 2*1 + sj
                    gi = 2 * ci + si
                    gj = 2 * 1 + sj
                    U_row[gi * m + gj] = coeff
                
                for (sj, sk), coeff in st['V'].items():
                    # B_block(1, ck): local (sj, sk) maps to global row 2*1 + sj, col 2*ck + sk
                    gj = 2 * 1 + sj
                    gk = 2 * ck + sk
                    V_row[gj * p + gk] = coeff
                
                for (si, sk), coeff in st['W'].items():
                    # C_block(ci, ck): local (si, sk) maps to global row 2*ci + si, col 2*ck + sk
                    gi = 2 * ci + si
                    gk = 2 * ck + sk
                    W_row[gk * n + gi] = coeff
                
                full_U.append(U_row)
                full_V.append(V_row)
                full_W.append(W_row)
    
    U_strassen = np.array(full_U, dtype=float)
    V_strassen = np.array(full_V, dtype=float)
    W_strassen = np.array(full_W, dtype=float)
    
    rank_strassen = len(full_U)
    
    # Verify the Strassen decomposition
    # Check: sum_t U[:,t] * V[:,t]^T should give the multiplication tensor
    # More precisely: for all i,j,k: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    
    def verify_decomposition(U, V, W, n, m, p):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for t in range(rank):
                        total += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                    if abs(total - 1) > 1e-6:
                        return False
        return True
    
    if verify_decomposition(U_strassen, V_strassen, W_strassen, n, m, p):
        U_list = U_strassen.tolist()
        V_list = V_strassen.tolist()
        W_list = W_strassen.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U_list = U_school.tolist()
    V_list = V_school.tolist()
    W_list = W_school.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
