"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import itertools

def _build_block_strassen():
    """
    Construct a Block-Strassen decomposition for 6x6x6 matrix multiplication.
    
    We partition each 6x6 matrix into 2x2 blocks of size 3x3.
    Strassen's algorithm uses 7 multiplications of 3x3 blocks instead of 8.
    Each 3x3 multiplication is done via schoolbook (27 scalar mults).
    Total rank: 7 * 27 = 189.
    
    The block structure:
    A = [[A11, A12], [A21, A22]]  where each Aij is 3x3
    B = [[B11, B12], [B21, B22]]  where each Bij is 3x3
    C = [[C11, C12], [C21, C22]]  where each Cij is 3x3
    
    Strassen:
    M1 = (A21 + A22) * B12
    M2 = A22 * (B11 - B12)
    M3 = A11 * (B22 + B21)
    M4 = A11 * (B11 - B22)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    
    C11 = M1 + M3 - M5 + M7
    C12 = M5 + M6
    C21 = M2 + M3
    C22 = M1 - M2 + M4 + M6
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Block size is 3 (since 6/2 = 3)
    bs = 3
    
    # We'll build the decomposition column by column.
    # Each Strassen multiplication M_k involves adding/subtracting blocks of A and B,
    # then doing a 3x3x3 schoolbook multiplication (27 terms).
    
    # For the tensor decomposition, we need:
    # U: shape (n*m, rank) = (36, rank)
    # V: shape (m*p, rank) = (36, rank)  
    # W: shape (p*n, rank) = (36, rank)
    
    # The tensor T[i,j,k] = 1 if C[i,k] depends on A[i,j]*B[j,k]
    # Under the trace convention: W[k*n + i, r]
    # U[i*m + j, r] corresponds to A[i,j]
    # V[j*p + k, r] corresponds to B[j,k]
    # W[k*n + i, r] corresponds to C[i,k]
    
    # For each Strassen term M_k = (sum of A blocks) * (sum of B blocks):
    # The product is a 3x3 matrix. Each element (a,b) of the product
    # gets its contribution from all (j) in the inner dimension of the 3x3 blocks.
    
    # Let's define the block positions.
    # Block (br, bc) of a 6x6 matrix covers rows [br*3:(br+1)*3], cols [bc*3:(bc+1)*3]
    # A block A(br, bc) has elements A[br*3+a, bc*3+b] for a,b in 0..2
    # In U indexing: U[(br*3+a)*6 + (bc*3+b), col]
    
    # B block B(br, bc) has elements B[br*3+a, bc*3+b] for a,b in 0..2
    # In V indexing: V[(br*3+a)*6 + (bc*3+b), col]
    
    # C block C(br, bc) has elements C[br*3+a, bc*3+b] for a,b in 0..2
    # In W indexing: W[(bc*3+b)*6 + (br*3+a), col]  (k*n+i where k=col, i=row)
    
    # Strassen definitions:
    # M1 = (A21 + A22) * B12
    #   A blocks: (1,0) and (1,1) -> rows 3-5, cols 0-2 and 3-5
    #   B block: (0,1) -> rows 0-2, cols 3-5
    # M2 = A22 * (B11 - B12)
    #   A block: (1,1)
    #   B blocks: (0,0) and (0,1)
    # M3 = A11 * (B22 + B21)
    #   A block: (0,0)
    #   B blocks: (1,1) and (1,0)
    # M4 = A11 * (B11 - B22)
    #   A block: (0,0)
    #   B blocks: (0,0) and (1,1)
    # M5 = (A11 + A12) * B22
    #   A blocks: (0,0) and (0,1)
    #   B block: (1,1)
    # M6 = (A21 - A11) * (B11 + B12)
    #   A blocks: (1,0) and (0,0)
    #   B blocks: (0,0) and (0,1)
    # M7 = (A12 - A22) * (B21 + B22)
    #   A blocks: (0,1) and (1,1)
    #   B blocks: (1,0) and (1,1)
    
    # C combinations:
    # C11 = M1 + M3 - M5 + M7  -> block (0,0)
    # C12 = M5 + M6            -> block (0,1)
    # C21 = M2 + M3            -> block (1,0)
    # C22 = M1 - M2 + M4 + M6  -> block (1,1)
    
    # For each M_k, we do a 3x3 schoolbook multiplication.
    # M_k = A_comb * B_comb where A_comb is a 3x3 (sum/diff of blocks) and B_comb is 3x3.
    # Element (a,b) of M_k = sum_{j=0}^{2} A_comb[a,j] * B_comb[j,b]
    
    # In the tensor decomposition, each scalar product in the schoolbook part
    # becomes one rank term.
    
    # Let's define the Strassen terms more precisely.
    # Each M_k has:
    #   A-side: a set of (block_row, block_col, sign) for A
    #   B-side: a set of (block_row, block_col, sign) for B
    
    strassen_terms = [
        # (name, A_blocks, B_blocks)
        # A_blocks: list of (block_row, block_col, sign)
        # B_blocks: list of (block_row, block_col, sign)
        ('M1', [(1,0,1), (1,1,1)], [(0,1,1)]),
        ('M2', [(1,1,1)], [(0,0,1), (0,1,-1)]),
        ('M3', [(0,0,1)], [(1,1,1), (1,0,1)]),
        ('M4', [(0,0,1)], [(0,0,1), (1,1,-1)]),
        ('M5', [(0,0,1), (0,1,1)], [(1,1,1)]),
        ('M6', [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        ('M7', [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
    ]
    
    # C block combinations:
    # (c_block_row, c_block_col): list of (M_index, sign)
    c_combos = {
        (0,0): [(0,1), (2,1), (4,-1), (6,1)],   # M1 + M3 - M5 + M7
        (0,1): [(4,1), (5,1)],                    # M5 + M6
        (1,0): [(1,1), (2,1)],                    # M2 + M3
        (1,1): [(0,1), (1,-1), (3,1), (5,1)],     # M1 - M2 + M4 + M6
    }
    
    rank = 0
    U = []
    V = []
    W = []
    
    # For each Strassen term M_k and each (a,b,j) in the 3x3 schoolbook:
    # We create one rank column.
    # U gets the A-side coefficients for element (block_a, a, block_b, j)
    # V gets the B-side coefficients for element (block_b, j, block_c, b)
    # W gets the C-side coefficients for element (block_c, b, block_d, a)
    
    # Wait, let me reconsider. The schoolbook multiplication for 3x3:
    # M[a,b] = sum_{j=0}^{2} A_comb[a,j] * B_comb[j,b]
    # A_comb[a,j] = sum over A blocks (br, bc, s): s * A[br*3+a, bc*3+j]
    # B_comb[j,b] = sum over B blocks (br, bc, s): s * B[br*3+j, bc*3+b]
    
    # So M[a,b] = sum_{j=0}^{2} sum_{(br1,bc1,s1) in A_blocks} sum_{(br2,bc2,s2) in B_blocks}
    #             s1*s2 * A[br1*3+a, bc1*3+j] * B[br2*3+j, bc2*3+b]
    
    # For the tensor decomposition, each (j, br1, bc1, br2, bc2) combination
    # in the double sum is a separate rank term. But actually, we can factor it differently.
    
    # Actually, the standard way: for each M_k, the 3x3 multiplication is itself
    # a tensor. The rank of 3x3x3 multiplication is 27 (schoolbook).
    # So each M_k contributes 27 rank terms.
    
    # For M_k with A_blocks and B_blocks:
    # M_k[a,b] = sum_{j=0}^{2} (sum_{(br1,bc1,s1)} s1*A[br1*3+a, bc1*3+j]) * (sum_{(br2,bc2,s2)} s2*B[br2*3+j, bc2*3+b])
    
    # Expanding: M_k[a,b] = sum_{j} sum_{(br1,bc1,s1)} sum_{(br2,bc2,s2)} s1*s2 * A[br1*3+a, bc1*3+j] * B[br2*3+j, bc2*3+b]
    
    # Each term A[br1*3+a, bc1*3+j] * B[br2*3+j, bc2*3+b] contributes to C elements
    # through the c_combos.
    
    # For the rank decomposition, each (j, br1, bc1, br2, bc2) is one rank term.
    # U[br1*3+a, bc1*3+j, col] = s1
    # V[br2*3+j, bc2*3+b, col] = s2
    # W gets contributions from c_combos.
    
    # But wait - the C element C[c_row*3+ca, c_col*3+cb] receives contributions from
    # multiple M_k terms. Let me restructure.
    
    # Actually, let me think of it as: the full 6x6x6 multiplication tensor has
    # T[i,j,k] = 1 for all i,j,k (216 entries). We want to decompose this into
    # rank terms.
    
    # The Strassen approach: we group the 216 terms into 7 groups of 27 terms each.
    # Group k (for M_k) covers specific (i,j,k) patterns.
    
    # Let me just build it directly.
    
    # For each M_k, for each (a, b, j) in 0..2:
    #   The A-side element is at row (br1*3+a), col (bc1*3+j) for each (br1,bc1,s1) in A_blocks
    #   The B-side element is at row (br2*3+j), col (bc2*3+b) for each (br2,bc2,s2) in B_blocks
    #   The C-side element is at row (c_br*3+a), col (c_bc*3+b) for each c_block in c_combos
    #     where M_k appears with sign s_c
    
    # Actually, I think the cleanest way is:
    # For each M_k, we have a 3x3 "virtual" matrix multiplication.
    # The A virtual matrix has entries A_v[a,j] = sum_{(br1,bc1,s1)} s1 * A[br1*3+a, bc1*3+j]
    # The B virtual matrix has entries B_v[j,b] = sum_{(br2,bc2,s2)} s2 * B[br2*3+j, bc2*3+b]
    # M_v[a,b] = sum_j A_v[a,j] * B_v[j,b]
    
    # In the tensor decomposition, M_v[a,b] contributes to C[c_br*3+a, c_bc*3+b]
    # with coefficient s_c for each (c_br, c_bc, s_c) in c_combos where M_k appears.
    
    # The rank terms for M_k: for each (a, b, j), one rank term where:
    #   U has s1 at position (br1*3+a, bc1*3+j) for each (br1,bc1,s1) in A_blocks
    #   V has s2 at position (br2*3+j, bc2*3+b) for each (br2,bc2,s2) in B_blocks
    #   W has s_c at position (c_bc*3+b, c_br*3+a) for each (c_br,c_bc,s_c) in c_combos[c_block]
    #     where M_k appears
    
    # Wait, W indexing: W[k*n + i, col] where C[i,k] is the element.
    # C[c_br*3+a, c_bc*3+b] -> i = c_br*3+a, k = c_bc*3+b
    # W index = k*n + i = (c_bc*3+b)*6 + (c_br*3+a)
    
    # Let's build it:
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for mk_idx, (name, a_blocks, b_blocks) in enumerate(strassen_terms):
        for a in range(bs):
            for b in range(bs):
                for j in range(bs):
                    u_col = [0] * (n * m)  # 36
                    v_col = [0] * (m * p)  # 36
                    w_col = [0] * (p * n)  # 36
                    
                    # A-side
                    for (br1, bc1, s1) in a_blocks:
                        row = br1 * bs + a
                        col = bc1 * bs + j
                        u_col[row * m + col] = s1
                    
                    # B-side
                    for (br2, bc2, s2) in b_blocks:
                        row = br2 * bs + j
                        col = bc2 * bs + b
                        v_col[row * p + col] = s2
                    
                    # C-side: find which C blocks M_k contributes to
                    for (c_br, c_bc, s_c) in c_combos.items():
                        for (m_idx, sign) in s_c:
                            if m_idx == mk_idx:
                                i = c_br * bs + a
                                k = c_bc * bs + b
                                w_idx = k * n + i
                                w_col[w_idx] += sign * 1  # s1*s2 already in U and V
                                break
                    
                    U_cols.append(u_col)
                    V_cols.append(v_col)
                    W_cols.append(w_col)
    
    rank = len(U_cols)
    
    # Build the matrices
    U = [U_cols[r][:] for r in range(rank)]  # transposed: U is (n*m) x rank
    # Actually, U_cols[r] is a vector of length n*m for rank term r.
    # U[i*m+j, r] = U_cols[r][i*m+j]
    U = [[U_cols[r][idx] for r in range(rank)] for idx in range(n * m)]
    V = [[V_cols[r][idx] for r in range(rank)] for idx in range(m * p)]
    W = [[W_cols[r][idx] for r in range(rank)] for idx in range(p * n)]
    
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(result, n=6, m=6, p=6):
    """Verify that the decomposition is correct."""
    U = result["U"]
    V = result["V"]
    W = result["W"]
    rank = len(U[0]) if U else 0
    
    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    # Verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(total - 1) > 1e-9:
                    return False
    return True


def construct():
    start = time.time()
    deadline = start + 110.0
    
    result = _build_block_strassen()
    
    # Verify correctness
    if not _verify_decomposition(result):
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        result = {"U": U, "V": V, "W": W}
    
    if time.time() > deadline:
        pass
    
    return result
# EVOLVE-BLOCK-END
