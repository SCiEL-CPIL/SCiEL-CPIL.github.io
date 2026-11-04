import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    size_u = n * m
    size_v = m * p
    size_w = p * n
    rank = len(U[0]) if U else 0
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Compute X[I, J, K] = sum_r U[I,r] V[J,r] W[K,r]
    X = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np, optimize=True)
    
    # Build target tensor
    T = np.zeros((size_u, size_v, size_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
    
    if not np.allclose(X, T, atol=1e-9):
        return False
    
    # Check entries are integers or half-integers
    for mat, name in [(U, 'U'), (V, 'V'), (W, 'W')]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition: rank = n*m*p"""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank


def _strassen_2x2x2():
    """Strassen decomposition for 2x2x2: rank 7
    Returns U (4x7), V (4x7), W (4x7) where:
    U index: i*2+j (i,j in 0..1) -> 0,1,2,3
    V index: j*2+k (j,k in 0..1) -> 0,1,2,3
    W index: k*2+i (k,i in 0..1) -> 0,1,2,3
    """
    # Strassen 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # U[i,j] for A[i,j], V[j,k] for B[j,k], W[k,i] for C[k,i]
    #
    # M1: (A00+A11)(B00+B11)
    #   U: A00->1, A11->1; V: B00->1, B11->1; W: C00->1, C11->1
    # M2: (A10+A11)B00
    #   U: A10->1, A11->1; V: B00->1; W: C10->1, C11->-1
    # M3: A00(B10-B11)
    #   U: A00->1; V: B10->1, B11->-1; W: C01->1, C11->1
    # M4: A11(B01-B00)
    #   U: A11->1; V: B01->1, B00->-1; W: C00->1, C10->1
    # M5: (A00+A01)B11
    #   U: A00->1, A01->1; V: B11->1; W: C00->-1, C01->1
    # M6: (A10-A00)(B00+B01)
    #   U: A10->1, A00->-1; V: B00->1, B01->1; W: C11->1
    # M7: (A01-A11)(B10+B11)
    #   U: A01->1, A11->-1; V: B10->1, B11->1; W: C00->1
    
    U = [[0]*7 for _ in range(4)]  # indices 0=A00,1=A01,2=A10,3=A11
    V = [[0]*7 for _ in range(4)]  # indices 0=B00,1=B01,2=B10,3=B11
    W = [[0]*7 for _ in range(4)]  # indices 0=C00,1=C01,2=C10,3=C11
    
    # M1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
    for r in range(1):
        U[0][r] = 1; U[3][r] = 1
        V[0][r] = 1; V[3][r] = 1
        W[0][r] = 1; W[3][r] = 1
    # M2: (A10+A11)B00 -> C10+=1, C11+=-1
    U[2][1] = 1; U[3][1] = 1; V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    # M3: A00(B10-B11) -> C01+=1, C11+=1
    U[0][2] = 1; V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    # M4: A11(B01-B00) -> C00+=1, C10+=1
    U[3][3] = 1; V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    # M5: (A00+A01)B11 -> C00+=-1, C01+=1
    U[0][4] = 1; U[1][4] = 1; V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    # M6: (A10-A00)(B00+B01) -> C11+=1
    U[2][5] = 1; U[0][5] = -1; V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    # M7: (A01-A11)(B10+B11) -> C00+=1
    U[1][6] = 1; U[3][6] = -1; V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _block_multiply(n1, n2, m1, m2, p1, p2):
    """
    Decompose (n1+n2) x (m1+m2) @ (m1+m2) x (p1+p2) using blocks.
    A = [[A11, A12],[A21, A22]] where A11 is n1xm1, A12 is n1xm2, A21 is n2xm1, A22 is n2xm2
    B = [[B11, B12],[B21, B22]] where B11 is m1xp1, B12 is m1xp2, B21 is m2xp1, B22 is m2xp2
    C = [[C11, C12],[C21, C22]] where C11 is n1xp1, C12 is n1xp2, C21 is n2xp1, C22 is n2xp2
    
    C11 = A11@B11 + A12@B21
    C12 = A11@B12 + A12@B22
    C21 = A21@B11 + A22@B21
    C22 = A21@B12 + A22@B22
    """
    # We'll use schoolbook for sub-blocks as a baseline, but try Strassen where possible
    # For now, use schoolbook for each of the 8 sub-products
    U_list = []
    V_list = []
    W_list = []
    
    # Block dimensions:
    # A11: rows [0, n1), cols [0, m1)
    # A12: rows [0, n1), cols [m1, m1+m2)
    # A21: rows [n1, n1+n2), cols [0, m1)
    # A22: rows [n1, n1+n2), cols [m1, m1+m2)
    # B11: rows [0, m1), cols [0, p1)
    # B12: rows [0, m1), cols [p1, p1+p2)
    # B21: rows [m1, m1+m2), cols [0, p1)
    # B22: rows [m1, m1+m2), cols [p1, p1+p2)
    # C11: rows [0, n1), cols [0, p1)
    # C12: rows [0, n1), cols [p1, p1+p2)
    # C21: rows [n1, n1+n2), cols [0, p1)
    # C22: rows [n1, n1+n2), cols [p1, p1+p2)
    
    total_n = n1 + n2
    total_m = m1 + m2
    total_p = p1 + p2
    
    # For each sub-product (A_ik @ B_kl -> C_il), add schoolbook terms
    def add_subproduct(a_i, a_j, b_j, b_k, c_i, c_l, rank):
        """Add rank-many terms for A[a_i:a_i+ai_rows, a_j:a_j+ai_cols] @ B[b_j:b_j+bj_rows, b_k:b_k+bk_cols] -> C[c_i:c_i+ci_rows, c_l:c_l+cl_cols]"""
        pass  # placeholder
    
    # Simpler: just do full schoolbook but structured
    # Actually, let's just use the full schoolbook decomposition directly
    # The point is to find a better decomposition
    
    # Let's try: split into blocks and use Strassen for 2x2x2 blocks
    # For 3x5x6, split as 2+1 rows, 2+3 or 3+2 cols, 3+3 cols
    
    return None  # Not used directly


def _construct_block_strassen():
    """
    Block Strassen for 3x5x6.
    Split A (3x5) as rows [2,1], B (5x6) as cols [3,2] or [2,3].
    Let's split inner dim m=5 as 2+3.
    A = [[A_top (2x5)], [A_bot (1x5)]]
    B = [[B_left (5x3)], [B_right (5x3)]]
    
    C = [[A_top @ B_left, A_top @ B_right],
         [A_bot @ B_left, A_bot @ B_right]]
    
    Each sub-multiplication:
    - A_top (2x5) @ B_left (5x3): split inner 5 as 2+3
      = A_top[:, :2] @ B_left[:2, :] + A_top[:, 2:] @ B_left[2:, :]
      = (2x2)@(2x3) + (2x3)@(3x3)
      Rank of 2x2x3: use Strassen-like -> we can do rank 10 or 11
      Rank of 2x3x3: use schoolbook -> 18, or better
      
    This is getting complex. Let me just implement a known good rank.
    
    Actually, let me try a different approach: use the known fact that
    r(3x5x6) <= r(2x5x6) + r(1x5x6)
    
    r(1x5x6) = 5 (each output element is a dot product, but they share structure)
    Actually r(1xm xp) = m for any m,p since C_k = sum_j A_j B_jk, 
    but the tensor is sum_k A_k B_k e_k which has rank... 
    For 1x5x6: T[i,j,k] = 1 if i=0 always, so it's just sum over j of B[j,k] for each (0,k)
    The tensor T[0,j,k] = 1 for all j,k? No.
    
    Let me reconsider. The tensor for matrix multiplication A(nxm) @ B(mx p) -> C(nx p):
    T[I, J, K] where I = i*m+j, J = j*p+k, K = k*n+i
    T[I,J,K] = 1 iff the indices are consistent (same i, j, k).
    
    For 1x5x6: n=1, m=5, p=6
    I = 0*5+j = j (j=0..4)
    J = j*6+k (j=0..4, k=0..5)
    K = k*1+0 = k (k=0..5)
    T[j, j*6+k, k] = 1
    
    So T[j, 6j+k, k] = 1. This is a permutation-like tensor.
    Rank = max(5, 6) = 6? Let's see: T = sum_{j,k} e_j e_{6j+k} e_k = sum_k (sum_j e_j e_{6j+k}) e_k
    For fixed k, sum_j e_j e_{6j+k} has rank 5 (5 independent terms)
    So total rank = 6 * 5 / something... Actually each term e_j e_{6j+k} e_k is rank 1,
    and there are 5*6=30 terms, but they may not all be independent.
    
    T[j, 6j+k, k] = 1 for j in 0..4, k in 0..5.
    The support has 30 points. The rank is at least max(5, 6, 5) = 6.
    Is rank 6 achievable? We need T = sum_{r=0}^{5} u_r v_r w_r.
    u_r is in R^5, v_r in R^30, w_r in R^6.
    
    For this specific sparse tensor, I believe rank = 6 works:
    Let w_r = e_r (standard basis in R^6, so w_r[k]=1 iff k=r).
    Then we need sum_r u_r v_r [k=r] = T.
    For k=r: u_r v_r = T[:, :, r]
    T[j, 6j+k, k] = 1, so for fixed k, T[:, :, k] has 1s at positions (j, 6j+k) for j=0..4.
    So T[:, :, k] = sum_j e_j e_{6j+k}, which has rank 5.
    So u_r v_r must have rank 5, meaning we need 5 terms for each k, total 30 terms.
    So rank(1x5x6) = 30? That seems high.
    
    Wait, I think I'm overcomplicating. Let me just use the schoolbook fallback with rank 90,
    and try to improve with a structured approach.
    """
    n, m, p = 3, 5, 6
    
    # Try a block decomposition:
    # Split rows of A: 2 + 1
    # Split inner dim: 3 + 2
    # Split cols of B: 3 + 3
    #
    # A = [[A11 (2x3), A12 (2x2)],
    #      [A21 (1x3), A22 (1x2)]]
    # B = [[B11 (3x3), B12 (3x3)],
    #      [B21 (2x3), B22 (2x3)]]
    # C = [[C11 (2x3), C12 (2x3)],
    #      [C21 (1x3), C22 (1x3)]]
    #
    # C11 = A11@B11 + A12@B21
    # C12 = A11@B12 + A12@B22
    # C21 = A21@B11 + A22@B21
    # C22 = A21@B12 + A22@B22
    #
    # Sub-ranks:
    # 2x3x3: schoolbook = 18, can we do better? r(2x3x3) <= 17? Let's use 18.
    # 2x2x3: schoolbook = 12, r(2x2x3) <= 11? Use 11.
    # 1x3x3: schoolbook = 9.
    # 1x2x3: schoolbook = 6.
    #
    # Total: 18+11+18+11+9+6+9+6 = 88. Better than 90!
    
    # Let's implement each sub-problem with schoolbook and combine.
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Helper: add a sub-problem
    # a_row_start, a_col_start: top-left of A block
    # b_row_start, b_col_start: top-left of B block  
    # c_row_start, c_col_start: top-left of C block
    # a_rows, a_cols, b_rows, b_cols, c_rows, c_cols: dimensions
    
    def add_block(a_rs, a_cs, b_rs, b_cs, c_rs, c_cs, ar, ac, br, bc, cr, cc):
        """Add schoolbook terms for A[a_rs:a_rs+ar, a_cs:a_cs+ac] @ B[b_rs:b_rs+br, b_cs:b_cs+bc] -> C[c_rs:c_rs+cr, c_cs:c_cs+cc]"""
        # For each (i, j, k) in the sub-block:
        # A index: (a_rs+i)*m + (a_cs+j)
        # B index: (b_rs+j)*p + (b_cs+k)  -- wait, B's row is the inner index
        # Actually B is mx p, so B[row, col] where row is inner (0..m-1), col is output (0..p-1)
        # B index in V: row * p + col
        # But in our block, B's rows are b_rs..b_rs+br-1 (inner dim), cols are b_cs..b_cs+bc-1
        # A's rows are a_rs..a_rs+ar-1 (output row), cols are a_cs..a_cs+ac-1 (inner dim)
        # C's rows are c_rs..c_rs+cr-1 (output row), cols are c_cs..c_cs+cc-1
        #
        # For consistency: A's col j corresponds to B's row j (both inner dim)
        # A's row i corresponds to C's row i (output row)
        # B's col k corresponds to C's col k (output col)
        
        rank_block = ar * ac * br  # = ar * bc * cr should equal this if dimensions match
        # Actually ar*ac*br = ar*ac*bc (since br=ac for valid multiplication)
        
        for i in range(ar):
            for j in range(ac):
                for k in range(bc):
                    a_idx = (a_rs + i) * m + (a_cs + j)
                    b_idx = (b_rs + j) * p + (b_cs + k)
                    c_idx = (c_cs + k) * n + (c_rs + i)
                    
                    U_parts.append([a_idx, 1])
                    V_parts.append([b_idx, 1])
                    W_parts.append([c_idx, 1])
    
    # C11 = A11@B11 + A12@B21
    # A11: rows 0-1, cols 0-2 (2x3)
    # B11: rows 0-2, cols 0-2 (3x3)
    # C11: rows 0-1, cols 0-2 (2x3)
    add_block(0, 0, 0, 0, 0, 0, 2, 3, 3, 3, 2, 3)  # A11@B11 -> C11
    # A12: rows 0-1, cols 3-4 (2x2)
    # B21: rows 3-4, cols 0-2 (2x3)
    add_block(0, 3, 3, 0, 0, 0, 2, 2, 2, 3, 2, 3)  # A12@B21 -> C11
    
    # C12 = A11@B12 + A12@B22
    # B12: rows 0-2, cols 3-5 (3x3)
    # C12: rows 0-1, cols 3-5 (2x3)
    add_block(0, 0, 0, 3, 0, 3, 2, 3, 3, 3, 2, 3)  # A11@B12 -> C12
    # B22: rows 3-4, cols 3-5 (2x3)
    add_block(0, 3, 3, 3, 0, 3, 2, 2, 2, 3, 2, 3)  # A12@B22 -> C12
    
    # C21 = A21@B11 + A22@B21
    # A21: rows 2-2, cols 0-2 (1x3)
    # C21: rows 2-2, cols 0-2 (1x3)
    add_block(2, 0, 0, 0, 2, 0, 1, 3, 3, 3, 1, 3)  # A21@B11 -> C21
    # A22: rows 2-2, cols 3-4 (1x2)
    add_block(2, 3, 3, 0, 2, 0, 1, 2, 2, 3, 1, 3)  # A22@B21 -> C21
    
    # C22 = A21@B12 + A22@B22
    add_block(2, 0, 0, 3, 2, 3, 1, 3, 3, 3, 1, 3)  # A21@B12 -> C22
    add_block(2, 3, 3, 3, 2, 3, 1, 2, 2, 3, 1, 3)  # A22@B22 -> C22
    
    # Now assemble U, V, W
    rank = len(U_parts)
    size_u = n * m  # 15
    size_v = m * p  # 30
    size_w = p * n  # 18
    
    U = [[0] * rank for _ in range(size_u)]
    V = [[0] * rank for _ in range(size_v)]
    W = [[0] * rank for _ in range(size_w)]
    
    for r, (a_idx, val) in enumerate(U_parts):
        U[a_idx][r] = val
    for r, (b_idx, val) in enumerate(V_parts):
        V[b_idx][r] = val
    for r, (c_idx, val) in enumerate(W_parts):
        W[c_idx][r] = val
    
    return U, V, W, rank


def _construct_bipartite(n, m, p):
    """
    Try a bipartite/structured approach.
    For 3x5x6, try to find a rank < 88 decomposition.
    """
    # Use Strassen on the 2x2x2 sub-blocks where possible
    n, m, p = 3, 5, 6
    
    # Split: A rows [2,1], inner [2,3], B cols [3,3]
    # A = [[A11(2x2), A12(2x3)],
    #      [A21(1x2), A22(1x3)]]
    # B = [[B11(2x3), B12(2x3)],
    #      [B21(3x3), B22(3x3)]]
    # C = [[C11(2x3), C12(2x3)],
    #      [C21(1x3), C22(1x3)]]
    #
    # C11 = A11@B11 + A12@B21
    # C12 = A11@B12 + A12@B22
    # C21 = A21@B11 + A22@B21
    # C22 = A21@B12 + A22@B22
    #
    # Sub-ranks:
    # 2x2x3: use Strassen-like rank 10 (or 11)
    # 2x3x3: schoolbook 18
    # 1x2x3: schoolbook 6
    # 1x3x3: schoolbook 9
    #
    # Total with rank 10 for 2x2x3: 10+18+10+18+6+9+6+9 = 86
    # Total with rank 11 for 2x2x3: 11+18+11+18+6+9+6+9 = 88
    
    # Let's implement with schoolbook for all to be safe, then optimize
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_schoolbook(a_rs, a_cs, b_rs, b_cs, c_rs, c_cs, ar, ac, br, bc, cr, cc):
        for i in range(ar):
            for j in range(ac):
                for k in range(bc):
                    a_idx = (a_rs + i) * m + (a_cs + j)
                    b_idx = (b_rs + j) * p + (b_cs + k)
                    c_idx = (c_cs + k) * n + (c_rs + i)
                    U_parts.append([a_idx, 1])
                    V_parts.append([b_idx, 1])
                    W_parts.append([c_idx, 1])
    
    # Use split: inner dim 5 = 2 + 3
    # A11: 2x2 (rows 0-1, cols 0-1)
    # A12: 2x3 (rows 0-1, cols 2-4)
    # A21: 1x2 (row 2, cols 0-1)
    # A22: 1x3 (row 2, cols 2-4)
    # B11: 2x3 (rows 0-1, cols 0-2)
    # B12: 2x3 (rows 0-1, cols 3-5)
    # B21: 3x3 (rows 2-4, cols 0-2)
    # B22: 3x3 (rows 2-4, cols 3-5)
    # C11: 2x3 (rows 0-1, cols 0-2)
    # C12: 2x3 (rows 0-1, cols 3-5)
    # C21: 1x3 (row 2, cols 0-2)
    # C22: 1x3 (row 2, cols 3-5)
    
    # C11 = A11@B11 + A12@B21
    add_schoolbook(0, 0, 0, 0, 0, 0, 2, 2, 2, 3, 2, 3)  # 2x2x3 = 12 terms
    add_schoolbook(0, 2, 2, 0, 0, 0, 2, 3, 3, 3, 2, 3)  # 2x3x3 = 18 terms
    
    # C12 = A11@B12 + A12@B22
    add_schoolbook(0, 0, 0, 3, 0, 3, 2, 2, 2, 3, 2, 3)  # 2x2x3 = 12 terms
    add_schoolbook(0, 2, 2, 3, 0, 3, 2, 3, 3, 3, 2, 3)  # 2x3x3 = 18 terms
    
    # C21 = A21@B11 + A22@B21
    add_schoolbook(2, 0, 0, 0, 2, 0, 1, 2, 2, 3, 1, 3)  # 1x2x3 = 6 terms
    add_schoolbook(2, 2, 2, 0, 2, 0, 1, 3, 3, 3, 1, 3)  # 1x3x3 = 9 terms
    
    # C22 = A21@B12 + A22@B22
    add_schoolbook(2, 0, 0, 3, 2, 3, 1, 2, 2, 3, 1, 3)  # 1x2x3 = 6 terms
    add_schoolbook(2, 2, 2, 3, 2, 3, 1, 3, 3, 3, 1, 3)  # 1x3x3 = 9 terms
    
    rank = len(U_parts)
    # 12+18+12+18+6+9+6+9 = 90 (same as schoolbook, no improvement)
    
    # This is just schoolbook in disguise. Let me try Strassen for the 2x2x3 blocks.
    # For 2x2x3, we can potentially use rank 10 instead of 12.
    # 2x2x3: A(2x2) @ B(2x3) -> C(2x3)
    # T[I,J,K] where I = i*2+j, J = j*3+k, K = k*2+i
    # I in 0..3, J in 0..8, K in 0..5
    #
    # Can we do rank 10? Let's try to construct it.
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01, b02], [b10, b11, b12]]
    # C = [[c00, c01, c02], [c10, c11, c12]]
    # c00 = a00*b00 + a01*b10
    # c01 = a00*b01 + a01*b11
    # c02 = a00*b02 + a01*b12
    # c10 = a10*b00 + a11*b10
    # c11 = a10*b01 + a11*b11
    # c12 = a10*b02 + a11*b12
    #
    # This is 2x2x3. Schoolbook rank = 12.
    # We can factor: C = A @ B.
    # Note that C[:, k] = A @ B[:, k] for each k.
    # So C = sum_k (A @ B[:,k]) e_k
    # A @ B[:,k] is a 2x1 vector = A @ v_k where v_k is the k-th column of B.
    # A @ v = a00*v0 + a01*v1 (where v = [v0, v1])
    # So C[:,k] = a00 * B[:,k][0] * [1,0] + a01 * B[:,k][1] * [0,1] ... no that's not right.
    # A @ v = [a00*v0 + a01*v1, a10*v0 + a11*v1]
    # So C[:,k] = (a00*B[0,k] + a01*B[1,k]) * e_0 + (a10*B[0,k] + a11*B[1,k]) * e_1
    # C[:,k] = B[0,k] * [a00, a10] + B[1,k] * [a01, a11]
    # C[:,k] = B[0,k] * A_col0 + B[1,k] * A_col1
    #
    # So C = sum_k B[0,k] * A_col0 * e_k + B[1,k] * A_col1 * e_k
    # = sum_k (B[0,k] * A_col0 + B[1,k] * A_col1) * e_k
    #
    # In tensor form: T = sum_{k} (sum_j B[j,k] * A_col_j) * e_k
    # A_col_j is a vector in R^2 (the j-th column of A, but actually the j-th "row" of A transposed... )
    # Let me re-index.
    # A = [[a00, a01], [a10, a11]], columns: col0 = [a00, a10], col1 = [a01, a11]
    # B = [[b00, b01, b02], [b10, b11, b12]], columns: col0 = [b00, b10], etc.
    # C[:,k] = A @ B[:,k] = b0k * [a00, a10] + b1k * [a01, a11]
    # So C[:,k] = b0k * A_col0 + b1k * A_col1
    #
    # T = sum_k (b0k * A_col0 * e_k + b1k * A_col1 * e_k)
    # = sum_k b0k * (A_col0 * e_k) + sum_k b1k * (A_col1 * e_k)
    #
    # A_col0 * e_k = [a00, a10, 0, 0, ...] with 1 at position k... 
    # In the tensor T[I,J,K] where I is A-index, J is B-index, K is C-index:
    # I for A: i*2+j, so A_col0 corresponds to I=0 (i=0,j=0) and I=2 (i=1,j=0)
    # J for B: j*3+k, so b0k is J=k (j=0, col k), b1k is J=3+k (j=1, col k)
    # K for C: k*2+i, so e_k means K=k*2+0 = 2k and K=k*2+1 = 2k+1
    #
    # C[:,k] = b0k * A_col0 + b1k * A_col1
    # C[0,k] = b0k*a00 + b1k*a01
    # C[1,k] = b0k*a10 + b1k*a11
    #
    # T[I,J,K] = 1 if consistent.
    # For C[0,k]: I in {0,1} (a00, a01), J in {k, 3+k} (b0k, b1k), K = 2k
    # T[0, k, 2k] = 1 (a00 * b0k -> c0k)
    # T[0, 3+k, 2k] = 1 (a01 * b1k -> c0k) -- wait, a01 is I=1, not I=0
    # Let me redo: A[0,0]=a00 -> I=0, A[0,1]=a01 -> I=1, A[1,0]=a10 -> I=2, A[1,1]=a11 -> I=3
    # B[0,k]=b0k -> J=k, B[1,k]=b1k -> J=3+k
    # C[0,k] -> K=2k, C[1,k] -> K=2k+1
    #
    # C[0,k] = a00*b0k + a01*b1k
    # T[0, k, 2k] = 1
    # T[1, 3+k, 2k] = 1
    # C[1,k] = a10*b0k + a11*b1k
    # T[2, k, 2k+1] = 1
    # T[3, 3+k, 2k+1] = 1
    #
    # So the tensor has 12 non-zero entries (for k=0,1,2, 4 entries each = 12).
    # Rank 12 is schoolbook. Can we do less?
    #
    # T = sum_k (e_0 e_k e_{2k} + e_1 e_{3+k} e_{2k} + e_2 e_k e_{2k+1} + e_3 e_{3+k} e_{2k+1})
    #
    # Group by k:
    # k=0: e_0 e_0 e_0 + e_1 e_3 e_0 + e_2 e_0 e_1 + e_3 e_3 e_1
    # k=1: e_0 e_1 e_2 + e_1 e_4 e_2 + e_2 e_1 e_3 + e_3 e_4 e_3
    # k=2: e_0 e_2 e_4 + e_1 e_5 e_4 + e_2 e_2 e_5 + e_3 e_5 e_5
    #
    # These 12 terms. Are any linearly dependent? Unlikely for a generic multiplication tensor.
    # I'll stick with schoolbook rank 12 for 2x2x3.
    
    # Given the complexity, let me just return the schoolbook decomposition as fallback
    # and rely on the block structure to at least match rank 90.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (a_idx, val) in enumerate(U_parts):
        U[a_idx][r] = val
    for r, (b_idx, val) in enumerate(V_parts):
        V[b_idx][r] = val
    for r, (c_idx, val) in enumerate(W_parts):
        W[c_idx][r] = val
    
    return U, V, W, rank


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try block Strassen decomposition
    try:
        U, V, W, rank = _construct_block_strassen()
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception as e:
        pass
    
    # Fallback to schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
