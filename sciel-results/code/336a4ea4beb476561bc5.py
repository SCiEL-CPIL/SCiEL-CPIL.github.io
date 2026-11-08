"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    # Build the tensor T[i, j, k] = delta_{i,j,k}
    # The decomposition says: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta[i,j,k]
    # where delta[i,j,k] = 1 if we're checking the (i,j,k) entry, but actually:
    # For matrix multiplication C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is T[i,j,k] = 1 for all (i,j,k) since it's the identity tensor for multiplication.
    # Wait, the multiplication tensor has T[i,j,k] = 1 for all i,j,k.
    # Let me re-derive: sum_c U[i,j,c] V[j,k,c] W[k,i,c] should equal 1 for all i,j,k?
    # No. The tensor T_{ijk} for matrix mult is T_{ijk} = 1 for all i,j,k.
    # Actually no. The matrix multiplication tensor is T_{ijk} where the i-th row of A, j-th column of B...
    # Let's think: C_{ik} = sum_j A_{ij} B_{jk}
    # The tensor has entries T_{ijk} = 1 for all (i,j,k), and the multiplication is sum_j T_{ijk} A_{ij} B_{jk} = sum_j A_{ij} B_{jk} = C_{ik}.
    # So the decomposition requires: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for ALL i in [0,n), j in [0,m), k in [0,p).
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if s != 1:
                    return False
    return True

def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return [U.tolist(), V.tolist(), W.tolist()]

def _strassen_step(n):
    """
    Given a decomposition for n x n x n with U (n^2 x r), V (n^2 x r), W (n^2 x r),
    produce a decomposition for 2n x 2n x 2n.
    
    Strassen 7 products:
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
    # For each of the 7 Strassen products, we need to combine the local n x n decompositions.
    # Each product M_l involves specific row combinations of A and B blocks.
    # In the tensor decomposition, each "multiplication" uses the full U, V, W of the base case.
    
    r_base = 0  # Will be set
    # The 7 products and their A-row coefficients and B-row coefficients:
    # A is 2n x 2n, split into A11, A12, A21, A22 each n x n
    # B is 2n x 2n, split into B11, B12, B21, B22 each n x n
    
    # For product l, we have A_l = sum of certain A blocks with coefficients, B_l = sum of certain B blocks.
    # The U matrix for product l: rows (I, i_local, J, j_local) where I,J in {1,2} block indices
    # Actually U index is (i, j) in the 2n x 2n matrix.
    # i = (I-1)*n + i_local, j = (J-1)*n + j_local
    
    # Let me define the 7 products more carefully:
    # M1: A = A11+A22, B = B11+B22
    # M2: A = A21+A22, B = B11
    # M3: A = A11, B = B12-B22
    # M4: A = A22, B = B21-B11
    # M5: A = A11+A12, B = B22
    # M6: A = A21-A11, B = B11+B12
    # M7: A = A12-A22, B = B21+B22
    
    # C reconstruction:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For the decomposition, we need to build U, V, W for the 2n case.
    # Each of the 7 products contributes r_base columns.
    # Total rank = 7 * r_base.
    
    # U_{2n}[i,j,c] where c = l * r_base + c_local for product l
    # For product l, U_l[i,j,c_local] depends on which A blocks are involved.
    
    # Let me think about this differently.
    # The base decomposition gives us: for n x n x n, sum_c U[i,j,c] V[j',k',c] W[k'',i'',c] = 1 for all i,j',k',k'',i''
    # Wait, that's not right. The base decomposition is for multiplying two n x n matrices.
    # U has rows indexed by (i, j) where i is row of first matrix, j is column of first matrix (and row of second).
    # V has rows indexed by (j, k) where j is row of second matrix, k is column of second matrix.
    # W has rows indexed by (k, i) where k is column of second, i is row of first.
    # And sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k.
    
    # So for the 2n case, we have:
    # U_{2n}[(I-1)*n + i, (J-1)*n + j, c]
    # V_{2n}[(J-1)*n + j, (K-1)*n + k, c]
    # W_{2n}[(K-1)*n + k, (I-1)*n + i, c]
    
    # For product l, the A factor involves certain (I,J) blocks and the B factor involves certain (J,K) blocks.
    # But wait, in Strassen, the "row index" of the combined A matrix and the "row index" of the combined B matrix
    # must match for the multiplication to work. The A factor has a "column" index j and the B factor has a "row" index j'.
    # In the base case, U[i,j,c] and V[j,k,c] share the j index.
    
    # In the 2n case, for product l:
    # A_l = sum_{(I,J) in S_A(l)} coef_A(l, I, J) * A[I, J]
    # B_l = sum_{(J,K) in S_B(l)} coef_B(l, J, K) * B[J, K]
    # The j index in U and V must be consistent. A[I,J] has row I, column J. B[J,K] has row J, column K.
    # So the shared index is J.
    
    # For U_{2n}: row is (i, j) in 2n x 2n, so (I, i_local, J, j_local) where I, J are block indices.
    # For product l, U_l[i, j, c_local] = sum over (I,J) in S_A(l) of coef_A(l, I, J) * U_base[i_local, j_local, c_local]
    # where i_local, j_local are within the n x n block.
    
    # Wait, that's not quite right either. U_base[i, j, c] has i as row of first matrix and j as column of first matrix.
    # In the 2n case, the first matrix is 2n x 2n. Row i in 2n is (I-1)*n + i_local, column j in 2n is (J-1)*n + j_local.
    # So U_{2n}[(I-1)*n + i_local, (J-1)*n + j_local, c] for product l should be:
    # If (I, J) is in S_A(l), then coef_A(l, I, J) * U_base[i_local, j_local, c_local]
    # Otherwise 0.
    
    # Similarly for V_{2n}: row is (j, k) in 2n x 2n, so (J, j_local, K, k_local).
    # V_{2n}[(J-1)*n + j_local, (K-1)*n + k_local, c] for product l should be:
    # If (J, K) is in S_B(l), then coef_B(l, J, K) * V_base[j_local, k_local, c_local]
    # Otherwise 0.
    
    # And W_{2n}: row is (k, i) in 2n x 2n, so (K, k_local, I, i_local).
    # W_{2n}[(K-1)*n + k_local, (I-1)*n + i_local, c] for product l should be:
    # The coefficient for C[I, K] from product l.
    # C[I, K] = sum_l coef_C(l, I, K) * M_l
    # So W_{2n} for product l at position (K, k_local, I, i_local) should be coef_C(l, I, K) * W_base[k_local, i_local, c_local]
    
    # Let me define the coefficients:
    # S_A(l): set of (I, J) block pairs with coefficients for A factor of product l
    # S_B(l): set of (J, K) block pairs with coefficients for B factor of product l
    # coef_C(l, I, K): coefficient of M_l in C[I, K]
    
    # M1: A = A11+A22 -> S_A = {(1,1):1, (2,2):1}
    #     B = B11+B22 -> S_B = {(1,1):1, (2,2):1}
    #     C11 += M1, C22 += M1 -> coef_C(M1, 1,1)=1, coef_C(M1, 2,2)=1
    # M2: A = A21+A22 -> S_A = {(2,1):1, (2,2):1}
    #     B = B11 -> S_B = {(1,1):1}
    #     C21 += M2, C22 -= M2 -> coef_C(M2, 2,1)=1, coef_C(M2, 2,2)=-1
    # M3: A = A11 -> S_A = {(1,1):1}
    #     B = B12-B22 -> S_B = {(1,2):1, (2,2):-1}
    #     C12 += M3, C22 += M3 -> coef_C(M3, 1,2)=1, coef_C(M3, 2,2)=1
    # M4: A = A22 -> S_A = {(2,2):1}
    #     B = B21-B11 -> S_B = {(2,1):1, (1,1):-1}
    #     C11 += M4, C21 += M4 -> coef_C(M4, 1,1)=1, coef_C(M4, 2,1)=1
    # M5: A = A11+A12 -> S_A = {(1,1):1, (1,2):1}
    #     B = B22 -> S_B = {(2,2):1}
    #     C11 -= M5, C12 += M5 -> coef_C(M5, 1,1)=-1, coef_C(M5, 1,2)=1
    # M6: A = A21-A11 -> S_A = {(2,1):1, (1,1):-1}
    #     B = B11+B12 -> S_B = {(1,1):1, (1,2):1}
    #     C22 += M6 -> coef_C(M6, 2,2)=1
    # M7: A = A12-A22 -> S_A = {(1,2):1, (2,2):-1}
    #     B = B21+B22 -> S_B = {(2,1):1, (2,2):1}
    #     C11 += M7 -> coef_C(M7, 1,1)=1
    
    # Note: I, J, K are 1-indexed in the math above, 0-indexed in code.
    # Let me convert to 0-indexed:
    # Blocks: 00, 01, 10, 11 (where first index is row block, second is column block)
    
    products = [
        # (S_A, S_B, C_coefs)
        # S_A: dict from (I,J) to coef
        # S_B: dict from (J,K) to coef
        # C_coefs: dict from (I,K) to coef
        (
            {(0,0): 1, (1,1): 1},  # A11+A22
            {(0,0): 1, (1,1): 1},  # B11+B22
            {(0,0): 1, (1,1): 1}   # C11+M1, C22+M1
        ),
        (
            {(1,0): 1, (1,1): 1},  # A21+A22
            {(0,0): 1},             # B11
            {(1,0): 1, (1,1): -1}  # C21+M2, C22-M2
        ),
        (
            {(0,0): 1},             # A11
            {(0,1): 1, (1,1): -1},  # B12-B22
            {(0,1): 1, (1,1): 1}    # C12+M3, C22+M3
        ),
        (
            {(1,1): 1},             # A22
            {(1,0): 1, (0,0): -1},  # B21-B11
            {(0,0): 1, (1,0): 1}    # C11+M4, C21+M4
        ),
        (
            {(0,0): 1, (0,1): 1},   # A11+A12
            {(1,1): 1},             # B22
            {(0,0): -1, (0,1): 1}   # C11-M5, C12+M5
        ),
        (
            {(1,0): 1, (0,0): -1},  # A21-A11
            {(0,0): 1, (0,1): 1},   # B11+B12
            {(1,1): 1}              # C22+M6
        ),
        (
            {(0,1): 1, (1,1): -1},  # A12-A22
            {(1,0): 1, (1,1): 1},   # B21+B22
            {(0,0): 1}              # C11+M7
        ),
    ]
    
    return products

def _apply_strassen_step(U_base, V_base, W_base, n):
    """
    Apply one Strassen step to go from n x n x n to 2n x 2n x 2n.
    U_base: (n^2, r) list of lists
    V_base: (n^2, r) list of lists
    W_base: (n^2, r) list of lists
    Returns: (U_new, V_new, W_new) for 2n x 2n x 2n
    """
    r = len(U_base[0])
    products = _strassen_step(n)
    
    U_new = [[0] * (7 * r) for _ in range(4 * n * n)]
    V_new = [[0] * (7 * r) for _ in range(4 * n * n)]
    W_new = [[0] * (7 * r) for _ in range(4 * n * n)]
    
    for l, (S_A, S_B, C_coefs) in enumerate(products):
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for c in range(r):
                        u_val = U_base[i * n + j][c]
                        v_val = V_base[j * n + k][c]
                        w_val = W_base[k * n + i][c]
                        
                        col = l * r + c
                        
                        # U_new rows: (I, i, J, j) where I, J are block indices
                        for (I, J), coef_a in S_A.items():
                            u_idx = (I * n + i) * (2 * n) + (J * n + j)
                            U_new[u_idx][col] = coef_a * u_val
                        
                        # V_new rows: (J, j, K, k) where J, K are block indices
                        for (J, K), coef_b in S_B.items():
                            v_idx = (J * n + j) * (2 * n) + (K * n + k)
                            V_new[v_idx][col] = coef_b * v_val
                        
                        # W_new rows: (K, k, I, i) where K, I are block indices
                        for (I, K), coef_c in C_coefs.items():
                            w_idx = (K * n + k) * (2 * n) + (I * n + i)
                            W_new[w_idx][col] = coef_c * w_val
    
    return U_new, V_new, W_new

def _strassen_2x2_to_4x4():
    """
    Build Strassen decomposition for 4x4x4.
    Start with 2x2x2 (rank 7, from Strassen's original 7-product formula for 2x2).
    Then apply one Strassen step to get 4x4x4 (rank 49).
    """
    # 2x2x2 Strassen decomposition
    # The 7 products for 2x2:
    # M1 = (a+g)(e+k)
    # M2 = (g+h)e
    # M3 = a(f-k)
    # M4 = h(d-a)
    # M5 = (a+b)k
    # M6 = (g-a)(e+f)
    # M7 = (b-h)(d+k)
    # s = M1+M4-M5+M7
    # t = M3+M5
    # u = M2+M4
    # v = M1-M2+M3+M6
    
    # For 2x2, the base case is the schoolbook 8-term, but Strassen gives 7.
    # Let me just use the known 7-term Strassen for 2x2.
    # Actually, for the recursive approach, I need a decomposition for 2x2x2.
    # The minimal rank for 2x2x2 is 7 (Strassen).
    # Let me construct it.
    
    n = 2
    r = 7
    U = [[0] * r for _ in range(n * n)]
    V = [[0] * r for _ in range(n * n)]
    W = [[0] * r for _ in range(n * n)]
    
    # For 2x2, indices:
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # i, j, k in {0, 1}
    
    # Let me use the standard Strassen 7 products for 2x2:
    # A = [[a, b], [g, h]], B = [[e, f], [k, l]]
    # C = A*B = [[ae+bk, af+bl], [ge+hk, gf+hl]]
    
    # M1 = (a+g)(e+k) -> U rows: (0,0),(1,0) with coef 1; V rows: (0,0),(0,1) with coef 1
    # Wait, I need to be more careful.
    
    # Let me just use the block Strassen approach starting from 1x1x1.
    # 1x1x1: rank 1, U=[[1]], V=[[1]], W=[[1]]
    # Apply Strassen step to get 2x2x2: rank 7
    # Apply Strassen step to get 4x4x4: rank 49
    
    # 1x1x1
    U_1 = [[1]]
    V_1 = [[1]]
    W_1 = [[1]]
    
    # Apply Strassen step: 1x1 -> 2x2
    U_2, V_2, W_2 = _apply_strassen_step(U_1, V_1, W_1, 1)
    
    # Apply Strassen step: 2x2 -> 4x4
    U_4, V_4, W_4 = _apply_strassen_step(U_2, V_2, W_2, 2)
    
    return U_4, V_4, W_4

def _block_mult(n, m, p, U_base, V_base, W_base):
    """
    Given a decomposition for n x n x n, produce a decomposition for n x m x p
    by splitting B into blocks.
    C = A * B where A is n x m, B is m x p.
    Split B into column blocks of size n (last block may be smaller).
    C = A1 * B1 + A2 * B2 + ... where Ai is n x ni, Bi is ni x pi
    But this doesn't directly work because A is n x m and we need to split by columns of A and rows of B.
    
    Actually, the correct block decomposition is:
    Split A into row blocks and B into corresponding column/row blocks.
    But A is n x m and B is m x p. The shared dimension is m.
    So split m into chunks. For each chunk of size s, we have A_chunk (n x s) and B_chunk (s x p).
    But A_chunk is n x s, not n x n.
    
    A better approach: use the 4x4x4 Strassen decomposition for the "square" part,
    and handle the rectangular part separately.
    
    For 4x4x7:
    Split B (4x7) into B1 (4x4) and B2 (4x3).
    C = A*B1 + A*B2
    A*B1: 4x4 * 4x4 -> use Strassen 4x4x4 (rank 49)
    A*B2: 4x4 * 4x3 -> need a decomposition for this
    
    For 4x4x3:
    We can use the fact that 4x4x3 has known bounds.
    Actually, let me just use schoolbook for the 4x4x3 part.
    Schoolbook rank for 4x4x3 = 4*4*3 = 48.
    Total: 49 + 48 = 97.
    
    Hmm, that's not as good as I hoped. Let me think of a better way.
    
    Alternative: use Strassen for 4x4x4 and then for the remaining 3 columns,
    use a more efficient method.
    
    Actually, for 4x4x3, we can use the 2x2x2 Strassen recursively.
    4x4x3: split into 2x2 blocks.
    A is 4x4, B is 4x3.
    Split A into A11 (2x2), A12 (2x2), A21 (2x2), A22 (2x2).
    Split B into B11 (2x2), B12 (2x1), B21 (2x2), B22 (2x1).
    C = A*B = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
               [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    This gives 8 sub-problems, each of size 2x2x2 or 2x2x1.
    2x2x2: rank 7 (Strassen)
    2x2x1: rank 4 (schoolbook) or less
    
    Total: 4*7 + 4*4 = 28 + 16 = 44.
    Total for 4x4x7: 49 + 44 = 93.
    """
    pass

def construct():
    deadline = time.time() + 140
    
    n, m, p = DIMENSIONS
    
    # Try Strassen for 4x4x4 + optimized for 4x4x3
    try:
        U_4, V_4, W_4 = _strassen_2x2_to_4x4()
        
        if time.time() > deadline:
            return {"U": _make_schoolbook(n, m, p)[0], "V": _make_schoolbook(n, m, p)[1], "W": _make_schoolbook(n, m, p)[2]}
        
        # For 4x4x3, use block decomposition with 2x2 Strassen
        # A (4x4) * B (4x3)
        # Split into 2x2 blocks
        # A = [[A00, A01], [A10, A11]] where each is 2x2
        # B = [[B00, B01], [B10, B11]] where B00 is 2x2, B01 is 2x1, B10 is 2x2, B11 is 2x1
        # C = A*B = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
        #            [A10*B00 + A11*B10, A10*B01 + A11*B11]]
        
        # Sub-problems:
        # 1. A00*B00: 2x2 * 2x2 (rank 7)
        # 2. A01*B10: 2x2 * 2x2 (rank 7)
        # 3. A00*B01: 2x2 * 2x1 (rank 4)
        # 4. A01*B11: 2x2 * 2x1 (rank 4)
        # 5. A10*B00: 2x2 * 2x2 (rank 7)
        # 6. A11*B10: 2x2 * 2x2 (rank 7)
        # 7. A10*B01: 2x2 * 2x1 (rank 4)
        # 8. A11*B11: 2x2 * 2x1 (rank 4)
        
        # Get 2x2x2 Strassen decomposition
        U_1 = [[1]]
        V_1 = [[1]]
        W_1 = [[1]]
        U_2, V_2, W_2 = _apply_strassen_step(U_1, V_1, W_1, 1)
        
        # Get 2x2x1 schoolbook decomposition
        # U: 2x2=4 rows, V: 2x1=2 rows, W: 1x2=2 rows
        U_21 = [[0]*4 for _ in range(4)]
        V_21 = [[0]*4 for _ in range(2)]
        W_21 = [[0]*4 for _ in range(2)]
        for i in range(2):
            for j in range(2):
                for k in range(1):
                    col = i*2 + j
                    U_21[i*2+j][col] = 1
                    V_21[j*1+k][col] = 1
                    W_21[k*2+i][col] = 1
        
        # Now build 4x4x3
        # Total rank: 4*7 + 4*4 = 28 + 16 = 44
        rank_443 = 44
        U_443 = [[0]*rank_443 for _ in range(16)]  # 4*4
        V_443 = [[0]*rank_443 for _ in range(12)]  # 4*3
        W_443 = [[0]*rank_443 for _ in range(12)]  # 3*4
        
        col_offset = 0
        
        # For each sub-problem, map local indices to global indices
        # Sub-problem (I, J) where I is row block of A (0 or 1), J is col block of B (0, 1, or 2)
        # A block (I, J) is rows I*2..I*2+1, cols J*2..J*2+1 (if J=0 or 1) or cols 4..4 (if J=2)
        # B block (J, K) is rows J*2..J*2+1, cols K*2..K*2+1 (if K=0) or cols 2..2 (if K=1)
        # C block (I, K) is rows I*2..I*2+1, cols K*2..K*2+1 (if K=0) or cols 2..2 (if K=1)
        
        # Sub-problems:
        # (I=0, J=0, K=0): A00*B00 -> C00
        # (I=0, J=1, K=0): A01*B10 -> C00
        # (I=0, J=0, K=1): A00*B01 -> C01
        # (I=0, J=1, K=1): A01*B11 -> C01
        # (I=1, J=0, K=0): A10*B00 -> C10
        # (I=1, J=1, K=0): A11*B10 -> C10
        # (I=1, J=0, K=1): A10*B01 -> C10
        # (I=1, J=1, K=1): A11*B11 -> C11
        
        sub_problems = [
            (0, 0, 0, 'sq'),  # A00*B00
            (0, 1, 0, 'sq'),  # A01*B10
            (0, 0, 1, 'rect'),  # A00*B01
            (0, 1, 1, 'rect'),  # A01*B11
            (1, 0, 0, 'sq'),  # A10*B00
            (1, 1, 0, 'sq'),  # A11*B10
            (1, 0, 1, 'rect'),  # A10*B01
            (1, 1, 1, 'rect'),  # A11*B11
        ]
        
        for I, J, K, typ in sub_problems:
            if typ == 'sq':
                U_sub, V_sub, W_sub = U_2, V_2, W_2
                r_sub = 7
            else:
                U_sub, V_sub, W_sub = U_21, V_21, W_21
                r_sub = 4
            
            for local_i in range(2):
                for local_j in range(2):
                    for local_k in range(2 if typ == 'sq' else 1):
                        for c in range(r_sub):
                            col = col_offset + c
                            
                            # U index: global row I*2+local_i, global col J*2+local_j (if sq) or J*2+local_j (if rect, same)
                            # Wait, for rect (2x2x1), local_k only goes 0, and B has 1 column
                            # U row: (I*2+local_i, J*2+local_j)
                            u_row = (I*2 + local_i) * 4 + (J*2 + local_j)
                            U_443[u_row][col] = U_sub[local_i*2 + local_j][c]
                            
                            # V row: (J*2+local_j, K*2+local_k)
                            # For sq: K*2+local_k where local_k in 0,1
                            # For rect: K*1+local_k where local_k in 0
                            if typ == 'sq':
                                v_row = (J*2 + local_j) * 3 + (K*2 + local_k)
                            else:
                                v_row = (J*2 + local_j) * 3 + (K*2 + local_k)
                            V_443[v_row][col] = V_sub[local_j*2 + local_k][c]
                            
                            # W row: (K*2+local_k, I*2+local_i) for sq, (K*1+local_k, I*2+local_i) for rect
                            if typ == 'sq':
                                w_row = (K*2 + local_k) * 4 + (I*2 + local_i)
                            else:
                                w_row = (K*2 + local_k) * 4 + (I*2 + local_i)
                            W_443[w_row][col] = W_sub[local_k*2 + local_i][c]
            
            col_offset += r_sub
        
        # Combine 4x4x4 and 4x4x3
        # 4x4x4: U (16 x 49), V (16 x 49), W (16 x 49)
        # 4x4x3: U (16 x 44), V (12 x 44), W (12 x 44)
        # Total: U (16 x 93), V (28 x 93), W (28 x 93)
        # V: rows 0-15 from 4x4x4, rows 16-27 from 4x4x3
        # W: rows 0-15 from 4x4x4, rows 16-27 from 4x4x3
        # Wait, W for 4x4x4 is (4*4=16 x 49) indexed by (k, i) where k in 0..3, i in 0..3
        # W for 4x4x3 is (3*4=12 x 44) indexed by (k, i) where k in 0..2, i in 0..3
        # Total W: (7*4=28 x 93) indexed by (k, i) where k in 0..6, i in 0..3
        # So W rows 0-15 are from 4x4x4 (k=0..3), W rows 16-27 are from 4x4x3 (k=0..2 mapped to k=4..6)
        
        # V for 4x4x4: (4*4=16 x 49) indexed by (j, k) where j in 0..3, k in 0..3
        # V for 4x4x3: (4*3=12 x 44) indexed by (j, k) where j in 0..3, k in 0..2
        # Total V: (4*7=28 x 93) indexed by (j, k) where j in 0..3, k in 0..6
        # V rows 0-15 from 4x4x4 (k=0..3), V rows 16-27 from 4x4x3 (k=0..2 mapped to k=4..6)
        
        rank_total = 49 + 44
        U_total = [[0]*rank_total for _ in range(16)]
        V_total = [[0]*rank_total for _ in range(28)]
        W_total = [[0]*rank_total for _ in range(28)]
        
        # Copy 4x4x4
        for i in range(16):
            for c in range(49):
                U_total[i][c] = U_4[i][c]
            for k in range(4):
                V_total[i*1 + k][c] = V_4[i*1 + k][c]  # V_4 row i is (j=k, k_local=?) wait
                # V_4 is (16 x 49) indexed by (j, k) where j, k in 0..3
                # So V_4[j*4+k, c]
                # In total, V_total[j*7+k, c] for k in 0..3
        # Let me redo this more carefully.
        
        for i in range(16):
            for c in range(49):
                U_total[i][c] = U_4[i][c]
        
        for j in range(4):
            for k in range(4):
                for c in range(49):
                    V_total[j*7 + k][c] = V_4[j*4 + k][c]
        
        for k in range(4):
            for i in range(4):
                for c in range(49):
                    W_total[k*4 + i][c] = W_4[k*4 + i][c]
        
        # Copy 4x4x3 (offset by 49)
        for i in range(16):
            for c in range(44):
                U_total[i][49 + c] = U_443[i][c]
        
        for j in range(4):
            for k in range(3):
                for c in range(44):
                    V_total[j*7 + 4 + k][c] = V_443[j*3 + k][49 - 49 + c]  # Wait, V_443 is (12 x 44)
                    # V_443[j*3+k, c]
        
        for k in range(3):
            for i in range(4):
                for c in range(44):
                    W_total[(4+k)*4 + i][49 + c] = W_443[k*4 + i][c]
        
        result = {"U": U_total, "V": V_total, "W": W_total}
        
        if time.time() < deadline:
            if _verify(U_total, V_total, W_total, 4, 4, 7):
                return result
        
        # If verification fails, return schoolbook
        school = _make_schoolbook(4, 4, 7)
        return {"U": school[0], "V": school[1], "W": school[2]}
    
    except Exception:
        school = _make_schoolbook(4, 4, 7)
        return {"U": school[0], "V": school[1], "W": school[2]}

# EVOLVE-BLOCK-END
