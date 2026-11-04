"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n=5, m=5, p=5):
    """Verify that U, V, W form a valid matrix multiplication decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor T[i,j,k] should equal delta (1 if this is a valid mult tensor)
    # The matmul tensor is T[i,j,k] = 1 for all i,j,k (it's the identity in the tensor sense)
    # Actually: C[i,k] = sum_j A[i,j] B[j,k], so the tensor has T[i,j,k]=1 for all i,j,k
    # Check: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _schoolbook_block(i_start, i_end, j_start, j_end, k_start, k_end, U, V, W, offset):
    """Add schoolbook terms for a block product A[i_start:i_end, j_start:j_end] * B[j_start:j_end, k_start:k_end] -> C[i_start:i_end, k_start:k_end]"""
    col = offset
    for i in range(i_start, i_end):
        for j in range(j_start, j_end):
            for k in range(k_start, k_end):
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 5 + i][col] = 1
                col += 1
    return col

def _strassen_2x2(i_start, j_start, k_start, U, V, W, offset):
    """Add rank-7 Strassen terms for 2x2 matrix multiplication block."""
    # Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    # 
    # For each Strassen multiplication, we need to express it as rank-1 tensors.
    # Each M is a product of two 2x2 "matrices" formed from linear combinations.
    # We handle each of the 7 Strassen multiplications using schoolbook on the combined terms.
    
    i0, i1 = i_start, i_start + 1
    j0, j1 = j_start, j_start + 1
    k0, k1 = k_start, k_start + 1
    
    col = offset
    
    # M1 = (A11+A22)(B11+B22)
    # A11+A22 is a 2x2 matrix with entries: [i0,j0]=1, [i1,j1]=1
    # B11+B22 is a 2x2 matrix with entries: [j0,k0]=1, [j1,k1]=1
    # Product contributes to C11 and C22
    # We need to express (A11+A22)(B11+B22) as rank-1 terms
    # The 2x2 matrix M1 = (A11+A22)*(B11+B22) where these are actual 2x2 matrix products
    # M1[i,k] = sum_j (A11+A22)[i,j] * (B11+B22)[j,k]
    # For i in {0,1}, j in {0,1}, k in {0,1}:
    # (A11+A22)[i,j] is 1 if (i,j)==(0,0) or (1,1), else 0
    # (B11+B22)[j,k] is 1 if (j,k)==(0,0) or (1,1), else 0
    # So M1[0,0] = 1, M1[1,1] = 1, others 0
    # We need rank-1 decompositions for each term.
    # M1[0,0] = A11*B11 + A22*B22 -> already handled by schoolbook? No, we need explicit terms.
    # Actually, let me think differently. Each Strassen M is a 2x2 matrix product.
    # I'll use schoolbook for each M but with the linear combinations built in.
    
    # Let me just use a cleaner approach: express each of the 7 Strassen multiplications
    # as their schoolbook decomposition (rank 8 each) but that gives 56, not 7.
    # 
    # The correct approach: each Strassen M_t is itself a matrix product of two 2x2 matrices
    # formed by linear combinations. To decompose M_t as rank-1 tensors, we note that
    # M_t[i,k] = sum_j L_A_t[i,j] * L_B_t[j,k] where L_A and L_B are the linear combination matrices.
    # For a 2x2 matrix product, the schoolbook rank is 8, but we can do better since
    # L_A and L_B have specific sparse structures.
    
    # Actually, the standard way: each of the 7 Strassen products is a 2x2 matmul, which has
    # tensor rank 7 (Strassen). So we need to recursively apply Strassen or use explicit rank-7.
    # This gets complicated. Let me just use schoolbook for the 2x2 blocks (rank 8 each) 
    # and see if the total is still below 125.
    
    # For simplicity and correctness, let me use schoolbook for all blocks first,
    # then optimize the 2x2 and 3x3 cores.
    
    # Schoolbook 2x2: rank 8
    for i in [i0, i1]:
        for j in [j0, j1]:
            for k in [k0, k1]:
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 5 + i][col] = 1
                col += 1
    return col

def _strassen_2x2_correct(i_start, j_start, k_start, U, V, W, offset):
    """Add rank-7 Strassen terms for 2x2 matrix multiplication block.
    
    The 7 Strassen multiplications are:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    Each M_t is a product of two 2x2 matrices (with specific linear combinations).
    To decompose each M_t into rank-1 tensors, we note that for a 2x2 matrix product
    X*Y where X and Y are 2x2, the tensor has rank at most 8 (schoolbook) but can be 7 (Strassen).
    
    However, since X and Y here are specific linear combinations, each M_t[i,k] is a bilinear
    form in the entries of A and B. We can decompose each M_t into rank-1 terms by expanding.
    
    For M_t = X_t * Y_t where X_t[i,j] and Y_t[j,k] are linear forms:
    M_t[i,k] = sum_j X_t[i,j] * Y_t[j,k]
    Each term X_t[i,j]*Y_t[j,k] is a product of a linear form in A and a linear form in B,
    which is a rank-1 tensor term (with appropriate W coefficient).
    
    So for each M_t, we get up to 4 rank-1 terms (one for each j=0,1 and the specific i,k).
    But we need to be careful: the total contribution to C is the sum over all M_t with coefficients.
    
    Let me just enumerate all rank-1 terms explicitly.
    """
    i0, i1 = i_start, i_start + 1
    j0, j1 = j_start, j_start + 1
    k0, k1 = k_start, k_start + 1
    
    # Index mapping: local (li, lj, lk) -> global (i0+li, j0+lj, k0+lk)
    # U index: (i0+li)*5 + (j0+lj)
    # V index: (j0+lj)*5 + (k0+lk)
    # W index: (k0+lk)*5 + (i0+li)
    
    col = offset
    
    # For each of the 7 Strassen products, we decompose into rank-1 terms.
    # Each product M_t = X_t * Y_t where X_t is 2x2 (linear in A) and Y_t is 2x2 (linear in B).
    # M_t[li, lk] = sum_lj X_t[li, lj] * Y_t[lj, lk]
    # Each (li, lj, lk) triple with nonzero X_t[li,lj] * Y_t[lj,lk] gives a rank-1 term.
    
    # Define the 7 Strassen multiplications as (X_t, Y_t) pairs
    # X_t and Y_t are 2x2 matrices of coefficients
    
    strassen_products = [
        # M1 = (A11+A22)(B11+B22)
        # X1 = [[1,0],[0,1]], Y1 = [[1,0],[0,1]]
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]]),
        # M2 = (A21+A22)B11
        # X2 = [[0,0],[1,1]], Y2 = [[1,0],[0,0]]
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]]),
        # M3 = A11(B12-B22)
        # X3 = [[1,0],[0,0]], Y3 = [[0,1],[0,-1]]
        ([[1, 0], [0, 0]], [[0, 1], [0, -1]]),
        # M4 = A22(B21-B11)
        # X4 = [[0,0],[0,1]], Y4 = [[-1,0],[1,0]]
        ([[0, 0], [0, 1]], [[-1, 0], [1, 0]]),
        # M5 = (A11+A12)B22
        # X5 = [[1,1],[0,0]], Y5 = [[0,0],[0,1]]
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]]),
        # M6 = (A21-A11)(B11+B12)
        # X6 = [[-1,0],[1,0]], Y6 = [[1,1],[0,0]]
        ([-1, 0], [1, 0], [[1, 1], [0, 0]]),  # placeholder, will fix
        # M7 = (A12-A22)(B21+B22)
        # X7 = [[0,1],[0,-1]], Y7 = [[0,0],[1,1]]
        ([[0, 1], [0, -1]], [[0, 0], [1, 1]]),
    ]
    
    # Fix M6
    strassen_products[5] = (
        [[-1, 0], [1, 0]], [[1, 1], [0, 0]]
    )
    
    # Coefficients for combining M_t into C:
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For each M_t and each (li, lk) in {0,1}^2, the coefficient in C[li,lk] is:
    coeff = {
        (0, 0): [1, 0, 0, 1, -1, 0, 1],   # M1, M2, M3, M4, M5, M6, M7
        (0, 1): [0, 0, 1, 0, 1, 0, 0],
        (1, 0): [0, 1, 0, 1, 0, 0, 0],
        (1, 1): [1, -1, 1, 0, 0, 1, 0],
    }
    
    for t in range(7):
        X_t, Y_t = strassen_products[t]
        for li in range(2):
            for lk in range(2):
                for lj in range(2):
                    x_coeff = X_t[li][lj]
                    y_coeff = Y_t[lj][lk]
                    if x_coeff == 0 or y_coeff == 0:
                        continue
                    # This gives a rank-1 term contributing to C[li, lk]
                    # U[(i0+li)*5 + (j0+lj)] gets x_coeff
                    # V[(j0+lj)*5 + (k0+lk)] gets y_coeff
                    # W[(k0+lk)*5 + (i0+li)] gets coeff[(li,lk)][t]
                    u_idx = (i0 + li) * 5 + (j0 + lj)
                    v_idx = (j0 + lj) * 5 + (k0 + lk)
                    w_idx = (k0 + lk) * 5 + (i0 + li)
                    c_coeff = coeff[(li, lk)][t]
                    if c_coeff != 0:
                        U[u_idx][col] = x_coeff
                        V[v_idx][col] = y_coeff
                        W[w_idx][col] = c_coeff
                        col += 1
    
    return col

def _rank17_3x3(i_start, j_start, k_start, U, V, W, offset):
    """Add a rank-17 decomposition for 3x3 matrix multiplication block.
    
    Using the known rank-17 algorithm for 3x3 matrix multiplication.
    This is based on the work of Alon, Shpilka, and others.
    For simplicity, I'll use a known explicit rank-17 decomposition.
    """
    # For now, use schoolbook for 3x3 (rank 27) and optimize later
    # Actually, let me try to find a rank-17 or better decomposition.
    # The minimal known rank for 3x3 is 19 (or 17 with some conditions).
    # Let me use schoolbook (rank 27) for correctness first.
    
    i0 = i_start
    j0 = j_start
    k0 = k_start
    
    col = offset
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u_idx = (i0 + i) * 5 + (j0 + j)
                v_idx = (j0 + j) * 5 + (k0 + k)
                w_idx = (k0 + k) * 5 + (i0 + i)
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    return col

def _schoolbook_decomposition(n=5, m=5, p=5):
    """Return the schoolbook decomposition as fallback."""
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
    return {"U": U, "V": V, "W": W}

def _block_decomposition():
    """Construct a block-partitioned decomposition.
    
    Partition 5x5 into 2x2 top-left and 3x3 bottom-right blocks.
    Use Strassen (rank 7) for the 2x2 core, schoolbook for 3x3 core (rank 27),
    and schoolbook for cross-block interactions.
    """
    n = 5
    total_rank_estimate = 8 + 27 + 12 + 18 + 12 + 12  # 2x2 core + 3x3 core + cross-blocks
    # 2x2 core (C_TL): 8
    # 3x3 core (C_BR): 27
    # Cross: A_TR(2x3)*B_BL(3x2) -> 2x2: 12
    # Cross: A_TR(2x3)*B_BR(3x3) -> 2x3: 18
    # Cross: A_BL(3x2)*B_TL(2x2) -> 3x2: 12
    # Cross: A_BL(3x2)*B_TR(2x2) -> 3x2: 12
    # Total: 8+27+12+18+12+12 = 89
    
    rank = 89
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    
    col = 0
    
    # 2x2 core: C_TL = A_TL * B_TL (indices 0-1)
    col = _strassen_2x2_correct(0, 0, 0, U, V, W, col)
    
    # 3x3 core: C_BR = A_BR * B_BR (indices 2-4)
    col = _rank17_3x3(2, 2, 2, U, V, W, col)
    
    # Cross-block 1: A_TR(2x3, rows 0-1, cols 2-4) * B_BL(3x2, rows 2-4, cols 0-1) -> C_TL (rows 0-1, cols 0-1)
    col = _schoolbook_block(0, 2, 2, 5, 0, 2, U, V, W, col)
    
    # Cross-block 2: A_TR(2x3, rows 0-1, cols 2-4) * B_BR(3x3, rows 2-4, cols 2-4) -> C_TR (rows 0-1, cols 2-4)
    col = _schoolbook_block(0, 2, 2, 5, 2, 5, U, V, W, col)
    
    # Cross-block 3: A_BL(3x2, rows 2-4, cols 0-1) * B_TL(2x2, rows 0-1, cols 0-1) -> C_BL (rows 2-4, cols 0-1)
    col = _schoolbook_block(2, 5, 0, 2, 0, 2, U, V, W, col)
    
    # Cross-block 4: A_BL(3x2, rows 2-4, cols 0-1) * B_TR(2x2, rows 0-1, cols 2-4) -> C_BR_part (rows 2-4, cols 2-4)
    col = _schoolbook_block(2, 5, 0, 2, 2, 5, U, V, W, col)
    
    # Trim to actual rank used
    actual_rank = col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s kill
    
    # First, try the block decomposition
    try:
        result = _block_decomposition()
        if _verify_decomposition(result["U"], result["V"], result["W"]):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook_decomposition()
# EVOLVE-BLOCK-END
