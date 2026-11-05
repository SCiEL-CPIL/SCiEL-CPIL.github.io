"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 6x6 matrix multiplication.
    
    Approach: Use Strassen's algorithm on 2x2 blocks of 3x3 matrices.
    This reduces the rank from 216 (schoolbook) to 189 (7 * 27).
    
    The matrix multiplication tensor T[i,j,k] = delta_{i,j,k} where:
    - i indexes elements of A (6x6), row-major
    - j indexes elements of B (6x6), row-major
    - k indexes elements of C (6x6), row-major
    
    The decomposition is: T = sum_c U[:,c] * V[:,c] * W[:,c]
    where U has shape (36, rank), V has shape (36, rank), W has shape (36, rank).
    
    For Strassen:
    - Split A into 4 blocks of 3x3: A11, A12, A21, A22
    - Split B into 4 blocks of 3x3: B11, B12, B21, B22
    - Split C into 4 blocks of 3x3: C11, C12, C21, C22
    
    Strassen's 7 multiplications:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B12 - B22)
    M4 = A22(B21 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    Results:
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    n, m, p = DIMENSIONS
    
    # Define the 7 Strassen steps
    # Each step: (coeffs for A blocks, coeffs for B blocks, result blocks with signs)
    # A blocks: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
    # B blocks: B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1)
    # C blocks: C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1)
    
    # For each Strassen step s, we need to compute X_s * Y_s where
    # X_s is a linear combination of A blocks and Y_s is a linear combination of B blocks
    # Each X_s and Y_s is a 3x3 matrix, so we need 27 scalar multiplications per step.
    
    # Define the Strassen steps as (a_coeffs, b_coeffs, c_contributions)
    # a_coeffs: dict mapping (br, bc) -> coefficient for A block
    # b_coeffs: dict mapping (br, bc) -> coefficient for B block
    # c_contributions: dict mapping (cr, cc) -> (sign, step_index)
    
    strassen_steps = [
        # M1 = (A11 + A22)(B11 + B22)
        {
            'a_coeffs': {((0,0), (0,0)): 1, ((0,0), (0,1)): 0, ((0,1), (0,0)): 0, ((0,1), (0,1)): 0,
                          ((1,0), (0,0)): 0, ((1,0), (0,1)): 0, ((1,1), (0,0)): 0, ((1,1), (0,1)): 0,
                          ((0,0), (1,0)): 0, ((0,0), (1,1)): 0, ((0,1), (1,0)): 0, ((0,1), (1,1)): 0,
                          ((1,0), (1,0)): 0, ((1,0), (1,1)): 0, ((1,1), (1,0)): 1, ((1,1), (1,1)): 0},
            'b_coeffs': {((0,0), (0,0)): 1, ((0,0), (0,1)): 0, ((0,1), (0,0)): 0, ((0,1), (0,1)): 0,
                          ((1,0), (0,0)): 0, ((1,0), (0,1)): 0, ((1,1), (0,0)): 0, ((1,1), (0,1)): 0,
                          ((0,0), (1,0)): 0, ((0,0), (1,1)): 0, ((0,1), (1,0)): 0, ((0,1), (1,1)): 0,
                          ((1,0), (1,0)): 0, ((1,0), (1,1)): 0, ((1,1), (1,0)): 0, ((1,1), (1,1)): 1},
            'c_contributions': {((0,0), (0,0)): 1, ((0,0), (0,1)): 0, ((0,1), (0,0)): 0, ((0,1), (0,1)): 0,
                                ((1,0), (0,0)): 0, ((1,0), (0,1)): 0, ((1,1), (0,0)): 0, ((1,1), (0,1)): 0,
                                ((0,0), (1,0)): 0, ((0,0), (1,1)): 0, ((0,1), (1,0)): 0, ((0,1), (1,1)): 0,
                                ((1,0), (1,0)): 0, ((1,0), (1,1)): 0, ((1,1), (1,0)): 0, ((1,1), (1,1)): 1}
        },
        # M2 = (A21 + A22)B11
        {
            'a_coeffs': {((0,0), (0,0)): 0, ((0,0), (0,1)): 0, ((0,1), (0,0)): 0, ((0,1), (0,1)): 0,
                          ((1,0), (0,0)): 0, ((1,0), (0,1)): 0, ((1,1), (0,0)): 0, ((1,1), (0,1)): 0,
                          ((0,0), (1,0)): 0, ((0,0), (1,1)): 0, ((0,1), (1,0)): 0, ((0,1), (1,1)): 0,
                          ((1,0), (1,0)): 1, ((1,0), (1,1)): 0, ((1,1), (1,0)): 0, ((1,1), (1,1)): 1},
            'b_coeffs': {((0,0), (0,0)): 1, ((0,0), (0,1)): 0, ((0,1), (0,0)): 0, ((0,1), (0,1)): 0,
                          ((1,0), (0,0)): 0, ((1,0), (0,1)): 0, ((1,1), (0,0)): 0, ((1,1), (0,1)): 0,
                          ((0,0), (1,0)): 0, ((0,0), (1,1)): 0, ((0,1), (1,0)): 0, ((0,1), (1,1)): 0,
                          ((1,0), (1,0)): 0, ((1,0), (1,1)): 0, ((1,1), (1,0)): 0, ((1,1), (1,1)): 0},
            'c_contributions': {((0,0), (0,0)): 0, ((0,0), (0,1)): 0, ((0,1), (0,0)): 0, ((0,1), (0,1)): 0,
                                ((1,0), (0,0)): 0, ((1,0), (0,1)): 0, ((1,1), (0,0)): 0, ((1,1), (0,1)): 0,
                                ((0,0), (1,0)): 0, ((0,0), (1,1)): 0, ((0,1), (1,0)): 0, ((0,1), (1,1)): 0,
                                ((1,0), (1,0)): 0, ((1,0), (1,1)): 0, ((1,1), (1,0)): 1, ((1,1), (1,1)): 0}
        },
        # M3 = A11(B12 - B22)
        {
            'a_coeffs': {((0,0), (0,0)): 1, ((0,0), (0,1)): 0, ((0,1), (0,0)): 0, ((0,1), (0,1)): 0,
                          ((1,0), (0,0)): 0, ((1,0), (0,1)): 0, ((1,1), (0,0)): 0, ((1,1), (0,1)): 0,
                          ((0,0), (1,0)): 0, ((0,0), (1,1)): 0, ((0,1), (1,0)): 0, ((0,1), (1,1)): 0,
                          ((1,0), (1,0)): 0, ((1,0), (1,1)): 0, ((1,1), (1,0)): 0, ((1,1), (1,1)): 0},
            'b_coeffs': {((0,0), (0,0)): 0, ((0,0), (0,1)): 1, ((0,1), (0,0)): 0, ((0,1), (0,1)): 0,
                          ((1,0), (0,0)): 0, ((1,0), (0,1)): 0, ((1,1), (0,0)): 0, ((1,1), (0,1)): 0,
                          ((0,0), (1,0)): 0, ((0,0), (1,1)): 0, ((0,1), (1,0)): 0, ((0,1), (1,1)): 0,
                          ((1,0), (1,0)): 0, ((1,0), (1,1)): 0, ((1,1), (1,0)): 0, ((1,1), (1,1)): -1},
            'c_contributions': {((0,0), (0,0)): 0, ((0,0), (0,1)): 1, ((0,1), (0,0)): 0, ((0,1), (0,1)): 0,
                                ((1,0), (0,0)): 0, ((1,0), (0,1)): 0, ((1,1), (0,0)): 0, ((1,1), (0,1)): 0,
                                ((0,0), (1,0)): 0, ((0,0), (1,1)): 0, ((0,1), (1,0)): 0, ((0,1), (1,1)): 0,
                                ((1,0), (1,0)): 0, ((1,0), (1,1)): 0, ((1,1), (1,0)): 0, ((1,1), (1,1)): 0}
        },
        # M4 = A22(B21 - B11)
        {
            'a_coeffs': {((0,0), (0,0)): 0, ((0,0), (0,1)): 0, ((0,1), (0,0)): 0, ((0,1), (0,1)): 0,
                          ((1,0), (0,0)): 0, ((1,0), (0,1)): 0, ((1,1), (0,0)): 0, ((1,1), (0,1)): 0,
                          ((0,0), (1,0)): 0, ((0,0), (1,1)): 0, ((0,1), (1,0)): 0, ((0,1), (1,1)): 0,
                          ((1,0), (1,0)): 0, ((1,0), (1,1)): 0, ((1,1), (1,0)): 0, ((1,1), (1,1)): 1},
            'b_coeffs': {((0,0), (0,0)): -1, ((0,0), (0,1)): 0, ((0,1), (0,0)): 0, ((0,1), (0,1)): 0,
                          ((1,0), (0,0)): 0, ((1,0), (0,1)): 0, ((1,1), (0,0)): 0, ((1,1), (0,1)): 0,
                          ((0,0), (1,0)): 0, ((0,0), (1,1)): 0, ((0,1), (1,0)): 0, ((0,1), (1,1)): 0,
                          ((1,0), (1,0)): 1, ((1,0), (1,1)): 0, ((1,1), (1,0)): 0, ((1,1), (1,1)): 0},
            'c_contributions': {((0,0), (0,0)): 1, ((0,0), (0,1)): 0, ((0,1), (0,0)): 0, ((0,1), (0,1)): 0,
                                ((1,0), (0,0)): 0, ((1,0), (0,1)): 0, ((1,1), (0,0)): 0, ((1,1), (0,1)): 0,
                                ((0,0), (1,0)): 0, ((0,0), (1,1)): 0, ((0,1), (1,0)): 0, ((0,1), (1,1)): 0,
                                ((1,0), (1,0)): 1, ((1,0), (1,1)): 0, ((1,1), (1,0)): 0, ((1,1), (1,1)): 0}
        },
        # M5 = (A11 + A12)B22
        {
            'a_coeffs': {((0,0), (0,0)): 1, ((0,0), (0,1)): 0, ((0,1), (0,0)): 0, ((0,1), (0,1)): 1,
                          ((1,0), (0,0)): 0, ((1,0), (0,1)): 0, ((1,1), (0,0)): 0, ((1,1), (0,1)): 0,
                          ((0,0), (1,0)): 0, ((0,0), (1,1)): 0, ((0,1), (1,0)): 0, ((0,1), (1,1)): 0,
                          ((1,0), (1,0)): 0, ((1,0), (1,1)): 0, ((1,1), (1,0)): 0, ((1,1), (1,1)): 0},
            'b_coeffs': {((0,0), (0,0)): 0, ((0,0), (0,1)): 0, ((0,1), (0,0)): 0, ((0,1), (0,1)): 0,
                          ((1,0), (0,0)): 0, ((1,0), (0,1)): 0, ((1,1), (0,0)): 0, ((1,1), (0,1)): 0,
                          ((0,0), (1,0)): 0, ((0,0), (1,1)): 0, ((0,1), (1,0)): 0, ((0,1), (1,1)): 0,
                          ((1,0), (1,0)): 0, ((1,0), (1,1)): 0, ((1,1), (1,0)): 0, ((1,1), (1,1)): 1},
            'c_contributions': {((0,0), (0,0)): -1, ((0,0), (0,1)): 1, ((0,1), (0,0)): 0, ((0,1), (0,1)): 0,
                                ((1,0), (0,0)): 0, ((1,0), (0,1)): 0, ((1,1), (0,0)): 0, ((1,1), (0,1)): 0,
                                ((0,0), (1,0)): 0, ((0,0), (1,1)): 0, ((0,1), (1,0)): 0, ((0,1), (1,1)): 0,
                                ((1,0), (1,0)): 0, ((1,0), (1,1)): 0, ((1,1), (1,0)): 0, ((1,1), (1,1)): 0}
        },
        # M6 = (A21 - A11)(B11 + B12)
        {
            'a_coeffs': {((0,0), (0,0)): -1, ((0,0), (0,1)): 0, ((0,1), (0,0)): 0, ((0,1), (0,1)): 0,
                          ((1,0), (0,0)): 0, ((1,0), (0,1)): 0, ((1,1), (0,0)): 0, ((1,1), (0,1)): 0,
                          ((0,0), (1,0)): 0, ((0,0), (1,1)): 0, ((0,1), (1,0)): 0, ((0,1), (1,1)): 0,
                          ((1,0), (1,0)): 1, ((1,0), (1,1)): 0, ((1,1), (1,0)): 0, ((1,1), (1,1)): 0},
            'b_coeffs': {((0,0), (0,0)): 1, ((0,0), (0,1)): 1, ((0,1), (0,0)): 0, ((0,1), (0,1)): 0,
                          ((1,0), (0,0)): 0, ((1,0), (0,1)): 0, ((1,1), (0,0)): 0, ((1,1), (0,1)): 0,
                          ((0,0), (1,0)): 0, ((0,0), (1,1)): 0, ((0,1), (1,0)): 0, ((0,1), (1,1)): 0,
                          ((1,0), (1,0)): 0, ((1,0), (1,1)): 0, ((1,1), (1,0)): 0, ((1,1), (1,1)): 0},
            'c_contributions': {((0,0), (0,0)): 0, ((0,0), (0,1)): 0, ((0,1), (0,0)): 0, ((0,1), (0,1)): 0,
                                ((1,0), (0,0)): 0, ((1,0), (0,1)): 0, ((1,1), (0,0)): 0, ((1,1), (0,1)): 0,
                                ((0,0), (1,0)): 0, ((0,0), (1,1)): 0, ((0,1), (1,0)): 0, ((0,1), (1,1)): 0,
                                ((1,0), (1,0)): 0, ((1,0), (1,1)): 0, ((1,1), (1,0)): 0, ((1,1), (1,1)): 1}
        },
        # M7 = (A12 - A22)(B21 + B22)
        {
            'a_coeffs': {((0,0), (0,0)): 0, ((0,0), (0,1)): 0, ((0,1), (0,0)): 0, ((0,1), (0,1)): 1,
                          ((1,0), (0,0)): 0, ((1,0), (0,1)): 0, ((1,1), (0,0)): 0, ((1,1), (0,1)): 0,
                          ((0,0), (1,0)): 0, ((0,0), (1,1)): 0, ((0,1), (1,0)): 0, ((0,1), (1,1)): 0,
                          ((1,0), (1,0)): 0, ((1,0), (1,1)): 0, ((1,1), (1,0)): 0, ((1,1), (1,1)): -1},
            'b_coeffs': {((0,0), (0,0)): 0, ((0,0), (0,1)): 0, ((0,1), (0,0)): 0, ((0,1), (0,1)): 0,
                          ((1,0), (0,0)): 0, ((1,0), (0,1)): 0, ((1,1), (0,0)): 0, ((1,1), (0,1)): 0,
                          ((0,0), (1,0)): 0, ((0,0), (1,1)): 0, ((0,1), (1,0)): 0, ((0,1), (1,1)): 0,
                          ((1,0), (1,0)): 1, ((1,0), (1,1)): 0, ((1,1), (1,0)): 0, ((1,1), (1,1)): 1},
            'c_contributions': {((0,0), (0,0)): 1, ((0,0), (0,1)): 0, ((0,1), (0,0)): 0, ((0,1), (0,1)): 0,
                                ((1,0), (0,0)): 0, ((1,0), (0,1)): 0, ((1,1), (0,0)): 0, ((1,1), (0,1)): 0,
                                ((0,0), (1,0)): 0, ((0,0), (1,1)): 0, ((0,1), (1,0)): 0, ((0,1), (1,1)): 0,
                                ((1,0), (1,0)): 0, ((1,0), (1,1)): 0, ((1,1), (1,0)): 0, ((1,1), (1,1)): 0}
        }
    ]
    
    rank = 7 * 27  # 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for s, step in enumerate(strassen_steps):
        # For each scalar multiplication in the 3x3 block product
        for u in range(3):
            for v in range(3):
                for w in range(3):
                    # This term computes X[u,v] * Y[v,w] contributing to Z[u,w]
                    # X[u,v] = sum over A blocks: a_coeff[br,bc] * A[br*3+u, bc*3+v]
                    # Y[v,w] = sum over B blocks: b_coeff[br,bc] * B[br*3+v, bc*3+w]
                    # Z[u,w] contributes to C blocks via c_contributions
                    
                    # For this scalar multiplication, we need to find the U, V, W entries
                    # U[i*m+j, c] = coefficient of A[i,j] in X[u,v]
                    # V[j*p+k, c] = coefficient of B[j,k] in Y[v,w]
                    # W[k*n+i, c] = coefficient of C[i,k] in Z[u,w] (from c_contributions)
                    
                    # Compute U entries: for each element (ai, aj) of A, what's its coefficient in X[u,v]?
                    # X[u,v] = sum_{br,bc} a_coeffs[br,bc] * A[br*3+u, bc*3+v]
                    for br in range(2):
                        for bc in range(2):
                            a_idx = br * 3 + u
                            j_idx = bc * 3 + v
                            coeff = step['a_coeffs'].get(((br, bc)), 0)
                            if coeff != 0:
                                # A element at (a_idx, j_idx)
                                a_flat = a_idx * m + j_idx
                                U[a_flat][col] = coeff
                    
                    # Compute V entries: for each element (bi, bj) of B, what's its coefficient in Y[v,w]?
                    # Y[v,w] = sum_{br,bc} b_coeffs[br,bc] * B[br*3+v, bc*3+w]
                    for br in range(2):
                        for bc in range(2):
                            bi_idx = br * 3 + v
                            bj_idx = bc * 3 + w
                            coeff = step['b_coeffs'].get(((br, bc)), 0)
                            if coeff != 0:
                                # B element at (bi_idx, bj_idx)
                                b_flat = bi_idx * p + bj_idx
                                V[b_flat][col] = coeff
                    
                    # Compute W entries: for each element (ci, cj) of C, what's its coefficient?
                    # Z[u,w] contributes to C blocks. The C block at (cr, cc) gets contribution from Z[u,w]
                    # where u = cr*3 + something, w = cc*3 + something
                    # Actually, Z[u,w] is the (u,w) element of the 3x3 result matrix
                    # This corresponds to C block element at (cr, cc) where cr = u//3, cc = w//3? No.
                    # Let me reconsider.
                    
                    # The 3x3 block product X*Y gives a 3x3 matrix Z.
                    # Z[r, c] = sum_u X[r, u] * Y[u, c]
                    # In our loop, we're computing the term X[u, v] * Y[v, w] which contributes to Z[u, w]
                    
                    # Z[u, w] is the (u, w) element of the 3x3 result.
                    # This 3x3 result is added to specific C blocks with specific signs.
                    
                    # For c_contributions, the key (cr, cc) refers to which C block (0=top, 1=bottom)
                    # and the value is the sign.
                    
                    # Z[u, w] corresponds to C block element at row = u, col = w within that block.
                    # So the full C index is: row = cr*3 + u, col = cc*3 + w where (cr, cc) is the block.
                    
                    for (cr, cc), sign in step['c_contributions'].items():
                        if sign != 0:
                            c_row = cr * 3 + u
                            c_col = cc * 3 + w
                            c_flat = c_row * n + c_col
                            W[c_flat][col] = sign
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
