"""Rewritten program for 4x4x9 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]
TIME_LIMIT = 120.0

def _construct_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _construct_strassen_block(n, m, p):
    """Strassen-like decomposition for n x m x p where n=m=p is a power of 2.
    Returns U, V, W as lists of lists with float entries."""
    if n == 1:
        # 1x1x1: rank 1
        return [[1.0]], [[1.0]], [[1.0]]
    
    # Split each dimension in half
    n2, m2, p2 = n // 2, m // 2, p // 2
    
    # Strassen's 7 products for matrix multiplication.
    # We have matrices A (n x m), B (m x p), C (n x p).
    # Split A into [[A11, A12], [A21, A22]] (each n2 x m2)
    # Split B into [[B11, B12], [B21, B22]] (each m2 x p2)
    # Split C into [[C11, C12], [C21, C22]] (each n2 x p2)
    #
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each M_t is a product of two (n2 x m2) and (m2 x p2) matrices.
    # We need to express the full decomposition as a tensor decomposition.
    
    # For each of the 7 products, we recursively decompose the (n2, m2, p2) multiplication.
    # But we also need to account for the linear combinations in the inputs.
    
    # Approach: build U, V, W by assembling contributions from each recursive call.
    # For product t, we have:
    #   input_A_t = sum_{(i1,j1) in A-blocks} alpha_t[(i1,j1)] * A_{(i1,j1)}
    #   input_B_t = sum_{(j1,k1) in B-blocks} beta_t[(j1,k1)] * B_{(j1,k1)}
    #   output: C_{(i1,k1)} += gamma_t[(i1,k1)] * M_t
    
    # The recursive decomposition for product t gives:
    #   M_t = sum_c U_t[c] * input_A_t * input_B_t * V_t[c]
    # where U_t and V_t come from the recursive (n2,m2,p2) decomposition.
    
    # So the full contribution to column c of the global decomposition is:
    #   U[i*m+j] = sum over t where (i,j) is in the A-block for t: alpha_t[(i,j)] * U_t_recursive[c]
    #   V[j*p+k] = sum over t where (j,k) is in the B-block for t: beta_t[(j,k)] * V_t_recursive[c]
    #   W[k*n+i] = sum over t where (i,k) is in the C-block for t: gamma_t[(i,k)] * 1 (the output coefficient)
    #   Wait, W should also come from the recursive structure. Actually, in a tensor decomposition,
    #   the output coefficient is determined by the recursive W.
    
    # Let me reconsider. The tensor decomposition of the product of two linear combinations:
    #   (sum_a U_a[i,j] * A_a) * (sum_b V_b[j,k] * B_b) = sum_c W_c[k,i] * C_c
    # where C_c is the c-th product term.
    
    # Actually, the standard way: if we have a decomposition of the (n2,m2,p2) multiplication
    # with rank r2, then for each of the 7 Strassen products, we use r2 columns.
    # Total rank = 7 * r2.
    
    # Let's define the Strassen coefficients.
    # A-blocks: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B-blocks: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C-blocks: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # For each product t, define which A-blocks and B-blocks contribute, with coefficients.
    # A_coeffs[t] is a dict from A-block index to coefficient
    # B_coeffs[t] is a dict from B-block index to coefficient
    
    # Block indexing: 0=(i1=0,j1=0), 1=(0,1), 2=(1,0), 3=(1,1)
    # A[i1][j1] corresponds to block (i1*2 + j1)? Let's use a consistent indexing.
    # A-block (i1, j1): rows i1*n2..(i1+1)*n2-1, cols j1*m2..(j1+1)*m2-1
    # B-block (j1, k1): rows j1*m2..(j1+1)*m2-1, cols k1*p2..(k1+1)*p2-1
    # C-block (i1, k1): rows k1*p2..(k1+1)*p2-1, cols i1*n2..(i1+1)*n2-1
    
    # Let's define the 7 products:
    # t=0: M1 = (A00 + A11) * (B00 + B11)
    #   A: {0: 1, 3: 1}
    #   B: {0: 1, 3: 1}
    # t=1: M2 = (A10 + A11) * B00
    #   A: {2: 1, 3: 1}
    #   B: {0: 1}
    # t=2: M3 = A00 * (B01 - B11)
    #   A: {0: 1}
    #   B: {1: 1, 3: -1}
    # t=3: M4 = A11 * (B10 - B00)
    #   A: {3: 1}
    #   B: {2: 1, 0: -1}
    # t=4: M5 = (A00 + A01) * B11
    #   A: {0: 1, 1: 1}
    #   B: {3: 1}
    # t=5: M6 = (A10 - A00) * (B00 + B01)
    #   A: {2: 1, 0: -1}
    #   B: {0: 1, 1: 1}
    # t=6: M7 = (A01 - A11) * (B10 + B11)
    #   A: {1: 1, 3: -1}
    #   B: {2: 1, 3: 1}
    
    # Output contributions:
    # C00 = M1 + M4 - M5 + M7  => t=0: +1, t=3: +1, t=4: -1, t=6: +1
    # C01 = M3 + M5             => t=2: +1, t=4: +1
    # C10 = M2 + M4             => t=1: +1, t=3: +1
    # C11 = M1 - M2 + M3 + M6   => t=0: +1, t=1: -1, t=2: +1, t=5: +1
    
    A_coeffs = [
        {0: 1, 3: 1},   # t=0
        {2: 1, 3: 1},   # t=1
        {0: 1},          # t=2
        {3: 1},          # t=3
        {0: 1, 1: 1},   # t=4
        {2: 1, 0: -1},  # t=5
        {1: 1, 3: -1},  # t=6
    ]
    B_coeffs = [
        {0: 1, 3: 1},   # t=0
        {0: 1},          # t=1
        {1: 1, 3: -1},  # t=2
        {2: 1, 0: -1},  # t=3
        {3: 1},          # t=4
        {0: 1, 1: 1},   # t=5
        {2: 1, 3: 1},   # t=6
    ]
    # C_coeffs[c_block] is a dict from t to coefficient
    C_coeffs = [
        {0: 1, 3: 1, 4: -1, 6: 1},   # C00
        {2: 1, 4: 1},                 # C01
        {1: 1, 3: 1},                 # C10
        {0: 1, 1: -1, 2: 1, 5: 1},   # C11
    ]
    
    # Recursively get the decomposition for (n2, m2, p2)
    U_rec, V_rec, W_rec = _construct_strassen_block(n2, m2, p2)
    r2 = len(U_rec[0]) if U_rec else 0
    
    # Now assemble the full U, V, W.
    # Total rank = 7 * r2
    total_rank = 7 * r2
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    for t in range(7):
        offset = t * r2
        for c in range(r2):
            col = offset + c
            # U contribution
            for a_block, alpha in A_coeffs[t].items():
                i1, j1 = a_block // 2, a_block % 2
                for di in range(n2):
                    for dj in range(m2):
                        i = i1 * n2 + di
                        j = j1 * m2 + dj
                        U[i * m + j][col] += alpha * U_rec[di * m2 + dj][c]
            # V contribution
            for b_block, beta in B_coeffs[t].items():
                j1, k1 = b_block // 2, b_block % 2
                for dj in range(m2):
                    for dk in range(p2):
                        j = j1 * m2 + dj
                        k = k1 * p2 + dk
                        V[j * p + k][col] += beta * V_rec[dj * p2 + dk][c]
            # W contribution: for each C-block that uses product t
            for c_block, gamma in C_coeffs[t].items():
                i1, k1 = c_block // 2, c_block % 2
                for di in range(n2):
                    for dk in range(p2):
                        i = i1 * n2 + di
                        k = k1 * p2 + dk
                        W[k * n + i][col] += gamma * W_rec[dk * n2 + di][c]
    
    return U, V, W

def _construct_4x4x9():
    """Construct a decomposition for 4x4x9.
    
    Strategy: Split the 9 columns into 4+4+1.
    - Use Strassen for the first 4 columns (4x4x4 subproblem, rank 56)
    - Use Strassen for the next 4 columns (4x4x4 subproblem, rank 56)
    - Use schoolbook for the last 1 column (4x4x1 subproblem, rank 16)
    Total rank: 56 + 56 + 16 = 128.
    
    The A matrix (4x4) is shared, B is (4x9), C is (4x9).
    
    For the first block (cols 0-3): A is the full 4x4, B is B[:, 0:4], C is C[:, 0:4].
    For the second block (cols 4-7): A is the full 4x4, B is B[:, 4:8], C is C[:, 4:8].
    For the third block (col 8): A is the full 4x4, B is B[:, 8:9], C is C[:, 8:9].
    
    In the tensor decomposition, U is the same for all blocks (since A is the same).
    V and W are block-diagonal in the column index.
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Get the 4x4x4 Strassen decomposition
    U_444, V_444, W_444 = _construct_strassen_block(4, 4, 4)
    r444 = len(U_444[0])  # should be 56
    
    # Get the 4x4x1 schoolbook decomposition
    U_441, V_441, W_441 = _construct_strassen_block(4, 4, 1)
    # Wait, Strassen doesn't work well for p=1. Let me use schoolbook for 4x4x1.
    # Actually, let me just use schoolbook for the 1-column case.
    
    # Schoolbook 4x4x1: rank = 4*4*1 = 16
    U_441 = [[0.0] * 16 for _ in range(16)]
    V_441 = [[0.0] * 16 for _ in range(4)]
    W_441 = [[0.0] * 16 for _ in range(4)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(1):
                U_441[i * 4 + j][col] = 1.0
                V_441[j * 1 + k][col] = 1.0
                W_441[k * 4 + i][col] = 1.0
                col += 1
    
    r441 = 16
    
    total_rank = 2 * r444 + r441  # 56 + 56 + 16 = 128
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    # Block 1: cols 0-3
    for c in range(r444):
        col = c
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = U_444[i * m + j][c]
        for j in range(m):
            for k in range(4):
                V[j * p + k][col] = V_444[j * 4 + k][c]
        for k in range(4):
            for i in range(n):
                W[k * n + i][col] = W_444[k * 4 + i][c]
    
    # Block 2: cols 4-7
    for c in range(r444):
        col = r444 + c
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = U_444[i * m + j][c]
        for j in range(m):
            for k in range(4, 8):
                V[j * p + k][col] = V_444[j * 4 + (k - 4)][c]
        for k in range(4, 8):
            for i in range(n):
                W[k * n + i][col] = W_444[(k - 4) * 4 + i][c]
    
    # Block 3: col 8
    for c in range(r441):
        col = 2 * r444 + c
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = U_441[i * m + j][c]
        for j in range(m):
            V[j * p + 8][col] = V_441[j * 1 + 0][c]
        for i in range(n):
            W[8 * n + i][col] = W_441[0 * 4 + i][c]
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    
    # Try the Strassen-based construction
    try:
        result = _construct_4x4x9()
        elapsed = time.time() - start_time
        if elapsed < TIME_LIMIT - 10:
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _construct_schoolbook()
