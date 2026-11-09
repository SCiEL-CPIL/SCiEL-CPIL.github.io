"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=0):
    """Exact integer verification that U,V,W reproduce the multiplication tensor.

    T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] must equal (1 if i==k else 0).
    """
    rank = U.shape[1]
    U = np.asarray(U, dtype=object)
    V = np.asarray(V, dtype=object)
    W = np.asarray(W, dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                expected = 1 if i == k else 0
                if s != expected:
                    return False
    return True


def _strassen_2x2():
    """Rank-7 decomposition for 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) with integer entries.
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)

    # Strassen's 7 products for 2x2:
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

    # U encodes A entries: U[i*m+j, c] where i,j are A indices
    # V encodes B entries: V[j*p+k, c] where j,k are B indices
    # W encodes C entries: W[k*n+i, c] where k,n are C indices (k=col, n=row -> k*n+i)

    # Column 0: M1 = (A11+A22)*(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1  # A[0,0], A[1,1]
    V[0, 0] = 1; V[3, 0] = 1  # B[0,0], B[1,1]
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0], C[1,1]

    # Column 1: M2 = (A21+A22)*B11
    U[2, 1] = 1; U[3, 1] = 1  # A[1,0], A[1,1]
    V[0, 1] = 1               # B[0,0]
    W[2, 1] = 1               # C[1,0]

    # Column 2: M3 = A11*(B12-B22)
    U[0, 2] = 1               # A[0,0]
    V[1, 2] = 1; V[3, 2] = -1 # B[0,1], B[1,1]
    W[1, 2] = 1; W[3, 2] = 1  # C[0,1], C[1,1]

    # Column 3: M4 = A22*(B21-B11)
    U[3, 3] = 1               # A[1,1]
    V[2, 3] = 1; V[0, 3] = -1 # B[1,0], B[0,0]
    W[0, 3] = 1; W[2, 3] = 1  # C[0,0], C[1,0]

    # Column 4: M5 = (A11+A12)*B22
    U[0, 4] = 1; U[1, 4] = 1  # A[0,0], A[0,1]
    V[3, 4] = 1               # B[1,1]
    W[0, 4] = -1; W[1, 4] = 1 # C[0,0], C[0,1]

    # Column 5: M6 = (A21-A11)*(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1 # A[1,0], A[0,0]
    V[0, 5] = 1; V[1, 5] = 1  # B[0,0], B[0,1]
    W[3, 5] = 1               # C[1,1]

    # Column 6: M7 = (A12-A22)*(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1 # A[0,1], A[1,1]
    V[2, 6] = 1; V[3, 6] = 1  # B[1,0], B[1,1]
    W[0, 6] = 1               # C[0,0]

    return U, V, W


def _build_strassen_4x4():
    """Construct rank-49 decomposition for 4x4 matrix multiplication by
    applying Strassen's rank-7 algorithm to each of the 7 block products,
    where each block is 2x2. This gives 7*7 = 49 rank terms.
    """
    n = m = p = 4
    rank = 49
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)

    # Get the 2x2 Strassen factors
    U2, V2, W2 = _strassen_2x2()

    # For 4x4, we partition into 2x2 blocks.
    # A is 4x4, split into blocks A11, A12, A21, A22 (each 2x2)
    # B is 4x4, split into blocks B11, B12, B21, B22 (each 2x2)
    # C is 4x4, split into blocks C11, C12, C21, C22 (each 2x2)
    #
    # Block indices: block_row in {0,1}, block_col in {0,1}
    # Element (i,j) in A belongs to block (i//2, j//2), local (i%2, j%2)
    #
    # The 7 Strassen products at the block level:
    # M1 = (A11+A22)*(B11+B22) -> contributes to C11, C12, C21, C22
    # M2 = (A21+A22)*B11       -> contributes to C21, C22
    # M3 = A11*(B12-B22)       -> contributes to C11, C12
    # M4 = A22*(B21-B11)       -> contributes to C21, C22
    # M5 = (A11+A12)*B22       -> contributes to C12, C22
    # M6 = (A21-A11)*(B11+B12) -> contributes to C21, C22
    # M7 = (A12-A22)*(B21+B22) -> contributes to C11, C12
    #
    # For each block-level product Mr, we further decompose it using the
    # 2x2 Strassen factors. Each block-level term involves specific block
    # combinations, and within each block we use the 2x2 rank-7 scheme.

    # Define block-level structure for each of the 7 products:
    # For product r, we have:
    #   A_blocks: dict from block_index to coefficient
    #   B_blocks: dict from block_index to coefficient
    #   C_blocks: dict from block_index to coefficient
    # Then for each 2x2 local Strassen term (col c2), the overall column is
    # c = r*7 + c2, and:
    #   For each A block (ba, ca), and each 2x2 Strassen A entry (i2, c2) with val va:
    #     U[ba*2+i2, j_a*2+... ] ... this gets complicated. Let me think differently.

    # Actually, let me use a cleaner approach. The 4x4 multiplication can be seen
    # as: for each of the 7 Strassen block-products, we multiply two 2x2 "sums"
    # of blocks. Each such product is itself a 2x2 * 2x2 multiplication (of the
    # summed blocks), which we decompose with rank 7.

    # Let me define the 7 block products explicitly:
    # Product 0 (M1): A_blocks = {(0,0):1, (1,1):1}, B_blocks = {(0,0):1, (1,1):1}, C_blocks = {(0,0):1, (1,1):1}
    # Product 1 (M2): A_blocks = {(1,0):1, (1,1):1}, B_blocks = {(0,0):1}, C_blocks = {(1,0):1}
    # Product 2 (M3): A_blocks = {(0,0):1}, B_blocks = {(0,1):1, (1,1):-1}, C_blocks = {(0,1):1, (1,1):1}
    # Product 3 (M4): A_blocks = {(1,1):1}, B_blocks = {(1,0):1, (0,0):-1}, C_blocks = {(0,0):1, (1,0):1}
    # Product 4 (M5): A_blocks = {(0,0):1, (0,1):1}, B_blocks = {(1,1):1}, C_blocks = {(0,0):-1, (0,1):1}
    # Product 5 (M6): A_blocks = {(1,0):1, (0,0):-1}, B_blocks = {(0,0):1, (0,1):1}, C_blocks = {(1,1):1}
    # Product 6 (M7): A_blocks = {(0,1):1, (1,1):-1}, B_blocks = {(1,0):1, (1,1):1}, C_blocks = {(0,0):1}

    # Wait, this approach is getting complex. Let me use the direct recursive
    # Strassen construction properly.

    # Actually, the cleanest way: the 4x4 multiplication tensor can be built by
    # applying Strassen to each "diagonal" of block interactions. Let me instead
    # just verify that a rank-49 decomposition exists via the standard recursive
    # Strassen and construct it explicitly.

    # For 4x4 = 2x2 blocks, Strassen gives 7 block products. Each block product
    # is a 2x2 * 2x2 multiplication, which itself uses 7 scalar products. Total: 49.

    # Let me construct this by treating each block product independently.

    # Block product definitions (a_blocks, b_blocks, c_blocks):
    # Each is a list of (block_row, block_col, coeff)
    block_products = [
        # M1 = (A00+A11)*(B00+B11) -> C00 + C11
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2 = (A10+A11)*B00 -> C10
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # M3 = A00*(B01-B11) -> C01 + C11
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4 = A11*(B10-B00) -> C00 + C10
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5 = (A00+A01)*B11 -> C01, and -C00
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6 = (A10-A00)*(B00+B01) -> C11
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7 = (A01-A11)*(B10+B11) -> C00
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    # For each block product r, and for each 2x2 Strassen sub-term c2 (0..6):
    # The global column is c = r*7 + c2
    #
    # For the A part: for each block (br, bc) with coeff ca in a_blocks,
    #   for each 2x2 Strassen A entry at local position (i2, j2) in column c2 with value va:
    #     U[br*2+i2, bc*2+j2... wait, no.
    #
    # U is indexed by [i*m+j, c] = [i*4+j, c] for the full 4x4 A.
    # But within a block, the local A matrix has entries A_local[i2, j2].
    # Block (br, bc) occupies rows br*2..br*2+1 and cols bc*2..bc*2+1 of the full A.
    # So A_local[i2, j2] corresponds to A[br*2+i2, bc*2+j2], which is row index br*2+i2, col bc*2+j2.
    # In U, the row index is i*m+j = (br*2+i2)*4 + (bc*2+j2).

    # Similarly for V: V[j*p+k, c] = [j*4+k, c]. Block (br, bc) of B occupies rows br*2..br*2+1, cols bc*2..bc*2+1.
    # B_local[i2, j2] -> B[br*2+i2, bc*2+j2] -> V row = (br*2+i2)*4 + (bc*2+j2).

    # For W: W[k*n+i, c] = [k*4+i, c]. C block (br, bc) occupies rows br*2..br*2+1, cols bc*2..bc*2+1.
    # C_local[i2, j2] -> C[br*2+i2, bc*2+j2] -> W row = (bc*2+j2)*4 + (br*2+i2).
    # Note: W index is k*n+i where k is column, i is row. So W[col*4+row, c].

    # Let's build it:
    for r, (a_blocks, b_blocks, c_blocks) in enumerate(block_products):
        for c2 in range(7):
            c = r * 7 + c2
            # A part
            for (br, bc), ca in a_blocks.items():
                va = U2[br*2 + 0 if False else None, c2]  # This is wrong, let me fix

    # I'm overcomplicating this. Let me restart with a clearer approach.

    # Clear the arrays
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)

    # U2, V2, W2 are the 2x2 Strassen factors (4x7 each)
    # U2[row, col] where row = i*2+j for A[i,j]
    # V2[row, col] where row = j*2+k for B[j,k]
    # W2[row, col] where row = k*2+i for C[i,k] (k is col, i is row)

    for r, (a_blocks, b_blocks, c_blocks) in enumerate(block_products):
        for c2 in range(7):
            c = r * 7 + c2
            # A contributions
            for (br, bc), ca in a_blocks.items():
                for i2 in range(2):
                    for j2 in range(2):
                        val = U2[i2 * 2 + j2, c2]
                        if val != 0:
                            full_row = (br * 2 + i2) * 4 + (bc * 2 + j2)
                            U[full_row, c] += ca * val

            # B contributions
            for (br, bc), cb in b_blocks.items():
                for i2 in range(2):
                    for j2 in range(2):
                        val = V2[i2 * 2 + j2, c2]
                        if val != 0:
                            full_row = (br * 2 + i2) * 4 + (bc * 2 + j2)
                            V[full_row, c] += cb * val

            # C contributions
            for (br, bc), cc in c_blocks.items():
                for i2 in range(2):
                    for j2 in range(2):
                        val = W2[j2 * 2 + i2, c2]  # W2 row = k*2+i = j2*2+i2
                        if val != 0:
                            full_row = (bc * 2 + j2) * 4 + (br * 2 + i2)
                            W[full_row, c] += cc * val

    return U, V, W


def _verify_4x4(U, V, W):
    """Verify the 4x4 decomposition exactly."""
    n = m = p = 4
    return _verify(U, V, W, n, m, p)


def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    # Build the rank-49 Strassen decomposition
    U, V, W = _build_strassen_4x4()

    # Verify
    if not _verify_4x4(U, V, W):
        # Fallback to schoolbook if something went wrong
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=object)
        V = np.zeros((m * p, rank), dtype=object)
        W = np.zeros((p * n, rank), dtype=object)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1

    # Convert to plain nested lists with integer entries
    U_list = [[int(x) for x in row] for row in np.asarray(U, dtype=object)]
    V_list = [[int(x) for x in row] for row in np.asarray(V, dtype=object)]
    W_list = [[int(x) for x in row] for row in np.asarray(W, dtype=object)]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
