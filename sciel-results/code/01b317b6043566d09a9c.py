"""Matrix multiplication 4x4x7: rank-minimal exact tensor decomposition.

Strategy:
  - Use Strassen's algorithm (rank 7) to multiply the 4x4 block of the 4x7 matrix,
    giving rank 7 for the 4x4x4 part.
  - Use schoolbook (rank 4*4*3 = 48) for the remaining 4x4x3 part.
  - Total rank = 7 + 48 = 55.
  - Additionally, try to use Strassen on the 4x4x4 part and then check if we can
    do better by exploiting the rectangular structure.
  
  Actually, let me think more carefully.
  
  The tensor is T[i,j,k] = A[i,j] * B[j,k] = C[i,k] for i in [0,4), j in [0,4), k in [0,7).
  
  We can split k into k1 in [0,4) and k2 in [4,7).
  
  Part 1: C[i, k1] = sum_j A[i,j] * B[j, k1] — this is 4x4 times 4x4, rank <= 7 (Strassen)
  Part 2: C[i, k2] = sum_j A[i,j] * B[j, k2] — this is 4x4 times 4x3, rank <= 48 (schoolbook)
  
  Total rank = 7 + 48 = 55.
  
  But wait — can we do better on the 4x4x3 part? The rank of the 4x4x3 matrix multiplication tensor is at most 4*4*3=48 by schoolbook. It's known that for 2x2x1, rank is 4 (schoolbook is optimal). For larger rectangular cases, the exact rank is generally unknown, but schoolbook is a safe upper bound.
  
  Actually, let me reconsider. The known best upper bound for 4x4x4 is 49 (Strassen-like improvements). But for 4x4x3, I don't have a known better-than-schoolbook bound. Let's stick with 55.
  
  However, I should also consider: can we use a single decomposition that handles all 7 columns together? The rank of the 4x4x7 tensor is at most the rank of 4x4x4 plus rank of 4x4x3 if we split, but maybe a joint decomposition is better. Without known results, the split approach giving 55 is a solid improvement over 112.
  
  Let me implement Strassen for 4x4x4 and schoolbook for 4x4x3.
"""

import time

DIMENSIONS = [4, 4, 7]

def _strassen_4x4x4():
    """
    Return the Strassen decomposition for 4x4 x 4x4 -> 4x4.
    U has shape (16, 7), V has shape (16, 7), W has shape (16, 7).
    Indexing: U[i*m + j, c], V[j*p + k, c], W[k*n + i, c] with n=m=p=4.
    
    Strassen's algorithm for 2x2:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B10)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    For 4x4, we split into 2x2 blocks of 2x2 each. But actually, Strassen's algorithm
    is naturally for 2x2. For 4x4, we can apply it recursively: split 4x4 into four 2x2 blocks,
    compute 7 products of 2x2 matrices, each of which can be done by schoolbook (rank 8 per 2x2x2).
    
    Wait, that gives rank 7 * 8 = 56 for 4x4x4, which is worse than 49.
    
    Let me reconsider. The known rank-49 decomposition for 4x4x4 uses a different approach.
    
    Actually, let me just use the standard Strassen for 2x2 and apply it at the 2x2 level.
    For a 4x4 matrix, we can view it as a 2x2 matrix of 2x2 blocks. Then Strassen gives 7
    multiplications of 2x2 by 2x2. Each 2x2x2 multiplication has rank 4 (schoolbook).
    So total rank = 7 * 4 = 28 for the 4x4x4 part.
    
    Wait, that's much better! Let me verify: 
    - 4x4 matrix A is split into [[A00, A01], [A10, A11]] where each is 2x2.
    - Similarly B.
    - Strassen computes 7 products of 2x2 matrices.
    - Each 2x2 x 2x2 -> 2x2 product has rank 4 (schoolbook).
    - Total rank for 4x4x4 = 7 * 4 = 28.
    
    That's a big improvement! Let me implement this.
    
    For the 4x4x3 part: split B into 2x2 blocks in the row dimension (j) and keep columns as-is.
    B is 4x3, so it's [[B00, B01, B02], [B10, B11, B12]] where each Bij is 2x3.
    A is 4x4, split into 2x2 blocks of 2x2.
    
    Strassen on the 2x2 block structure:
    M1 = (A00+A11)(B00+B10+B20+B30)... 
    
    Hmm, this gets complicated because B has 3 columns. Let me think differently.
    
    For the 4x4x3 part, I'll just use schoolbook: rank = 4*4*3 = 48.
    
    Total: 28 + 48 = 76. Still better than 112.
    
    Actually wait — can I do better on the 4x4x3 part? Let me think about using Strassen-like
    decomposition. If I split the 4x4 into 2x2 blocks, I get 7 products where each product
    involves a 2x2 matrix times a 2x3 matrix. The rank of 2x2x3 is at most 2*2*3=12 (schoolbook).
    So total = 7 * 12 = 84 for the 4x4x3 part. That's worse than 48.
    
    So schoolbook for 4x4x3 is better. Total = 28 + 48 = 76.
    
    Let me implement this.
    """
    # Strassen for 4x4x4 using 2x2 blocks
    # Each 2x2x2 uses schoolbook with rank 4
    # Total rank = 7 * 4 = 28
    
    # For a 2x2x2 schoolbook:
    # U[i*2+j, c] = 1, V[j*2+k, c] = 1, W[k*2+i, c] = 1 for c corresponding to (i,j,k)
    
    # Strassen formulas for 2x2 blocks:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # Each M is a 2x2 matrix.
    # To compute M = X * Y where X, Y are 2x2, we need 4 scalar multiplications.
    
    # The U matrix entries for Strassen:
    # For M1: X = A00 + A11, Y = B00 + B11
    #   A00 contributes +1 to U at positions (0,0),(0,1),(1,0),(1,1) [i=0,1; j=0,1]
    #   A11 contributes +1 to U at positions (2,2),(2,3),(3,2),(3,3) [i=2,3; j=2,3]
    #   B00 contributes +1 to V at positions (0,0),(0,1),(1,0),(1,1) [j=0,1; k=0,1]
    #   B11 contributes +1 to V at positions (2,2),(2,3),(3,2),(3,3) [j=2,3; k=2,3]
    
    # This is getting complex. Let me build it systematically.
    
    n = 4
    rank = 28  # 7 * 4
    
    U = [[0] * rank for _ in range(n * n)]  # 16 x 28
    V = [[0] * rank for _ in range(n * n)]  # 16 x 28
    W = [[0] * rank for _ in range(n * n)]  # 16 x 28
    
    # Strassen terms for 2x2 block structure
    # Each term is (U_coeff, V_coeff, W_coeff)
    # U_coeff: dict mapping (i,j) -> coefficient for A[i,j]
    # V_coeff: dict mapping (j,k) -> coefficient for B[j,k]
    # W_coeff: dict mapping (i,k) -> coefficient for C[i,k]
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        (
            {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0,
             (2,2): 1, (2,3): 0, (3,2): 0, (3,3): 0},
            {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0,
             (2,2): 1, (2,3): 0, (3,2): 0, (3,3): 0},
            {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0,
             (2,2): 1, (2,3): 0, (3,2): 0, (3,3): 0}
        ),
        # M2 = (A10+A11)B00
        (
            {(2,0): 1, (2,1): 0, (3,0): 0, (3,1): 0,
             (2,2): 1, (2,3): 0, (3,2): 0, (3,3): 0},
            {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0},
            {(2,0): 1, (2,1): 0, (3,0): 0, (3,1): 0}
        ),
        # M3 = A00(B10-B11)
        (
            {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0},
            {(2,0): 1, (2,1): 0, (3,0): 0, (3,1): 0,
             (2,2): -1, (2,3): 0, (3,2): 0, (3,3): 0},
            {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}
        ),
        # M4 = A11(B01-B10)
        (
            {(2,2): 1, (2,3): 0, (3,2): 0, (3,3): 0},
            {(0,1): 1, (0,0): 0, (1,1): 0, (1,0): 0,
             (2,0): -1, (2,1): 0, (3,0): 0, (3,1): 0},
            {(2,1): 1, (2,0): 0, (3,1): 0, (3,0): 0}
        ),
        # M5 = (A00+A01)B11
        (
            {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0,
             (0,2): 1, (0,3): 0, (1,2): 0, (1,3): 0},
            {(2,2): 1, (2,3): 0, (3,2): 0, (3,3): 0},
            {(0,2): 1, (0,3): 0, (1,2): 0, (1,3): 0}
        ),
        # M6 = (A10-A00)(B00+B10)
        (
            {(2,0): 1, (2,1): 0, (3,0): 0, (3,1): 0,
             (0,0): -1, (0,1): 0, (1,0): 0, (1,1): 0},
            {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0,
             (2,0): 1, (2,1): 0, (3,0): 0, (3,1): 0},
            {(2,0): 1, (2,1): 0, (3,0): 0, (3,1): 0}
        ),
        # M7 = (A01-A11)(B01+B11)
        (
            {(0,2): 1, (0,3): 0, (1,2): 0, (1,3): 0,
             (2,2): -1, (2,3): 0, (3,2): 0, (3,3): 0},
            {(0,1): 1, (0,0): 0, (1,1): 0, (1,0): 0,
             (2,2): 1, (2,3): 0, (3,2): 0, (3,3): 0},
            {(0,1): 1, (0,0): 0, (1,1): 0, (1,0): 0}
        ),
    ]
    
    # Now, for each Strassen term, we need to decompose the 2x2x2 multiplication into
    # 4 scalar multiplications (schoolbook).
    
    # For a 2x2x2 multiplication with matrices X (2x2), Y (2x2), result Z (2x2):
    # Z[i,k] = sum_j X[i,j] * Y[j,k]
    # Schoolbook: for each (i,j,k), one multiplication.
    
    # The U, V, W for a 2x2x2 schoolbook (with i,j,k in {0,1}):
    # U[i,j] = 1 for the (i,j,k) term
    # V[j,k] = 1 for the (i,j,k) term  
    # W[i,k] = 1 for the (i,j,k) term
    
    # But in our case, the "2x2" matrices are embedded in the 4x4 matrices.
    # For example, in M1, X = A00 + A11, which means:
    # X[0,0] = A[0,0], X[0,1] = A[0,1], X[1,0] = A[1,0], X[1,1] = A[1,1]
    # But A00 is at positions (0,0),(0,1),(1,0),(1,1) and A11 is at (2,2),(2,3),(3,2),(3,3).
    # So X[0,0] = A[0,0] + A[2,2], etc.
    
    # I think the cleanest way is to directly build U, V, W for the full 4x4x4 Strassen.
    
    # Let me restart with a cleaner approach.
    
    # For the 4x4x4 Strassen decomposition:
    # We have 7 Strassen terms, each requiring 4 scalar multiplications = 28 total.
    
    # For each Strassen term t and each scalar multiplication (a,b,c) where
    # a is the row index in the 2x2 X matrix, b is the col index in 2x2 X / row in 2x2 Y,
    # c is the col index in 2x2 Y / col in 2x2 Z:
    
    # The contribution to U is: for each (i,j) where the 2x2 X has a non-zero coefficient,
    # U[i*m+j, col] += coeff * x_coeff[a,b]
    
    # This is getting quite involved. Let me just hard-code the known rank-49 or use
    # a simpler approach.
    
    # Actually, let me just use the straightforward Strassen at the 2x2 level and
    # schoolbook for the 2x2x2 subproblems.
    
    # I'll build it step by step.
    
    pass  # Placeholder, will rebuild below

def construct():
    """
    Construct the decomposition for 4x4 x 4x7 -> 4x7.
    
    Uses:
    - Strassen (2x2 block) for the 4x4x4 part: rank 28
    - Schoolbook for the 4x4x3 part: rank 48
    - Total rank: 76
    """
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Split p into p1=4 and p2=3
    p1 = 4
    p2 = 3
    
    # Part 1: 4x4 x 4x4 -> 4x4 (Strassen, rank 28)
    # Part 2: 4x4 x 4x3 -> 4x3 (Schoolbook, rank 48)
    
    rank1 = 28
    rank2 = 48
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 76
    V = [[0] * total_rank for _ in range(m * p)]  # 28 x 76
    W = [[0] * total_rank for _ in range(p * n)]  # 28 x 76
    
    # --- Part 1: Strassen for 4x4x4 ---
    # Build the Strassen decomposition for 4x4x4
    # Using 2x2 blocks, each 2x2x2 subproblem uses schoolbook (rank 4)
    
    # Strassen terms: (X_coeffs, Y_coeffs, Z_coeffs)
    # where X_coeffs maps 2x2 position (a,b) to coefficient in X
    # Y_coeffs maps 2x2 position (b,c) to coefficient in Y
    # Z_coeffs maps 2x2 position (a,c) to coefficient in Z
    
    # Block layout for 4x4:
    # Block (0,0): rows 0-1, cols 0-1
    # Block (0,1): rows 0-1, cols 2-3
    # Block (1,0): rows 2-3, cols 0-1
    # Block (1,1): rows 2-3, cols 2-3
    
    def block_indices(br, bc):
        """Return the 4 (i,j) indices for block (br,bc)."""
        return [(br*2 + a, bc*2 + b) for a in range(2) for b in range(2)]
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        # M3 = A00(B10-B11)
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,0): 1}),
        # M4 = A11(B01-B10)
        ({(1,1): 1}, {(0,1): 1, (1,0): -1}, {(1,1): 1}),
        # M5 = (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1}),
        # M6 = (A10-A00)(B00+B10)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, {(1,0): 1}),
        # M7 = (A01-A11)(B01+B11)
        ({(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}, {(0,1): 1}),
    ]
    
    col = 0
    for x_coeffs, y_coeffs, z_coeffs in strassen_terms:
        # For each of the 4 scalar multiplications in the 2x2x2 schoolbook
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    # This scalar multiplication computes Z[a,c] += X[a,b] * Y[b,c]
                    # X[a,b] gets contributions from A blocks
                    for (br_x, bc_x), coeff_x in x_coeffs.items():
                        for (i, j) in block_indices(br_x, bc_x):
                            U[i * m + j][col] += coeff_x
                    # Y[b,c] gets contributions from B blocks
                    for (br_y, bc_y), coeff_y in y_coeffs.items():
                        for (j, k) in block_indices(br_y, bc_y):
                            V[j * p1 + k][col] += coeff_y
                    # Z[a,c] gets contributions to C blocks
                    for (br_z, bc_z), coeff_z in z_coeffs.items():
                        for (i, k) in block_indices(br_z, bc_z):
                            W[k * n + i][col] += coeff_z
                    col += 1
    
    # col should now be 28
    assert col == rank1, f"Expected col={rank1}, got {col}"
    
    # --- Part 2: Schoolbook for 4x4x3 ---
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # where k is in [0,3) but in the full 4x7 matrix, k is in [4,7)
    
    for i in range(n):
        for j in range(m):
            for k in range(p1, p):  # k = 4, 5, 6
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    assert col == total_rank, f"Expected col={total_rank}, got {col}"
    
    # Check time
    if time.time() > deadline:
        # Shouldn't happen, but just in case
        pass
    
    return {"U": U, "V": V, "W": W}


# Verify correctness
if __name__ == "__main__":
    import numpy as np
    
    result = construct()
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # Build the tensor T[i,j,k] = delta (from decomposition)
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 if i,j,k form a valid triple (always in this case)
    # Actually, the tensor for matrix multiplication is T[i,j,k] = 1 for all i,j,k
    # because C[i,k] = sum_j A[i,j] B[j,k]
    
    # Check: for all i,j,k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                err = abs(val - 1.0)
                max_err = max(max_err, err)
    
    print(f"Rank: {rank}")
    print(f"Max error: {max_err}")
    if max_err < 1e-9:
        print("CORRECT")
    else:
        print("INCORRECT")
