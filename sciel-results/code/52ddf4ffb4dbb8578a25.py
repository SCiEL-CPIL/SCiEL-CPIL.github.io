"""Optimized tensor decomposition for 4x4 * 4x9 matrix multiplication."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 9
    
    # We'll use a block decomposition approach.
    # Split the 4x9 matrix into blocks: 4x4, 4x4, 4x1
    # For 4x4 * 4x4, we use a rank 49 decomposition (Strassen-like or similar).
    # For 4x4 * 4x1, we can use rank 16 (schoolbook) or less.
    
    # Actually, let's use a known good decomposition for 4x4 matrices.
    # The Strassen algorithm gives rank 7 for 2x2.
    # For 4x4, we can use Strassen recursively: rank 7^2 = 49.
    
    # Strassen 2x2 decomposition:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's build the U, V, W matrices for Strassen 2x2.
    # U has shape (2*2, 7) = (4, 7)
    # V has shape (2*2, 7) = (4, 7)
    # W has shape (2*2, 7) = (4, 7)
    
    # For Strassen, the factor matrices are:
    # U (for A): each column corresponds to one Strassen multiplication
    # The A-part of each Strassen multiplication is a 2x2 matrix with entries in {-1, 0, 1}
    
    # Let's define the 7 Strassen multiplications for A and B parts:
    # M1: A-part = [[1,0],[0,1]] (A11 + A22), B-part = [[1,0],[0,1]] (B11 + B22)
    # M2: A-part = [[0,0],[1,1]] (A21 + A22), B-part = [[1,0],[0,0]] (B11)
    # M3: A-part = [[1,0],[0,0]] (A11), B-part = [[0,1],[0,-1]] (B12 - B22)
    # M4: A-part = [[0,0],[0,1]] (A22), B-part = [[0,-1],[1,0]] (B21 - B11)
    # M5: A-part = [[1,1],[0,0]] (A11 + A12), B-part = [[0,0],[0,1]] (B22)
    # M6: A-part = [[-1,0],[1,0]] (A21 - A11), B-part = [[1,1],[0,0]] (B11 + B12)
    # M7: A-part = [[0,1],[-1,0]] (A12 - A22), B-part = [[0,0],[1,1]] (B21 + B22)
    
    # W (for C):
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's build these matrices.
    # U: 4x7 matrix, U[i*2+j, c] is the (i,j) entry of the A-part for multiplication c
    # V: 4x7 matrix, V[i*2+j, c] is the (i,j) entry of the B-part for multiplication c
    # W: 4x7 matrix, W[i*2+j, c] is the coefficient of multiplication c in C[i,j]
    
    # A parts (2x2 each, flattened to 4 entries):
    A_parts = [
        [1, 0, 0, 1],  # M1: [[1,0],[0,1]]
        [0, 0, 1, 1],  # M2: [[0,0],[1,1]]
        [1, 0, 0, 0],  # M3: [[1,0],[0,0]]
        [0, 0, 0, 1],  # M4: [[0,0],[0,1]]
        [1, 1, 0, 0],  # M5: [[1,1],[0,0]]
        [-1, 0, 1, 0], # M6: [[-1,0],[1,0]]
        [0, 1, -1, 0], # M7: [[0,1],[-1,0]]
    ]
    
    # B parts (2x2 each, flattened to 4 entries):
    B_parts = [
        [1, 0, 0, 1],  # M1: [[1,0],[0,1]]
        [1, 0, 0, 0],  # M2: [[1,0],[0,0]]
        [0, 1, 0, -1], # M3: [[0,1],[0,-1]]
        [0, -1, 1, 0], # M4: [[0,-1],[1,0]]
        [0, 0, 0, 1],  # M5: [[0,0],[0,1]]
        [1, 1, 0, 0],  # M6: [[1,1],[0,0]]
        [0, 0, 1, 1],  # M7: [[0,0],[1,1]]
    ]
    
    # C parts (2x2 each, flattened to 4 entries):
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    C_parts = [
        [1, 0, 0, 1],  # M1: [[1,0],[0,1]]
        [0, 0, 1, 0],  # M2: [[0,0],[1,0]]
        [0, 1, 0, 0],  # M3: [[0,1],[0,0]]
        [0, 0, 0, 1],  # M4: [[0,0],[0,1]]
        [-1, 0, 0, -1],# M5: [[-1,0],[0,-1]]
        [0, 0, 0, 1],  # M6: [[0,0],[0,1]]
        [1, 0, 0, -1], # M7: [[1,0],[0,-1]]
    ]
    
    # Now, for 4x4 * 4x4, we apply Strassen recursively.
    # Split 4x4 into four 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] where each is 2x2.
    # B = [[B11, B12], [B21, B22]] where each is 2x2.
    # C = A * B = [[C11, C12], [C21, C22]] where each is 2x2.
    
    # Each Cij is computed using the 7 Strassen multiplications,
    # but each "multiplication" is actually a 2x2 * 2x2 matrix multiplication,
    # which itself requires 7 scalar multiplications (Strassen).
    
    # So for each of the 7 Strassen multiplications at the top level,
    # we need 7 scalar multiplications, giving 7*7 = 49 total.
    
    # Let's build the 4x4 * 4x4 decomposition.
    # U_44: 16 x 49
    # V_44: 16 x 49
    # W_44: 16 x 49
    
    # The 4x4 matrix A is split into 2x2 blocks:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    
    # Similarly for B and C.
    
    # For each top-level Strassen multiplication M_c (c=0..6),
    # we have A-part and B-part which are 2x2 matrices.
    # But we need to express these in terms of the 4x4 matrix entries.
    
    # Let's think differently. The 4x4 Strassen decomposition can be built
    # by applying the 2x2 Strassen decomposition to each 2x2 block.
    
    # Actually, let's just use the schoolbook decomposition for simplicity,
    # but with a better structure. The schoolbook rank is 4*4*4 = 64 for 4x4*4x4.
    # With Strassen, we can get rank 49.
    
    # Let me construct the 4x4 Strassen decomposition explicitly.
    
    # For the 4x4 case, we can use the following approach:
    # The tensor M_{4,4,4} can be decomposed using 49 multiplications.
    
    # Let's build it step by step.
    # We'll create U, V, W of shape (16, 49).
    
    # Initialize
    U_44 = [[0] * 49 for _ in range(16)]
    V_44 = [[0] * 49 for _ in range(16)]
    W_44 = [[0] * 49 for _ in range(16)]
    
    # For each of the 7 top-level Strassen multiplications (c=0..6),
    # and for each of the 7 bottom-level Strassen multiplications (d=0..6),
    # we have a scalar multiplication.
    
    # The index in U, V, W is c * 7 + d.
    
    # For top-level multiplication c:
    # The A-part is a 2x2 matrix, say A_c = [[a_c11, a_c12], [a_c21, a_c22]]
    # The B-part is a 2x2 matrix, say B_c = [[b_c11, b_c12], [b_c21, b_c22]]
    
    # These are linear combinations of the 2x2 blocks of A and B.
    
    # For the bottom-level Strassen multiplication d:
    # The A-part is a 2x2 matrix, say A_d = [[a_d11, a_d12], [a_d21, a_d22]]
    # The B-part is a 2x2 matrix, say B_d = [[b_d11, b_d12], [b_d21, b_d22]]
    
    # The actual scalar multiplication is between (A_c * A_d) and (B_c * B_d),
    # where A_c * A_d means the element-wise product or something?
    
    # Actually, no. The top-level Strassen computes linear combinations of 2x2 blocks.
    # Each 2x2 block multiplication is then done using Strassen.
    
    # Let me think again. The 4x4 matrix A has entries A[i,j] for i,j in 0..3.
    # We split it into 2x2 blocks:
    # A11 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # A12 = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]
    # A21 = [[A[2,0], A[2,1]], [A[3,0], A[3,1]]]
    # A22 = [[A[2,2], A[2,3]], [A[3,2], A[3,3]]]
    
    # For top-level Strassen multiplication c, we compute a linear combination
    # of these 2x2 blocks. For example, for c=0 (M1), we compute A11 + A22.
    # This gives a 2x2 matrix:
    # M_A_c = [[A11[0,0] + A22[0,0], A11[0,1] + A22[0,1]],
    #           [A11[1,0] + A22[1,0], A11[1,1] + A22[1,1]]]
    #        = [[A[0,0] + A[2,2], A[0,1] + A[2,3]],
    #           [A[1,0] + A[3,2], A[1,1] + A[3,3]]]
    
    # This is a 2x2 matrix, which we then multiply by the corresponding B-part
    # using Strassen (7 scalar multiplications).
    
    # So for each (c, d) pair, we have a scalar multiplication between
    # one entry of M_A_c and one entry of M_B_c.
    
    # Let's define the 2x2 matrices for each top-level Strassen multiplication.
    # A_c is a 2x2 matrix of linear combinations of A's 2x2 blocks.
    # B_c is a 2x2 matrix of linear combinations of B's 2x2 blocks.
    
    # For c=0 (M1): A_c = A11 + A22, B_c = B11 + B22
    # For c=1 (M2): A_c = A21 + A22, B_c = B11
    # For c=2 (M3): A_c = A11, B_c = B12 - B22
    # For c=3 (M4): A_c = A22, B_c = B21 - B11
    # For c=4 (M5): A_c = A11 + A12, B_c = B22
    # For c=5 (M6): A_c = A21 - A11, B_c = B11 + B12
    # For c=6 (M7): A_c = A12 - A22, B_c = B21 + B22
    
    # Now, for each c, we need to express A_c and B_c as 2x2 matrices
    # with entries that are linear combinations of the original matrix entries.
    
    # Let's define the coefficients.
    # A_c[i,j] is a linear combination of A[p,q] for p,q in 0..3.
    # We can represent this as a 2x2x4x4 tensor, or just a 4x4 matrix
    # (flattened 2x2 blocks).
    
    # Actually, let's just build the U, V, W matrices directly.
    
    # For each (c, d) where c in 0..6 and d in 0..6:
    # The scalar multiplication is between one entry of A_c and one entry of B_c.
    # The entry of A_c is a linear combination of A's entries.
    # The entry of B_c is a linear combination of B's entries.
    
    # Let's define the 2x2 matrices A_c and B_c explicitly.
    
    # A_c matrices (2x2 each, entries are linear combinations of A's 2x2 blocks):
    # We'll represent each A_c as a 4x4 matrix where the (i,j) entry
    # is the coefficient of A[i,j] in the (row,col) entry of A_c.
    
    # Actually, let's use a simpler representation.
    # A_c is a 2x2 matrix. Each entry A_c[r,c] is a linear combination
    # of the 4x4 matrix A. We can represent this as a 2x2x4x4 array,
    # or as a 4x4 matrix for each of the 4 entries of A_c.
    
    # Let's just define the coefficients directly.
    
    # For A_c, the 2x2 matrix has entries that are linear combinations
    # of A's 2x2 blocks. Let's denote the 2x2 blocks of A as A11, A12, A21, A22.
    
    # A_c[r,c] = sum of coefficients * A_block
    
    # For c=0: A_c = A11 + A22
    #   A_c[0,0] = A11[0,0] + A22[0,0] = A[0,0] + A[2,2]
    #   A_c[0,1] = A11[0,1] + A22[0,1] = A[0,1] + A[2,3]
    #   A_c[1,0] = A11[1,0] + A22[1,0] = A[1,0] + A[3,2]
    #   A_c[1,1] = A11[1,1] + A22[1,1] = A[1,1] + A[3,3]
    
    # For the bottom-level Strassen multiplication d, we multiply
    # A_c[r1,c1] by B_c[r2,c2] for some (r1,c1) and (r2,c2).
    
    # This is getting complex. Let me just use the schoolbook decomposition
    # for now, which is simpler and guaranteed to work.
    
    # Schoolbook for 4x4 * 4x9:
    # Rank = 4 * 4 * 9 = 144
    # U[i*4+j, k] = 1 if i=i, j=j, k=k, else 0
    # V[j*9+k, k] = 1 if j=j, k=k, else 0
    # W[k*4+i, i] = 1 if k=k, i=i, else 0
    
    n, m, p = 4, 4, 9
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
# EVOLVE-BLOCK-END
