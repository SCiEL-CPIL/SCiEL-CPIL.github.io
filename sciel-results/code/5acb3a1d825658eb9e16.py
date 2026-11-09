"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _schoolbook():
    """Returns U, V, W for schoolbook (rank n*m*p) decomposition."""
    n, m, p = 4, 4, 6
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """
    Returns U, V, W for rank-7 decomposition of 2x2x2 matrix multiplication.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices: i*2+j for A, i*2+j for B, i*2+j for C.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (A00 + A11)(B00 + B11)
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1]
    
    # P2 = (A10 + A11) B00
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 0, 1, 0]
    
    # P3 = A00 (B01 - B11)
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 1, 0, -1]
    W[:, 2] = [0, 1, 0, 0]
    
    # P4 = A11 (B10 - B00)
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [0, 0, 1, -1]
    W[:, 3] = [1, 0, 0, 0]
    
    # P5 = (A00 + A01) B11
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [-1, 0, 0, 0]
    
    # P6 = (A10 - A00)(B00 + B01)
    U[:, 5] = [-1, 0, 1, 0]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, 0, 0, 1]
    
    # P7 = (A01 - A11)(B10 + B11)
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [1, 0, 0, 0]
    
    return U, V, W

def _verify_decomposition(U, V, W):
    """
    Verify that the decomposition T = sum_r u_r v_r w_r is valid.
    T[a, b, c] = 1 if C[c] = A[a] * B[b] is part of the product.
    For n=4, m=4, p=6:
    A index: i*m + j where i in [0,4), j in [0,4)
    B index: j*p + k where j in [0,4), k in [0,6)
    C index: k*n + i where k in [0,6), i in [0,4)
    T[a,b,c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k.
    """
    n, m, p = 4, 4, 6
    # Build the multiplication tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Compute sum of u_r v_r w_r
    T_computed = np.zeros_like(T)
    for r in range(U.shape[1]):
        T_computed += np.einsum('r,cr,br,ar->abc', W[:, r], W[:, r], V[:, r], U[:, r])
        # Actually, T_computed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Let's do it properly
    T_computed = np.einsum('ar,br,cr->abc', U, V, W)
    
    return np.allclose(T_computed, T, atol=1e-10)

def _try_strassen_block_decomposition():
    """
    Try to build a rank-49 decomposition for 4x4x4 and rank-32 for 4x4x2,
    then combine them for 4x4x6.
    """
    # First, let's implement the 4x4x4 Strassen decomposition properly.
    # The 4x4x4 Strassen uses 7 blocks, each of size 2x2x2.
    # But the blocks are not independent; they share indices.
    
    # Actually, the standard way to get a 4x4x4 rank-49 decomposition is to use
    # the 2x2x2 Strassen decomposition as a block decomposition.
    # The 4x4 matrix is divided into four 2x2 blocks: A00, A01, A10, A11.
    # Similarly for B and C.
    # The product C = A*B is computed as:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Using Strassen's algorithm for 2x2 blocks:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B01 - B11)
    # P4 = A11(B10 - B00)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Now, each Pk is a product of two 2x2 matrices (or linear combinations).
    # We can decompose each 2x2x2 multiplication using the 7-term Strassen.
    # This gives 7 * 7 = 49 terms total.
    
    # Let's implement this.
    U_base, V_base, W_base = _strassen_2x2x2()
    
    # Define the block mappings.
    # For a 4x4 matrix divided into 2x2 blocks:
    # Block 00: indices [0,1,4,5] (row-major: A[0,0], A[0,1], A[1,0], A[1,1])
    # Block 01: indices [2,3,6,7] (A[0,2], A[0,3], A[1,2], A[1,3])
    # Block 10: indices [8,9,12,13] (A[2,0], A[2,1], A[3,0], A[3,1])
    # Block 11: indices [10,11,14,15] (A[2,2], A[2,3], A[3,2], A[3,3])
    
    # The 2x2 Strassen decomposition operates on indices [0,1,2,3] of a 2x2 matrix.
    # We need to map each base index to the corresponding block index.
    
    # Let's define the block index mapping for each of the 7 Strassen steps.
    # For step s, we have linear combinations of blocks.
    # We need to express the u, v, w vectors in the 4x4 space.
    
    # This is complex. Let's instead construct the 49 terms directly.
    # Each term (s, r) where s in 0..6 (Strassen step) and r in 0..6 (base term).
    # u_{s,r} = sum over blocks of coefficient * (U_base[r] mapped to block)
    
    # Actually, it's easier to think of it this way:
    # The 4x4 multiplication tensor is the sum of 7 rank-7 tensors (one for each Strassen step).
    # Each rank-7 tensor corresponds to a 2x2x2 multiplication of linear combinations of blocks.
    
    # Let's define the block linear combinations for each Strassen step.
    # For step s, we have:
    # A_s = sum_i alpha_{s,i} * A_i (where A_i are the 2x2 blocks)
    # B_s = sum_j beta_{s,j} * B_j
    # C_s = sum_k gamma_{s,k} * C_k (with coefficients)
    
    # The multiplication of A_s * B_s gives C_s (with coefficients).
    # We decompose this 2x2x2 multiplication using the base Strassen.
    
    # The rank-1 term for step s, base term r is:
    # u = (sum_i alpha_{s,i} * U_base[r]_i) where U_base[r]_i is the i-th component
    # Wait, no. The base U, V, W are vectors in the 2x2 space (4 elements).
    # When we have a linear combination of blocks, the resulting vector in the 4x4 space
    # is the concatenation of the scaled block vectors.
    
    # Let's define the block index sets.
    # Block 00: [0,1,4,5]
    # Block 01: [2,3,6,7]
    # Block 10: [8,9,12,13]
    # Block 11: [10,11,14,15]
    
    # For each Strassen step s, we have coefficients for A_blocks, B_blocks, C_blocks.
    # Let's define them.
    
    # Step 0: P1 = (A00 + A11)(B00 + B11) -> contributes to C00 and C11
    # A_coeffs: [1, 0, 0, 1] (for blocks 00, 01, 10, 11)
    # B_coeffs: [1, 0, 0, 1]
    # C_coeffs: [1, 0, 0, 1] (for C00, C01, C10, C11)
    
    # Step 1: P2 = (A10 + A11)B00 -> contributes to C10
    # A_coeffs: [0, 0, 1, 1]
    # B_coeffs: [1, 0, 0, 0]
    # C_coeffs: [0, 0, 1, 0]
    
    # Step 2: P3 = A00(B01 - B11) -> contributes to C01
    # A_coeffs: [1, 0, 0, 0]
    # B_coeffs: [0, 1, 0, -1]
    # C_coeffs: [0, 1, 0, 0]
    
    # Step 3: P4 = A11(B10 - B00) -> contributes to C00
    # A_coeffs: [0, 0, 0, 1]
    # B_coeffs: [0, 0, 1, -1]
    # C_coeffs: [1, 0, 0, 0]
    
    # Step 4: P5 = (A00 + A01)B11 -> contributes to C00 with -1
    # A_coeffs: [1, 1, 0, 0]
    # B_coeffs: [0, 0, 0, 1]
    # C_coeffs: [-1, 0, 0, 0]
    
    # Step 5: P6 = (A10 - A00)(B00 + B01) -> contributes to C11
    # A_coeffs: [-1, 0, 1, 0]
    # B_coeffs: [1, 1, 0, 0]
    # C_coeffs: [0, 0, 0, 1]
    
    # Step 6: P7 = (A01 - A11)(B10 + B11) -> contributes to C00
    # A_coeffs: [0, 1, 0, -1]
    # B_coeffs: [0, 0, 1, 1]
    # C_coeffs: [1, 0, 0, 0]
    
    A_coeff_steps = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    B_coeff_steps = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    C_coeff_steps = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Block index sets for 4x4 matrix (16 elements)
    block_indices = [
        [0, 1, 4, 5],   # Block 00
        [2, 3, 6, 7],   # Block 01
        [8, 9, 12, 13], # Block 10
        [10, 11, 14, 15] # Block 11
    ]
    
    U_4 = np.zeros((16, 49))
    V_4 = np.zeros((16, 49))
    W_4 = np.zeros((16, 49))
    
    col = 0
    for s in range(7):
        for r in range(7):
            # u vector for this term
            u = np.zeros(16)
            v = np.zeros(16)
            w = np.zeros(16)
            
            # For A: u[indices of block i] += A_coeff_steps[s][i] * U_base[r][j]
            for i in range(4):
                for j in range(4):
                    idx = block_indices[i][j]
                    u[idx] += A_coeff_steps[s][i] * U_base[r][j]
                    v[idx] += B_coeff_steps[s][i] * V_base[r][j]
                    w[idx] += C_coeff_steps[s][i] * W_base[r][j]
            
            U_4[:, col] = u
            V_4[:, col] = v
            W_4[:, col] = w
            col += 1
    
    # Verify the 4x4x4 decomposition
    if not _verify_decomposition(U_4, V_4, W_4):
        # Fallback to schoolbook
        return None
    
    # Now, we need to extend this to 4x4x6.
    # The 4x4x6 multiplication can be split into 4x4x4 + 4x4x2.
    # The 4x4x4 part is handled by the above (rank 49).
    # The 4x4x2 part needs a rank-32 decomposition (schoolbook).
    
    # Let's build the 4x4x2 schoolbook decomposition.
    U_2 = np.zeros((16, 32))
    V_2 = np.zeros((16, 32))
    W_2 = np.zeros((24, 32))  # p=6, n=4, so p*n=24
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U_2[i * 4 + j, col] = 1
                V_2[j * 6 + k, col] = 1  # Wait, V is for B, which is 4x6, so index j*6+k
                W_2[k * 4 + i, col] = 1
                col += 1
    
    # Now, combine the 4x4x4 and 4x4x2 decompositions.
    # The full U has shape (16, 49+32) = (16, 81)
    # The full V has shape (24, 49+32) = (24, 81)
    # The full W has shape (24, 49+32) = (24, 81)
    
    U_full = np.zeros((16, 81))
    V_full = np.zeros((24, 81))
    W_full = np.zeros((24, 81))
    
    # For the 4x4x4 part (columns 0..48):
    U_full[:, 0:49] = U_4
    V_full[0:16, 0:49] = V_4  # V_4 is for the first 4 columns of B (4x4 part)
    W_full[0:16, 0:49] = W_4  # W_4 is for the first 4 rows of C (4x4 part)
    
    # For the 4x4x2 part (columns 49..80):
    U_full[:, 49:81] = U_2
    V_full[:, 49:81] = V_2  # V_2 is for the last 2 columns of B (4x2 part)
    W_full[:, 49:81] = W_2  # W_2 is for the last 2 rows of C (2x4 part)
    
    # Verify the full decomposition
    if not _verify_decomposition(U_full, V_full, W_full):
        return None
    
    return U_full, V_full, W_full

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    start_time = time.time()
    
    # Try the Strassen-based decomposition (rank 81)
    result = _try_strassen_block_decomposition()
    if result is not None:
        U, V, W = result
        elapsed = time.time() - start_time
        if elapsed < 120:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook (rank 96)
    U, V, W = _schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

