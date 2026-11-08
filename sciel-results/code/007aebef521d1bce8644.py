"""Matrix multiplication 4,4,4 - Strassen rank-49 decomposition."""
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 matrix multiplication.
    Uses Strassen's algorithm for 4x4x4 multiplication, achieving rank 49.
    
    The decomposition uses the standard Strassen block decomposition:
    - Split 4x4 matrices into 2x2 blocks of 2x2 submatrices
    - Apply Strassen's 7-product formula at the block level
    - Apply Strassen's 7-product formula at the sub-block level
    - Total rank: 7 * 7 = 49
    """
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 10  # Very fast construction, just in case
    
    # 2x2 Strassen coefficients
    # Indexing: [term][element]
    # Elements: 00, 01, 10, 11 -> indices 0, 1, 2, 3
    
    # A coefficients for the 7 Strassen products
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)(B00)
    # P3 = (A00)(B01-B11)
    # P4 = (A11)(B10-B00)
    # P5 = (A00+A01)(B11)
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    S_A = [
        [1, 0, 0, 1], # P1: A00 + A11
        [0, 0, 1, 1], # P2: A10 + A11
        [1, 0, 0, 0], # P3: A00
        [0, 0, 0, 1], # P4: A11
        [1, 1, 0, 0], # P5: A00 + A01
        [-1, 0, 1, 0],# P6: A10 - A00
        [0, 1, 0, -1] # P7: A01 - A11
    ]
    
    S_B = [
        [1, 0, 0, 1], # P1: B00 + B11
        [1, 0, 0, 0], # P2: B00
        [0, 1, 0, -1],# P3: B01 - B11
        [-1, 0, 1, 0],# P4: B10 - B00
        [0, 0, 0, 1], # P5: B11
        [1, 1, 0, 0], # P6: B00 + B01
        [0, 0, 1, 1]  # P7: B10 + B11
    ]
    
    # C coefficients
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    S_C = [
        [1, 0, 0, 1], # C00: P1 + P4 - P5 + P7 -> wait, let's map carefully
        # The decomposition is: C = sum_k U_k * V_k * W_k
        # For a specific C element, the coefficient is the sum of W_k for that element.
        # In the tensor decomposition U, V, W, W[k][c] indicates how much product k contributes to C element c.
        # So W_c is a vector of length 7.
        # C00 = 1*P1 + 0*P2 + 0*P3 + 1*P4 - 1*P5 + 0*P6 + 1*P7
        # C01 = 0*P1 + 0*P2 + 1*P3 + 0*P4 + 1*P5 + 0*P6 + 0*P7
        # C10 = 0*P1 + 1*P2 + 0*P3 + 1*P4 + 0*P5 + 0*P6 + 0*P7
        # C11 = 1*P1 - 1*P2 + 1*P3 + 0*P4 + 0*P5 + 1*P6 + 0*P7
        
    ]
    
    # Let's define W coefficients as [c_element][p_term]
    # c_element: 0=C00, 1=C01, 2=C10, 3=C11
    # p_term: 0..6
    
    W_coeffs = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # The 4x4 matrices are split into 2x2 blocks.
    # Block indices: 00, 01, 10, 11 -> indices 0, 1, 2, 3
    # The Strassen formula at the block level uses the SAME coefficients S_A, S_B, W_coeffs.
    # So the "Outer" coefficients are identical to the "Inner" coefficients.
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Construct the decomposition
    # Column index r = outer_term * 7 + inner_term
    
    for K in range(7): # Outer Strassen term (block level)
        for l in range(7): # Inner Strassen term (sub-block level)
            r = K * 7 + l
            
            # Fill U
            # U[i*m + j, r]
            for i in range(n):
                for j in range(m):
                    # Determine which 2x2 block (i,j) belongs to
                    rb = i // 2
                    cb = j // 2
                    block_idx = rb * 2 + cb
                    
                    # Determine local index within the 2x2 block
                    li = i % 2
                    lj = j % 2
                    local_idx = li * 2 + lj
                    
                    # Coefficient is Outer_K[block_idx] * Inner_l[local_idx]
                    coeff = S_A[K][block_idx] * S_A[l][local_idx]
                    if coeff != 0:
                        U[i * m + j][r] = coeff
                        
            # Fill V
            # V[j*p + k, r]
            for j in range(m):
                for k in range(p):
                    rb = j // 2
                    cb = k // 2
                    block_idx = rb * 2 + cb
                    
                    li = j % 2
                    lj = k % 2
                    local_idx = li * 2 + lj
                    
                    coeff = S_B[K][block_idx] * S_B[l][local_idx]
                    if coeff != 0:
                        V[j * p + k][r] = coeff
                        
            # Fill W
            # W[k*n + i, r]
            # Note: The problem statement says W[k*n+i,r]
            # C[i,k] is the output.
            # In the tensor product sum_r U[i,j,r] V[j,k,r] W[k,i,r] (or similar)
            # The W matrix maps the product index to the output C[i,k].
            # The index in W is k*n + i.
            # The coefficient for C[i,k] from product r is W_coeffs[c_idx][r_inner] * W_coeffs[c_block_idx][r_outer]
            # c_idx corresponds to local position in C block
            # c_block_idx corresponds to which block of C
            
            for k in range(p):
                for i in range(n):
                    # C element is C[i,k]
                    rb = i // 2
                    cb = k // 2
                    block_idx = rb * 2 + cb
                    
                    li = i % 2
                    lk = k % 2
                    local_idx = li * 2 + lk
                    
                    # Outer coefficient: contribution of block K to C block
                    # Inner coefficient: contribution of term l to C element
                    coeff = W_coeffs[block_idx][K] * W_coeffs[local_idx][l]
                    
                    if coeff != 0:
                        W[k * n + i][r] = coeff
                        
    # Verification (optional but good for debugging, skipped in final for speed if confident)
    # The construction is mathematically sound based on Strassen's algorithm.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
