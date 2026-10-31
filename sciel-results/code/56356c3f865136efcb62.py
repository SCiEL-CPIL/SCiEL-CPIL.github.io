import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the rank-49 tensor decomposition for 4x4 matrix multiplication
    using the Kronecker product of the rank-7 Strassen decomposition for 2x2 matrices.
    
    The Strassen algorithm for 2x2 matrices uses 7 multiplications.
    We define the U, V, W factors for the 2x2 case as sparse matrices or lists,
    then use the Kronecker product to build the 4x4 factors.
    """
    n, m, p = DIMENSIONS
    
    # Define Strassen factors for 2x2 multiplication.
    # U (4x7): Maps input X (flattened 2x2) to 7 products.
    # V (4x7): Maps input Y (flattened 2x2) to 7 products.
    # W (4x7): Maps 7 products to output Z (flattened 2x2).
    
    # Strassen formula:
    # m1 = (X00 + X11) * (Y00 + Y11)
    # m2 = (X10 + X11) * Y00
    # m3 = X00 * (Y01 - Y11)
    # m4 = X11 * (Y10 - Y00)
    # m5 = (X00 + X01) * Y11
    # m6 = (X10 - X00) * (Y00 + Y01)
    # m7 = (X01 - X11) * (Y10 + Y11)
    
    # Z00 = m1 + m4 - m5 + m7
    # Z01 = m3 + m5
    # Z10 = m2 + m4
    # Z11 = m1 - m2 + m3 + m6
    
    # Flattening order: row-major. X = [[X00, X01], [X10, X11]] -> [X00, X01, X10, X11]
    # U matrix columns correspond to m1...m7. Rows to X00, X01, X10, X11.
    U_2x2 = np.array([
        [1, 0, 1, 0, 1, 1, 0],  # X00: m1, m3, m5, m6
        [0, 1, 0, 0, 1, 0, 1],  # X01: m5, m7
        [0, 1, 0, 0, 0, 1, 0],  # X10: m2, m6
        [1, 1, 0, 1, 0, 0, -1]  # X11: m1, m2, m4, m7
    ]).T # Transpose to get (7, 4) if we view it as coefficients for products? 
    # Wait, standard form: U is (n*m, rank) = (4, 7).
    # Let's build it explicitly to avoid confusion.
    
    U_2x2 = np.zeros((4, 7), dtype=int)
    # m1
    U_2x2[0, 0] += 1 # X00
    U_2x2[3, 0] += 1 # X11
    # m2
    U_2x2[2, 1] += 1 # X10
    U_2x2[3, 1] += 1 # X11
    # m3
    U_2x2[0, 2] += 1 # X00
    # m4
    U_2x2[3, 3] += 1 # X11
    # m5
    U_2x2[0, 4] += 1 # X00
    U_2x2[1, 4] += 1 # X01
    # m6
    U_2x2[2, 5] += 1 # X10
    U_2x2[0, 5] += -1 # X00
    # m7
    U_2x2[1, 6] += 1 # X01
    U_2x2[3, 6] += -1 # X11
    
    V_2x2 = np.zeros((4, 7), dtype=int)
    # m1
    V_2x2[0, 0] += 1 # Y00
    V_2x2[3, 0] += 1 # Y11
    # m2
    V_2x2[0, 1] += 1 # Y00
    # m3
    V_2x2[1, 2] += 1 # Y01
    V_2x2[3, 2] += -1 # Y11
    # m4
    V_2x2[2, 3] += 1 # Y10
    V_2x2[0, 3] += -1 # Y00
    # m5
    V_2x2[3, 4] += 1 # Y11
    # m6
    V_2x2[0, 5] += 1 # Y00
    V_2x2[1, 5] += 1 # Y01
    # m7
    V_2x2[2, 6] += 1 # Y10
    V_2x2[3, 6] += 1 # Y11
    
    W_2x2 = np.zeros((4, 7), dtype=int)
    # Output Z is flattened [Z00, Z01, Z10, Z11]
    # m1
    W_2x2[0, 0] += 1 # Z00
    W_2x2[3, 0] += 1 # Z11
    # m2
    W_2x2[2, 1] += 1 # Z10
    W_2x2[3, 1] += -1 # Z11
    # m3
    W_2x2[1, 2] += 1 # Z01
    W_2x2[3, 2] += 1 # Z11
    # m4
    W_2x2[0, 3] += 1 # Z00
    W_2x2[2, 3] += 1 # Z10
    # m5
    W_2x2[0, 4] += -1 # Z00
    W_2x2[1, 4] += 1 # Z01
    # m6
    W_2x2[3, 5] += 1 # Z11
    # m7
    W_2x2[0, 6] += 1 # Z00
    
    # Now, the 4x4 multiplication tensor is the Kronecker product of the 2x2 tensor with itself.
    # U_4x4 = kron(U_2x2, U_2x2) ?? 
    # Let's verify the indices.
    # Input X is 4x4. Flattened as [X00, X01, X02, X03, X10, ...]
    # This corresponds to a 2x2 block matrix where each block is 2x2.
    # The Kronecker product of matrices A (n x r) and B (m x s) results in (nm) x (rs).
    # The entries are A[i,k] * B[j,l].
    # This matches the structure of recursive Strassen.
    
    U = np.kron(U_2x2, U_2x2)
    V = np.kron(V_2x2, V_2x2)
    W = np.kron(W_2x2, W_2x2)
    
    # The rank is 7 * 7 = 49.
    # Shapes: U (16, 49), V (16, 49), W (16, 49).
    # This matches the required interface.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
