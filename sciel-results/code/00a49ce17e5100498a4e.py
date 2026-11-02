import numpy as np
import time

# EVOLVE-BLOCK-START

def _strassen_decomposition(n):
    """
    Recursively generates U, V, W for n x n matrix multiplication using Strassen's algorithm.
    Returns U, V, W as numpy arrays of shape (n*n, rank).
    """
    if n == 1:
        # Base case: 1x1 matrix multiplication
        # A * B = C
        # U[0,0] = 1, V[0,0] = 1, W[0,0] = 1
        U = np.array([[1.0]])
        V = np.array([[1.0]])
        W = np.array([[1.0]])
        return U, V, W
    
    # Strassen's algorithm for 2x2 block matrices
    # We construct the 7 products for the 2x2 case
    # Then we recursively solve for the 2x2 subproblems? 
    # No, Strassen's algorithm applies to 2x2 blocks.
    # For n=4, we treat it as 2x2 blocks of size 2x2.
    # The rank of 2x2 is 7.
    # The rank of 4x4 is 7 * (rank of 2x2 subproblem? No).
    # Actually, Strassen's algorithm for n=2^k has rank R(n) = 7 * R(n/2).
    # R(1) = 1.
    # R(2) = 7.
    # R(4) = 49.
    
    # We need to construct the U, V, W matrices for the 4x4 case.
    # The indices for U are (i, k) where i, k in 0..3.
    # The indices for V are (k, j) where k, j in 0..3.
    # The indices for W are (j, i) where j, i in 0..3.
    # The rank dimension is the 7 products from the 2x2 Strassen step, 
    # combined with the recursive structure.
    
    # It is easier to construct the 4x4 decomposition by combining 7 copies of the 2x2 decomposition?
    # No, the 7 products in Strassen are linear combinations of the sub-matrices.
    
    # Let's use the explicit 4x4 Strassen formulas derived from 2x2 Strassen.
    # This is what the current program does, but hardcoded.
    # To make it "structural", I will define the 2x2 Strassen patterns and combine them.
    
    # 2x2 Strassen patterns:
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
    
    # For n=4, we have 2x2 blocks of size 2x2.
    # Let A be divided into A11, A12, A21, A22 (each 2x2).
    # Let B be divided into B11, B12, B21, B22 (each 2x2).
    # The products M1..M7 are 2x2 matrix multiplications.
    # Each 2x2 multiplication has rank 7.
    # So total rank is 7 * 7 = 49.
    
    # We can construct the 4x4 U, V, W by "inflation" of the 2x2 patterns.
    
    U_2, V_2, W_2 = _strassen_decomposition(2)
    
    # U_2 is (4, 7), V_2 is (4, 7), W_2 is (4, 7).
    # Wait, for n=2, U is (4, 7).
    # For n=4, U is (16, 49).
    
    # The 2x2 Strassen coefficients for A (input to U):
    # A11, A12, A21, A22 are 2x2 blocks.
    # In the 4x4 matrix A (flattened), the blocks are at indices:
    # A11: 0..3, A12: 4..7, A21: 8..11, A22: 12..15.
    
    # The 7 products M1..M7 require specific linear combinations of these blocks.
    # For each product Mk, we need to form the "effective A block" and "effective B block".
    # Then we use the 2x2 decomposition to multiply them.
    
    # Let's define the 2x2 Strassen coefficients for A and B.
    # A_coefs: 7 x 4 matrix (for the 4 blocks of A)
    # B_coefs: 7 x 4 matrix (for the 4 blocks of B)
    # C_coefs: 4 x 7 matrix (for the 4 blocks of C)
    
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # A_coefs (rows are M1..M7, cols are A11, A12, A21, A22)
    A_coefs = np.array([
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ], dtype=float)
    
    # B_coefs (rows are M1..M7, cols are B11, B12, B21, B22)
    B_coefs = np.array([
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ], dtype=float)
    
    # C_coefs (rows are C11, C12, C21, C22, cols are M1..M7)
    C_coefs = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C11 = M1 + M4 - M5 + M7
        [0, 0, 1, 0, 1, 0, 0],   # C12 = M3 + M5
        [0, 1, 0, 1, 0, 0, 0],   # C21 = M2 + M4
        [1, -1, 1, 0, 0, 1, 0]   # C22 = M1 - M2 + M3 + M6
    ], dtype=float)
    
    # Now, construct U_4, V_4, W_4.
    # U_4 has shape (16, 49).
    # The 49 dimensions are (m_idx, sub_idx) where m_idx in 0..6 (the 7 products) and sub_idx in 0..6 (the 7 sub-products of the 2x2 multiplication).
    
    U_4 = np.zeros((16, 49))
    V_4 = np.zeros((16, 49))
    W_4 = np.zeros((16, 49))
    
    for m_idx in range(7):
        for sub_idx in range(7):
            rank_idx = m_idx * 7 + sub_idx
            
            # U_4 contribution:
            # U_4[i, rank_idx] = sum over A_blocks b of A_coefs[m_idx, b] * U_2[i_in_block_b, sub_idx]
            # i_in_block_b is the index within the 2x2 block.
            # The global index i is b*4 + i_in_block_b.
            for b in range(4):
                coeff = A_coefs[m_idx, b]
                if coeff != 0:
                    # U_2 has shape (4, 7). Column sub_idx.
                    # We add coeff * U_2[:, sub_idx] to U_4[4*b : 4*b+4, rank_idx]
                    U_4[4*b : 4*b+4, rank_idx] += coeff * U_2[:, sub_idx]
            
            # V_4 contribution:
            for b in range(4):
                coeff = B_coefs[m_idx, b]
                if coeff != 0:
                    V_4[4*b : 4*b+4, rank_idx] += coeff * V_2[:, sub_idx]
            
            # W_4 contribution:
            # W_4[j, rank_idx] = sum over C_blocks c of C_coefs[c, m_idx] * W_2[j_in_block_c, sub_idx]
            for c in range(4):
                coeff = C_coefs[c, m_idx]
                if coeff != 0:
                    W_4[4*c : 4*c+4, rank_idx] += coeff * W_2[:, sub_idx]
                    
    return U_4, V_4, W_4

def _verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    rank = U.shape[1]
    U_r = np.asarray(U, dtype=float).reshape(n, m, rank)
    V_r = np.asarray(V, dtype=float).reshape(m, p, rank)
    W_r = np.asarray(W, dtype=float).reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    # Target tensor for matrix multiplication:
    # T[i, j, k] should be 1 if the term A[i, k] * B[k, j] is present?
    # No, the tensor T_{ijk} represents the coefficient of A_{ik} B_{kj} in C_{ij}.
    # For standard matrix multiplication, C_{ij} = sum_k A_{ik} B_{kj}.
    # So the tensor is 1 for all i, j, k.
    target = np.ones((n, m, p))
    return bool(np.allclose(T, target, atol=atol))

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 4, 4, 4
    
    # Generate Strassen decomposition
    U, V, W = _strassen_decomposition(n)
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
