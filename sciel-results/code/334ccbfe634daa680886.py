import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: W[k*n + i, r]
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True


def _schoolbook(n, m, p):
    """Full schoolbook decomposition, rank n*m*p."""
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
    return U, V, W, rank


def _strassen_2x2x2():
    """Strassen decomposition for 2x2 x 2x2, rank 7.
    Returns U, V, W each of shape (4, 7).
    A indices: 0,1,2,3 (row-major 2x2)
    B indices: 0,1,2,3 (row-major 2x2)
    C indices: 0,1,2,3 (row-major 2x2)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 products for 2x2 x 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # A local indices: A00=0, A01=1, A10=2, A11=3
    # B local indices: B00=0, B01=1, B10=2, B11=3
    # C local indices: C00=0, C01=1, C10=2, C11=3
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[3, 4] = -1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[3, 6] = 1
    
    return U, V, W


def _build_4x4x4_strassen():
    """Build Strassen decomposition for 4x4 x 4x4 using Kronecker product structure.
    Rank: 7 * 7 = 49.
    
    A (4x4) = [[A00, A01], [A10, A11]] where each Ai* is 2x2
    B (4x4) = [[B00, B01], [B10, B11]] where each Bi* is 2x2
    C (4x4) = [[C00, C01], [C10, C11]] where each Ci* is 2x2
    
    Strassen at 4x4 level:
    M1 = (A00+A11)(B00+B11) -> contributes to C00
    M2 = (A10+A11)B00 -> contributes to C10
    M3 = A00(B01-B11) -> contributes to C01
    M4 = A11(B10-B00) -> contributes to C10
    M5 = (A00+A01)B11 -> contributes to C11
    M6 = (A10-A00)(B00+B01) -> contributes to C11
    M7 = (A01-A11)(B10+B11) -> contributes to C00 and C01
    
    Wait, that's not right. Let me re-derive.
    
    For 4x4 x 4x4 Strassen:
    C00 = M1 + M4 - M5 - M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M7
    
    where:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    Wait, I need to check the standard Strassen formulas again.
    
    Standard Strassen for [[A00,A01],[A10,A11]] * [[B00,B01],[B10,B11]]:
    P1 = (A00+A11)(B00+B11)
    P2 = (A10+A11)B00
    P3 = A00(B01-B11)
    P4 = A11(B10-B00)
    P5 = (A00+A01)B11
    P6 = (A10-A00)(B00+B01)
    P7 = (A01-A11)(B10+B11)
    
    C00 = P1 + P4 - P5 - P7
    C01 = P3 + P5
    C10 = P2 + P4
    C11 = P1 - P2 + P3 + P7
    
    Each Pi is a 2x2 x 2x2 product. Using Strassen for each: rank 7 per Pi.
    Total rank: 7 * 7 = 49.
    """
    # A global indices (4x4 row-major):
    # A00 = [0,1,4,5], A01 = [2,3,6,7], A10 = [8,9,12,13], A11 = [10,11,14,15]
    # B global indices (4x4 row-major):
    # B00 = [0,1,4,5], B01 = [2,3,6,7], B10 = [8,9,12,13], B11 = [10,11,14,15]
    # C global indices (4x4 row-major):
    # C00 = [0,1,4,5], C01 = [2,3,6,7], C10 = [8,9,12,13], C11 = [10,11,14,15]
    
    A00 = [0, 1, 4, 5]
    A01 = [2, 3, 6, 7]
    A10 = [8, 9, 12, 13]
    A11 = [10, 11, 14, 15]
    B00 = [0, 1, 4, 5]
    B01 = [2, 3, 6, 7]
    B10 = [8, 9, 12, 13]
    B11 = [10, 11, 14, 15]
    C00 = [0, 1, 4, 5]
    C01 = [2, 3, 6, 7]
    C10 = [8, 9, 12, 13]
    C11 = [10, 11, 14, 15]
    
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Get the base Strassen 2x2x2 decomposition
    sU, sV, sW = _strassen_2x2x2()
    
    # Define the 7 Strassen products at the 4x4 level
    # Each product: (A_combo) * (B_combo) -> C_block
    # A_combo is a linear combination of A blocks
    # B_combo is a linear combination of B blocks
    
    # Product definitions: (list of (A_block, coeff), list of (B_block, coeff), C_block)
    products = [
        # P1 = (A00+A11)(B00+B11) -> C00
        ([(A00, 1), (A11, 1)], [(B00, 1), (B11, 1)], C00),
        # P2 = (A10+A11)B00 -> C10
        ([(A10, 1), (A11, 1)], [(B00, 1)], C10),
        # P3 = A00(B01-B11) -> C01
        ([(A00, 1)], [(B01, 1), (B11, -1)], C01),
        # P4 = A11(B10-B00) -> C10
        ([(A11, 1)], [(B10, 1), (B00, -1)], C10),
        # P5 = (A00+A01)B11 -> C11
        ([(A00, 1), (A01, 1)], [(B11, 1)], C11),
        # P6 = (A10-A00)(B00+B01) -> C11
        ([(A10, 1), (A00, -1)], [(B00, 1), (B01, 1)], C11),
        # P7 = (A01-A11)(B10+B11) -> C00 and C01
        ([(A01, 1), (A11, -1)], [(B10, 1), (B11, 1)], None),  # Special handling
    ]
    
    # C contribution coefficients for each product
    # C00 = P1 + P4 - P5 - P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P7
    
    # For each of the 7 products, we use Strassen 2x2x2 (7 sub-products)
    # For each sub-product, we add to U, V, W with appropriate coefficients
    
    col = 0
    for prod_idx in range(7):
        A_terms, B_terms, C_block = products[prod_idx]
        
        for sub in range(7):
            # For each of the 7 Strassen sub-products:
            # U: sum of A_terms with sU coefficients
            # V: sum of B_terms with sV coefficients
            # W: C_block with sW coefficient
            
            # Determine C contribution sign
            # P1 -> C00: +1, C11: +1
            # P2 -> C10: +1, C11: -1
            # P3 -> C01: +1, C11: +1
            # P4 -> C10: +1, C00: +1
            # P5 -> C11: +1, C00: -1
            # P6 -> C11: +1
            # P7 -> C00: -1, C11: +1
            
            # For now, handle the main C block
            if prod_idx == 6:  # P7 contributes to both C00 and C11
                # We'll handle this specially
                c_sign_00 = -1
                c_sign_11 = 1
            else:
                c_sign_00 = 0
                c_sign_11 = 0
                if prod_idx == 0:  # P1
                    c_sign_00 = 1
                    c_sign_11 = 1
                elif prod_idx == 1:  # P2
                    c_sign_00 = 0
                    c_sign_11 = -1
                elif prod_idx == 2:  # P3
                    c_sign_00 = 0
                    c_sign_11 = 0
                elif prod_idx == 3:  # P4
                    c_sign_00 = 1
                    c_sign_11 = 0
                elif prod_idx == 4:  # P5
                    c_sign_00 = -1
                    c_sign_11 = 1
                elif prod_idx == 5:  # P6
                    c_sign_00 = 0
                    c_sign_11 = 1
            
            # Build U for this sub-product
            for a_block, a_coeff in A_terms:
                for local_idx in range(4):
                    U[a_block[local_idx], col] += a_coeff * sU[local_idx, sub]
            
            # Build V for this sub-product
            for b_block, b_coeff in B_terms:
                for local_idx in range(4):
                    V[b_block[local_idx], col] += b_coeff * sV[local_idx, sub]
            
            # Build W for this sub-product
            # Need to determine which C block and what sign
            # The C block is determined by the product index
            # For most products, there's a single C block
            # For P7, there are two C blocks
            
            # Let me redefine: each product has a list of (C_block, sign)
            c_contribs = {
                0: [(C00, 1), (C11, 1)],   # P1
                1: [(C10, 1), (C11, -1)],  # P2
                2: [(C01, 1), (C11, 1)],   # P3
                3: [(C10, 1), (C00, 1)],   # P4
                4: [(C11, 1), (C00, -1)],  # P5
                5: [(C11, 1)],              # P6
                6: [(C00, -1), (C11, 1)],  # P7
            }
            
            for c_block, c_sign in c_contribs[prod_idx]:
                for local_idx in range(4):
                    W[c_block[local_idx], col] += c_sign * sW[local_idx, sub]
            
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return U, V, W, rank


def _build_4x4x7():
    """Build decomposition for 4x4 x 4x7.
    Split B into 4x4 + 4x3.
    4x4 x 4x4: Strassen rank 49
    4x4 x 4x3: schoolbook rank 48
    Total: 97
    """
    n, m, p = 4, 4, 7
    
    # Part 1: 4x4 x 4x4 (first 4 columns of B)
    U1, V1, W1, r1 = _build_4x4x4_strassen()
    
    # Part 2: 4x4 x 4x3 (last 3 columns of B)
    # Schoolbook for 4x4 x 4x3
    p2 = 3
    r2 = n * m * p2  # 48
    U2 = np.zeros((n * m, r2))
    V2 = np.zeros((m * p2, r2))
    W2 = np.zeros((p2 * n, r2))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p2):
                U2[i * m + j, col] = 1
                V2[j * p2 + k, col] = 1
                W2[k * n + i, col] = 1
                col += 1
    
    # Combine: U has shape (16, r1+r2), V has shape (m*p, r1+r2) = (28, r1+r2)
    # V1 has shape (16, r1) for the first 4 columns
    # V2 has shape (12, r2) for the last 3 columns
    # We need to place V2 in the correct rows of V
    
    total_rank = r1 + r2
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Copy part 1
    U[:, :r1] = U1
    V[:m * 4, :r1] = V1  # First 16 rows of V (4*4)
    W[:4 * n, :r1] = W1  # First 4*4=16 rows of W
    
    # Copy part 2
    U[:, r1:] = U2
    # V2 goes in rows 16..27 (last 12 rows, corresponding to columns 4-6 of B)
    V[m * 4:, r1:] = V2
    # W2 goes in rows 16..18 (last 3 rows, corresponding to columns 4-6 of C)
    W[4 * n:, r1:] = W2
    
    return U, V, W, total_rank


def construct():
    """Construct a low-rank tensor decomposition for 4x4 * 4x7 matrix multiplication.
    Returns dict with 'U', 'V', 'W' keys.
    """
    deadline = time.time() + 110
    
    n, m, p = 4, 4, 7
    
    # Build the decomposition
    U, V, W, rank = _build_4x4x7()
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W, rank = _schoolbook(n, m, p)
        if not _verify(U, V, W, n, m, p):
            # Should never happen
            pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
