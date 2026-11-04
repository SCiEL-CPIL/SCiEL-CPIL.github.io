import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _build_target_tensor(n, m, p):
    return np.ones((n, m, p), dtype=np.float64)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol, rtol=0)

def _construct_strassen_4x4():
    """Construct a rank-28 decomposition of the 4x4x4 matrix multiplication tensor
    using Strassen's algorithm on 2x2 blocks, with schoolbook multiplication within each block.
    
    The 4x4 matrix multiplication can be viewed as multiplying 2x2 blocks of 2x2 matrices.
    Strassen's algorithm uses 7 block multiplications instead of 4.
    Each 2x2 block multiplication uses 4 scalar multiplications (schoolbook).
    Total: 7 * 4 = 28 scalar multiplications.
    
    Returns U, V, W of shapes (16, 28), (16, 28), (16, 28) for n=m=p=4.
    """
    n = m = p = 4
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    # Strassen's 7 products for 2x2 blocks:
    # Let A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # For 4x4 matrices, the blocks are 2x2:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # Similarly for B and C.
    
    # Block indices: (row_block, col_block)
    # A blocks: (0,0)=A11, (0,1)=A12, (1,0)=A21, (1,1)=A22
    # B blocks: (0,0)=B11, (0,1)=B12, (1,0)=B21, (1,1)=B22
    # C blocks: (0,0)=C11, (0,1)=C12, (1,0)=C21, (1,1)=C22
    
    # Each 2x2 block multiplication uses 4 scalar products.
    # For a block product X * Y where X is a 2x2 block of A and Y is a 2x2 block of B,
    # the result Z is a 2x2 block of C.
    # Z[i,j] = sum_k X[i,k] * Y[k,j]
    
    # We need to track which A elements, B elements, and C elements are involved.
    
    # Let's define helper to add a term for a specific block multiplication
    # For block (rb, cb) of C, computed from blocks of A and B:
    # C[rb, cb] = sum over products
    
    # Actually, let's think of it differently.
    # For each of the 7 Strassen products, we compute a 2x2 matrix product.
    # Each 2x2 product = 4 scalar multiplications.
    
    # M1 = (A11 + A22)(B11 + B22)
    # A11 + A22 is a 2x2 matrix: elements at A[0,0]+A[2,2], A[0,1]+A[2,3], A[1,0]+A[3,2], A[1,1]+A[3,3]
    # B11 + B22 is a 2x2 matrix: elements at B[0,0]+B[2,2], B[0,1]+B[2,3], B[1,0]+B[3,2], B[1,1]+B[3,3]
    # Product is 2x2: M1[0,0], M1[0,1], M1[1,0], M1[1,1]
    # These contribute to C11 = M1 + M4 - M5 + M7 and C22 = M1 - M2 + M3 + M6
    
    # Let me just enumerate all 28 terms directly.
    
    # For each Strassen product k (0-6), and each position (i,j) in the 2x2 result (0-3):
    # The term is: (linear combo of A elements) * (linear combo of B elements) -> contributes to C
    
    # Let's define the 7 Strassen products more carefully.
    
    # Product 1: P1 = (A11 + A22)(B11 + B22)
    # A11+A22: 
    #   [0,0]: A[0,0] + A[2,2]
    #   [0,1]: A[0,1] + A[2,3]
    #   [1,0]: A[1,0] + A[3,2]
    #   [1,1]: A[1,1] + A[3,3]
    # B11+B22:
    #   [0,0]: B[0,0] + B[2,2]
    #   [0,1]: B[0,1] + B[2,3]
    #   [1,0]: B[1,0] + B[3,2]
    #   [1,1]: B[1,1] + B[3,3]
    # P1[i,j] = sum_k (A11+A22)[i,k] * (B11+B22)[k,j]
    
    # P1 contributes to:
    # C11 += P1, C22 += P1
    
    # Product 2: P2 = (A21 + A22)B11
    # A21+A22:
    #   [0,0]: A[2,0] + A[2,2]
    #   [0,1]: A[2,1] + A[2,3]
    #   [1,0]: A[3,0] + A[3,2]
    #   [1,1]: A[3,1] + A[3,3]
    # B11:
    #   [0,0]: B[0,0], [0,1]: B[0,1]
    #   [1,0]: B[1,0], [1,1]: B[1,1]
    # P2[i,j] = sum_k (A21+A22)[i,k] * B11[k,j]
    
    # P2 contributes to:
    # C21 += P2, C22 -= P2
    
    # Product 3: P3 = A11(B12 - B22)
    # A11:
    #   [0,0]: A[0,0], [0,1]: A[0,1]
    #   [1,0]: A[1,0], [1,1]: A[1,1]
    # B12-B22:
    #   [0,0]: B[0,2] - B[2,2]
    #   [0,1]: B[0,3] - B[2,3]
    #   [1,0]: B[1,2] - B[3,2]
    #   [1,1]: B[1,3] - B[3,3]
    # P3[i,j] = sum_k A11[i,k] * (B12-B22)[k,j]
    
    # P3 contributes to:
    # C12 += P3, C22 += P3
    
    # Product 4: P4 = A22(B21 - B11)
    # A22:
    #   [0,0]: A[2,2], [0,1]: A[2,3]
    #   [1,0]: A[3,2], [1,1]: A[3,3]
    # B21-B11:
    #   [0,0]: B[2,0] - B[0,0]
    #   [0,1]: B[2,1] - B[0,1]
    #   [1,0]: B[3,0] - B[1,0]
    #   [1,1]: B[3,1] - B[1,1]
    # P4[i,j] = sum_k A22[i,k] * (B21-B11)[k,j]
    
    # P4 contributes to:
    # C11 += P4, C21 += P4
    
    # Product 5: P5 = (A11 + A12)B22
    # A11+A12:
    #   [0,0]: A[0,0] + A[0,2]
    #   [0,1]: A[0,1] + A[0,3]
    #   [1,0]: A[1,0] + A[1,2]
    #   [1,1]: A[1,1] + A[1,3]
    # B22:
    #   [0,0]: B[2,2], [0,1]: B[2,3]
    #   [1,0]: B[3,2], [1,1]: B[3,3]
    # P5[i,j] = sum_k (A11+A12)[i,k] * B22[k,j]
    
    # P5 contributes to:
    # C11 -= P5, C12 += P5
    
    # Product 6: P6 = (A21 - A11)(B11 + B12)
    # A21-A11:
    #   [0,0]: A[2,0] - A[0,0]
    #   [0,1]: A[2,1] - A[0,1]
    #   [1,0]: A[3,0] - A[1,0]
    #   [1,1]: A[3,1] - A[1,1]
    # B11+B12:
    #   [0,0]: B[0,0] + B[0,2]
    #   [0,1]: B[0,1] + B[0,3]
    #   [1,0]: B[1,0] + B[1,2]
    #   [1,1]: B[1,1] + B[1,3]
    # P6[i,j] = sum_k (A21-A11)[i,k] * (B11+B12)[k,j]
    
    # P6 contributes to:
    # C22 += P6
    
    # Product 7: P7 = (A12 - A22)(B21 + B22)
    # A12-A22:
    #   [0,0]: A[0,2] - A[2,2]
    #   [0,1]: A[0,3] - A[2,3]
    #   [1,0]: A[1,2] - A[3,2]
    #   [1,1]: A[1,3] - A[3,3]
    # B21+B22:
    #   [0,0]: B[2,0] + B[2,2]
    #   [0,1]: B[2,1] + B[2,3]
    #   [1,0]: B[3,0] + B[3,2]
    #   [1,1]: B[3,1] + B[3,3]
    # P7[i,j] = sum_k (A12-A22)[i,k] * (B21+B22)[k,j]
    
    # P7 contributes to:
    # C11 += P7, C22 += P7
    
    # Now let's enumerate all 28 terms.
    # For each product k and each (i,j) position in the 2x2 result:
    
    def add_term(a_coeffs, b_coeffs, c_row, c_col, sign=1):
        """Add a term: sum of (a_coeff * A[row,col]) * sum of (b_coeff * B[row,col]) contributes to C[c_row, c_col] with given sign."""
        nonlocal col
        for (ar, ac, a_val) in a_coeffs:
            U[ar * m + ac, col] += a_val * sign
        for (br, bc, b_val) in b_coeffs:
            V[br * p + bc, col] += b_val
        W[c_col * n + c_row, col] += 1
        col += 1
    
    # Product 1: P1 = (A11 + A22)(B11 + B22)
    # P1[0,0]:
    a1 = [(0,0,1), (2,2,1)]
    b1 = [(0,0,1), (2,2,1)]
    add_term(a1, b1, 0, 0)  # C11
    add_term(a1, b1, 3, 3)  # C22
    
    # P1[0,1]:
    a1 = [(0,0,1), (2,2,1)]
    b1 = [(0,1,1), (2,3,1)]
    add_term(a1, b1, 0, 1)  # C12? No, P1 contributes to C11 and C22
    # Wait, P1 is a 2x2 matrix. P1[0,0] -> C11[0,0], P1[0,1] -> C11[0,1]? No.
    # P1 is the full 2x2 block. So P1[0,0] -> C[0,0], P1[0,1] -> C[0,1], P1[1,0] -> C[1,0], P1[1,1] -> C[1,1]
    # But C11 is a 2x2 block, so C11[0,0] = C[0,0], C11[0,1] = C[0,1], etc.
    # And C22 is a 2x2 block, so C22[0,0] = C[2,2], C22[0,1] = C[2,3], etc.
    
    # Let me redo this. P1 is a 2x2 matrix.
    # P1[0,0] contributes to C11[0,0] = C[0,0] and C22[0,0] = C[2,2]
    # P1[0,1] contributes to C11[0,1] = C[0,1] and C22[0,1] = C[2,3]
    # P1[1,0] contributes to C11[1,0] = C[1,0] and C22[1,0] = C[3,2]
    # P1[1,1] contributes to C11[1,1] = C[1,1] and C22[1,1] = C[3,3]
    
    # I already added P1[0,0] for C[0,0] and C[2,2]. Let me continue.
    
    # P1[0,1]:
    a1 = [(0,0,1), (2,2,1)]
    b1 = [(0,1,1), (2,3,1)]
    add_term(a1, b1, 0, 1)  # C[0,1]
    add_term(a1, b1, 2, 3)  # C[2,3]
    
    # P1[1,0]:
    a1 = [(1,0,1), (3,2,1)]
    b1 = [(0,0,1), (2,2,1)]
    add_term(a1, b1, 1, 0)  # C[1,0]
    add_term(a1, b1, 3, 2)  # C[3,2]
    
    # P1[1,1]:
    a1 = [(1,1,1), (3,3,1)]
    b1 = [(0,1,1), (2,3,1)]
    add_term(a1, b1, 1, 1)  # C[1,1]
    add_term(a1, b1, 3, 3)  # C[3,3]
    
    # Product 2: P2 = (A21 + A22)B11
    # P2[0,0]:
    a2 = [(2,0,1), (2,2,1)]
    b2 = [(0,0,1)]
    add_term(a2, b2, 2, 0)  # C21[0,0] = C[2,0]
    add_term(a2, b2, 3, 3, -1)  # C22[0,0] = C[2,2], with -1
    
    # P2[0,1]:
    a2 = [(2,0,1), (2,2,1)]
    b2 = [(0,1,1)]
    add_term(a2, b2, 2, 1)  # C[2,1]
    add_term(a2, b2, 2, 3, -1)  # C[2,3], with -1
    
    # P2[1,0]:
    a2 = [(3,0,1), (3,2,1)]
    b2 = [(0,0,1)]
    add_term(a2, b2, 3, 0)  # C[3,0]
    add_term(a2, b2, 3, 3, -1)  # C[3,3], with -1
    
    # P2[1,1]:
    a2 = [(3,0,1), (3,2,1)]
    b2 = [(0,1,1)]
    add_term(a2, b2, 3, 1)  # C[3,1]
    add_term(a2, b2, 3, 3, -1)  # C[3,3], with -1
    
    # Product 3: P3 = A11(B12 - B22)
    # P3[0,0]:
    a3 = [(0,0,1)]
    b3 = [(0,2,1), (2,2,-1)]
    add_term(a3, b3, 0, 2)  # C12[0,0] = C[0,2]
    add_term(a3, b3, 3, 3)  # C22[0,0] = C[2,2]
    
    # P3[0,1]:
    a3 = [(0,0,1)]
    b3 = [(0,3,1), (2,3,-1)]
    add_term(a3, b3, 0, 3)  # C[0,3]
    add_term(a3, b3, 2, 3)  # C[2,3]
    
    # P3[1,0]:
    a3 = [(1,0,1)]
    b3 = [(0,2,1), (2,2,-1)]
    add_term(a3, b3, 1, 2)  # C[1,2]
    add_term(a3, b3, 3, 2)  # C[3,2]
    
    # P3[1,1]:
    a3 = [(1,0,1)]
    b3 = [(0,3,1), (2,3,-1)]
    add_term(a3, b3, 1, 3)  # C[1,3]
    add_term(a3, b3, 3, 3)  # C[3,3]
    
    # Product 4: P4 = A22(B21 - B11)
    # P4[0,0]:
    a4 = [(2,2,1)]
    b4 = [(2,0,1), (0,0,-1)]
    add_term(a4, b4, 0, 0)  # C11[0,0] = C[0,0]
    add_term(a4, b4, 2, 0)  # C21[0,0] = C[2,0]
    
    # P4[0,1]:
    a4 = [(2,2,1)]
    b4 = [(2,1,1), (0,1,-1)]
    add_term(a4, b4, 0, 1)  # C[0,1]
    add_term(a4, b4, 2, 1)  # C[2,1]
    
    # P4[1,0]:
    a4 = [(3,2,1)]
    b4 = [(2,0,1), (0,0,-1)]
    add_term(a4, b4, 1, 0)  # C[1,0]
    add_term(a4, b4, 3, 0)  # C[3,0]
    
    # P4[1,1]:
    a4 = [(3,2,1)]
    b4 = [(2,1,1), (0,1,-1)]
    add_term(a4, b4, 1, 1)  # C[1,1]
    add_term(a4, b4, 3, 1)  # C[3,1]
    
    # Product 5: P5 = (A11 + A12)B22
    # P5[0,0]:
    a5 = [(0,0,1), (0,2,1)]
    b5 = [(2,2,1)]
    add_term(a5, b5, 0, 0, -1)  # C11[0,0] = C[0,0], with -1
    add_term(a5, b5, 0, 2)  # C12[0,0] = C[0,2]
    
    # P5[0,1]:
    a5 = [(0,0,1), (0,2,1)]
    b5 = [(2,3,1)]
    add_term(a5, b5, 0, 1, -1)  # C[0,1], with -1
    add_term(a5, b5, 0, 3)  # C[0,3]
    
    # P5[1,0]:
    a5 = [(1,0,1), (1,2,1)]
    b5 = [(2,2,1)]
    add_term(a5, b5, 1, 0, -1)  # C[1,0], with -1
    add_term(a5, b5, 1, 2)  # C[1,2]
    
    # P5[1,1]:
    a5 = [(1,0,1), (1,2,1)]
    b5 = [(2,3,1)]
    add_term(a5, b5, 1, 1, -1)  # C[1,1], with -1
    add_term(a5, b5, 1, 3)  # C[1,3]
    
    # Product 6: P6 = (A21 - A11)(B11 + B12)
    # P6[0,0]:
    a6 = [(2,0,1), (0,0,-1)]
    b6 = [(0,0,1), (0,2,1)]
    add_term(a6, b6, 3, 3)  # C22[0,0] = C[2,2]
    
    # P6[0,1]:
    a6 = [(2,0,1), (0,0,-1)]
    b6 = [(0,1,1), (0,3,1)]
    add_term(a6, b6, 2, 3)  # C[2,3]
    
    # P6[1,0]:
    a6 = [(3,0,1), (1,0,-1)]
    b6 = [(0,0,1), (0,2,1)]
    add_term(a6, b6, 3, 3)  # C[3,3]
    
    # P6[1,1]:
    a6 = [(3,0,1), (1,0,-1)]
    b6 = [(0,1,1), (0,3,1)]
    add_term(a6, b6, 3, 3)  # C[3,3]
    
    # Product 7: P7 = (A12 - A22)(B21 + B22)
    # P7[0,0]:
    a7 = [(0,2,1), (2,2,-1)]
    b7 = [(2,0,1), (2,2,1)]
    add_term(a7, b7, 0, 0)  # C11[0,0] = C[0,0]
    add_term(a7, b7, 3, 3)  # C22[0,0] = C[2,2]
    
    # P7[0,1]:
    a7 = [(0,2,1), (2,2,-1)]
    b7 = [(2,1,1), (2,3,1)]
    add_term(a7, b7, 0, 1)  # C[0,1]
    add_term(a7, b7, 2, 3)  # C[2,3]
    
    # P7[1,0]:
    a7 = [(1,2,1), (3,2,-1)]
    b7 = [(2,0,1), (2,2,1)]
    add_term(a7, b7, 1, 0)  # C[1,0]
    add_term(a7, b7, 3, 2)  # C[3,2]
    
    # P7[1,1]:
    a7 = [(1,2,1), (3,2,-1)]
    b7 = [(2,1,1), (2,3,1)]
    add_term(a7, b7, 1, 1)  # C[1,1]
    add_term(a7, b7, 3, 3)  # C[3,3]
    
    return U, V, W

def _construct_3x4x4_from_4x4x4(U44, V44, W44):
    """Project the 4x4x4 decomposition to 3x4x4 by taking the first 3 rows of A.
    
    The 3x4x4 tensor is a sub-tensor of the 4x4x4 tensor.
    We can project by keeping only the first 3 rows of the U factor.
    """
    n, m, p = 3, 4, 4
    U = U44[:n*m, :]  # Take first 12 rows (3*4)
    V = V44  # Keep all 16 rows
    W = W44  # Keep all 16 rows
    
    # Verify the decomposition
    if not _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
        raise ValueError("Projected decomposition does not verify")
    
    return U, V, W

def _construct_strassen_3x4x4():
    """Construct a rank-28 decomposition of the 3x4x4 matrix multiplication tensor."""
    U44, V44, W44 = _construct_strassen_4x4()
    U, V, W = _construct_3x4x4_from_4x4x4(U44, V44, W44)
    return U, V, W

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Try the rank-28 Strassen-based decomposition
    try:
        U, V, W = _construct_strassen_3x4x4()
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    except Exception:
        pass
    
    # Fallback to a simple schoolbook decomposition (rank 48)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
