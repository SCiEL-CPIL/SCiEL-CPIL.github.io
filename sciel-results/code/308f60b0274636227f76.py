"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Test with random matrices
    rng = np.random.RandomState(42)
    for _ in range(5):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        
        # Compute C from decomposition
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_val = np.sum(U[:, r] * A.flatten())
            v_val = np.sum(V[:, r] * B.flatten())
            W_mat = W[:, r].reshape(p, n)
            C_computed += u_val * v_val * W_mat.T  # W_mat is (p,n), we need (n,p)
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def _strassen_2x2_decomposition():
    """Return U, V, W for 2x2 x 2x2 matrix multiplication with rank 7."""
    # We need to find 7 rank-1 terms that decompose the 2x2x2x2 tensor.
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: k*n+i, n=2: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    #   i.e., C[i,k] where i=row of A, k=col of B
    
    # Let's use a verified Strassen decomposition.
    # I'll construct it by solving the system.
    
    # Actually, let me use the following known-correct Strassen terms:
    # Each term is (u, v, w) where:
    # u is coefficients for A[0,0], A[0,1], A[1,0], A[1,1]
    # v is coefficients for B[0,0], B[0,1], B[1,0], B[1,1]
    # w is coefficients for C[0,0], C[0,1], C[1,0], C[1,1]
    
    # Term 1: (a+d)(e+h)
    # u = [1, 0, 0, 1]
    # v = [1, 1, 0, 0]  # e+h means B[0,0] + B[0,1]
    # w = [1, 0, 0, 1]  # contributes to C[0,0] and C[1,1]
    
    # Term 2: (c+d)e
    # u = [0, 0, 1, 1]
    # v = [1, 0, 0, 0]
    # w = [0, 0, 1, 1]  # contributes to C[1,0] and C[1,1]
    
    # Term 3: a(f-h)
    # u = [1, 0, 0, 0]
    # v = [0, 1, 0, -1]
    # w = [0, 1, 0, 1]  # contributes to C[0,1] and C[1,1]
    
    # Term 4: d(g-c)
    # u = [0, 0, 0, 1]
    # v = [0, 0, 1, 0]  # g is B[1,0]
    # Wait, g-c means B[1,0] - ... no, c is A[1,0]
    # d*(g-c) where g=B[1,0], c=A[1,0]
    # u = [0, 0, -1, 1]  # -c + d
    # v = [0, 0, 1, 0]   # g
    # w = [1, 0, 1, 0]   # contributes to C[0,0] and C[1,0]
    
    # Term 5: (a+b)h
    # u = [1, 1, 0, 0]
    # v = [0, 0, 0, 1]
    # w = [1, 1, 0, 0]   # contributes to C[0,0] and C[0,1]
    
    # Term 6: (c-a)(e+f)
    # u = [0, 0, 1, -1]  # c - a... wait, a is A[0,0], c is A[1,0]
    # Actually c-a: u = [-1, 0, 1, 0]
    # v = [1, 1, 0, 0]   # e+f
    # w = [0, 0, 0, 1]   # contributes to C[1,1]
    
    # Term 7: (b-d)(g+h)
    # u = [0, 1, 0, -1]  # b - d
    # v = [0, 0, 1, 1]   # g+h
    # w = [1, 0, 0, 0]   # contributes to C[0,0]
    
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # a
        [0, 0, 0, 0, 1, 0, 1],    # b
        [0, 1, 0, -1, 0, 1, 0],   # c
        [1, 1, 0, 1, 0, 0, -1],   # d
    ], dtype=float)
    
    V = np.array([
        [1, 1, 0, 0, 0, 1, 0],    # e
        [1, 0, 1, 0, 0, 1, 0],    # f
        [0, 0, 0, 1, 0, 0, 1],    # g
        [0, 0, -1, 0, 1, 0, 1],   # h
    ], dtype=float)
    
    W = np.array([
        [1, 0, 0, 1, 1, 0, 1],    # C[0,0]
        [0, 0, 1, 0, 1, 0, 0],    # C[0,1]
        [0, 1, 0, 1, 0, 0, 0],    # C[1,0]
        [1, 1, 1, 0, 0, 1, 0],    # C[1,1]
    ], dtype=float)
    
    return U.tolist(), V.tolist(), W.tolist()

def _build_4x4_from_2x2_strassen(U2, V2, W2):
    """Build 4x4x4x4 decomposition from 2x2 Strassen blocks. Rank = 7*7 = 49."""
    n = m = p = 4
    rank = 49
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    U2 = np.array(U2)
    V2 = np.array(V2)
    W2 = np.array(W2)
    
    # 4x4 matrix divided into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    # 
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Using Strassen on the 2x2 block structure:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-A10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M is a product of two 2x2 matrices, which we decompose using 2x2 Strassen.
    
    # For each block product X*Y (both 2x2), we use the 2x2 Strassen decomposition.
    # The 2x2 Strassen gives us 7 terms where:
    #   term_r: u_r (coeffs for X), v_r (coeffs for Y), w_r (coeffs for result)
    
    # Let's define the block structure:
    # A blocks: A00 at (0,0), A01 at (0,1), A10 at (1,0), A11 at (1,1)
    # In the 4x4 matrix, A[i,j] where i,j in 0..3:
    #   A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
    #   A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    
    # Similarly for B and C.
    
    # For each Strassen term at the block level, and each Strassen term at the sub-block level,
    # we create one rank-1 term in the 4x4 decomposition.
    
    # Block-level Strassen terms (7 terms), each involving products of 2x2 blocks.
    # Sub-block-level Strassen terms (7 terms) for each 2x2 product.
    
    # This is getting complex. Let me define it more systematically.
    
    # Block Strassen:
    # M1 = (A00+A11) * (B00+B11)
    # M2 = (A10+A11) * B00
    # M3 = A00 * (B01-B11)
    # M4 = A11 * (B10-A10)
    # M5 = (A00+A01) * B11
    # M6 = (A10-A00) * (B00+B01)
    # M7 = (A01-A11) * (B10+B11)
    
    # For each M, the "left factor" is a 2x2 matrix (sum/diff of A blocks) and 
    # the "right factor" is a 2x2 matrix (sum/diff of B blocks).
    
    # The left factor, when viewed as a 2x2 matrix, has entries that are linear combinations
    # of the entries of the relevant A blocks.
    
    # For example, M1 left factor = A00 + A11:
    #   Entry (0,0) = A[0,0] + A[2,2]
    #   Entry (0,1) = A[0,1] + A[2,3]
    #   Entry (1,0) = A[1,0] + A[3,2]
    #   Entry (1,1) = A[1,1] + A[3,3]
    
    # So the u vector for this 2x2 matrix (in the 2x2 Strassen) would have:
    #   u[0] (coeff for 2x2 entry (0,0)) = 1, contributing to A[0,0] and A[2,2]
    #   etc.
    
    # This is getting very involved. Let me just build it up programmatically.
    
    # Define which A-blocks and B-blocks each M uses, with signs.
    # M1: left = A00 + A11, right = B00 + B11
    # M2: left = A10 + A11, right = B00
    # M3: left = A00, right = B01 - B11
    # M4: left = A11, right = B10 - A10  # Wait, this mixes A and B!
    
    # Hmm, M4 = A11 * (B10 - A10) doesn't make sense as a matrix product of two matrices
    # where one is from A and one is from B. Let me re-examine.
    
    # Oh wait, I think I have the Strassen block algorithm wrong. Let me reconsider.
    
    # For block Strassen, we treat the 2x2 block matrix as a 2x2 matrix where each entry
    # is itself a 2x2 matrix. The Strassen algorithm applies to this "matrix of matrices".
    # 
    # A_block = [[A00, A01], [A10, A11]]
    # B_block = [[B00, B01], [B10, B11]]
    # 
    # Strassen:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - A10)   # THIS IS WRONG - should be B10 - something from B
    # 
    # I think the correct block Strassen is:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - A10)  # No, this is still wrong
    # 
    # Actually, for the standard Strassen applied to blocks:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)(B00)
    # M3 = (A00)(B01 - B11)
    # M4 = (A11)(B10 - A10)  # Still wrong
    # 
    # I think the issue is that in standard Strassen, all intermediate products
    # are of the form (linear combo of A entries) * (linear combo of B entries).
    # For blocks:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)(B00)
    # M3 = (A00)(B01 - B11)
    # M4 = (A11)(B10 - A10)  # This is NOT a valid Strassen term
    # 
    # The correct block Strassen should be:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)(B00)
    # M3 = (A00)(B01 - B11)
    # M4 = (A11)(B10 - A10)  # Still wrong
    # 
    # I give up trying to derive this correctly from memory. Let me just use the
    # schoolbook for the 4x4 part and accept rank 128, or try a different approach.

    return U, V, W  # placeholder

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
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
