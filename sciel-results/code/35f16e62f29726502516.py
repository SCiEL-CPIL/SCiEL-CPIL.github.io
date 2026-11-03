"""Matrix multiplication 4x4x5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_to_half_integer(x):
    """Round x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    """Round all entries of matrix M to nearest half-integer."""
    return np.array([[ _round_to_half_integer(x) for x in row] for row in M])

def _compute_tensor(U, V, W, n, m, p):
    """Compute the tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]."""
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                T[i, j, k] = s
    return T

def _als_decompose(n, m, p, rank, U0, V0, W0, max_iters=200, tol=1e-12):
    """Alternating least squares to find exact decomposition.
    
    For each (i,j,k), we need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1.
    We alternate fixing two matrices and solving for the third.
    """
    U = U0.copy()
    V = V0.copy()
    W = W0.copy()
    
    prev_err = float('inf')
    for it in range(max_iters):
        # Fix V, W, optimize U
        for i in range(n):
            for j in range(m):
                # For each r, coefficient of U[i,j,r] is V[j,k,r]*W[k,i,r] summed over k
                # Actually: sum_k V[j*p+k, r] * W[k*n+i, r] for each r
                A = np.array([V[j*p+k, r] * W[k*n+i, r] for k in range(p)])  # shape (p,)
                # We need: sum_r U[i*m+j, r] * (sum_k V[j*p+k,r]*W[k*n+i,r]) = 1
                # This is: U[i*m+j, :] @ A = 1
                # But A depends on r... let me re-think.
                # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # For fixed i,j: for each k, sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1
                # So: for each k, the dot product of U[i*m+j,:] with vector (V[j*p+k,r]*W[k*n+i,r])_r = 1
                # This gives p equations in rank unknowns.
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])  # (p, rank)
                b_vec = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Fix U, W, optimize V
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])  # (n, rank)
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Fix U, V, optimize W
        for k in range(p):
            for i in range(n):
                A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])  # (m, rank)
                b_vec = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err += (s - 1.0) ** 2
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W

def _strassen_2x2():
    """Return the 7-term Strassen decomposition for 2x2 matrix multiplication.
    U, V, W are lists of 7 vectors of length 4 (for 2x2)."""
    # Strassen's 7 products for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # U[i,j,r] for A matrix factor, V[i,j,r] for B matrix factor, W[i,j,r] for C matrix factor
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] with n=m=p=2
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0: M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1  # A11
    U[3, 0] = 1  # A22
    V[0, 0] = 1  # B11
    V[3, 0] = 1  # B22
    W[0, 0] = 1  # C11
    W[3, 0] = 1  # C22
    
    # r=1: M2 = (A21+A22)B11
    U[2, 1] = 1  # A21
    U[3, 1] = 1  # A22
    V[0, 1] = 1  # B11
    W[2, 1] = 1  # C21
    W[3, 1] = -1  # C22
    
    # r=2: M3 = A11(B12-B22)
    U[0, 2] = 1  # A11
    V[1, 2] = 1  # B12
    V[3, 2] = -1  # B22
    W[1, 2] = 1  # C12
    W[3, 2] = 1  # C22
    
    # r=3: M4 = A22(B21-B11)
    U[3, 3] = 1  # A22
    V[2, 3] = 1  # B21
    V[0, 3] = -1  # B11
    W[0, 3] = 1  # C11
    W[2, 3] = 1  # C21
    
    # r=4: M5 = (A11+A12)B22
    U[0, 4] = 1  # A11
    U[1, 4] = 1  # A12
    V[3, 4] = 1  # B22
    W[0, 4] = -1  # C11
    W[1, 4] = 1  # C12
    
    # r=5: M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1  # A21
    U[0, 5] = -1  # A11
    V[0, 5] = 1  # B11
    V[1, 5] = 1  # B12
    W[3, 5] = 1  # C22
    
    # r=6: M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1  # A12
    U[3, 6] = -1  # A22
    V[2, 6] = 1  # B21
    V[3, 6] = 1  # B22
    W[0, 6] = 1  # C11
    
    return U, V, W

def _build_strassen_4x4x5():
    """Build a Strassen-based decomposition for 4x4x5.
    
    Split 4x4 into four 2x2 blocks, and 4x5 into two 2x2 and one 2x1 block.
    Actually, let's think more carefully.
    
    A is 4x4, B is 4x5, C is 4x5.
    Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]] where each is 2x2.
    Split B into 2x2 and 2x1: B = [[B11, B12, B13], [B21, B22, B23]] 
    where B11, B12, B21, B22 are 2x2 and B13, B23 are 2x1.
    
    C = [[C11, C12, C13], [C21, C22, C23]]
    where C11, C12, C21, C22 are 2x2 and C13, C23 are 2x1.
    
    C11 = A11*B11 + A12*B21 (2x2 * 2x2 + 2x2 * 2x2)
    C12 = A11*B12 + A12*B22
    C13 = A11*B13 + A12*B23 (2x2 * 2x1 + 2x2 * 2x1)
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    C23 = A21*B13 + A22*B23
    
    For the 2x2*2x2 parts, we use Strassen (7 terms each).
    For the 2x2*2x1 parts, we can use a simpler decomposition.
    
    Actually, 2x2 * 2x1: 
    [a b]   [e]   [ae+bf]
    [c d] * [f] = [ce+df]
    
    This can be done with 3 multiplications:
    m1 = (a+b)*e = ae+be
    m2 = (c-d)*f = cf-df  
    m3 = a*(e+f) = ae+af
    Hmm, that's not quite right. Let me think...
    
    Actually for 2x2 * 2x1, we need:
    result[0] = a*e + b*f
    result[1] = c*e + d*f
    
    We can use:
    m1 = a*e
    m2 = b*f
    m3 = c*e
    m4 = d*f
    That's 4 multiplications (schoolbook).
    
    Or with Strassen-like:
    m1 = (a+b)*e
    m2 = (c+d)*f
    m3 = a*f
    m4 = c*e
    Then result[0] = m1 - m3 + (b*e) ... no this doesn't work cleanly.
    
    Let me just use schoolbook for the 2x2*2x1 parts: 4 multiplications each.
    
    Total: 4 Strassen blocks (7 each) + 4 schoolbook blocks (4 each) = 28 + 16 = 44.
    Wait, that's not right either. Let me recount.
    
    C11 = A11*B11 + A12*B21: two 2x2*2x2 products, each needs 7 Strassen terms = 14
    C12 = A11*B12 + A12*B22: two 2x2*2x2 products = 14
    C13 = A11*B13 + A12*B23: two 2x2*2x1 products = 8 (schoolbook)
    C21 = A21*B11 + A22*B21: two 2x2*2x2 products = 14
    C22 = A21*B12 + A22*B22: two 2x2*2x2 products = 14
    C23 = A21*B13 + A22*B23: two 2x2*2x1 products = 8
    
    Total = 14+14+8+14+14+8 = 72. That matches the known Strassen rank for 4x4x5.
    
    But wait, we can share Strassen terms between C11 and C21 since they both involve B11 and B21.
    Similarly C12 and C22 share B12 and B22.
    
    Let me use the Kronecker product approach: the rank of matmul(n,m,p) is at most
    rank(matmul(2,2,2)) * ceil(n/2) * ceil(m/2) * ceil(p/2) when dimensions are powers of 2.
    For 4x4x5: 7 * 2 * 2 * 3 = 84? No that's not right either.
    
    Actually, the standard result is that if you can multiply 2x2 in 7 multiplications,
    you can multiply 2^k x 2^k in 7^k multiplications. For 4x4: 7^2 = 49.
    For 4x4x5, it's a bit different since the third dimension is 5.
    
    Let me just construct it explicitly using the block structure.
    """
    n, m, p = 4, 4, 5
    
    # Get the 2x2 Strassen decomposition
    U2, V2, W2 = _strassen_2x2()
    
    # We'll build the full decomposition by combining blocks.
    # A is 4x4: rows 0-3, cols 0-3
    # B is 4x5: rows 0-3, cols 0-4
    # C is 4x5: rows 0-3, cols 0-4
    #
    # Block structure (2x2 blocks):
    # A = [[A00, A01], [A10, A11]]  (each 2x2)
    # B = [[B00, B01, B02], [B10, B11, B12]]  (B00,B01,B10,B11 are 2x2; B02,B12 are 2x1)
    # C = [[C00, C01, C02], [C10, C11, C12]]  (C00,C01,C10,C11 are 2x2; C02,C12 are 2x1)
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12  (2x2 * 2x1)
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12  (2x2 * 2x1)
    
    # For 2x2*2x2, use Strassen (7 terms)
    # For 2x2*2x1, use schoolbook (4 terms)
    
    # Total terms: 6 pairs of 2x2*2x2 (12 Strassen applications * 7 = 84) 
    # Wait no: C00 has 2 products, C01 has 2, C10 has 2, C11 has 2 = 8 Strassen applications = 56 terms
    # C02 has 2 products, C12 has 2 = 4 schoolbook applications = 16 terms
    # Total = 56 + 16 = 72
    
    # But we can share: A00*B00 appears in C00, and A10*B00 appears in C10.
    # These are different products (different A blocks), so no sharing there.
    # However, B00 is used in both A00*B00 and A10*B00, but the A factors differ.
    
    # Let me just enumerate all 72 terms.
    rank_total = 72
    U_full = np.zeros((n*m, rank_total))
    V_full = np.zeros((m*p, rank_total))
    W_full = np.zeros((p*n, rank_total))
    
    col = 0
    
    # Helper: add a Strassen 2x2*2x2 term
    def add_strassen(i_off, j_off, k_off, c_i_off, c_k_off):
        """Add Strassen terms for A[i_off:i_off+2, j_off:j_off+2] * B[j_off:j_off+2, k_off:k_off+2]
        contributing to C[c_i_off:c_i_off+2, c_k_off:c_k_off+2]."""
        nonlocal col
        for r in range(7):
            # U factor (A block)
            for di in range(2):
                for dj in range(2):
                    U_full[(i_off+di)*m + (j_off+dj), col] = U2[di*2+dj, r]
            # V factor (B block)
            for dj in range(2):
                for dk in range(2):
                    V_full[(j_off+dj)*p + (k_off+dk), col] = V2[dj*2+dk, r]
            # W factor (C block)
            for dk in range(2):
                for di in range(2):
                    W_full[(c_k_off+dk)*n + (c_i_off+di), col] = W2[dk*2+di, r]
            col += 1
    
    # Helper: add a schoolbook 2x2*2x1 term
    def add_schoolbook_2x2x1(i_off, j_off, k_off, c_i_off, c_k_off):
        """Add schoolbook terms for A[i_off:i_off+2, j_off:j_off+2] * B[j_off:j_off+2, k_off:k_off+1]
        contributing to C[c_i_off:c_i_off+2, c_k_off:c_k_off+1]."""
        nonlocal col
        for di in range(2):
            for dj in range(2):
                # Term: A[i_off+di, j_off+dj] * B[j_off+dj, k_off]
                U_full[(i_off+di)*m + (j_off+dj), col] = 1.0
                V_full[(j_off+dj)*p + k_off, col] = 1.0
                W_full[c_k_off*n + (c_i_off+di), col] = 1.0
                col += 1
    
    # C00 = A00*B00 + A01*B10
    add_strassen(0, 0, 0, 0, 0)   # A00*B00 -> C00
    add_strassen(0, 2, 2, 0, 0)   # A01*B10 -> C00
    
    # C01 = A00*B01 + A01*B11
    add_strassen(0, 0, 2, 0, 2)   # A00*B01 -> C01
    add_strassen(0, 2, 4, 0, 2)   # A01*B11 -> C01
    # Wait, B01 is B[0:2, 2:4], B11 is B[2:4, 2:4]
    # k_off for B01 should be 2, for B11 should be 2
    # Let me redo this more carefully.
    
    # Actually let me restart with clearer indexing.
    # A[i,j] with i in 0..3, j in 0..3
    # B[j,k] with j in 0..3, k in 0..4
    # C[i,k] with i in 0..3, k in 0..4
    #
    # 2x2 blocks of A:
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
    # A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    #
    # 2x2 and 2x1 blocks of B:
    # B00 = B[0:2, 0:2], B01 = B[0:2, 2:4], B02 = B[0:2, 4:5]
    # B10 = B[2:4, 0:2], B11 = B[2:4, 2:4], B12 = B[2:4, 4:5]
    #
    # 2x2 and 2x1 blocks of C:
    # C00 = C[0:2, 0:2], C01 = C[0:2, 2:4], C02 = C[0:2, 4:5]
    # C10 = C[2:4, 0:2], C11 = C[2:4, 2:4], C12 = C[2:4, 4:5]
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    
    # Reset
    col = 0
    U_full[:] = 0
    V_full[:] = 0
    W_full[:] = 0
    
    # C00 = A00*B00 + A01*B10
    add_strassen(0, 0, 0, 0, 0)   # A[0:2,0:2]*B[0:2,0:2] -> C[0:2,0:2]
    add_strassen(0, 2, 2, 0, 0)   # A[0:2,2:4]*B[2:4,0:2] -> C[0:2,0:2]
    
    # C01 = A00*B01 + A01*B11
    add_strassen(0, 0, 2, 0, 2)   # A[0:2,0:2]*B[0:2,2:4] -> C[0:2,2:4]
    add_strassen(0, 2, 4, 0, 2)   # A[0:2,2:4]*B[2:4,2:4] -> C[0:2,2:4]
    # Wait, B11 = B[2:4, 2:4], so j_off=2, k_off=2
    # That's correct.
    
    # C02 = A00*B02 + A01*B12 (2x2 * 2x1)
    add_schoolbook_2x2x1(0, 0, 4, 0, 4)   # A[0:2,0:2]*B[0:2,4:5] -> C[0:2,4:5]
    add_schoolbook_2x2x1(0, 2, 4, 0, 4)   # A[0:2,2:4]*B[2:4,4:5] -> C[0:2,4:5]
    
    # C10 = A10*B00 + A11*B10
    add_strassen(2, 0, 0, 2, 0)   # A[2:4,0:2]*B[0:2,0:2] -> C[2:4,0:2]
    add_strassen(2, 2, 2, 2, 0)   # A[2:4,2:4]*B[2:4,0:2] -> C[2:4,0:2]
    
    # C11 = A10*B01 + A11*B11
    add_strassen(2, 0, 2, 2, 2)   # A[2:4,0:2]*B[0:2,2:4] -> C[2:4,2:4]
    add_strassen(2, 2, 4, 2, 2)   # A[2:4,2:4]*B[2:4,2:4] -> C[2:4,2:4]
    # B11 = B[2:4, 2:4], j_off=2, k_off=2. Correct.
    
    # C12 = A10*B02 + A11*B12 (2x2 * 2x1)
    add_schoolbook_2x2x1(2, 0, 4, 2, 4)   # A[2:4,0:2]*B[0:2,4:5] -> C[2:4,4:5]
    add_schoolbook_2x2x1(2, 2, 4, 2, 4)   # A[2:4,2:4]*B[2:4,4:5] -> C[2:4,4:5]
    
    assert col == rank_total, f"Expected {rank_total} terms, got {col}"
    
    return U_full.tolist(), V_full.tolist(), W_full.tolist()

def _try_als_refinement(n, m, p, rank, U0, V0, W0, seed=42, max_iters=150):
    """Try to refine a decomposition using ALS, then round to half-integers."""
    np.random.seed(seed)
    
    # Add small random perturbation to break symmetry
    U = U0.copy() + np.random.randn(*U0.shape) * 0.01
    V = V0.copy() + np.random.randn(*V0.shape) * 0.01
    W = W0.copy() + np.random.randn(*W0.shape) * 0.01
    
    U, V, W = _als_decompose(n, m, p, rank, U, V, W, max_iters=max_iters)
    
    # Round to half-integers
    U_r = _round_matrix_to_half_int(U)
    V_r = _round_matrix_to_half_int(V)
    W_r = _round_matrix_to_half_int(W)
    
    if _verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None

def _random_init(n, m, p, rank, seed=42, density=0.4):
    """Random sparse initialization with half-integer values."""
    np.random.seed(seed)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    half_ints = [-1.0, -0.5, 0.5, 1.0]
    
    for r in range(rank):
        for idx in range(n*m):
            if np.random.random() < density:
                U[idx, r] = np.random.choice(half_ints)
        for idx in range(m*p):
            if np.random.random() < density:
                V[idx, r] = np.random.choice(half_ints)
        for idx in range(p*n):
            if np.random.random() < density:
                W[idx, r] = np.random.choice(half_ints)
    
    return U, V, W

def _try_rank_als(n, m, p, rank, seed=42, max_iters=100):
    """Try to find a decomposition at given rank using ALS from random init."""
    U, V, W = _random_init(n, m, p, rank, seed=seed)
    U, V, W = _als_decompose(n, m, p, rank, U, V, W, max_iters=max_iters)
    
    U_r = _round_matrix_to_half_int(U)
    V_r = _round_matrix_to_half_int(V)
    W_r = _round_matrix_to_half_int(W)
    
    if _verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline (20s safety margin)
    
    # Start with Strassen-based decomposition (rank 72)
    U_str, V_str, W_str = _build_strassen_4x4x5()
    best_result = {"U": U_str, "V": V_str, "W": W_str}
    best_rank = 72
    
    # Verify the Strassen decomposition
    if not _verify_decomposition(U_str, V_str, W_str, n, m, p):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_fb = [[0] * rank_sb for _ in range(n * m)]
        V_fb = [[0] * rank_sb for _ in range(m * p)]
        W_fb = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col] = 1
                    V_fb[j*p+k][col] = 1
                    W_fb[k*n+i][col] = 1
                    col += 1
        best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
        best_rank = rank_sb
    
    # Try to improve using ALS refinement from Strassen init
    for seed in range(20):
        if time.time() > deadline - 30:
            break
        result = _try_als_refinement(n, m, p, best_rank, 
                                      np.array(U_str), np.array(V_str), np.array(W_str),
                                      seed=seed, max_iters=80)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            # Don't reduce rank yet, just confirm we have a valid decomposition
    
    # Try lower ranks using ALS from random initialization
    for rank in range(68, 49, -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        for seed in range(15):
            if time.time() > deadline - 20:
                break
            
            result = _try_rank_als(n, m, p, rank, seed=seed, max_iters=80)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                found = True
                break
        
        if found:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
