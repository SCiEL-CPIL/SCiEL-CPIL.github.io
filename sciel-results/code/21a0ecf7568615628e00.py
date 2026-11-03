"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = 140  # seconds, safety margin before 150s kill


def _build_schoolbook(n, m, p):
    """Return (U, V, W) as numpy arrays for the schoolbook decomposition (rank = n*m*p)."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = U.shape[1]
    # The multiplication tensor T[i,j,k] = delta_{i,k} in the sense that
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # In tensor form: T is a 3-tensor where T[i, j, k] contributes to C[i,k] via A[i,j]*B[j,k]
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Build the tensor from the decomposition
    T_decomp = np.zeros((n, m, p))
    for r in range(rank):
        u_vec = U[:, r]  # shape (n*m,)
        v_vec = V[:, r]  # shape (m*p,)
        w_vec = W[:, r]  # shape (p*n,)
        
        # u_vec[i*m+j] = U[i*m+j, r]
        # v_vec[j*p+k] = V[j*p+k, r]
        # w_vec[k*n+i] = W[k*n+i, r]
        
        u_mat = u_vec.reshape(n, m)
        v_mat = v_vec.reshape(m, p)
        w_mat = w_vec.reshape(p, n)
        
        # T_decomp[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_decomp[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
    
    # The true tensor: T[i,j,k] = 1 if the indices are consistent with matrix mult
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = 1 for all valid (i,j,k) since each term A[i,j]*B[j,k] appears exactly once
    T_true = np.ones((n, m, p))
    
    if not np.allclose(T_decomp, T_true, atol=tol):
        return False
    return True


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using matrix operations."""
    rank = U.shape[1]
    # T_decomp[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # We can compute this as: for each r, outer product structure
    
    # Alternative: check that for random A, B, the product matches
    # C = A @ B should equal sum_r (U[:,r] reshaped as n x m) * (V[:,r] reshaped as m x p) ... no
    # Actually: C[i,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    #           = sum_r W[k*n+i,r] * sum_j U[i*m+j,r] * V[j*p+k,r]
    # Let M_r = U[:,r].reshape(n,m) @ V[:,r].reshape(m,p)  -- no, that's not right either
    # M_r[i,k] = sum_j U[i*m+j,r] * V[j*p+k,r]
    # Then C[i,k] = sum_r W[k*n+i,r] * M_r[i,k]
    
    # Let's verify with random matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_decomp = np.zeros((n, p))
    for r in range(rank):
        u_mat = U[:, r].reshape(n, m)
        v_mat = V[:, r].reshape(m, p)
        w_mat = W[:, r].reshape(p, n)
        # M_r[i,k] = sum_j u_mat[i,j] * v_mat[j,k]
        M_r = u_mat @ v_mat
        # C_decomp[i,k] += w_mat[k,i] * M_r[i,k]
        C_decomp += w_mat.T * M_r
    
    if not np.allclose(C_decomp, C_expected, atol=tol):
        return False
    
    # Try a few more random seeds
    for seed in [123, 456]:
        np.random.seed(seed)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            u_mat = U[:, r].reshape(n, m)
            v_mat = V[:, r].reshape(m, p)
            w_mat = W[:, r].reshape(p, n)
            M_r = u_mat @ v_mat
            C_decomp += w_mat.T * M_r
        if not np.allclose(C_decomp, C_expected, atol=tol):
            return False
    
    return True


def _strassen_2x2(A, B):
    """Multiply two 2x2 matrices using Strassen's algorithm (7 multiplications)."""
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]
    e, f = B[0, 0], B[0, 1]
    g, h = B[1, 0], B[1, 1]
    
    M1 = (a + d) * (e + h)
    M2 = (c + d) * e
    M3 = a * (f - h)
    M4 = d * (g - e)
    M5 = (a + b) * h
    M6 = (c - a) * (e + f)
    M7 = (b - d) * (g + h)
    
    C = np.zeros((2, 2))
    C[0, 0] = M1 + M4 - M5 + M7
    C[0, 1] = M3 + M5
    C[1, 0] = M2 + M4
    C[1, 1] = M1 - M2 + M3 + M6
    return C


def _build_strassen_decomposition(n, m, p):
    """
    Try to build a Strassen-based decomposition.
    For 4x4 * 4x8:
    View as 2x2 blocks of 2x2 matrices.
    A = [[A11, A12], [A21, A22]]
    B = [[B11, B12, B13, B14], [B21, B22, B23, B24]]
    C = [[C11, C12, C13, C14], [C21, C22, C23, C24]]
    
    C1j = A11*B1j + A12*B2j
    C2j = A21*B1j + A22*B2j
    
    For each column block j of B, we have a (2,2,2) problem at the block level:
    C1j = A11*B1j + A12*B2j  (but these are 2x2 matrix products)
    C2j = A21*B1j + A22*B2j
    
    Actually, let's think of it differently.
    
    The full problem is 4x4 * 4x8.
    Using Strassen at the top level with 2x2 blocks:
    We need to compute 8 block products C1j and C2j (j=1..4).
    
    For C1j = A11*B1j + A12*B2j: this is a (2,4,2) sub-problem (2x2 * 4x2 = 2x2)
    For C2j = A21*B1j + A22*B2j: same.
    
    Total: 8 sub-problems of type (2,4,2).
    
    For (2,4,2): C (2x2) = M (2x4) * N (4x2)
    We can split M = [M1, M2] (each 2x2) and N = [N1; N2] (each 2x2)
    C = M1*N1 + M2*N2
    
    This is equivalent to computing two 2x2*2x2 products and adding.
    Using Strassen for each: 7 + 7 = 14 multiplications, plus 2 additions.
    But the additions are free in tensor rank (they're just additions, not multiplications).
    
    Wait, in tensor decomposition, each "multiplication" in Strassen corresponds to one rank-1 term.
    The additions are handled by the structure of U, V, W entries.
    
    So for one (2,4,2) sub-problem: rank = 14 (two Strassen multiplications of 2x2 matrices).
    Total rank = 8 * 14 = 112.
    
    Hmm, that's worse than 128? No, 112 < 128. So this is an improvement!
    
    Let me implement this carefully.
    """
    # This is getting complex. Let me think about the exact tensor decomposition.
    
    # Actually, let me try a different approach. Let's just do the schoolbook for now
    # and try to find a better decomposition through search.
    pass


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = n * m * p  # 128
    
    # Try to improve using Strassen-like decomposition
    # For 4x4 * 4x8, we can use a hierarchical Strassen approach.
    
    # Let's try to build a decomposition with rank 112 using Strassen on (2,4,2) sub-problems.
    
    # The key insight: C[i,k] = sum_j A[i,j]*B[j,k]
    # Split i into (i1, i0) where i = 2*i1 + i0, i1 in {0,1}, i0 in {0,1}
    # Split k into (k1, k0) where k = 2*k1 + k0, k1 in {0,1,2,3}, k0 in {0,1}
    # Split j into (j1, j0) where j = 2*j1 + j0, j1 in {0,1}, j0 in {0,1}
    
    # C[2*i1+i0, 2*k1+k0] = sum_{j1=0}^{1} sum_{j0=0}^{1} A[2*i1+i0, 2*j1+j0] * B[2*j1+j0, 2*k1+k0]
    
    # For fixed (i1, k1):
    # C_block[i0, k0] = sum_{j1=0}^{1} sum_{j0=0}^{1} A_block[i1, j1, i0, j0] * B_block[j1, k1, j0, k0]
    
    # Let's define for each (i1, k1):
    # M[i0, j1, j0] = A[2*i1+i0, 2*j1+j0]  (2x2x2 tensor)
    # N[j1, j0, k0] = B[2*j1+j0, 2*k1+k0]  (2x2x2 tensor)
    # C_block[i0, k0] = sum_{j1,j0} M[i0, j1, j0] * N[j1, j0, k0]
    
    # This is a (2, 4, 2) matrix multiplication: M (2x4) * N (4x2) = C (2x2)
    # where M[i0, j1*2+j0] = A[2*i1+i0, 2*j1+j0]
    #       N[j1*2+j0, k0] = B[2*j1+j0, 2*k1+k0]
    
    # For the (2,4,2) sub-problem, use Strassen:
    # M = [M1, M2] where M1 = M[:, 0:2], M2 = M[:, 2:4] (each 2x2)
    # N = [N1; N2] where N1 = N[0:2, :], N2 = N[2:4, :] (each 2x2)
    # C = M1*N1 + M2*N2
    
    # Strassen for M1*N1: 7 rank-1 terms
    # Strassen for M2*N2: 7 rank-1 terms
    # Total for this sub-problem: 14 rank-1 terms
    
    # There are 2 (i1) * 4 (k1) = 8 sub-problems
    # Total rank: 8 * 14 = 112
    
    rank = 112
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i1 in range(2):
        for k1 in range(4):
            # M[i0, j1*2+j0] = A[2*i1+i0, 2*j1+j0]
            # N[j1*2+j0, k0] = B[2*j1+j0, 2*k1+k0]
            # C[i0, k0] = sum_{j1,j0} M[i0, j1*2+j0] * N[j1*2+j0, k0]
            
            # M1 = M[:, 0:2] = A[2*i1+i0, 0:2] for i0 in {0,1}
            # M2 = M[:, 2:4] = A[2*i1+i0, 2:4] for i0 in {0,1}
            # N1 = N[0:2, :] = B[0:2, 2*k1+k0] for k0 in {0,1}
            # N2 = N[2:4, :] = B[2:4, 2*k1+k0] for k0 in {0,1}
            
            # Strassen for M1*N1:
            # M1 = [[m11, m12], [m21, m22]] where m11 = A[2*i1, 0], m12 = A[2*i1, 1], m21 = A[2*i1+1, 0], m22 = A[2*i1+1, 1]
            # N1 = [[n11, n12], [n21, n22]] where n11 = B[0, 2*k1], n12 = B[0, 2*k1+1], n21 = B[1, 2*k1], n22 = B[1, 2*k1+1]
            
            m11_idx = (2*i1) * m + 0
            m12_idx = (2*i1) * m + 1
            m21_idx = (2*i1+1) * m + 0
            m22_idx = (2*i1+1) * m + 1
            
            n11_idx = 0 * p + (2*k1)
            n12_idx = 0 * p + (2*k1+1)
            n21_idx = 1 * p + (2*k1)
            n22_idx = 1 * p + (2*k1+1)
            
            # Strassen products for M1*N1:
            # P1 = (m11+m22)*(n11+n22)
            # P2 = (m21+m22)*n11
            # P3 = m11*(n12-n22)
            # P4 = m22*(n21-n11)
            # P5 = (m11+m12)*n22
            # P6 = (m21-m11)*(n11+n12)
            # P7 = (m12-m22)*(n21+n22)
            
            # Each product corresponds to one rank-1 term.
            # For P1: U has +1 at m11_idx and +1 at m22_idx, V has +1 at n11_idx and +1 at n22_idx
            # W has +1 at the appropriate C indices.
            
            # C = M1*N1 + M2*N2
            # C11 = P1 + P4 - P5 + P7 (from M1*N1) + ... (from M2*N2)
            # This is getting complex. Let me just set up the rank-1 terms directly.
            
            # For Strassen M1*N1:
            strassen_terms_M1N1 = [
                # (u_indices, v_indices, w_coefficients)
                # P1 = (m11+m22)*(n11+n22)
                ([m11_idx, m22_idx], [n11_idx, n22_idx], [(0,0,1), (0,1,1), (1,0,1), (1,1,-1)]),
                # P2 = (m21+m22)*n11
                ([m21_idx, m22_idx], [n11_idx], [(0,0,0), (0,1,0), (1,0,-1), (1,1,0)]),
                # P3 = m11*(n12-n22)
                ([m11_idx], [n12_idx, n22_idx], [(0,0,0), (0,1,1), (1,0,0), (1,1,0)]),
                # P4 = m22*(n21-n11)
                ([m22_idx], [n21_idx, n11_idx], [(0,0,0), (0,1,0), (1,0,1), (1,1,-1)]),
                # P5 = (m11+m12)*n22
                ([m11_idx, m12_idx], [n22_idx], [(0,0,-1), (0,1,0), (1,0,0), (1,1,0)]),
                # P6 = (m21-m11)*(n11+n12)
                ([m21_idx, m11_idx], [n11_idx, n12_idx], [(0,0,0), (0,1,0), (1,0,0), (1,1,1)]),
                # P7 = (m12-m22)*(n21+n22)
                ([m12_idx, m22_idx], [n21_idx, n22_idx], [(0,0,0), (0,1,0), (1,0,0), (1,1,1)]),
            ]
            
            # Strassen for M2*N2:
            # M2 = [[M2_11, M2_12], [M2_21, M2_22]] where M2_11 = A[2*i1, 2], M2_12 = A[2*i1, 3], M2_21 = A[2*i1+1, 2], M2_22 = A[2*i1+1, 3]
            # N2 = [[N2_11, N2_12], [N2_21, N2_22]] where N2_11 = B[2, 2*k1], N2_12 = B[2, 2*k1+1], N2_21 = B[3, 2*k1], N2_22 = B[3, 2*k1+1]
            
            M2_11_idx = (2*i1) * m + 2
            M2_12_idx = (2*i1) * m + 3
            M2_21_idx = (2*i1+1) * m + 2
            M2_22_idx = (2*i1+1) * m + 3
            
            N2_11_idx = 2 * p + (2*k1)
            N2_12_idx = 2 * p + (2*k1+1)
            N2_21_idx = 3 * p + (2*k1)
            N2_22_idx = 3 * p + (2*k1+1)
            
            strassen_terms_M2N2 = [
                # Q1 = (M2_11+M2_22)*(N2_11+N2_22)
                ([M2_11_idx, M2_22_idx], [N2_11_idx, N2_22_idx], [(0,0,1), (0,1,1), (1,0,1), (1,1,-1)]),
                # Q2 = (M2_21+M2_22)*N2_11
                ([M2_21_idx, M2_22_idx], [N2_11_idx], [(0,0,0), (0,1,0), (1,0,-1), (1,1,0)]),
                # Q3 = M2_11*(N2_12-N2_22)
                ([M2_11_idx], [N2_12_idx, N2_22_idx], [(0,0,0), (0,1,1), (1,0,0), (1,1,0)]),
                # Q4 = M2_22*(N2_21-N2_11)
                ([M2_22_idx], [N2_21_idx, N2_11_idx], [(0,0,0), (0,1,0), (1,0,1), (1,1,-1)]),
                # Q5 = (M2_11+M2_12)*N2_22
                ([M2_11_idx, M2_12_idx], [N2_22_idx], [(0,0,-1), (0,1,0), (1,0,0), (1,1,0)]),
                # Q6 = (M2_21-M2_11)*(N2_11+N2_12)
                ([M2_21_idx, M2_11_idx], [N2_11_idx, N2_12_idx], [(0,0,0), (0,1,0), (1,0,0), (1,1,1)]),
                # Q7 = (M2_12-M2_22)*(N2_21+N2_22)
                ([M2_12_idx, M2_22_idx], [N2_21_idx, N2_22_idx], [(0,0,0), (0,1,0), (1,0,0), (1,1,1)]),
            ]
            
            all_terms = strassen_terms_M1N1 + strassen_terms_M2N2
            
            for u_indices, v_indices, w_coeffs in all_terms:
                # u_indices: list of indices into U (n*m dimension)
                # v_indices: list of indices into V (m*p dimension)
                # w_coeffs: list of (i0, k0, coeff) for W (p*n dimension)
                
                for ui in u_indices:
                    U[ui, col] = 1.0
                for vi in v_indices:
                    V[vi, col] = 1.0
                for i0, k0, coeff in w_coeffs:
                    # W index: k*n + i where k = 2*k1+k0, i = 2*i1+i0
                    k_global = 2*k1 + k0
                    i_global = 2*i1 + i0
                    w_idx = k_global * n + i_global
                    W[w_idx, col] += coeff
                
                col += 1
    
    # Verify
    if _verify_fast(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
