import time
import numpy as np
import itertools

DIMENSIONS = [6, 6, 6]

# Standard Strassen steps for 2x2 matrix multiplication
# Each step: (A_combinations, B_combinations, C_combinations)
# A_combinations: list of (row, col, coeff) for the A-side
# B_combinations: list of (row, col, coeff) for the B-side
# C_combinations: list of (row, col, coeff) for the C-side
STRASSEN_STEPS = [
    # M1 = (A11+A22)(B11+B22)
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
    # M2 = (A21+A22)B11
    ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,1)]),
    # M3 = A11(B21-B11)
    ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,0,-1),(0,1,1)]),
    # M4 = A22(B12-B22)
    ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,1,-1),(1,1,-1)]),
    # M5 = (A11+A12)B22
    ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,1,1),(1,1,1)]),
    # M6 = (A21-A11)(B11+B12)
    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,0,1)]),
    # M7 = (A12-A22)(B21+B22)
    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
]

def _build_rank49_decomposition():
    """
    Build a rank-49 decomposition for 4x4 matrix multiplication using
    the Strassen tensor square. The 4x4 matrices are partitioned into
    2x2 blocks of 2x2 sub-blocks, and Strassen is applied at both levels.
    
    Returns U, V, W matrices of shape (16, 49) each.
    """
    rank = 49  # 7 * 7
    U = np.zeros((16, rank), dtype=np.float64)
    V = np.zeros((16, rank), dtype=np.float64)
    W = np.zeros((16, rank), dtype=np.float64)
    
    col_idx = 0
    for step_a in STRASSEN_STEPS:
        a_terms_a, b_terms_a, c_terms_a = step_a
        for step_b in STRASSEN_STEPS:
            a_terms_b, b_terms_b, c_terms_b = step_b
            
            # For each pair of Strassen steps, we have a rank-1 term
            # The A-side combines terms from both levels
            # The B-side combines terms from both levels
            # The C-side combines terms from both levels
            
            # A-side: for each term in a_terms_a (outer) and a_terms_b (inner)
            # The outer step determines which 2x2 block of the 4x4 A matrix
            # The inner step determines which 2x2 sub-block within that block
            
            for I_outer, J_outer, ca_outer in a_terms_a:
                for I_inner, J_inner, ca_inner in a_terms_b:
                    # Global row = I_outer * 2 + I_inner
                    # Global col = J_outer * 2 + J_inner
                    row = I_outer * 2 + I_inner
                    col = J_outer * 2 + J_inner
                    coeff = ca_outer * ca_inner
                    U[row * 4 + col, col_idx] += coeff
            
            # B-side: for each term in b_terms_a (outer) and b_terms_b (inner)
            for J_outer, K_outer, cb_outer in b_terms_a:
                for J_inner, K_inner, cb_inner in b_terms_b:
                    # Global row = J_outer * 2 + J_inner
                    # Global col = K_outer * 2 + K_inner
                    row = J_outer * 2 + J_inner
                    col = K_outer * 2 + K_inner
                    coeff = cb_outer * cb_inner
                    V[row * 4 + col, col_idx] += coeff
            
            # C-side: for each term in c_terms_a (outer) and c_terms_b (inner)
            for I_outer, K_outer, cc_outer in c_terms_a:
                for I_inner, K_inner, cc_inner in c_terms_b:
                    # Global row = K_outer * 2 + K_inner
                    # Global col = I_outer * 2 + I_inner
                    row = K_outer * 2 + K_inner
                    col = I_outer * 2 + I_inner
                    coeff = cc_outer * cc_inner
                    W[row * 4 + col, col_idx] += coeff
            
            col_idx += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, sample_points=50):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = U.shape[1]
    # Check shapes
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    # Sample random test cases
    np.random.seed(42)
    for _ in range(sample_points):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute using decomposition
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_vec = U[:, r]
            v_vec = V[:, r]
            w_vec = W[:, r]
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            w_mat = w_vec.reshape(p, n)
            # The contribution is (u_mat @ A @ v_mat) * w_mat
            # But actually: C[i,k] = sum_r U[i*m+k, r] * (A @ B) ... 
            # Let me think about this more carefully.
            # The tensor is T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # So C[i,k] = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k]
            # = sum_r (sum_{i,j} U[i*m+j, r] * A[i,j]) * (sum_{j,k} V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
            # This is more complex. Let me just directly compute.
            pass
        
        # Direct verification: for each (i,k), check C[i,k]
        # C[i,k] should equal sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k]
        # where T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for k in range(p):
                val = 0.0
                for j in range(m):
                    t_val = 0.0
                    for r in range(rank):
                        t_val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    val += A[i, j] * B[j, k] * t_val
                if abs(val - C_expected[i, k]) > 1e-8:
                    return False
    return True

def _build_direct_decomposition(n, m, p):
    """Build the direct (naive) rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1
                V[j*p + k, idx] = 1
                W[k*n + i, idx] = 1
                idx += 1
    
    return U, V, W

def _build_strassen_decomposition(n, m, p):
    """Build a Strassen-based decomposition for n=m=p=6."""
    # For 6x6, partition into 3x3 blocks of 2x2 sub-blocks
    # Use Strassen at the 3x3 block level (but Strassen is for 2x2, not 3x3)
    # Instead, use direct multiplication for now as a fallback
    return _build_direct_decomposition(n, m, p)

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    dim = n  # 6
    
    # First, try the Strassen-based approach
    U, V, W = _build_rank49_decomposition()
    
    # Verify the 4x4 Strassen decomposition
    if not _verify_decomposition(U, V, W, 4, 4, 4, sample_points=10):
        # Fall back to direct decomposition for 6x6
        U, V, W = _build_direct_decomposition(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If 4x4 Strassen works, build the full 6x6 decomposition
    total_rank = 49 + 152
    
    U_full = np.zeros((dim * dim, total_rank), dtype=np.float64)
    V_full = np.zeros((dim * dim, total_rank), dtype=np.float64)
    W_full = np.zeros((dim * dim, total_rank), dtype=np.float64)
    
    # First, add the rank-49 decomposition for A11*B11 (top-left 4x4)
    U_full[:16, :49] = U
    V_full[:16, :49] = V
    W_full[:16, :49] = W
    
    col_idx = 49
    
    # Now add direct multiplication terms for the remaining blocks
    # A12 (rows 0-3, cols 4-5) * B21 (rows 4-5, cols 0-3) -> C11 (rows 0-3, cols 0-3)
    for i in range(4):
        for j in range(2):
            for k in range(4):
                u_idx = i * dim + (4 + j)
                v_idx = (4 + j) * dim + k
                w_idx = k * dim + i
                U_full[u_idx, col_idx] = 1
                V_full[v_idx, col_idx] = 1
                W_full[w_idx, col_idx] = 1
                col_idx += 1
    
    # A11 (rows 0-3, cols 0-3) * B12 (rows 0-3, cols 4-5) -> C12 (rows 0-3, cols 4-5)
    for i in range(4):
        for j in range(4):
            for k in range(2):
                u_idx = i * dim + j
                v_idx = j * dim + (4 + k)
                w_idx = (4 + k) * dim + i
                U_full[u_idx, col_idx] = 1
                V_full[v_idx, col_idx] = 1
                W_full[w_idx, col_idx] = 1
                col_idx += 1
    
    # A12 (rows 0-3, cols 4-5) * B22 (rows 4-5, cols 4-5) -> C12 (rows 0-3, cols 4-5)
    for i in range(4):
        for j in range(2):
            for k in range(2):
                u_idx = i * dim + (4 + j)
                v_idx = (4 + j) * dim + (4 + k)
                w_idx = (4 + k) * dim + i
                U_full[u_idx, col_idx] = 1
                V_full[v_idx, col_idx] = 1
                W_full[w_idx, col_idx] = 1
                col_idx += 1
    
    # A21 (rows 4-5, cols 0-3) * B11 (rows 0-3, cols 0-3) -> C21 (rows 4-5, cols 0-3)
    for i in range(2):
        for j in range(4):
            for k in range(4):
                u_idx = (4 + i) * dim + j
                v_idx = j * dim + k
                w_idx = k * dim + (4 + i)
                U_full[u_idx, col_idx] = 1
                V_full[v_idx, col_idx] = 1
                W_full[w_idx, col_idx] = 1
                col_idx += 1
    
    # A22 (rows 4-5, cols 4-5) * B21 (rows 4-5, cols 0-3) -> C21 (rows 4-5, cols 0-3)
    for i in range(2):
        for j in range(2):
            for k in range(4):
                u_idx = (4 + i) * dim + (4 + j)
                v_idx = (4 + j) * dim + k
                w_idx = k * dim + (4 + i)
                U_full[u_idx, col_idx] = 1
                V_full[v_idx, col_idx] = 1
                W_full[w_idx, col_idx] = 1
                col_idx += 1
    
    # A21 (rows 4-5, cols 0-3) * B12 (rows 0-3, cols 4-5) -> C22 (rows 4-5, cols 4-5)
    for i in range(2):
        for j in range(4):
            for k in range(2):
                u_idx = (4 + i) * dim + j
                v_idx = j * dim + (4 + k)
                w_idx = (4 + k) * dim + (4 + i)
                U_full[u_idx, col_idx] = 1
                V_full[v_idx, col_idx] = 1
                W_full[w_idx, col_idx] = 1
                col_idx += 1
    
    # A22 (rows 4-5, cols 4-5) * B22 (rows 4-5, cols 4-5) -> C22 (rows 4-5, cols 4-5)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                u_idx = (4 + i) * dim + (4 + j)
                v_idx = (4 + j) * dim + (4 + k)
                w_idx = (4 + k) * dim + (4 + i)
                U_full[u_idx, col_idx] = 1
                V_full[v_idx, col_idx] = 1
                W_full[w_idx, col_idx] = 1
                col_idx += 1
    
    # Verify the full 6x6 decomposition
    if not _verify_decomposition(U_full, V_full, W_full, n, m, p, sample_points=20):
        # Fall back to direct decomposition
        U_direct, V_direct, W_direct = _build_direct_decomposition(n, m, p)
        return {"U": U_direct.tolist(), "V": V_direct.tolist(), "W": W_direct.tolist()}
    
    if time.time() > deadline:
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
