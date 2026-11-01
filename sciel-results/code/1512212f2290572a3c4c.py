import numpy as np
import time

DIMENSIONS = [2, 4, 7]
_DEADLINE = time.time() + 130.0  # safety margin under 150s wall-clock


def _now_ok():
    return time.time() < _DEADLINE


def _verify_decomposition(n, m, p, U, V, W, rank, num_tests=5, seed=12345):
    """Vectorized verification of tensor decomposition."""
    U_arr = np.array(U, dtype=np.float64).T  # shape (rank, n*m)
    V_arr = np.array(V, dtype=np.float64).T  # shape (rank, m*p)
    W_arr = np.array(W, dtype=np.float64).T  # shape (rank, p*n)
    
    rng = np.random.default_rng(seed)
    for _ in range(num_tests):
        A = rng.integers(-3, 4, size=(n, m)).astype(np.float64)
        B = rng.integers(-3, 4, size=(m, p)).astype(np.float64)
        C_true = A @ B
        
        # C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        C = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += np.dot(U_arr[:, i*m+j], V_arr[:, j*p+k] * W_arr[:, k*n+i])
                C[i, k] = s
        
        if not np.allclose(C, C_true, atol=1e-6):
            return False
    return True


def construct():
    n, m, p = DIMENSIONS
    
    # Block Strassen: split inner dim m=4 into two blocks of 2,
    # split output p=7 into three blocks of 2 plus one of 1.
    # For each (j_block, k_block) pair: 2x2x2 Strassen (rank 7) or 2x2x1 schoolbook (rank 4).
    # Total: 2 j-blocks * (3 * 7 + 1 * 4) = 2 * 25 = 50.
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_strassen222(i0, j0, k0):
        """Add a 2x2x2 Strassen block for rows i0,i0+1; inner j0,j0+1; cols k0,k0+1."""
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)
        
        def P(u_list, v_list, w_list):
            nonlocal col
            for (idx, val) in u_list:
                U[idx][col] += val
            for (idx, val) in v_list:
                V[idx][col] += val
            for (idx, val) in w_list:
                W[idx][col] += val
            col += 1
        
        # Standard Strassen 7 products for 2x2x2:
        # M1 = (a+d)(e+h) -> c00+=M1, c11+=M1
        P([(a_idx,1),(d_idx,1)], [(e_idx,1),(h_idx,1)], [(c00,1),(c11,1)])
        # M2 = (c+d)*e -> c10+=M2, c11-=M2
        P([(c_idx,1),(d_idx,1)], [(e_idx,1)], [(c10,1),(c11,-1)])
        # M3 = a*(f-h) -> c01+=M3, c11+=M3
        P([(a_idx,1)], [(f_idx,1),(h_idx,-1)], [(c01,1),(c11,1)])
        # M4 = d*(g-e) -> c00+=M4, c10+=M4
        P([(d_idx,1)], [(g_idx,1),(e_idx,-1)], [(c00,1),(c10,1)])
        # M5 = (a+b)*h -> c00-=M5, c01+=M5
        P([(a_idx,1),(b_idx,1)], [(h_idx,1)], [(c00,-1),(c01,1)])
        # M6 = (c-a)*(e+f) -> c11+=M6
        P([(c_idx,1),(a_idx,-1)], [(e_idx,1),(f_idx,1)], [(c11,1)])
        # M7 = (b-d)*(g+h) -> c00+=M7
        P([(b_idx,1),(d_idx,-1)], [(g_idx,1),(h_idx,1)], [(c00,1)])
    
    def add_schoolbook_221(i0, j0, k0):
        """Add a 2x2x1 schoolbook block for rows i0,i0+1; inner j0,j0+1; col k0."""
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)
        
        def P(u_list, v_list, w_list):
            nonlocal col
            for (idx, val) in u_list:
                U[idx][col] += val
            for (idx, val) in v_list:
                V[idx][col] += val
            for (idx, val) in w_list:
                W[idx][col] += val
            col += 1
        
        P([(a_idx,1)], [(e_idx,1)], [(c00,1)])
        P([(b_idx,1)], [(g_idx,1)], [(c00,1)])
        P([(c_idx,1)], [(e_idx,1)], [(c10,1)])
        P([(d_idx,1)], [(g_idx,1)], [(c10,1)])
    
    # Build: for each inner block j_start in {0, 2}:
    #   for each output block k_start in {0, 2, 4}: Strassen 2x2x2 (7 products)
    #   for output k=6: schoolbook 2x2x1 (4 products)
    for j_start in (0, 2):
        for k_start in (0, 2, 4):
            add_strassen222(0, j_start, k_start)
        add_schoolbook_221(0, j_start, 6)
    
    # Verify the decomposition
    if not _verify_decomposition(n, m, p, U, V, W, rank):
        return _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}
