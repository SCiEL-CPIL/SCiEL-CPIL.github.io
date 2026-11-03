# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = 140  # seconds, safety margin before 150s kill


def _build_strassen_2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    n=2, m=2, p=2.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    # Strassen products for [[a,b],[c,d]] * [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # U indices: 0=a(0,0), 1=b(0,1), 2=c(1,0), 3=d(1,1)
    # V indices: 0=e(0,0), 1=f(0,1), 2=g(1,0), 3=h(1,1)
    # W indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) -> k*n+i
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C[0,0] += M1, C[1,1] += M1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1 # C[1,0] += M2, C[1,1] -= M2
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1 # C[0,1] += M3, C[1,1] += M3
    
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1 # C[0,0] += M4, C[1,0] += M4
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1 # C[0,0] -= M5, C[0,1] += M5
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C[1,1] += M6
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C[0,0] += M7
    
    return U, V, W


def _build_recursive_strassen(n):
    """
    Build Strassen decomposition for n x n x n where n is a power of 2.
    Returns U, V, W.
    """
    if n == 2:
        return _build_strassen_2x2()
    
    h = n // 2
    U_sub, V_sub, W_sub = _build_recursive_strassen(h)
    rank_sub = U_sub.shape[1]
    rank = 7 * rank_sub
    
    U = np.zeros((n*n, rank), dtype=np.float64)
    V = np.zeros((n*n, rank), dtype=np.float64)
    W = np.zeros((n*n, rank), dtype=np.float64)
    
    # Strassen coefficients for blocks
    a_coeffs = [
        {(0,0): 1, (1,1): 1}, # M1
        {(1,0): 1, (1,1): 1}, # M2
        {(0,0): 1},           # M3
        {(1,1): 1},           # M4
        {(0,0): 1, (0,1): 1}, # M5
        {(1,0): 1, (0,0): -1},# M6
        {(0,1): 1, (1,1): -1} # M7
    ]
    b_coeffs = [
        {(0,0): 1, (1,1): 1}, # M1
        {(0,0): 1},           # M2
        {(1,1): 1, (0,0): -1},# M3
        {(1,0): 1, (0,0): -1},# M4
        {(1,1): 1},           # M5
        {(0,0): 1, (0,1): 1}, # M6
        {(1,0): 1, (1,1): 1}  # M7
    ]
    w_coeffs = [
        {(0,0): 1, (1,1): 1}, # M1
        {(1,0): 1, (1,1): -1},# M2
        {(0,1): 1, (1,1): 1}, # M3
        {(0,0): 1, (1,0): 1}, # M4
        {(0,0): -1, (0,1): 1},# M5
        {(1,1): 1},           # M6
        {(0,0): 1}            # M7
    ]
    
    col = 0
    for r in range(7):
        for s in range(rank_sub):
            # U
            for u in range(2):
                for v in range(2):
                    coeff_a = a_coeffs[r].get((u,v), 0)
                    if coeff_a == 0: continue
                    for i_sub in range(h):
                        for j_sub in range(h):
                            u_val = U_sub[i_sub*h + j_sub, s]
                            if u_val == 0: continue
                            i_global = u*h + i_sub
                            j_global = v*h + j_sub
                            U[i_global*n + j_global, col] += coeff_a * u_val
                            
            # V
            for u in range(2):
                for v in range(2):
                    coeff_b = b_coeffs[r].get((u,v), 0)
                    if coeff_b == 0: continue
                    for i_sub in range(h):
                        for j_sub in range(h):
                            v_val = V_sub[i_sub*h + j_sub, s]
                            if v_val == 0: continue
                            i_global = u*h + i_sub
                            j_global = v*h + j_sub
                            V[i_global*n + j_global, col] += coeff_b * v_val
                            
            # W
            for u in range(2):
                for v in range(2):
                    coeff_w = w_coeffs[r].get((u,v), 0)
                    if coeff_w == 0: continue
                    for k_sub in range(h):
                        for i_sub in range(h):
                            w_val = W_sub[k_sub*h + i_sub, s]
                            if w_val == 0: continue
                            k_global = u*h + k_sub
                            i_global = v*h + i_sub
                            W[k_global*n + i_global, col] += coeff_w * w_val
            
            col += 1
            
    return U, V, W


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using matrix operations."""
    rank = U.shape[1]
    np.random.seed(42)
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
    
    for seed in [123, 456, 789]:
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


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Use recursive Strassen for 4x4x4 sub-problems
    # Split k into two blocks of size 4
    U1, V1, W1 = _build_recursive_strassen(4)
    U2, V2, W2 = _build_recursive_strassen(4)
    
    # Combine
    rank = U1.shape[1] + U2.shape[1]
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    # First part: k in 0..3
    U[:, 0:49] = U1
    V[:, 0:49] = V1
    W[0:16, 0:49] = W1 # k in 0..3, i in 0..3 -> k*n+i in 0..15
    
    # Second part: k in 4..7
    # V2 needs to be mapped to correct V indices
    # V1 indices are j*p + k. Here p=4 for the sub-problem.
    # But global p=8.
    # V1[j*4 + k, col] corresponds to global V[j*8 + k, col] for k in 0..3.
    # V2[j*4 + k, col] corresponds to global V[j*8 + (4+k), col] for k in 0..3.
    
    for j in range(4):
        for k in range(4):
            V[j*8 + k, 49:98] = V1[j*4 + k, 0:49]
            V[j*8 + 4 + k, 49:98] = V2[j*4 + k, 0:49]
            
    # W2 indices: k in 0..3 for sub-problem. Global k in 4..7.
    # W1[k*4 + i, col] -> global W[k*4 + i, col]
    # W2[k*4 + i, col] -> global W[(4+k)*4 + i, col]
    
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 49:98] = W1[k*4 + i, 0:49]
            W[(4+k)*4 + i, 49:98] = W2[k*4 + i, 0:49]
            
    # U is same for both
    U[:, 49:98] = U2
    
    if _verify_fast(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    U_fb = np.zeros((n*m, n*m*p), dtype=np.float64)
    V_fb = np.zeros((m*p, n*m*p), dtype=np.float64)
    W_fb = np.zeros((p*n, n*m*p), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
