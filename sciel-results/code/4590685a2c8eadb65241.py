import numpy as np
import time
from scipy import linalg

DIMENSIONS = [2, 2, 11]

def _build_rank39_decomposition():
    """Build the rank-39 decomposition: 5 Strassen blocks + 1 schoolbook block."""
    n, m, p = 2, 2, 11
    total_rank = 5 * 7 + 1 * 4
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # V indices
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = 11 + k0  # B[1, k0]
        v_h = 11 + k1  # B[1, k1]
        
        # W indices
        w_c00 = 2 * k0     # C[0, k0]
        w_c10 = 2 * k0 + 1 # C[1, k0]
        w_c01 = 2 * k1     # C[0, k1]
        w_c11 = 2 * k1 + 1 # C[1, k1]
        
        # m1 = (a+d)(e+h)
        U[0, col] = 1; U[3, col] = 1
        V[v_e, col] = 1; V[v_h, col] = 1
        W[w_c00, col] = 1; W[w_c11, col] = 1
        col += 1
        
        # m2 = (c+d)e
        U[2, col] = 1; U[3, col] = 1
        V[v_e, col] = 1
        W[w_c10, col] = 1; W[w_c11, col] = -1
        col += 1
        
        # m3 = a(f-h)
        U[0, col] = 1
        V[v_f, col] = 1; V[v_h, col] = -1
        W[w_c01, col] = 1; W[w_c11, col] = 1
        col += 1
        
        # m4 = d(g-e)
        U[3, col] = 1
        V[v_g, col] = 1; V[v_e, col] = -1
        W[w_c00, col] = 1; W[w_c10, col] = 1
        col += 1
        
        # m5 = (a+b)h
        U[0, col] = 1; U[1, col] = 1
        V[v_h, col] = 1
        W[w_c00, col] = -1; W[w_c01, col] = 1
        col += 1
        
        # m6 = (c-a)(e+f)
        U[2, col] = 1; U[0, col] = -1
        V[v_e, col] = 1; V[v_f, col] = 1
        W[w_c11, col] = 1
        col += 1
        
        # m7 = (b-d)(g+h)
        U[1, col] = 1; U[3, col] = -1
        V[v_g, col] = 1; V[v_h, col] = 1
        W[w_c00, col] = 1
        col += 1
    
    # Single column k2 = 10
    k2 = 10
    v_e2 = k2
    v_g2 = 11 + k2
    w_c02 = 2 * k2
    w_c12 = 2 * k2 + 1
    
    # ae
    U[0, col] = 1
    V[v_e2, col] = 1
    W[w_c02, col] = 1
    col += 1
    
    # bg
    U[1, col] = 1
    V[v_g2, col] = 1
    W[w_c02, col] = 1
    col += 1
    
    # ce
    U[2, col] = 1
    V[v_e2, col] = 1
    W[w_c12, col] = 1
    col += 1
    
    # dg
    U[3, col] = 1
    V[v_g2, col] = 1
    W[w_c12, col] = 1
    col += 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    rank = U.shape[1]
    np.random.seed(42)
    for _ in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        
        C_dec = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for j in range(m):
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                C_dec[i, k] = val
        
        if not np.allclose(C, C_dec, atol=tol):
            return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, deadline, tol=1e-10):
    """Try to reduce the rank using SVD-based numerical rank detection and greedy elimination."""
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = current_U.shape[1]
    
    # Try SVD-based rank reduction
    # The decomposition is valid if sum_r U[:,r] * V[:,r] * W[:,r] = T
    # We can check if the columns are linearly dependent in some sense
    
    # First, check the numerical rank of each factor matrix
    for mat_name, mat in [('U', current_U), ('V', current_V), ('W', current_W)]:
        try:
            s = np.linalg.svd(mat, compute_uv=False)
            # Find effective rank
            effective_rank = np.sum(s > 1e-10 * s[0])
            if effective_rank < current_rank:
                # The matrix has numerical rank less than current_rank
                # This suggests some columns might be redundant
                pass  # We'll handle this with greedy elimination
        except:
            pass
    
    # Greedy elimination: try removing each column and check if we can still represent the tensor
    # This is expensive, so we'll do it smartly
    
    # Build the tensor from the current decomposition
    def compute_tensor(U_mat, V_mat, W_mat):
        rank = U_mat.shape[1]
        T = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U_mat[:,r], V_mat[:,r], W_mat[:,r])
        return T
    
    # Get the target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1
    
    # Try greedy elimination from the end
    best_U, best_V, best_W = current_U, current_V, current_W
    best_rank = current_rank
    
    # Try removing columns one by one from the end
    for r in range(current_rank - 1, -1, -1):
        if time.time() > deadline - 30:
            break
        
        # Try removing column r
        U_test = np.delete(current_U, r, axis=1)
        V_test = np.delete(current_V, r, axis=1)
        W_test = np.delete(current_W, r, axis=1)
        
        # Check if the remaining decomposition still matches
        T_test = compute_tensor(U_test, V_test, W_test)
        
        if np.allclose(T_test, T_target, atol=tol):
            current_U, current_V, current_W = U_test, V_test, W_test
            current_rank -= 1
            best_U, best_V, best_W = current_U, current_V, current_W
            best_rank = current_rank
            # Continue trying to remove more
        else:
            # Try to see if we can adjust the remaining columns to compensate
            # This is more complex; for now, just skip
            pass
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Build the rank-39 decomposition
    U, V, W = _build_rank39_decomposition()
    
    # Verify it's correct
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U_fb = np.zeros((n*m, rank))
        V_fb = np.zeros((m*p, rank))
        W_fb = np.zeros((p*n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, column] = 1
                    V_fb[j*p+k, column] = 1
                    W_fb[k*n+i, column] = 1
                    column += 1
        U, V, W = U_fb, V_fb, W_fb
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce the rank
    if time.time() < deadline - 60:
        U_red, V_red, W_red, red_rank = _try_reduce_rank(U, V, W, n, m, p, deadline)
        if red_rank < U.shape[1] and _verify(U_red, V_red, W_red, n, m, p):
            U, V, W = U_red, V_red, W_red
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
