import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [2, 4, 4]

def generate_strassen_28(n, m, p):
    rank = 28
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    def apply_strassen_block(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0, base_rank):
        a_idx = [
            a_r0 * m + a_c0, a_r0 * m + a_c0 + 1,
            (a_r0 + 1) * m + a_c0, (a_r0 + 1) * m + a_c0 + 1
        ]
        b_idx = [
            b_r0 * p + b_c0, b_r0 * p + b_c0 + 1,
            (b_r0 + 1) * p + b_c0, (b_r0 + 1) * p + b_c0 + 1
        ]
        c_idx = [
            c_c0 * n + c_r0, (c_c0 + 1) * n + c_r0,
            c_c0 * n + (c_r0 + 1), (c_c0 + 1) * n + (c_r0 + 1)
        ]
        
        r = base_rank
        # M1
        U[a_idx[0]][r] += 1; U[a_idx[3]][r] += 1
        V[b_idx[0]][r] += 1; V[b_idx[3]][r] += 1
        W[c_idx[0]][r] += 1; W[c_idx[3]][r] += 1
        r += 1
        # M2
        U[a_idx[2]][r] += 1; U[a_idx[3]][r] += 1
        V[b_idx[0]][r] += 1
        W[c_idx[2]][r] += 1; W[c_idx[3]][r] += -1
        r += 1
        # M3
        U[a_idx[0]][r] += 1
        V[b_idx[1]][r] += 1; V[b_idx[3]][r] += -1
        W[c_idx[1]][r] += 1; W[c_idx[3]][r] += 1
        r += 1
        # M4
        U[a_idx[3]][r] += 1
        V[b_idx[2]][r] += 1; V[b_idx[0]][r] += -1
        W[c_idx[0]][r] += 1; W[c_idx[2]][r] += 1
        r += 1
        # M5
        U[a_idx[0]][r] += 1; U[a_idx[1]][r] += 1
        V[b_idx[3]][r] += 1
        W[c_idx[0]][r] += -1; W[c_idx[1]][r] += 1
        r += 1
        # M6
        U[a_idx[2]][r] += 1; U[a_idx[0]][r] += -1
        V[b_idx[0]][r] += 1; V[b_idx[1]][r] += 1
        W[c_idx[3]][r] += 1
        r += 1
        # M7
        U[a_idx[1]][r] += 1; U[a_idx[3]][r] += -1
        V[b_idx[2]][r] += 1; V[b_idx[3]][r] += 1
        W[c_idx[0]][r] += 1
        r += 1
        return r

    r = apply_strassen_block(0, 0, 0, 0, 0, 0, 0)
    r = apply_strassen_block(0, 2, 2, 0, 0, 0, r)
    r = apply_strassen_block(0, 0, 0, 2, 0, 2, r)
    r = apply_strassen_block(0, 2, 2, 2, 0, 2, r)
    
    assert r == 28
    return U.tolist(), V.tolist(), W.tolist()

def verify_decomposition(U, V, W, n, m, p):
    """
    Verifies if U, V, W (lists of lists of floats/ints) form a valid decomposition.
    Assumes entries are close to half-integers.
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check if entries are close to half-integers
    # Multiply by 2 and check distance to nearest integer
    if np.max(np.abs(U_arr * 2 - np.round(U_arr * 2))) > 1e-6: return False
    if np.max(np.abs(V_arr * 2 - np.round(V_arr * 2))) > 1e-6: return False
    if np.max(np.abs(W_arr * 2 - np.round(W_arr * 2))) > 1e-6: return False
    
    # Convert to integers by multiplying by 2
    U_int = np.round(U_arr * 2).astype(np.int64)
    V_int = np.round(V_arr * 2).astype(np.int64)
    W_int = np.round(W_arr * 2).astype(np.int64)
    
    # Compute tensor T_recon * 8
    # T_ijk = sum_r U_ir V_jr W_kr
    # T_ijk * 8 = sum_r U'_ir V'_jr W'_kr
    T_recon = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int, optimize=True)
    
    # Target is all 8s
    target = np.full((n, m, p), 8, dtype=np.int64)
    
    return np.all(T_recon == target)

def residuals(x, n, m, p, rank):
    U = x[:n*m*rank].reshape(n*m, rank)
    V = x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
    W = x[n*m*rank + m*p*rank : ].reshape(p*n, rank)
    
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
    return T_recon.flatten() - 1.0

def try_find_low_rank(n, m, p, target_rank, deadline):
    size_U = n * m * target_rank
    size_V = m * p * target_rank
    size_W = p * n * target_rank
    total_size = size_U + size_V + size_W
    
    np.random.seed(42) # Reproducibility for debugging, but maybe random is better? Let's use random.
    
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        # Scale initialization to avoid large numbers
        scale = 1.0
        x0 = np.random.randn(total_size) * scale
        
        try:
            # Use least_squares
            # method='lm' is fast for underdetermined systems
            result = opt.least_squares(residuals, x0, args=(n, m, p, target_rank), method='lm', max_nfev=1000)
            
            if np.linalg.norm(result.fun) < 1e-6:
                # Extract U, V, W
                U = result.x[:size_U].reshape(n*m, target_rank)
                V = result.x[size_U : size_U + size_V].reshape(m*p, target_rank)
                W = result.x[size_U + size_V : ].reshape(p*n, target_rank)
                
                # Try to snap to half-integers
                U_snapped = np.round(U * 2) / 2.0
                V_snapped = np.round(V * 2) / 2.0
                W_snapped = np.round(W * 2) / 2.0
                
                if verify_decomposition(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), n, m, p):
                    return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
                    
        except Exception:
            pass
            
    return None, None, None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    
    # Generate fallback Rank 28
    U28, V28, W28 = generate_strassen_28(n, m, p)
    
    # Try to find Rank 24
    U24, V24, W24 = try_find_low_rank(n, m, p, 24, deadline)
    
    if U24 is not None:
        return {"U": U24, "V": V24, "W": W24}
        
    return {"U": U28, "V": V28, "W": W28}
