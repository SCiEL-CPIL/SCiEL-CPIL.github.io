import time
import sys
import numpy as np
import scipy.optimize as opt

START_TIME = time.time()
TIME_LIMIT = 140  # Safety margin before 150s kill

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def verify_decomposition(U, V, W, n=2, m=4, p=4):
    """Verify that the decomposition is correct using einsum for speed."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T_recon[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    T_target = np.ones((n, m, p), dtype=float)
    max_err = np.max(np.abs(T_recon - T_target))
    return max_err < 1e-9, max_err

def strassen_decomposition():
    """Return the Strassen-based decomposition (rank 28)."""
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    alpha = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (0,0): -1 },
        { (0,1): 1, (1,1): -1 }
    ]
    
    beta = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (0,1): 1, (1,1): -1 },
        { (1,0): 1, (0,0): -1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (1,1): 1 }
    ]
    
    w_coeff = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): -1 },
        { (0,1): 1, (1,1): 1 },
        { (0,0): 1, (1,0): 1 },
        { (0,0): -1, (0,1): 1 },
        { (1,1): 1 },
        { (0,0): 1 }
    ]
    
    for s in range(1, 8):
        for t in range(4):
            c = 4 * (s - 1) + t
            k_a = 0 if t in [0, 2] else 1
            r_b = 1 if t in [1, 3] else 0
            c_b = 1 if t in [2, 3] else 0
            k_c = 0 if t in [0, 1] else 1
            
            for (i, j), val in alpha[s].items():
                if val != 0:
                    u_idx = i * m + 2 * j + k_a
                    U[u_idx][c] += val
            
            for (i, j), val in beta[s].items():
                if val != 0:
                    v_row = 2 * i + r_b
                    v_col = 2 * j + c_b
                    v_idx = v_row * p + v_col
                    V[v_idx][c] += val
            
            for (i, j), val in w_coeff[s].items():
                if val != 0:
                    w_col = 2 * j + k_c
                    w_row = i
                    w_idx = w_col * n + w_row
                    W[w_idx][c] += val
    
    return U, V, W

def residuals_fast(x, n=2, m=4, p=4, rank=24):
    """Calculate residuals using einsum for speed."""
    size_U = n * m * rank
    U = x[:size_U].reshape(n*m, rank)
    V = x[size_U : size_U + m*p*rank].reshape(m*p, rank)
    W = x[size_U + m*p*rank : ].reshape(p*n, rank)
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return T_recon.flatten() - 1.0

def try_round_and_verify(U, V, W, n, m, p, rank):
    """Try rounding to half-integers and verify."""
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    valid, err = verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p)
    if valid:
        return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    return None

def find_low_rank_decomposition(target_rank, n=2, m=4, p=4, max_restarts=20):
    """Use scipy.optimize.least_squares to find a decomposition of target_rank."""
    size_U = n * m * target_rank
    size_V = m * p * target_rank
    size_W = p * n * target_rank
    total_size = size_U + size_V + size_W
    
    best_result = None
    best_error = float('inf')
    
    # Use a set of possible values for sparse initialization
    sparse_vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    for restart in range(max_restarts):
        if time_up():
            break
        
        # Random initialization with sparse structure
        x0 = np.zeros(total_size)
        
        # Randomly select a subset of entries to be non-zero
        num_nonzero = int(total_size * 0.3)  # 30% non-zero
        indices = np.random.choice(total_size, num_nonzero, replace=False)
        
        for idx in indices:
            x0[idx] = np.random.choice(sparse_vals)
        
        # Also try a dense random initialization occasionally
        if restart % 4 == 0:
            x0 = np.random.randn(total_size) * 0.5
        
        try:
            result = opt.least_squares(residuals_fast, x0, args=(n, m, p, target_rank), 
                                       method='trf', max_nfev=3000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            err = np.linalg.norm(result.fun)
            
            if err < best_error:
                best_error = err
                best_result = result.x
                
            if err < 1e-7:
                U_rec = best_result[:size_U].reshape(n*m, target_rank)
                V_rec = best_result[size_U : size_U + size_V].reshape(m*p, target_rank)
                W_rec = best_result[size_U + size_V : ].reshape(p*n, target_rank)
                
                # Try rounding
                rounded = try_round_and_verify(U_rec, V_rec, W_rec, n, m, p, target_rank)
                if rounded is not None:
                    return rounded
                    
                # If close but not rounded, try a polish pass
                if err < 1e-6:
                    # Do a high-precision polish
                    result2 = opt.least_squares(residuals_fast, best_result, args=(n, m, p, target_rank),
                                                method='trf', max_nfev=5000, ftol=1e-14, xtol=1e-14, gtol=1e-14)
                    err2 = np.linalg.norm(result2.fun)
                    if err2 < 1e-8:
                        U_rec2 = result2.x[:size_U].reshape(n*m, target_rank)
                        V_rec2 = result2.x[size_U : size_U + size_V].reshape(m*p, target_rank)
                        W_rec2 = result2.x[size_U + size_V : ].reshape(p*n, target_rank)
                        rounded2 = try_round_and_verify(U_rec2, V_rec2, W_rec2, n, m, p, target_rank)
                        if rounded2 is not None:
                            return rounded2
                    
        except Exception as e:
            pass
            
    return None

def construct():
    n, m, p = 2, 4, 4
    best_result = None
    best_rank = float('inf')
    
    # Start with Strassen
    U, V, W = strassen_decomposition()
    valid, err = verify_decomposition(U, V, W, n, m, p)
    if valid:
        best_rank = 28
        best_result = {"U": U, "V": V, "W": W}
    
    # Try to find lower rank decompositions
    # Try 24 first (most likely to be minimal), then 25, 26, 27
    for target_rank in [24, 25, 26, 27]:
        if time_up():
            break
        if best_rank <= target_rank:
            continue
        
        res = find_low_rank_decomposition(target_rank, n, m, p)
        if res is not None:
            U2, V2, W2 = res
            valid, err = verify_decomposition(U2, V2, W2, n, m, p)
            if valid:
                best_rank = target_rank
                best_result = {"U": U2, "V": V2, "W": W2}
                if target_rank == 24:
                    break  # Found the likely minimum
    
    if best_result is None:
        # Fallback: naive decomposition
        rank = 32
        U = [[0.0] * rank for _ in range(n*m)]
        V = [[0.0] * rank for _ in range(m*p)]
        W = [[0.0] * rank for _ in range(p*n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1.0
                    V[j*p+k][idx] = 1.0
                    W[k*n+i][idx] = 1.0
                    idx += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Best rank: {len(result['U'][0])}")
    valid, err = verify_decomposition(result['U'], result['V'], result['W'])
    print(f"Valid: {valid}, Max error: {err}")
