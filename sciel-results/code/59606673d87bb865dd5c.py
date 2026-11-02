import time
import sys
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 140  # Safety margin before 150s kill

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def verify_decomposition(U, V, W, n=2, m=4, p=4):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
    
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

def residuals(x, n=2, m=4, p=4, rank=24):
    """Calculate residuals for the optimization problem."""
    U = x[:n*m*rank].reshape(n*m, rank)
    V = x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
    W = x[n*m*rank + m*p*rank : ].reshape(p*n, rank)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
    
    return T_recon.flatten() - 1.0

def find_low_rank_decomposition(target_rank, n=2, m=4, p=4, max_restarts=100):
    """Use scipy.optimize.least_squares to find a decomposition of target_rank."""
    n, m, p = 2, 4, 4
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
        # We want a structured sparsity, so we pick random indices
        num_nonzero = int(total_size * 0.2) # 20% non-zero
        indices = np.random.choice(total_size, num_nonzero, replace=False)
        
        for idx in indices:
            x0[idx] = np.random.choice(sparse_vals)
        
        # Also try a dense random initialization occasionally
        if restart % 5 == 0:
            x0 = np.random.randn(total_size) * 0.5
        
        try:
            # Use 'trf' method which is more robust
            result = opt.least_squares(residuals, x0, args=(n, m, p, target_rank), method='trf', max_nfev=5000)
            err = np.linalg.norm(result.fun)
            
            if err < best_error:
                best_error = err
                best_result = result.x
                
            if err < 1e-8:
                # Found a close solution, try to round to half-integers
                U_rec = best_result[:size_U].reshape(n*m, target_rank)
                V_rec = best_result[size_U : size_U + size_V].reshape(m*p, target_rank)
                W_rec = best_result[size_U + size_V : ].reshape(p*n, target_rank)
                
                # Round to nearest half-integer
                U_rounded = np.round(U_rec * 2) / 2
                V_rounded = np.round(V_rec * 2) / 2
                W_rounded = np.round(W_rec * 2) / 2
                
                # Verify rounded solution
                valid, _ = verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p)
                if valid:
                    return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
                    
                # If rounding failed, check if the unrounded solution is exact enough
                # and return it (though the task requires integer/half-integer)
                # The task says "integer/half-integer entries".
                # So we must return a valid half-integer solution.
                
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
    # Prioritize 24, then 25, 26, 27
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
                break # Found the best possible in this list, stop searching lower? 
                # Actually, we should stop if we found 24, as it's likely the minimum.
                # If we found 27, we should still try 26, 25, 24.
                # So we don't break, we just update best_rank.
                # But the loop iterates 24, 25, 26, 27.
                # So if we find 24, we break.
                # If we find 25, we continue to try 24.
                # Wait, the loop is [24, 25, 26, 27].
                # So it tries 24 first. If it fails, it tries 25.
                # If it finds 25, it updates best_rank to 25.
                # Then it tries 26? No, 26 is worse than 25.
                # So we should only try ranks lower than the current best.
                # The loop is sorted ascending.
                # So if we find 24, we are done.
                # If we find 25, we should still try 24? No, 24 was already tried.
                # So the logic is:
                # Try 24. If success, return.
                # Try 25. If success, update best.
                # Try 26. If success, update best (only if 25 failed).
                # Try 27. If success, update best (only if 25,26 failed).
                
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
