import time
import sys
import numpy as np

START_TIME = time.time()
TIME_LIMIT = 110

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def verify_decomposition(U, V, W, n=2, m=4, p=4):
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

def try_remove_term(U_arr, V_arr, W_arr, col_to_remove, n=2, m=4, p=4):
    """Check if removing a specific column still gives valid decomposition."""
    mask = [i for i in range(U_arr.shape[1]) if i != col_to_remove]
    new_U = U_arr[:, mask]
    new_V = V_arr[:, mask]
    new_W = W_arr[:, mask]
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.sum(new_U[i*m+j, :] * new_V[j*p+k, :] * new_W[k*n+i, :])
    
    T_target = np.ones((n, m, p), dtype=float)
    return np.max(np.abs(T_recon - T_target)) < 1e-9

def systematic_rank_reduction(U, V, W, n=2, m=4, p=4):
    """Systematically try to remove terms one by one."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    current_U = U_arr.copy()
    current_V = V_arr.copy()
    current_W = W_arr.copy()
    current_rank = current_U.shape[1]
    
    # Try removing terms in order of smallest norm
    while current_rank > 0 and not time_up():
        norms = []
        for r in range(current_rank):
            norm = (np.linalg.norm(current_U[:, r]) + 
                   np.linalg.norm(current_V[:, r]) + 
                   np.linalg.norm(current_W[:, r]))
            norms.append(norm)
        
        # Try removing the term with smallest norm
        min_idx = np.argmin(norms)
        
        if try_remove_term(current_U, current_V, current_W, min_idx, n, m, p):
            mask = [i for i in range(current_rank) if i != min_idx]
            current_U = current_U[:, mask]
            current_V = current_V[:, mask]
            current_W = current_W[:, mask]
            current_rank = len(mask)
        else:
            # Try other terms
            found = False
            for r in range(current_rank):
                if r == min_idx:
                    continue
                if try_remove_term(current_U, current_V, current_W, r, n, m, p):
                    mask = [i for i in range(current_rank) if i != r]
                    current_U = current_U[:, mask]
                    current_V = current_V[:, mask]
                    current_W = current_W[:, mask]
                    current_rank = len(mask)
                    found = True
                    break
                if time_up():
                    break
            if not found and time_up():
                break
            if not found:
                break
    
    if current_rank < U_arr.shape[1]:
        return current_U.tolist(), current_V.tolist(), current_W.tolist(), current_rank
    
    return U, V, W, U_arr.shape[1]

def construct():
    n, m, p = 2, 4, 4
    best_result = None
    best_rank = float('inf')
    
    # Start with Strassen decomposition
    U, V, W = strassen_decomposition()
    valid, err = verify_decomposition(U, V, W, n, m, p)
    if valid:
        best_rank = 28
        best_result = {"U": U, "V": V, "W": W}
    
    # Try systematic rank reduction
    if not time_up() and best_result is not None:
        U2, V2, W2, rank2 = systematic_rank_reduction(U, V, W, n, m, p)
        if rank2 < best_rank:
            valid2, err2 = verify_decomposition(U2, V2, W2, n, m, p)
            if valid2:
                best_rank = rank2
                best_result = {"U": U2, "V": V2, "W": W2}
    
    # Fallback to trivial decomposition if needed
    if best_result is None:
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
