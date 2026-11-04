import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Target tensor for 3x3x5
    T_full = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_full[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook fallback
    rank_school = n * m * p
    U_sch = np.zeros((n*m, rank_school))
    V_sch = np.zeros((m*p, rank_school))
    W_sch = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i*m+j, col] = 1
                V_sch[j*p+k, col] = 1
                W_sch[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_sch.tolist(), "V": V_sch.tolist(), "W": W_sch.tolist()}
    best_rank = rank_school

    def check_exact(U, V, W, T_target):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_target, T_approx, atol=1e-9)

    def find_decomp(target_rank, max_restarts=20, max_iters=2000):
        vals = [-1, -0.5, 0, 0.5, 1]
        
        best_loss = np.inf
        best_U = None
        best_V = None
        best_W = None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            U = np.random.choice(vals, size=(n*m, target_rank))
            V = np.random.choice(vals, size=(m*p, target_rank))
            W = np.random.choice(vals, size=(p*n, target_rank))
            
            current_loss = np.sum((T_full - np.einsum('ar,br,cr->abc', U, V, W))**2)
            
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    break
                    
                if current_loss < 1e-9:
                    return U, V, W
                    
                mat_idx = np.random.randint(3)
                if mat_idx == 0:
                    r, c = np.random.randint(n*m), np.random.randint(target_rank)
                    old_val = U[r, c]
                    new_val = np.random.choice(vals)
                    if new_val == old_val: continue
                    U[r, c] = new_val
                    new_loss = np.sum((T_full - np.einsum('ar,br,cr->abc', U, V, W))**2)
                    if new_loss < current_loss:
                        current_loss = new_loss
                    else:
                        U[r, c] = old_val
                elif mat_idx == 1:
                    r, c = np.random.randint(m*p), np.random.randint(target_rank)
                    old_val = V[r, c]
                    new_val = np.random.choice(vals)
                    if new_val == old_val: continue
                    V[r, c] = new_val
                    new_loss = np.sum((T_full - np.einsum('ar,br,cr->abc', U, V, W))**2)
                    if new_loss < current_loss:
                        current_loss = new_loss
                    else:
                        V[r, c] = old_val
                else:
                    r, c = np.random.randint(p*n), np.random.randint(target_rank)
                    old_val = W[r, c]
                    new_val = np.random.choice(vals)
                    if new_val == old_val: continue
                    W[r, c] = new_val
                    new_loss = np.sum((T_full - np.einsum('ar,br,cr->abc', U, V, W))**2)
                    if new_loss < current_loss:
                        current_loss = new_loss
                    else:
                        W[r, c] = old_val
                        
            if current_loss < best_loss:
                best_loss = current_loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
        if best_loss < 1e-9:
            return best_U, best_V, best_W
        return None

    # Try to find lower rank decompositions for the full tensor
    # Start from rank 44 down to 40
    for rank in range(rank_school - 1, 39, -1):
        if time.time() > deadline - 10:
            break
            
        res = find_decomp(rank, max_restarts=10, max_iters=1000)
        if res is not None:
            U, V, W = res
            if check_exact(U, V, W, T_full):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # Fallback to schoolbook
    return best_result
