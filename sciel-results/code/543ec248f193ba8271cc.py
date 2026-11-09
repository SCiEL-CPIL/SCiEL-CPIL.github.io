import time
import numpy as np

DIMENSIONS = [3, 3, 4]

_DEADLINE = 120.0
_START = None

def _remaining():
    global _START
    if _START is None:
        return _DEADLINE
    return _DEADLINE - (time.time() - _START)

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _construct_rank34():
    n, m, p = DIMENSIONS
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal c
        if c >= rank:
            return
        U[u_idx][c] += u_val
        V[v_idx][c] += v_val
        W[w_idx][c] += w_val
        c += 1

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def c_idx(i, k): return k * n + i

    def add_strassen(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        nonlocal c
        U[a00][c] = 1; U[a11][c] = 1
        V[b00][c] = 1; V[b11][c] = 1
        W[c00][c] = 1; W[c11][c] = 1
        c += 1
        U[a10][c] = 1; U[a11][c] = 1
        V[b00][c] = 1
        W[c10][c] = 1; W[c11][c] = -1
        c += 1
        U[a00][c] = 1
        V[b01][c] = 1; V[b11][c] = -1
        W[c01][c] = 1; W[c11][c] = 1
        c += 1
        U[a11][c] = 1
        V[b10][c] = 1; V[b00][c] = -1
        W[c00][c] = 1; W[c10][c] = 1
        c += 1
        U[a00][c] = 1; U[a01][c] = 1
        V[b11][c] = 1
        W[c00][c] = -1; W[c01][c] = 1
        c += 1
        U[a10][c] = 1; U[a00][c] = -1
        V[b00][c] = 1; V[b01][c] = 1
        W[c11][c] = 1
        c += 1
        U[a01][c] = 1; U[a11][c] = -1
        V[b10][c] = 1; V[b11][c] = 1
        W[c00][c] = 1
        c += 1

    add_strassen(u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1), 
                 v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1), 
                 c_idx(0,0), c_idx(0,1), c_idx(1,0), c_idx(1,1))
    add_strassen(u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1), 
                 v_idx(0,2), v_idx(0,3), v_idx(1,2), v_idx(1,3), 
                 c_idx(0,2), c_idx(0,3), c_idx(1,2), c_idx(1,3))

    add_term(u_idx(0,2), 1, v_idx(2,0), 1, c_idx(0,0), 1)
    add_term(u_idx(0,2), 1, v_idx(2,1), 1, c_idx(0,1), 1)
    add_term(u_idx(1,2), 1, v_idx(2,0), 1, c_idx(1,0), 1)
    add_term(u_idx(1,2), 1, v_idx(2,1), 1, c_idx(1,1), 1)
    add_term(u_idx(0,2), 1, v_idx(2,2), 1, c_idx(0,2), 1)
    add_term(u_idx(0,2), 1, v_idx(2,3), 1, c_idx(0,3), 1)
    add_term(u_idx(1,2), 1, v_idx(2,2), 1, c_idx(1,2), 1)
    add_term(u_idx(1,2), 1, v_idx(2,3), 1, c_idx(1,3), 1)
    add_term(u_idx(2,0), 1, v_idx(0,0), 1, c_idx(2,0), 1)
    add_term(u_idx(2,0), 1, v_idx(0,1), 1, c_idx(2,1), 1)
    add_term(u_idx(2,1), 1, v_idx(1,0), 1, c_idx(2,0), 1)
    add_term(u_idx(2,1), 1, v_idx(1,1), 1, c_idx(2,1), 1)
    add_term(u_idx(2,0), 1, v_idx(0,2), 1, c_idx(2,2), 1)
    add_term(u_idx(2,0), 1, v_idx(0,3), 1, c_idx(2,3), 1)
    add_term(u_idx(2,1), 1, v_idx(1,2), 1, c_idx(2,2), 1)
    add_term(u_idx(2,1), 1, v_idx(1,3), 1, c_idx(2,3), 1)
    add_term(u_idx(2,2), 1, v_idx(2,0), 1, c_idx(2,0), 1)
    add_term(u_idx(2,2), 1, v_idx(2,1), 1, c_idx(2,1), 1)
    add_term(u_idx(2,2), 1, v_idx(2,2), 1, c_idx(2,2), 1)
    add_term(u_idx(2,2), 1, v_idx(2,3), 1, c_idx(2,3), 1)

    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(total - 1) > 1e-9:
                    return False
    return True

def _try_find_lower_rank(target_rank, time_limit=5.0):
    """
    Attempts to find a decomposition of rank < current best by searching 
    near the rank-34 solution.
    """
    n, m, p = DIMENSIONS
    # Target dimensions
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Start with rank-34 solution
    base = _construct_rank34()
    U0 = np.array(base['U'], dtype=float)
    V0 = np.array(base['V'], dtype=float)
    W0 = np.array(base['W'], dtype=float)
    
    # We want to reduce to target_rank.
    # We will try to zero out (34 - target_rank) terms and optimize the rest.
    # For simplicity, we'll try target_rank = 33 first.
    
    start_time = time.time()
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Try a few random subsets of terms to keep
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
        
        # Randomly select 'target_rank' indices out of 34
        indices = np.random.choice(34, size=target_rank, replace=False)
        
        U = np.zeros((dim_u, target_rank))
        V = np.zeros((dim_v, target_rank))
        W = np.zeros((dim_w, target_rank))
        
        # Initialize with values from base solution
        for new_idx, old_idx in enumerate(indices):
            U[:, new_idx] = U0[:, old_idx]
            V[:, new_idx] = V0[:, old_idx]
            W[:, new_idx] = W0[:, old_idx]
            
        # Simple gradient descent
        lr = 0.01
        for step in range(200):
            if time.time() - start_time > time_limit:
                break
            
            # Compute residual
            # Res[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1
            # We want to minimize sum Res^2
            
            # Gradient w.r.t U[i,j, r]:
            # dL/dU[i,j,r] = 2 * Res[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            # Efficiently compute gradients
            # Res[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            
            # This is a bit heavy to compute naively in Python loops.
            # Using numpy broadcasting:
            # U: (9, r), V: (12, r), W: (12, r)
            # We need to sum over r.
            
            # Let's just do a few steps with a small learning rate.
            # To keep it fast, we might skip full gradient computation if it's too slow.
            # Given 50 attempts * 200 steps * 36 entries, it's manageable.
            
            # Compute Res
            Res = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(target_rank):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        Res[i,j,k] = val - 1
                        
            # Compute Gradients
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res_val = Res[i,j,k]
                        for r in range(target_rank):
                            grad_U[i*m+j, r] += 2 * res_val * V[j*p+k, r] * W[k*n+i, r]
                            grad_V[j*p+k, r] += 2 * res_val * U[i*m+j, r] * W[k*n+i, r]
                            grad_W[k*n+i, r] += 2 * res_val * U[i*m+j, r] * V[j*p+k, r]
                            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            err = np.sum(Res**2)
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            if err < 1e-9:
                # Verify and return
                U_list = [[U[i, r] for r in range(target_rank)] for i in range(dim_u)]
                V_list = [[V[i, r] for r in range(target_rank)] for i in range(dim_v)]
                W_list = [[W[i, r] for r in range(target_rank)] for i in range(dim_w)]
                if _verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
    # If best_err is small, try to polish
    if best_err < 1e-6 and best_U is not None:
        # Convert to list and verify
        U_list = [[best_U[i, r] for r in range(target_rank)] for i in range(dim_u)]
        V_list = [[best_V[i, r] for r in range(target_rank)] for i in range(dim_v)]
        W_list = [[best_W[i, r] for r in range(target_rank)] for i in range(dim_w)]
        if _verify(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
            
    return None

def construct():
    global _START
    _START = time.time()
    
    # Try to find rank 33
    try:
        result33 = _try_find_lower_rank(33, time_limit=10.0)
        if result33 is not None:
            return result33
    except Exception as e:
        pass
        
    # Try to find rank 32 (less likely)
    try:
        result32 = _try_find_lower_rank(32, time_limit=10.0)
        if result32 is not None:
            return result32
    except Exception as e:
        pass
        
    # Fall back to rank 34
    result34 = _construct_rank34()
    if _verify(result34["U"], result34["V"], result34["W"]):
        return result34
        
    # Fall back to schoolbook
    return _schoolbook()
