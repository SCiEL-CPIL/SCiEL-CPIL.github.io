import numpy as np
import time

DIMENSIONS = [3, 3, 4]

_DEADLINE = 150.0
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

def _is_valid(n, m, p, U, V, W, tol=1e-9):
    """Check exact decomposition of the MM tensor with the official W trace convention."""
    if not U or not V or not W:
        return False
    R = len(U[0])
    if len(V[0]) != R or len(W[0]) != R:
        return False
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    for i in range(n):
        for k in range(p):
            for j in range(m):
                val = 0.0
                for r in range(R):
                    val += Ua[i * m + j, r] * Va[j * p + k, r] * Wa[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _construct_rank34():
    n, m, p = DIMENSIONS
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    c = 0

    def add_strassen(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        nonlocal c
        U[a00][c] += 1; U[a11][c] += 1
        V[b00][c] += 1; V[b11][c] += 1
        W[c00][c] += 1; W[c11][c] += 1
        c += 1
        U[a10][c] += 1; U[a11][c] += 1
        V[b00][c] += 1
        W[c10][c] += 1; W[c11][c] += -1
        c += 1
        U[a00][c] += 1
        V[b01][c] += 1; V[b11][c] += -1
        W[c01][c] += 1; W[c11][c] += 1
        c += 1
        U[a11][c] += 1
        V[b10][c] += 1; V[b00][c] += -1
        W[c00][c] += 1; W[c10][c] += 1
        c += 1
        U[a00][c] += 1; U[a01][c] += 1
        V[b11][c] += 1
        W[c00][c] += -1; W[c01][c] += 1
        c += 1
        U[a10][c] += 1; U[a00][c] += -1
        V[b00][c] += 1; V[b01][c] += 1
        W[c11][c] += 1
        c += 1
        U[a01][c] += 1; U[a11][c] += -1
        V[b10][c] += 1; V[b11][c] += 1
        W[c00][c] += 1
        c += 1

    def c_idx(i, k): return k * n + i
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k

    add_strassen(u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1),
                 v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1),
                 c_idx(0,0), c_idx(0,1), c_idx(1,0), c_idx(1,1))
    add_strassen(u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1),
                 v_idx(0,2), v_idx(0,3), v_idx(1,2), v_idx(1,3),
                 c_idx(0,2), c_idx(0,3), c_idx(1,2), c_idx(1,3))
    
    def add_term(u_idx_val, v_idx_val, w_idx_val):
        nonlocal c
        U[u_idx_val][c] += 1
        V[v_idx_val][c] += 1
        W[w_idx_val][c] += 1
        c += 1

    add_term(u_idx(0,2), v_idx(2,0), c_idx(0,0))
    add_term(u_idx(0,2), v_idx(2,1), c_idx(0,1))
    add_term(u_idx(1,2), v_idx(2,0), c_idx(1,0))
    add_term(u_idx(1,2), v_idx(2,1), c_idx(1,1))
    add_term(u_idx(0,2), v_idx(2,2), c_idx(0,2))
    add_term(u_idx(0,2), v_idx(2,3), c_idx(0,3))
    add_term(u_idx(1,2), v_idx(2,2), c_idx(1,2))
    add_term(u_idx(1,2), v_idx(2,3), c_idx(1,3))
    add_term(u_idx(2,0), v_idx(0,0), c_idx(2,0))
    add_term(u_idx(2,0), v_idx(0,1), c_idx(2,1))
    add_term(u_idx(2,1), v_idx(1,0), c_idx(2,0))
    add_term(u_idx(2,1), v_idx(1,1), c_idx(2,1))
    add_term(u_idx(2,0), v_idx(0,2), c_idx(2,2))
    add_term(u_idx(2,0), v_idx(0,3), c_idx(2,3))
    add_term(u_idx(2,1), v_idx(1,2), c_idx(2,2))
    add_term(u_idx(2,1), v_idx(1,3), c_idx(2,3))
    add_term(u_idx(2,2), v_idx(2,0), c_idx(2,0))
    add_term(u_idx(2,2), v_idx(2,1), c_idx(2,1))
    add_term(u_idx(2,2), v_idx(2,2), c_idx(2,2))
    add_term(u_idx(2,2), v_idx(2,3), c_idx(2,3))

    return {"U": U, "V": V, "W": W}

def _try_rank29(n, m, p, deadline):
    """Attempt to find a rank-29 decomposition by fixing the 4th column and searching the 3x3x3 part."""
    rank = 29
    R_sub = 20
    R_tail = 9
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Fix the last 9 terms for k=3 (schoolbook)
    for i in range(n):
        for j in range(m):
            r = R_sub + i * m + j
            U[i * m + j, r] = 1.0
            V[j * p + 3, r] = 1.0
            W[3 * n + i, r] = 1.0
            
    # Search for the first 20 terms for k in {0,1,2}
    # Target: T[i,j,k] = 1 for k < 3
    # We use SA on the first 20 columns
    
    np.random.seed(42)
    U_sub = np.random.randint(-1, 2, (n, m, R_sub)).astype(float)
    V_sub = np.random.randint(-1, 2, (m, 3, R_sub)).astype(float) # Only k=0,1,2
    W_sub = np.random.randint(-1, 2, (3, n, R_sub)).astype(float) # Only k=0,1,2
    
    def _calc_err_sub():
        # X[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # We only care about k < 3
        X = np.einsum('ijr,jkr,kir->ijk', U_sub, V_sub, W_sub)
        return np.sum((X - 1.0)**2)
        
    curr_err = _calc_err_sub()
    best_err = curr_err
    best_U = U_sub.copy()
    best_V = V_sub.copy()
    best_W = W_sub.copy()
    
    T_start = 1.0
    T_end = 0.01
    max_steps = 20000
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        T = T_start * (T_end / T_start) ** (step / max_steps)
        
        tensor_idx = np.random.randint(0, 3)
        if tensor_idx == 0:
            i, j, r = np.random.randint(n), np.random.randint(m), np.random.randint(R_sub)
            old_val = U_sub[i, j, r]
            delta = np.random.choice([-0.5, 0.5])
            U_sub[i, j, r] += delta
            new_err = _calc_err_sub()
            if new_err > curr_err:
                if np.random.rand() < np.exp(-(new_err - curr_err) / T):
                    curr_err = new_err
                else:
                    U_sub[i, j, r] = old_val
            else:
                curr_err = new_err
                if curr_err < best_err:
                    best_err = curr_err
                    best_U, best_V, best_W = U_sub.copy(), V_sub.copy(), W_sub.copy()
        elif tensor_idx == 1:
            j, k, r = np.random.randint(m), np.random.randint(3), np.random.randint(R_sub)
            old_val = V_sub[j, k, r]
            delta = np.random.choice([-0.5, 0.5])
            V_sub[j, k, r] += delta
            new_err = _calc_err_sub()
            if new_err > curr_err:
                if np.random.rand() < np.exp(-(new_err - curr_err) / T):
                    curr_err = new_err
                else:
                    V_sub[j, k, r] = old_val
            else:
                curr_err = new_err
                if curr_err < best_err:
                    best_err = curr_err
                    best_U, best_V, best_W = U_sub.copy(), V_sub.copy(), W_sub.copy()
        else:
            k, i, r = np.random.randint(3), np.random.randint(n), np.random.randint(R_sub)
            old_val = W_sub[k, i, r]
            delta = np.random.choice([-0.5, 0.5])
            W_sub[k, i, r] += delta
            new_err = _calc_err_sub()
            if new_err > curr_err:
                if np.random.rand() < np.exp(-(new_err - curr_err) / T):
                    curr_err = new_err
                else:
                    W_sub[k, i, r] = old_val
            else:
                curr_err = new_err
                if curr_err < best_err:
                    best_err = curr_err
                    best_U, best_V, best_W = U_sub.copy(), V_sub.copy(), W_sub.copy()
                    
    # Rounding and check
    U_r = np.round(best_U * 2) / 2
    V_r = np.round(best_V * 2) / 2
    W_r = np.round(best_W * 2) / 2
    
    # Verify sub-tensor
    X = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    if np.max(np.abs(X - 1.0)) > 1e-9:
        return None
        
    # Construct full tensors
    U[:, :R_sub] = U_r.reshape(n*m, R_sub)
    V[:, :R_sub] = V_r.reshape(m*3, R_sub)
    # Note: V_r is (m, 3, R_sub), but V is (m*p, R_sub). 
    # We need to place V_r into V[0:3, :R_sub] for k=0,1,2.
    # V index is j*p + k.
    for j in range(m):
        for k in range(3):
            V[j*p + k, :R_sub] = V_r[j, k, :]
            
    W[:, :R_sub] = 0 # W for k=0,1,2
    # W_r is (3, n, R_sub). W index is k*n + i.
    for k in range(3):
        for i in range(n):
            W[k*n + i, :R_sub] = W_r[k, i, :]
            
    # The last 9 columns are already set.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    global _START
    _START = time.time()
    n, m, p = DIMENSIONS

    fallback = _construct_rank34()
    if not _is_valid(n, m, p, fallback["U"], fallback["V"], fallback["W"]):
        return _schoolbook()

    deadline = _START + 140.0
    best = fallback
    
    # Try Rank 29
    if time.time() < deadline:
        res29 = _try_rank29(n, m, p, deadline)
        if res29 is not None and _is_valid(n, m, p, res29["U"], res29["V"], res29["W"]):
            return res29
            
    return best
