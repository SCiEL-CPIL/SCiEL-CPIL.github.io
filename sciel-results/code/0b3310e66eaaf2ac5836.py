"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                val = np.dot(np.multiply(U_arr[i * m + j, :], V_arr[j * p + k, :]), W_arr[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_decomposition():
    """
    Construct a Strassen-based decomposition for 2x4x8 matrix multiplication.
    Rank 56: 2 A-blocks * 4 B-blocks * 7 Strassen terms.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen's 7 products for C = A * B where A, B are 2x2:
    strassen_terms = [
        # M1 = (a00 + a11)(b00 + b11)
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        # M2 = (a10 + a11)b00
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1, (1,1): -1}
        ),
        # M3 = a00(b01 - b11)
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}
        ),
        # M4 = a11(b10 - b00)
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}
        ),
        # M5 = (a00 + a01)b11
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}
        ),
        # M6 = (a10 - a00)(b00 + b01)
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        # M7 = (a01 - a11)(b10 + b11)
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        ),
    ]
    
    col = 0
    for a_block in range(2):
        for b_block in range(4):
            for term in strassen_terms:
                A_coeffs, B_coeffs, C_coeffs = term
                for (li, lj), coeff in A_coeffs.items():
                    U[li * m + (2 * a_block + lj), col] += coeff
                for (lj, lk), coeff in B_coeffs.items():
                    V[(2 * a_block + lj) * p + (2 * b_block + lk), col] += coeff
                for (li, lk), coeff in C_coeffs.items():
                    W[(2 * b_block + lk) * n + li, col] += coeff
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook_decomposition():
    """Fallback schoolbook decomposition with rank n*m*p = 64."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(U, V, W, n, m, p, deadline, tol=1e-10):
    """Try to reduce the rank by identifying and removing linearly dependent terms."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    current_rank = rank
    active_cols = list(range(rank))
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        removed_any = False
        for idx in range(len(active_cols) - 1, -1, -1):
            if time.time() > deadline:
                break
            col = active_cols[idx]
            new_active = active_cols[:idx] + active_cols[idx+1:]
            if len(new_active) == 0:
                continue
            
            U_new = U_arr[:, new_active]
            V_new = V_arr[:, new_active]
            W_new = W_arr[:, new_active]
            
            ok = True
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        val = np.dot(np.multiply(U_new[i*m+j, :], V_new[j*p+k, :]), W_new[k*n+i, :])
                        if abs(val - 1.0) > tol:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                active_cols = new_active
                current_rank = len(active_cols)
                removed_any = True
                break
        
        if not removed_any:
            break
    
    if current_rank < rank:
        U_reduced = U_arr[:, active_cols].tolist()
        V_reduced = V_arr[:, active_cols].tolist()
        W_reduced = W_arr[:, active_cols].tolist()
        return U_reduced, V_reduced, W_reduced, current_rank
    
    return U, V, W, rank

def _randomized_search(U_init, V_init, W_init, n, m, p, deadline, max_iter=500, tol=1e-9):
    """
    Try to find a lower-rank decomposition by randomly perturbing and projecting.
    This is a heuristic search that may find a better decomposition.
    """
    U_arr = np.array(U_init, dtype=float)
    V_arr = np.array(V_init, dtype=float)
    W_arr = np.array(W_init, dtype=float)
    initial_rank = U_arr.shape[1]
    
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    best_rank = initial_rank
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try random subset of columns
        if best_rank <= 1:
            break
        
        trial_rank = max(1, best_rank - np.random.randint(1, max(2, best_rank // 10)))
        if trial_rank >= best_rank:
            continue
        
        cols = np.random.choice(best_rank, size=trial_rank, replace=False)
        U_try = best_U[:, cols]
        V_try = best_V[:, cols]
        W_try = best_W[:, cols]
        
        # Check if this subset works
        ok = True
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    val = np.dot(np.multiply(U_try[i*m+j, :], V_try[j*p+k, :]), W_try[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = trial_rank
    
    if best_rank < initial_rank:
        return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_rank
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Try Strassen decomposition first (rank 56)
    try:
        result = _strassen_decomposition()
        U, V, W = result["U"], result["V"], result["W"]
        if _verify(U, V, W, n, m, p, tol=1e-9):
            # Try to reduce rank
            U_red, V_red, W_red, new_rank = _try_reduce_rank(U, V, W, n, m, p, deadline)
            if new_rank < len(U[0]):
                if _verify(U_red, V_red, W_red, n, m, p, tol=1e-9):
                    return {"U": U_red, "V": V_red, "W": W_red}
            # Try randomized search for further reduction
            rand_result = _randomized_search(U_red, V_red, W_red, n, m, p, deadline)
            if rand_result is not None:
                U_r, V_r, W_r, r_rank = rand_result
                if _verify(U_r, V_r, W_r, n, m, p, tol=1e-9):
                    return {"U": U_r, "V": V_r, "W": W_r}
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 64)
    return _schoolbook_decomposition()

# EVOLVE-BLOCK-END
