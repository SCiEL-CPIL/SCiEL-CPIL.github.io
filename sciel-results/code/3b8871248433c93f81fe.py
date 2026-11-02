"""Matrix multiplication 4,4,8 rank-minimal search with Strassen decomposition and rank reduction."""
import random
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_2x2_factors():
    """
    Returns U, V, W for 2x2 matrix multiplication using Strassen's algorithm.
    U: 4x7, V: 4x7, W: 4x7
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22) -> contributes to C11, C22
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11 -> contributes to C21
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A11(B12-B22) -> contributes to C12
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = A22(B21-B11) -> contributes to C21
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1
    
    # M5 = (A11+A12)B22 -> contributes to C12
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12) -> contributes to C22
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22) -> contributes to C11
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _strassen_4x4_factors():
    """
    Returns U, V, W for 4x4 matrix multiplication using recursive Strassen.
    U: 16x49, V: 16x49, W: 16x49
    """
    U2, V2, W2 = _strassen_2x2_factors()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    high_level_factors = [
        {'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}},
        {'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(1,0): 1}},
        {'U': {(0,0): 1}, 'V': {(0,1): 1, (1,1): -1}, 'W': {(0,1): 1}},
        {'U': {(1,1): 1}, 'V': {(1,0): 1, (0,0): -1}, 'W': {(0,0): 1, (1,0): 1}},
        {'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,0): -1, (0,1): 1}},
        {'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(1,1): 1}},
        {'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1}},
    ]
    
    for h, hf in enumerate(high_level_factors):
        for l in range(7):
            factor_idx = h * 7 + l
            
            for (bi, bj), coeff_u in hf['U'].items():
                for li in range(2):
                    for lj in range(2):
                        a_row = bi * 2 + li
                        a_col = bj * 2 + lj
                        U_idx = a_row * 4 + a_col
                        U[U_idx, factor_idx] += coeff_u * U2[li * 2 + lj, l]
            
            for (bi, bj), coeff_v in hf['V'].items():
                for li in range(2):
                    for lj in range(2):
                        b_row = bi * 2 + li
                        b_col = bj * 2 + lj
                        V_idx = b_row * 4 + b_col
                        V[V_idx, factor_idx] += coeff_v * V2[li * 2 + lj, l]
            
            for (bi, bj), coeff_w in hf['W'].items():
                for li in range(2):
                    for lj in range(2):
                        c_row = bi * 2 + li
                        c_col = bj * 2 + lj
                        W_idx = c_row * 4 + c_col
                        W[W_idx, factor_idx] += coeff_w * W2[li * 2 + lj, l]
    
    return U, V, W

def _adapt_to_4x8(U4, V4, W4):
    """
    Adapt 4x4 Strassen factors to 4x8 multiplication.
    """
    rank4 = U4.shape[1]
    n, m, p = 4, 4, 8
    
    U = U4.copy()
    
    V = np.zeros((32, rank4))
    for j in range(4):
        for k in range(4):
            V[j*8 + k, :] = V4[j*4 + k, :]
            V[j*8 + (k+4), :] = V4[j*4 + k, :]
    
    W = np.zeros((32, rank4))
    for k in range(4):
        for i in range(4):
            W[k*4 + i, :] = W4[k*4 + i, :]
            W[(k+4)*4 + i, :] = W4[k*4 + i, :]
    
    return U, V, W

def _validate_fast(U, V, W, n, m, p):
    """Fast vectorized validation using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(np.dot(U_arr[a], V_arr[b]), W_arr[c])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _construct_strassen_4x8():
    """Construct Strassen-based decomposition for 4x4 * 4x8."""
    U4, V4, W4 = _strassen_4x4_factors()
    U, V, W = _adapt_to_4x8(U4, V4, W4)
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    col_sums = np.sum(np.abs(U_arr) * np.abs(V_arr) * np.abs(W_arr), axis=0)
    nonzero_cols = np.where(col_sums > 1e-12)[0]
    
    if len(nonzero_cols) < U_arr.shape[1]:
        U_arr = U_arr[:, nonzero_cols]
        V_arr = V_arr[:, nonzero_cols]
        W_arr = W_arr[:, nonzero_cols]
    
    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}

def _try_reduce_rank(U, V, W, n, m, p, deadline, max_iterations=2000):
    """
    Try to reduce rank by attempting to set some columns to zero 
    while maintaining exactness. Uses local search.
    """
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    current_rank = len(U[0])
    
    columns_to_try = list(range(current_rank))
    random.shuffle(columns_to_try)
    
    eliminated = set()
    for col in columns_to_try:
        if time.time() > deadline - 30:
            break
        
        U_save = [row[:] for row in U]
        V_save = [row[:] for row in V]
        W_save = [row[:] for row in W]
        
        for r in range(len(U)):
            U[r][col] = 0.0
        for r in range(len(V)):
            V[r][col] = 0.0
        for r in range(len(W)):
            W[r][col] = 0.0
        
        if _validate_fast(U, V, W, n, m, p):
            eliminated.add(col)
        else:
            U = U_save
            V = V_save
            W = W_save
    
    if len(eliminated) > 0:
        keep_cols = [c for c in range(current_rank) if c not in eliminated]
        U = [[row[c] for c in keep_cols] for row in U]
        V = [[row[c] for c in keep_cols] for row in V]
        W = [[row[c] for c in keep_cols] for row in W]
    
    return U, V, W

def _try_random_rank(r, n, m, p, deadline, max_attempts=10):
    """Try to find a rank-r decomposition via random search with local optimization."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 30:
            return None
        
        U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(nm)]
        V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(mp)]
        W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(pn)]
        
        for iteration in range(30):
            if time.time() > deadline - 30:
                break
            if _validate_fast(U, V, W, n, m, p):
                return U, V, W
            
            mat_name = random.choice(['U', 'V', 'W'])
            if mat_name == 'U':
                mat = U
            elif mat_name == 'V':
                mat = V
            else:
                mat = W
            
            row = random.randrange(len(mat))
            col = random.randrange(r)
            
            old_val = mat[row][col]
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val:
                new_val = (old_val + 1) % 3 - 1
            
            mat[row][col] = new_val
    
    return None

def _search_rank_32(n, m, p, deadline, max_restarts=5):
    """
    Search for a rank-32 decomposition using randomized local search.
    """
    rank = 32
    nm, mp, pn = n*m, m*p, p*n
    
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i*m+j, j*p+k, k*n+i))
                
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for restart in range(max_restarts):
        if time.time() > deadline - 30:
            break
            
        U = np.random.randint(-1, 2, (nm, rank))
        V = np.random.randint(-1, 2, (mp, rank))
        W = np.random.randint(-1, 2, (pn, rank))
        
        for iter in range(2000):
            if time.time() > deadline - 30:
                break
                
            error = 0.0
            for a, b, c in indices:
                val = np.dot(np.dot(U[a], V[b]), W[c])
                error += (val - 1.0) ** 2
            
            if error < 1e-9:
                return U.tolist(), V.tolist(), W.tolist()
            
            if error < best_error:
                best_error = error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            mat_idx = random.randrange(3)
            if mat_idx == 0:
                mat = U
                rows = nm
            elif mat_idx == 1:
                mat = V
                rows = mp
            else:
                mat = W
                rows = pn
                
            r = random.randrange(rows)
            c = random.randrange(rank)
            
            old_val = mat[r, c]
            candidates = [v for v in [-1, 0, 1] if v != old_val]
            random.shuffle(candidates)
            
            improved = False
            for new_val in candidates:
                mat[r, c] = new_val
                
                error_new = 0.0
                for a, b, c_idx in indices:
                    val = np.dot(np.dot(U[a], V[b]), W[c_idx])
                    error_new += (val - 1.0) ** 2
                    if error_new > error:
                        break
                
                if error_new < error:
                    error = error_new
                    improved = True
                    break
                else:
                    mat[r, c] = old_val
            
            if not improved:
                break
                
    if best_U is not None and best_error < 1e-9:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # First, try to find rank 32
    try:
        result = _search_rank_32(n, m, p, deadline)
        if result:
            U, V, W = result
            if _validate_fast(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Try to reduce Strassen rank
    try:
        strassen_result = _construct_strassen_4x8()
        U, V, W = strassen_result['U'], strassen_result['V'], strassen_result['W']
        
        # Try to reduce rank from 49
        U_reduced, V_reduced, W_reduced = _try_reduce_rank(U, V, W, n, m, p, deadline)
        
        if _validate_fast(U_reduced, V_reduced, W_reduced, n, m, p):
            return {"U": U_reduced, "V": V_reduced, "W": W_reduced}
    except Exception:
        pass
    
    # Fallback to Strassen (rank 49)
    try:
        result = _construct_strassen_4x8()
        U, V, W = result['U'], result['V'], result['W']
        
        if _validate_fast(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 128)
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
