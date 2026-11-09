import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_2x4x2():
    """
    Build a rank-14 decomposition for 2x4x2 matrix multiplication.
    Uses two independent Strassen 2x2x2 blocks (one for j=0,1 and one for j=2,3).
    """
    strassen_U = [
        [1,0,0,1],
        [0,0,1,1],
        [1,0,0,0],
        [0,0,0,1],
        [1,1,0,0],
        [-1,0,1,0],
        [0,1,0,-1],
    ]
    strassen_V = [
        [1,0,0,1],
        [1,0,0,0],
        [0,1,0,-1],
        [-1,0,1,0],
        [0,0,0,1],
        [1,1,0,0],
        [0,0,1,1],
    ]
    strassen_W = [
        [1,0,0,1],
        [0,1,0,-1],
        [0,0,1,1],
        [1,1,0,0],
        [-1,0,1,0],
        [0,0,0,1],
        [1,0,0,0],
    ]
    
    n, m, p = 2, 4, 2
    rank = 14
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for j0 in [0, 2]:
        for r in range(7):
            for li in range(2):
                for lj in range(2):
                    gi = li
                    gj = j0 + lj
                    U[gi*m + gj][col] = strassen_U[r][li*2 + lj]
            for lj in range(2):
                for lk in range(2):
                    gj = j0 + lj
                    gk = lk
                    V[gj*p + gk][col] = strassen_V[r][lj*2 + lk]
            for lk in range(2):
                for li in range(2):
                    gk = lk
                    gi = li
                    W[gk*n + gi][col] = strassen_W[r][lk*2 + li]
            col += 1
    
    return U, V, W, rank

def _try_rank_reduction(U, V, W, n, m, p, time_limit=30):
    """
    Attempt to reduce the rank by finding linear dependencies among columns.
    Uses a greedy approach with pair removal and smart verification.
    """
    start_time = time.time()
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_rank = rank
    
    def _verify_fast(U_t, V_t, W_t, keep_indices):
        """Fast verification using numpy matrix operations."""
        U_k = U_arr[:, keep_indices]
        V_k = V_arr[:, keep_indices]
        W_k = W_arr[:, keep_indices]
        
        # For each (i,j,k), check sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U_k[i*m+j] * V_k[j*p+k] * W_k[k*n+i])
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to remove columns greedily, one at a time
    removed = set()
    for c in range(rank):
        if time.time() - start_time > time_limit:
            break
        if c in removed:
            continue
        
        new_rank = rank - len(removed) - 1
        if new_rank < 20:  # Don't go too low
            break
        
        keep = [i for i in range(rank) if i != c and i not in removed]
        
        if _verify_fast(None, None, None, keep):
            removed.add(c)
            best_rank = len(keep)
            keep_arr = np.array(keep)
            best_U = U_arr[:, keep_arr].tolist()
            best_V = V_arr[:, keep_arr].tolist()
            best_W = W_arr[:, keep_arr].tolist()
    
    # Try removing pairs of columns
    if time.time() - start_time < time_limit - 5:
        current_keep = [i for i in range(rank) if i not in removed]
        for a in range(len(current_keep)):
            if time.time() - start_time > time_limit - 2:
                break
            for b in range(a + 1, len(current_keep)):
                if time.time() - start_time > time_limit - 2:
                    break
                
                ca = current_keep[a]
                cb = current_keep[b]
                new_keep = [i for i in current_keep if i != ca and i != cb]
                
                if _verify_fast(None, None, None, new_keep):
                    removed.add(ca)
                    removed.add(cb)
                    best_rank = len(new_keep)
                    keep_arr = np.array(new_keep)
                    best_U = U_arr[:, keep_arr].tolist()
                    best_V = V_arr[:, keep_arr].tolist()
                    best_W = W_arr[:, keep_arr].tolist()
                    current_keep = new_keep
                    a = -1  # Restart inner loop after modification
                    break
    
    if best_rank < rank:
        # Convert back to lists with proper types
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        
        # Try to round to integers or half-integers
        for mat in [U_list, V_list, W_list]:
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    val = mat[i][j]
                    # Round to nearest 0.5
                    rounded = round(val * 2) / 2
                    if abs(val - rounded) < 1e-9:
                        mat[i][j] = rounded
        
        return U_list, V_list, W_list, best_rank
    
    return None

def construct():
    n, m, p = 2, 4, 5
    start_time = time.time()
    
    # Build the base Strassen-based solution: rank 36
    strassen_U = [
        [1,0,0,1],
        [0,0,1,1],
        [1,0,0,0],
        [0,0,0,1],
        [1,1,0,0],
        [-1,0,1,0],
        [0,1,0,-1],
    ]
    strassen_V = [
        [1,0,0,1],
        [1,0,0,0],
        [0,1,0,-1],
        [-1,0,1,0],
        [0,0,0,1],
        [1,1,0,0],
        [0,0,1,1],
    ]
    strassen_W = [
        [1,0,0,1],
        [0,1,0,-1],
        [0,0,1,1],
        [1,1,0,0],
        [-1,0,1,0],
        [0,0,0,1],
        [1,0,0,0],
    ]
    
    # Total rank: 4 sub-problems * 7 + 8 (schoolbook for k=4) = 36
    rank = 4 * 7 + 8
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2]:
            for r in range(7):
                for li in range(2):
                    for lj in range(2):
                        gi = li
                        gj = j0 + lj
                        U[gi * m + gj][col] = strassen_U[r][li * 2 + lj]
                for lj in range(2):
                    for lk in range(2):
                        gj = j0 + lj
                        gk = k0 + lk
                        V[gj * p + gk][col] = strassen_V[r][lj * 2 + lk]
                for lk in range(2):
                    for li in range(2):
                        gk = k0 + lk
                        gi = li
                        W[gk * n + gi][col] = strassen_W[r][lk * 2 + li]
                col += 1
    
    # Schoolbook for k=4: 8 terms
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    # Verify the base solution
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank using greedy column removal
    time_remaining = 140 - (time.time() - start_time)
    if time_remaining > 10:
        result = _try_rank_reduction(U, V, W, n, m, p, time_limit=min(time_remaining - 5, 30))
        if result is not None:
            U_new, V_new, W_new, new_rank = result
            if new_rank < rank:
                # Verify the reduced solution
                if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                    U, V, W, rank = U_new, V_new, W_new, new_rank
    
    return {"U": U, "V": V, "W": W}
