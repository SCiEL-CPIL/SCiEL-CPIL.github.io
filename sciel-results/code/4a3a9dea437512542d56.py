"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 145.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _strassen_2x2_2():
    """Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen)."""
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # P1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _build_strassen_embedding(n, m, p, strassen_blocks):
    """
    Build decomposition using Strassen on specified 2x2x2 blocks, schoolbook elsewhere.
    """
    covered = set()
    for (i0, j0, k0) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    n_school = n*m*p - len(covered)
    rank = 7 * len(strassen_blocks) + n_school
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    sU, sV, sW = _strassen_2x2_2()
    for (i0, j0, k0) in strassen_blocks:
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj)][col+c] = sU[di*2+dj][c]
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk)][col+c] = sV[dj*2+dk][c]
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk)*n + (i0+di)][col+c] = sW[dk*2+di][c]
        col += 7
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_np(U, V, W, n, m, p, tol=1e-8):
    """Fast verification using numpy."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iters=20000):
    """
    Try to reduce rank from current to target_rank using coordinate descent.
    Start with the current decomposition, set extra columns to zero, and optimize.
    """
    current_rank = len(U[0])
    if current_rank <= target_rank:
        return U, V, W, True
    
    # Initialize: keep first target_rank columns, zero out the rest
    U_new = [[0]*target_rank for _ in range(n*m)]
    V_new = [[0]*target_rank for _ in range(m*p)]
    W_new = [[0]*target_rank for _ in range(p*n)]
    for i in range(n*m):
        for c in range(target_rank):
            U_new[i][c] = U[i][c]
    for j in range(m*p):
        for c in range(target_rank):
            V_new[j][c] = V[j][c]
    for k in range(p*n):
        for c in range(target_rank):
            W_new[k][c] = W[k][c]
    
    # Coordinate descent
    U_arr = np.array(U_new, dtype=np.float64)
    V_arr = np.array(V_new, dtype=np.float64)
    W_arr = np.array(W_new, dtype=np.float64)
    
    # Precompute all (i,j,k) triples
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    n_triples = len(triples)
    
    # Build index arrays for fast access
    U_idx = np.array([i*m+j for i, j, k in triples])
    V_idx = np.array([j*p+k for i, j, k in triples])
    W_idx = np.array([k*n+i for i, j, k in triples])
    
    best_loss = np.inf
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    
    for it in range(max_iters):
        if _time_left() < 10:
            break
        
        # Pick a random triple
        t = np.random.randint(n_triples)
        ui, vi, wi = U_idx[t], V_idx[t], W_idx[t]
        
        # Compute current product for this triple
        # residual = 1 - sum_c U[ui,c]*V[vi,c]*W[wi,c]
        prod = U_arr[ui] * V_arr[vi] * W_arr[wi]
        current_sum = np.sum(prod)
        residual = 1.0 - current_sum
        
        # Try updating U[ui,c] for random c
        c = np.random.randint(target_rank)
        
        # Optimal U[ui,c] given others:
        # We want sum_c' U[ui,c']*V[vi,c']*W[wi,c'] = 1
        # Let S = sum_{c' != c} U[ui,c']*V[vi,c']*W[wi,c']
        # Then U[ui,c] should be (1 - S) / (V[vi,c]*W[wi,c]) if V[vi,c]*W[wi,c] != 0
        Vw = V_arr[vi] * W_arr[wi]
        if abs(Vw[c]) < 1e-12:
            continue
        
        S = np.sum(prod) - U_arr[ui, c] * Vw[c]
        new_val = (1.0 - S) / Vw[c]
        
        # Update
        U_arr[ui, c] = new_val
        prod = U_arr[ui] * V_arr[vi] * W_arr[wi]
        
        # Occasionally check loss
        if it % 100 == 0:
            loss = 0.0
            for t2 in range(n_triples):
                p2 = U_arr[U_idx[t2]] * V_arr[V_idx[t2]] * W_arr[W_idx[t2]]
                loss += (1.0 - np.sum(p2))**2
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
                if best_loss < 1e-14:
                    break
    
    # Final check
    loss = 0.0
    for t2 in range(n_triples):
        p2 = best_U[U_idx[t2]] * best_V[V_idx[t2]] * best_W[W_idx[t2]]
        loss += (1.0 - np.sum(p2))**2
    
    if loss < 1e-10:
        # Convert back to lists with rounded values
        U_out = [[0]*target_rank for _ in range(n*m)]
        V_out = [[0]*target_rank for _ in range(m*p)]
        W_out = [[0]*target_rank for _ in range(p*n)]
        for i in range(n*m):
            for c in range(target_rank):
                U_out[i][c] = round(best_U[i, c], 10)
        for j in range(m*p):
            for c in range(target_rank):
                V_out[j][c] = round(best_V[j, c], 10)
        for k in range(p*n):
            for c in range(target_rank):
                W_out[k][c] = round(best_W[k, c], 10)
        return U_out, V_out, W_out, True
    
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p  # 72
    
    if _time_left() > 10:
        try:
            # 4 Strassen 2x2x2 blocks
            blocks = [
                (0, 0, 0),
                (0, 0, 2),
                (0, 0, 4),
                (0, 0, 6),
            ]
            U, V, W = _build_strassen_embedding(n, m, p, blocks)
            if _verify(U, V, W, n, m, p):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    # Try to reduce rank further using coordinate descent
    if _time_left() > 20 and best_rank > 67:
        try:
            for target in [best_rank - 1, best_rank - 2]:
                if target < 1:
                    break
                if _time_left() < 15:
                    break
                U_r, V_r, W_r, success = _try_reduce_rank(
                    best_result["U"], best_result["V"], best_result["W"],
                    n, m, p, target, max_iters=50000
                )
                if success and _verify(U_r, V_r, W_r, n, m, p):
                    best_rank = target
                    best_result = {"U": U_r, "V": V_r, "W": W_r}
                    break
        except Exception:
            pass
    
    return best_result
# EVOLVE-BLOCK-END
