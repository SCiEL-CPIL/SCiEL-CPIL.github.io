"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import itertools

DIMENSIONS = [2, 2, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0  # 5s margin before 150s kill

def _strassen_block_indices(k0, k1):
    """Return the 7 (u_entries, v_entries, w_entries) for one Strassen block."""
    p = DIMENSIONS[2]
    n = DIMENSIONS[0]
    # V indices
    v_e = 0 * p + k0
    v_f = 0 * p + k1
    v_g = 1 * p + k0
    v_h = 1 * p + k1
    # W indices
    w_c00 = k0 * n + 0
    w_c10 = k0 * n + 1
    w_c01 = k1 * n + 0
    w_c11 = k1 * n + 1

    blocks = []
    # P1 = (a+d)(e+h)
    blocks.append(([0,3],[v_e,v_h],[w_c00,w_c11]))
    # P2 = (c+d)e
    blocks.append(([2,3],[v_e],[w_c10,w_c11]))
    # P3 = a(f-h)
    blocks.append(([0],[v_f,v_h],[w_c01,w_c11]))
    # P4 = d(g-e)
    blocks.append(([3],[v_g,v_e],[w_c00,w_c10]))
    # P5 = (a+b)h
    blocks.append(([0,1],[v_h],[w_c00,w_c01]))
    # P6 = (c-a)(e+f)
    blocks.append(([2,0],[v_e,v_f],[w_c11]))
    # P7 = (b-d)(g+h)
    blocks.append(([1,3],[v_g,v_h],[w_c00]))
    return blocks

def _get_strassen_decomposition(n, m, p):
    """Build the standard 3-block Strassen decomposition, rank = 3*7 = 21."""
    rank = 3 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for b in range(3):
        k0 = 2 * b
        k1 = 2 * b + 1
        for step, (u_idx, v_idx, w_idx) in enumerate(_strassen_block_indices(k0, k1)):
            r = b * 7 + step
            for u in u_idx:
                U[u][r] = 1
            for v in v_idx:
                V[v][r] = 1
            for w in w_idx:
                W[w][r] = 1
            # Apply signs (the Strassen formulas have some -1 coefficients)
            # We hardcode the signs here
            if step == 1: W[w_idx[1]][r] = -1
            elif step == 2: V[v_idx[1]][r] = -1
            elif step == 3: V[v_idx[1]][r] = -1
            elif step == 4: W[w_idx[0]][r] = -1
            elif step == 5: U[u_idx[1]][r] = -1
            elif step == 6: U[u_idx[1]][r] = -1
    return U, V, W

def _quick_verify(U, V, W, n, m, p, num_trials=3):
    """Quick verification with early termination."""
    for _ in range(num_trials):
        A = [[random.randint(-3,3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3,3) for _ in range(p)] for _ in range(m)]
        C_true = [[sum(A[i][j]*B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        C_test = [[0]*p for _ in range(n)]
        rank = len(U[0])
        for r in range(rank):
            u_val = sum(U[i*m+j][r]*A[i][j] for i in range(n) for j in range(m))
            v_val = sum(V[j*p+k][r]*B[j][k] for j in range(m) for k in range(p))
            if u_val == 0 or v_val == 0:
                continue
            for i in range(n):
                for k in range(p):
                    C_test[i][k] += u_val * v_val * W[k*n+i][r]
        for i in range(n):
            for k in range(p):
                if C_test[i][k] != C_true[i][k]:
                    return False
    return True

def _try_fast_rank_reduction(n, m, p, U, V, W, current_rank, max_drops=3):
    """
    Fast rank reduction: iteratively try dropping one column at a time.
    Uses quick verification to stay well within time limits.
    """
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_rank = current_rank
    
    for _ in range(max_drops):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        dropped_any = False
        for col_to_drop in range(best_rank):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            
            # Build candidate without this column
            U_cand = [[U[r][c] for c in range(best_rank) if c != col_to_drop] for r in range(n*m)]
            V_cand = [[V[r][c] for c in range(best_rank) if c != col_to_drop] for r in range(m*p)]
            W_cand = [[W[r][c] for c in range(best_rank) if c != col_to_drop] for r in range(p*n)]
            
            # Quick verification
            valid = _quick_verify(U_cand, V_cand, W_cand, n, m, p, num_trials=3)
            
            if valid:
                # Confirm with more trials
                valid = _quick_verify(U_cand, V_cand, W_cand, n, m, p, num_trials=10)
                if valid:
                    U, V, W = U_cand, V_cand, W_cand
                    best_U, best_V, best_W = U_cand, V_cand, W_cand
                    best_rank -= 1
                    dropped_any = True
                    break
        
        if not dropped_any:
            break
    
    return best_U, best_V, best_W, best_rank

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    if time.time() - START_TIME > TIME_LIMIT:
        # Return the schoolbook decomposition as ultimate fallback
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
        return {"U": U, "V": V, "W": W}

    n, m, p = DIMENSIONS
    U, V, W = _get_strassen_decomposition(n, m, p)
    current_rank = len(U[0])

    # Attempt fast rank reduction
    if time.time() - START_TIME < TIME_LIMIT - 10:
        U, V, W, current_rank = _try_fast_rank_reduction(n, m, p, U, V, W, current_rank, max_drops=3)

    return {"U": U, "V": V, "W": W}

def _schoolbook_decomposition():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
