"""Matrix multiplication 2x2, 2x7 exact tensor decomposition search."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 7]

def _verify_fast(U, V, W, n, m, p):
    """Verify using numpy for speed."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = U[i*m+j] @ (V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(len(U[0])):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _get_strassen_decomp():
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, -1],
        [0, 1, 0, 0, 0, -1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    return U, V, W

def _get_schoolbook_1col_decomp():
    U = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    W = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    return U, V, W

def _construct_rank_25():
    n, m, p = DIMENSIONS
    R = 25
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    StrU, StrV, StrW = _get_strassen_decomp()
    SchU, SchV, SchW = _get_schoolbook_1col_decomp()
    
    col_offset = 0
    # Block 1: k=0,1
    for r in range(7):
        for i in range(4): U[i][col_offset + r] = StrU[i][r]
        V[0][col_offset + r] = StrV[0][r]
        V[1][col_offset + r] = StrV[1][r]
        V[7][col_offset + r] = StrV[2][r]
        V[8][col_offset + r] = StrV[3][r]
        W[0][col_offset + r] = StrW[0][r]
        W[1][col_offset + r] = StrW[1][r]
        W[2][col_offset + r] = StrW[2][r]
        W[3][col_offset + r] = StrW[3][r]
    col_offset += 7

    # Block 2: k=2,3
    for r in range(7):
        for i in range(4): U[i][col_offset + r] = StrU[i][r]
        V[2][col_offset + r] = StrV[0][r]
        V[3][col_offset + r] = StrV[1][r]
        V[9][col_offset + r] = StrV[2][r]
        V[10][col_offset + r] = StrV[3][r]
        W[4][col_offset + r] = StrW[0][r]
        W[5][col_offset + r] = StrW[1][r]
        W[6][col_offset + r] = StrW[2][r]
        W[7][col_offset + r] = StrW[3][r]
    col_offset += 7

    # Block 3: k=4,5
    for r in range(7):
        for i in range(4): U[i][col_offset + r] = StrU[i][r]
        V[4][col_offset + r] = StrV[0][r]
        V[5][col_offset + r] = StrV[1][r]
        V[11][col_offset + r] = StrV[2][r]
        V[12][col_offset + r] = StrV[3][r]
        W[8][col_offset + r] = StrW[0][r]
        W[9][col_offset + r] = StrW[1][r]
        W[10][col_offset + r] = StrW[2][r]
        W[11][col_offset + r] = StrW[3][r]
    col_offset += 7

    # Block 4: k=6
    for r in range(4):
        for i in range(4): U[i][col_offset + r] = SchU[i][r]
        V[6][col_offset + r] = SchV[0][r]
        V[13][col_offset + r] = SchV[1][r]
        W[12][col_offset + r] = SchW[0][r]
        W[13][col_offset + r] = SchW[1][r]
    col_offset += 4

    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return _build_schoolbook()

def _random_local_search_rank(R, U_init, V_init, W_init, n, m, p, time_budget):
    """Random local search to try to reduce rank or find valid decomposition."""
    start = time.time()
    deadline = start + time_budget
    best_rank = R
    best_result = {"U": [row[:] for row in U_init], "V": [row[:] for row in V_init], "W": [row[:] for row in W_init]}
    
    # Try to find a valid rank-R decomposition by perturbing
    U = np.array(U_init, dtype=float)
    V = np.array(V_init, dtype=float)
    W = np.array(W_init, dtype=float)
    
    # First verify initial
    if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        best_rank = R
    else:
        # Find a valid one first via small perturbations
        for attempt in range(100):
            if time.time() > deadline:
                break
            U_p = U + np.random.randn(*U.shape) * 0.01
            V_p = V + np.random.randn(*V.shape) * 0.01
            W_p = W + np.random.randn(*W.shape) * 0.01
            if _verify_fast(U_p.tolist(), V_p.tolist(), W_p.tolist(), n, m, p):
                U, V, W = U_p, V_p, W_p
                best_rank = R
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
    
    if best_rank > R or time.time() > deadline:
        return best_result, best_rank
    
    # Now try to reduce rank by zeroing columns
    for target_rank in range(R - 1, 20, -1):
        if time.time() > deadline:
            break
        success = False
        for attempt in range(200):
            if time.time() > deadline:
                break
            # Try zeroing a random column
            U_try = U.copy()
            V_try = V.copy()
            W_try = W.copy()
            
            # Zero out a random column
            col_to_zero = random.randint(0, R - 1)
            U_try[:, col_to_zero] = 0
            V_try[:, col_to_zero] = 0
            W_try[:, col_to_zero] = 0
            
            # Try to fix by adjusting other columns slightly
            # This is a heuristic - in reality we need exact solutions
            # For now, just check if zeroing works (unlikely)
            if _verify_fast(U_try.tolist(), V_try.tolist(), W_try.tolist(), n, m, p):
                # Trim to target_rank
                U_trim = np.delete(U_try, col_to_zero, axis=1)
                V_trim = np.delete(V_try, col_to_zero, axis=1)
                W_trim = np.delete(W_try, col_to_zero, axis=1)
                best_rank = target_rank
                best_result = {"U": U_trim.tolist(), "V": V_trim.tolist(), "W": W_trim.tolist()}
                success = True
                break
        
        if success:
            U = np.array(best_result["U"])
            V = np.array(best_result["V"])
            W = np.array(best_result["W"])
            continue
    
    return best_result, best_rank


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Primary: Rank 25
    res = _construct_rank_25()
    best_rank = len(res['U'][0])
    
    # Random local search with time budget
    remaining = deadline - time.time()
    if remaining > 10:
        time_budget = max(0, remaining - 10)
        U_init = res['U']
        V_init = res['V']
        W_init = res['W']
        
        new_res, new_rank = _random_local_search_rank(best_rank, U_init, V_init, W_init, n, m, p, time_budget)
        
        if new_rank < best_rank and _verify_fast(new_res['U'], new_res['V'], new_res['W'], n, m, p):
            res = new_res
            best_rank = new_rank

    return res

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
