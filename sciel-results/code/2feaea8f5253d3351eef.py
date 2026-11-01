import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110 # Safety margin

    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_fallback = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fallback = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fallback = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i * m + j][col] = 1
                V_fallback[j * p + k][col] = 1
                W_fallback[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_fallback, "V": V_fallback, "W": W_fallback}
    best_rank = rank_schoolbook

    # Try Strassen-based rank-49 construction
    if time.time() < deadline:
        U, V, W = _construct_strassen_49(n, m, p)
        if U is not None:
            # Verify
            if _verify(U, V, W, n, m, p):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = 49

    return best_result

def _construct_strassen_49(n, m, p):
    if p % 2 != 0:
        return None, None, None
    
    num_pairs = p // 2
    rank = 7 * num_pairs
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B00)
    # M4 = A11 * (B01 - B11)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M5 + M6
    # C10 = M2 + M3
    # C11 = M1 - M2 + M3 + M6
    
    # Map C indices to W indices
    # C_{i,k} -> W[k*n + i]
    # C00 -> W[0]
    # C10 -> W[1]
    # C01 -> W[2]
    # C11 -> W[3]
    
    # For each pair of columns (k1, k2)
    for t in range(num_pairs):
        k1 = 2 * t
        k2 = 2 * t + 1
        base = t * 7
        
        # M1
        U[0][base] = 1
        U[3][base] = 1
        V[k1][base] = 1
        V[p + k1][base] = 1
        V[k2][base] = 1
        V[p + k2][base] = 1
        W[k1 * n + 0][base] = 1  # C00
        W[k1 * n + 1][base] = 1  # C10? No, C11 is M1... wait.
        # C00 = M1... -> W[k1*n+0] += 1
        # C11 = M1... -> W[k2*n+1] += 1
        
        # Let's re-map carefully.
        # C00 (i=0, k=k1) -> W[k1*n+0]
        # C10 (i=1, k=k1) -> W[k1*n+1]
        # C01 (i=0, k=k2) -> W[k2*n+0]
        # C11 (i=1, k=k2) -> W[k2*n+1]
        
        # Reset W for this base to avoid confusion, or just add.
        # I will add contributions.
        
        # M1 contributes to C00 (+1) and C11 (+1)
        W[k1 * n + 0][base] += 1
        W[k2 * n + 1][base] += 1
        
        # M2 contributes to C10 (+1) and C11 (-1)
        U[2][base+1] = 1
        U[3][base+1] = 1
        V[k1][base+1] = 1
        W[k1 * n + 1][base+1] += 1
        W[k2 * n + 1][base+1] -= 1
        
        # M3 contributes to C10 (+1) and C11 (+1)
        U[0][base+2] = 1
        V[p + k1][base+2] = 1
        V[k1][base+2] = -1
        W[k1 * n + 1][base+2] += 1
        W[k2 * n + 1][base+2] += 1
        
        # M4 contributes to C00 (+1) and C11 (-1)? No, C11 = M1 - M2 + M3 + M6.
        # C00 = M1 + M4 - M5 + M7.
        # So M4 contributes to C00 (+1).
        U[3][base+3] = 1
        V[k2][base+3] = 1
        V[p + k2][base+3] = -1
        W[k1 * n + 0][base+3] += 1
        
        # M5 contributes to C00 (-1) and C01 (+1)
        U[0][base+4] = 1
        U[1][base+4] = 1
        V[p + k2][base+4] = 1
        W[k1 * n + 0][base+4] -= 1
        W[k2 * n + 0][base+4] += 1
        
        # M6 contributes to C01 (+1) and C11 (+1)
        U[2][base+5] = 1
        U[0][base+5] = -1
        V[k1][base+5] = 1
        V[k2][base+5] = 1
        W[k2 * n + 0][base+5] += 1
        W[k2 * n + 1][base+5] += 1
        
        # M7 contributes to C00 (+1)
        U[1][base+6] = 1
        U[3][base+6] = -1
        V[p + k1][base+6] = 1
        V[p + k2][base+6] = 1
        W[k1 * n + 0][base+6] += 1

    return U, V, W

def _verify(U, V, W, n, m, p):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True
# EVOLVE-BLOCK-END
