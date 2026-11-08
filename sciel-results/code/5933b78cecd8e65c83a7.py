import time
import sys

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0

def _check_deadline():
    return (time.time() - _start_time) > (_DEADLINE - _SAFETY_MARGIN)

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly represents matrix multiplication."""
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    # Build the multiplication tensor T[i,j,k] = delta
    # T should equal sum_r U[i,r] * V[j,r] * W[k,r]
    # where i indexes (i_row, i_col) of A, j indexes (j_row, j_col) of B, k indexes (k_row, k_col) of C
    # A is n x m, B is m x p, C is n x p
    # i = i_row * m + i_col, j = j_row * p + j_col, k = k_row * n + k_col
    # T[i,j,k] = 1 if i_col == j_row else 0
    
    for i_row in range(n):
        for i_col in range(m):
            for j_row in range(m):
                for j_col in range(p):
                    for k_row in range(n):
                        for k_col in range(p):
                            # Expected: 1 if i_col == j_row and i_row == k_row and j_col == k_col
                            expected = 1.0 if (i_col == j_row and i_row == k_row and j_col == k_col) else 0.0
                            
                            i = i_row * m + i_col
                            j = j_row * p + j_col
                            k = k_row * n + k_col
                            
                            # Compute sum_r U[i,r] * V[j,r] * W[k,r]
                            actual = 0.0
                            for r in range(rank):
                                actual += U_arr[i, r] * V_arr[j, r] * W_arr[k, r]
                            
                            if abs(actual - expected) > 1e-9:
                                return False
    return True

def construct():
    if _check_deadline():
        # Fallback: return naive rank-30 decomposition
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    n, m, p = DIMENSIONS
    
    # First, build the rank-28 decomposition and verify it
    def build_rank28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        
        # Part 1: A2 B2 (j=2, k=0..4)
        for i in range(n):
            for k in range(p):
                U[i * m + 2][r] = 1
                V[2 * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
                
        # Part 2: A1 B13 (j=0,1, k=4)
        for i in range(n):
            for j in range(2):
                U[i * m + j][r] = 1
                V[j * p + 4][r] = 1
                W[4 * n + i][r] = 1
                r += 1
                
        # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen
        # M1
        U[0 * m + 0][r] = 1
        U[1 * m + 1][r] = 1
        V[0 * p + 0][r] = 1
        V[1 * p + 1][r] = 1
        W[0 * n + 0][r] = 1
        W[1 * n + 1][r] = 1
        r += 1
        
        # M2
        U[1 * m + 0][r] = 1
        U[1 * m + 1][r] = 1
        V[0 * p + 0][r] = 1
        W[0 * n + 1][r] = 1
        W[1 * n + 1][r] = -1
        r += 1
        
        # M3
        U[0 * m + 0][r] = 1
        V[0 * p + 1][r] = 1
        V[1 * p + 1][r] = -1
        W[1 * n + 0][r] = 1
        W[1 * n + 1][r] = 1
        r += 1
        
        # M4
        U[1 * m + 1][r] = 1
        V[1 * p + 0][r] = 1
        V[0 * p + 0][r] = -1
        W[0 * n + 0][r] = 1
        W[0 * n + 1][r] = 1
        r += 1
        
        # M5
        U[0 * m + 0][r] = 1
        U[0 * m + 1][r] = 1
        V[1 * p + 1][r] = 1
        W[0 * n + 0][r] = -1
        W[1 * n + 0][r] = 1
        r += 1
        
        # M6
        U[1 * m + 0][r] = 1
        U[0 * m + 0][r] = -1
        V[0 * p + 0][r] = 1
        V[0 * p + 1][r] = 1
        W[1 * n + 1][r] = 1
        r += 1
        
        # M7
        U[0 * m + 1][r] = 1
        U[1 * m + 1][r] = -1
        V[1 * p + 0][r] = 1
        V[1 * p + 1][r] = 1
        W[0 * n + 0][r] = 1
        r += 1
        
        # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen
        # M1
        U[0 * m + 0][r] = 1
        U[1 * m + 1][r] = 1
        V[0 * p + 2][r] = 1
        V[1 * p + 3][r] = 1
        W[2 * n + 0][r] = 1
        W[3 * n + 1][r] = 1
        r += 1
        
        # M2
        U[1 * m + 0][r] = 1
        U[1 * m + 1][r] = 1
        V[0 * p + 2][r] = 1
        W[2 * n + 1][r] = 1
        W[3 * n + 1][r] = -1
        r += 1
        
        # M3
        U[0 * m + 0][r] = 1
        V[0 * p + 3][r] = 1
        V[1 * p + 3][r] = -1
        W[3 * n + 0][r] = 1
        W[3 * n + 1][r] = 1
        r += 1
        
        # M4
        U[1 * m + 1][r] = 1
        V[1 * p + 2][r] = 1
        V[0 * p + 2][r] = -1
        W[2 * n + 0][r] = 1
        W[2 * n + 1][r] = 1
        r += 1
        
        # M5
        U[0 * m + 0][r] = 1
        U[0 * m + 1][r] = 1
        V[1 * p + 3][r] = 1
        W[2 * n + 0][r] = -1
        W[3 * n + 0][r] = 1
        r += 1
        
        # M6
        U[1 * m + 0][r] = 1
        U[0 * m + 0][r] = -1
        V[0 * p + 2][r] = 1
        V[0 * p + 3][r] = 1
        W[3 * n + 1][r] = 1
        r += 1
        
        # M7
        U[0 * m + 1][r] = 1
        U[1 * m + 1][r] = -1
        V[1 * p + 2][r] = 1
        V[1 * p + 3][r] = 1
        W[2 * n + 0][r] = 1
        r += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Build and verify rank-28
    result = build_rank28()
    
    # Verify correctness
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # If verification fails, fall back to naive
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    return result
# EVOLVE-BLOCK-END
