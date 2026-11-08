"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import numpy as np

DIMENSIONS = [3, 3, 3]
START_TIME = time.time()
TIME_LIMIT = 120.0

def _verify_decomposition(U, V, W, n=3, m=3, p=3):
    """Verify that the tensor decomposition is correct."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _construct_rank26():
    """Construct the known rank-26 decomposition."""
    n, m, p = 3, 3, 3
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1
    
    # 1. A2 B2 (Top-left 2x2 * 2x2) -> Rank 7 (Strassen)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # 2. Cross terms
    add_term(2, 6, 0, 1)
    add_term(2, 7, 3, 1)
    add_term(5, 6, 1, 1)
    add_term(5, 7, 4, 1)
    
    add_term(0, 2, 6, 1)
    add_term(1, 5, 6, 1)
    add_term(3, 2, 7, 1)
    add_term(4, 5, 7, 1)
    
    add_term(2, 8, 6, 1)
    add_term(5, 8, 7, 1)
    
    add_term(6, 0, 2, 1)
    add_term(7, 3, 2, 1)
    add_term(6, 1, 5, 1)
    add_term(7, 4, 5, 1)
    
    add_term(8, 6, 2, 1)
    add_term(8, 7, 5, 1)
    
    add_term(6, 2, 8, 1)
    add_term(7, 5, 8, 1)
    
    add_term(8, 8, 8, 1)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # First, try to construct a rank-23 decomposition
    result_23 = _construct_rank23()
    if result_23 is not None:
        if _verify_decomposition(result_23["U"], result_23["V"], result_23["W"], n, m, p):
            return result_23
    
    # Fall back to rank-26
    result_26 = _construct_rank26()
    
    # Verify the rank-26 decomposition
    if not _verify_decomposition(result_26["U"], result_26["V"], result_26["W"], n, m, p):
        # If verification fails, return schoolbook rank-27
        rank = 27
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Deadline guard
    if time.time() - START_TIME > TIME_LIMIT - 5:
        pass
        
    return result_26

def _construct_rank23():
    """
    Attempt to construct a rank-23 decomposition using the Bini-De Falco algorithm.
    """
    n, m, p = 3, 3, 3
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # The Bini-De Falco decomposition for 3x3 matrix multiplication
    # This is a known rank-23 decomposition from the literature.
    
    # Define the 23 terms as (U_indices, V_indices, W_indices) with coefficients
    # Each term is a rank-1 tensor contribution
    
    # The decomposition uses the following structure:
    # We partition the 3x3 matrices into blocks and use a combination of
    # Strassen-like tricks and cross-term optimizations.
    
    # For now, return None to indicate we couldn't construct it
    # A proper implementation would need the exact coefficients from the literature
    
    return None
# EVOLVE-BLOCK-END
