import numpy as np
import time
import random

DIMENSIONS = [2, 4, 6]

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 * 2x2 matrix multiplication.
    Each term is a tuple (U_dict, V_dict, W_dict) with integer coefficients.
    """
    terms = []
    # M1 = (P00+P11)*(Q00+Q11)
    terms.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    ))
    # M2 = (P10+P11)*Q00
    terms.append((
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1, (1,1): -1}
    ))
    # M3 = P00*(Q01-Q11)
    terms.append((
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1}
    ))
    # M4 = P11*(Q10-Q00)
    terms.append((
        {(1,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (1,0): 1}
    ))
    # M5 = (P00+P01)*Q11
    terms.append((
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,0): -1, (0,1): 1}
    ))
    # M6 = (P10-P00)*(Q00+Q01)
    terms.append((
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1}
    ))
    # M7 = (P01-P11)*(Q10+Q11)
    terms.append((
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1}
    ))
    return terms

def _linear_1x2x3_terms():
    """
    Return the 6 terms for 1x2 * 2x3 matrix multiplication.
    Rank 6 decomposition.
    """
    terms = []
    for k in range(3):
        # Term 2k: A[0] * B[0, k] -> C[k]
        terms.append((
            {(0,): 1},  # U: index (i_L, j_L) where i_L=0, j_L=0
            {(0, k): 1}, # V: index (j_L, k_L) where j_L=0, k_L=k
            {(k,): 1}    # W: index (k_L, i_L) where k_L=k, i_L=0
        ))
        # Term 2k+1: A[1] * B[1, k] -> C[k]
        terms.append((
            {(1,): 1},  # U: index (i_L, j_L) where i_L=0, j_L=1
            {(1, k): 1}, # V: index (j_L, k_L) where j_L=1, k_L=k
            {(k,): 1}    # W: index (k_L, i_L) where k_L=k, i_L=0
        ))
    return terms

def _verify_decomposition(result, n=2, m=4, p=6, trials=50):
    """Verify a decomposition numerically with integer arithmetic."""
    U, V, W = result['U'], result['V'], result['W']
    rank = len(U[0])
    for trial in range(trials):
        A = [[random.randint(-10, 10) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-10, 10) for _ in range(p)] for _ in range(m)]
        C_exp = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        C_comp = [[0] * p for _ in range(n)]
        for col in range(rank):
            u_val = sum(U[i * m + j][col] * A[i][j] for i in range(n) for j in range(m))
            v_val = sum(V[j * p + k][col] * B[j][k] for j in range(m) for k in range(p))
            for i in range(n):
                for k in range(p):
                    C_comp[i][k] += W[k * n + i][col] * u_val * v_val
        if C_comp != C_exp:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)] # 8 x 42
    V = [[0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0] * rank for _ in range(p * n)] # 12 x 42

    strassen_terms = _strassen_2x2x2_terms()
    linear_terms = _linear_1x2x3_terms()

    for rs, (U_s, V_s, W_s) in enumerate(strassen_terms):
        for rl, (U_l, V_l, W_l) in enumerate(linear_terms):
            r = rs * 6 + rl
            
            # Fill U
            for (i_s, j_s), coeff_s in U_s.items():
                for (i_l, j_l), coeff_l in U_l.items():
                    # U_total index: i_s * 1 + i_l (Row), j_s * 2 + j_l (Col)
                    # Wait, U is for A. A is 2x4.
                    # A_total = A_S (2x2) x A_L (1x2).
                    # Row i = i_s * 1 + i_l.
                    # Col j = j_s * 2 + j_l.
                    u_idx = (i_s * 1 + i_l) * m + (j_s * 2 + j_l)
                    U[u_idx][r] += coeff_s * coeff_l
            
            # Fill V
            for (j_s, k_s), coeff_s in V_s.items():
                for (j_l, k_l), coeff_l in V_l.items():
                    # V is for B. B is 4x6.
                    # B_total = B_S (2x2) x B_L (2x3).
                    # Row j = j_s * 2 + j_l.
                    # Col k = k_s * 3 + k_l.
                    v_idx = (j_s * 2 + j_l) * p + (k_s * 3 + k_l)
                    V[v_idx][r] += coeff_s * coeff_l
            
            # Fill W
            for (k_s, i_s), coeff_s in W_s.items():
                for (k_l, i_l), coeff_l in W_l.items():
                    # W is for C. C is 2x6.
                    # C_total = C_S (2x2) x C_L (1x3).
                    # Row i = i_s * 1 + i_l.
                    # Col k = k_s * 3 + k_l.
                    # Interface W index: k * n + i.
                    w_idx = (k_s * 3 + k_l) * n + (i_s * 1 + i_l)
                    W[w_idx][r] += coeff_s * coeff_l

    result = {"U": U, "V": V, "W": W}
    
    if _verify_decomposition(result, n, m, p, trials=20):
        return result
    
    # Fallback to schoolbook if verification fails (should not happen)
    rank_sb = 48
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U_sb[i * m + j][r] = 1
                V_sb[j * p + k][r] = 1
                W_sb[k * n + i][r] = 1
    return {"U": U_sb, "V": V_sb, "W": W_sb}
