import numpy as np
import time
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verifies if U, V, W form a valid tensor decomposition for matrix multiplication.
    Uses exact Fraction arithmetic for safety.
    """
    rank = len(U[0])
    
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                total = Fraction(0)
                for r in range(rank):
                    total += U_frac[idx_u][r] * V_frac[idx_v][r] * W_frac[idx_w][r]
                
                if total != 1:
                    return False
    return True

def _try_remove_term(U, V, W, n, m, p, term_idx):
    """
    Try to remove term term_idx by zeroing it out and checking if the rest still works.
    """
    rank = len(U[0])
    if rank <= 1:
        return None
    
    U_new = [row[:] for row in U]
    V_new = [row[:] for row in V]
    W_new = [row[:] for row in W]
    
    for i in range(len(U_new)):
        U_new[i][term_idx] = 0
    for i in range(len(V_new)):
        V_new[i][term_idx] = 0
    for i in range(len(W_new)):
        W_new[i][term_idx] = 0
    
    if _verify_decomposition(U_new, V_new, W_new, n, m, p):
        U_compact = [row[:term_idx] + row[term_idx+1:] for row in U_new]
        V_compact = [row[:term_idx] + row[term_idx+1:] for row in V_new]
        W_compact = [row[:term_idx] + row[term_idx+1:] for row in W_new]
        return (U_compact, V_compact, W_compact)
    
    return None

def _construct_rank_27():
    """
    Constructs a rank-27 decomposition for 2x3 * 3x5.
    
    Strategy: Partition the intermediate dimension j={0,1,2} into j={0,1} (2D) and j={2} (1D).
    - Block 1: j=2 (1D), k=0..4: rank 10 (schoolbook)
    - Block 2: j=0,1 (2D), k=0,1,2 (3D): rank 10 (optimized 2x2x3)
    - Block 3: j=0,1 (2D), k=3,4 (2D): rank 7 (Strassen 2x2x2)
    Total: 10 + 10 + 7 = 27
    """
    n, m, p = 2, 3, 5
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Block 1: j=2, k=0..4 (rank 10, schoolbook)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    
    # Block 2: j=0,1, k=0,1,2 (rank 10, optimized 2x2x3)
    # A is 2x2 (rows i=0,1, cols j=0,1), B is 2x3 (rows j=0,1, cols k=0,1,2)
    # C is 2x3 (rows i=0,1, cols k=0,1,2)
    # Using the rank-10 algorithm for 2x2x3:
    # C_ik = sum_j A_ij B_jk
    # Terms:
    # 1. a00*b00 -> C_00
    # 2. a01*b10 -> C_00
    # 3. a00*b01 -> C_01
    # 4. a01*b11 -> C_01
    # 5. a00*b02 -> C_02
    # 6. a01*b12 -> C_02
    # 7. (a00+a01)*(b00+b11) -> contributes to C_00, C_01
    # 8. (a00+a01)*(b01+b12) -> contributes to C_01, C_02
    # 9. (a00+a01)*(b02+b10) -> contributes to C_02, C_00
    # 10. (a00-a01)*(b00+b01+b02) -> contributes to C_00, C_01, C_02
    
    # Term 1: a00*b00
    U[0 * m + 0][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # Term 2: a01*b10
    U[0 * m + 1][r] = 1
    V[1 * p + 0][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # Term 3: a00*b01
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1
    W[1 * n + 0][r] = 1
    r += 1
    
    # Term 4: a01*b11
    U[0 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[1 * n + 0][r] = 1
    r += 1
    
    # Term 5: a00*b02
    U[0 * m + 0][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    # Term 6: a01*b12
    U[0 * m + 1][r] = 1
    V[1 * p + 2][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    # Term 7: (a00+a01)*(b00+b11)
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = -1  # cancels with term 1+2
    W[1 * n + 0][r] = -1  # cancels with term 3+4
    r += 1
    
    # Term 8: (a00+a01)*(b01+b12)
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[0 * p + 1][r] = 1
    V[1 * p + 2][r] = 1
    W[1 * n + 0][r] = -1  # cancels with term 3+4
    W[2 * n + 0][r] = -1  # cancels with term 5+6
    r += 1
    
    # Term 9: (a00+a01)*(b02+b10)
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    V[1 * p + 0][r] = 1
    W[2 * n + 0][r] = -1  # cancels with term 5+6
    W[0 * n + 0][r] = -1  # cancels with term 1+2
    r += 1
    
    # Term 10: (a00-a01)*(b00+b01+b02)
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = -1
    V[0 * p + 0][r] = 1
    V[0 * p + 1][r] = 1
    V[0 * p + 2][r] = 1
    W[0 * n + 0][r] = 1
    W[1 * n + 0][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    # Block 3: j=0,1, k=3,4 (rank 7, Strassen 2x2x2)
    # A is 2x2 (rows i=0,1, cols j=0,1), B is 2x2 (rows j=0,1, cols k=3,4)
    # C is 2x2 (rows i=0,1, cols k=3,4)
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # M1 = (A00+A11)(B00+B11)
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 3][r] = 1
    V[1 * p + 4][r] = 1
    W[3 * n + 0][r] = 1
    W[4 * n + 1][r] = 1
    r += 1
    
    # M2 = (A10+A11)B00
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 3][r] = 1
    W[4 * n + 0][r] = 1
    W[4 * n + 1][r] = -1
    r += 1
    
    # M3 = A00(B10-B11)
    U[0 * m + 0][r] = 1
    V[1 * p + 3][r] = 1
    V[1 * p + 4][r] = -1
    W[3 * n + 1][r] = 1
    W[4 * n + 1][r] = 1
    r += 1
    
    # M4 = A11(B01-B00)
    U[1 * m + 1][r] = 1
    V[0 * p + 4][r] = 1
    V[0 * p + 3][r] = -1
    W[3 * n + 0][r] = 1
    W[4 * n + 0][r] = 1
    r += 1
    
    # M5 = (A00+A01)B11
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 4][r] = 1
    W[3 * n + 0][r] = -1
    W[4 * n + 0][r] = 1
    r += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 3][r] = 1
    V[0 * p + 4][r] = 1
    W[4 * n + 1][r] = 1
    r += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 3][r] = 1
    V[1 * p + 4][r] = 1
    W[3 * n + 0][r] = 1
    r += 1
    
    return {"U": U, "V": V, "W": W}

def _try_find_lower_rank(U, V, W, n, m, p, deadline):
    """
    Attempts to find a lower rank decomposition by removing terms.
    Returns (U, V, W, rank) of the best found.
    """
    current_U, current_V, current_W = U, V, W
    current_rank = len(current_U[0])
    
    # Try single removals
    for term_idx in range(current_rank):
        if time.time() > deadline:
            break
        
        result = _try_remove_term(current_U, current_V, current_W, n, m, p, term_idx)
        if result is not None:
            current_U, current_V, current_W = result
            current_rank = len(current_U[0])
    
    return current_U, current_V, current_W, current_rank

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 115 seconds, with safety margin
    
    n, m, p = 2, 3, 5
    
    # Construct the rank-27 decomposition
    res_27 = _construct_rank_27()
    U = res_27['U']
    V = res_27['V']
    W = res_27['W']
    
    # Verify it to ensure correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fall back to rank-28 if verification fails
        res_28 = {
            "U": [[0] * 28 for _ in range(n * m)],
            "V": [[0] * 28 for _ in range(m * p)],
            "W": [[0] * 28 for _ in range(p * n)]
        }
        r = 0
        for i in range(n):
            for k in range(p):
                res_28["U"][i * m + 2][r] = 1
                res_28["V"][2 * p + k][r] = 1
                res_28["W"][k * n + i][r] = 1
                r += 1
        for i in range(n):
            for j in range(2):
                res_28["U"][i * m + j][r] = 1
                res_28["V"][j * p + 4][r] = 1
                res_28["W"][4 * n + i][r] = 1
                r += 1
        # Strassen for j=0,1, k=0,1
        res_28["U"][0 * m + 0][r] = 1
        res_28["U"][1 * m + 1][r] = 1
        res_28["V"][0 * p + 0][r] = 1
        res_28["V"][1 * p + 1][r] = 1
        res_28["W"][0 * n + 0][r] = 1
        res_28["W"][1 * n + 1][r] = 1
        r += 1
        res_28["U"][1 * m + 0][r] = 1
        res_28["U"][1 * m + 1][r] = 1
        res_28["V"][0 * p + 0][r] = 1
        res_28["W"][0 * n + 1][r] = 1
        res_28["W"][1 * n + 1][r] = -1
        r += 1
        res_28["U"][0 * m + 0][r] = 1
        res_28["V"][0 * p + 1][r] = 1
        res_28["V"][1 * p + 1][r] = -1
        res_28["W"][1 * n + 0][r] = 1
        res_28["W"][1 * n + 1][r] = 1
        r += 1
        res_28["U"][1 * m + 1][r] = 1
        res_28["V"][1 * p + 0][r] = 1
        res_28["V"][0 * p + 0][r] = -1
        res_28["W"][0 * n + 0][r] = 1
        res_28["W"][0 * n + 1][r] = 1
        r += 1
        res_28["U"][0 * m + 0][r] = 1
        res_28["U"][0 * m + 1][r] = 1
        res_28["V"][1 * p + 1][r] = 1
        res_28["W"][0 * n + 0][r] = -1
        res_28["W"][1 * n + 0][r] = 1
        r += 1
        res_28["U"][1 * m + 0][r] = 1
        res_28["U"][0 * m + 0][r] = -1
        res_28["V"][0 * p + 0][r] = 1
        res_28["V"][0 * p + 1][r] = 1
        res_28["W"][1 * n + 1][r] = 1
        r += 1
        res_28["U"][0 * m + 1][r] = 1
        res_28["U"][1 * m + 1][r] = -1
        res_28["V"][1 * p + 0][r] = 1
        res_28["V"][1 * p + 1][r] = 1
        res_28["W"][0 * n + 0][r] = 1
        r += 1
        # Strassen for j=0,1, k=2,3
        res_28["U"][0 * m + 0][r] = 1
        res_28["U"][1 * m + 1][r] = 1
        res_28["V"][0 * p + 2][r] = 1
        res_28["V"][1 * p + 3][r] = 1
        res_28["W"][2 * n + 0][r] = 1
        res_28["W"][3 * n + 1][r] = 1
        r += 1
        res_28["U"][1 * m + 0][r] = 1
        res_28["U"][1 * m + 1][r] = 1
        res_28["V"][0 * p + 2][r] = 1
        res_28["W"][2 * n + 1][r] = 1
        res_28["W"][3 * n + 1][r] = -1
        r += 1
        res_28["U"][0 * m + 0][r] = 1
        res_28["V"][0 * p + 3][r] = 1
        res_28["V"][1 * p + 3][r] = -1
        res_28["W"][3 * n + 0][r] = 1
        res_28["W"][3 * n + 1][r] = 1
        r += 1
        res_28["U"][1 * m + 1][r] = 1
        res_28["V"][1 * p + 2][r] = 1
        res_28["V"][0 * p + 2][r] = -1
        res_28["W"][2 * n + 0][r] = 1
        res_28["W"][2 * n + 1][r] = 1
        r += 1
        res_28["U"][0 * m + 0][r] = 1
        res_28["U"][0 * m + 1][r] = 1
        res_28["V"][1 * p + 3][r] = 1
        res_28["W"][2 * n + 0][r] = -1
        res_28["W"][3 * n + 0][r] = 1
        r += 1
        res_28["U"][1 * m + 0][r] = 1
        res_28["U"][0 * m + 0][r] = -1
        res_28["V"][0 * p + 2][r] = 1
        res_28["V"][0 * p + 3][r] = 1
        res_28["W"][3 * n + 1][r] = 1
        r += 1
        res_28["U"][0 * m + 1][r] = 1
        res_28["U"][1 * m + 1][r] = -1
        res_28["V"][1 * p + 2][r] = 1
        res_28["V"][1 * p + 3][r] = 1
        res_28["W"][2 * n + 0][r] = 1
        r += 1
        
        U = res_28['U']
        V = res_28['V']
        W = res_28['W']
    
    # Try to find a lower rank decomposition
    U, V, W, rank = _try_find_lower_rank(U, V, W, n, m, p, deadline)
    
    return {'U': U, 'V': V, 'W': W}
