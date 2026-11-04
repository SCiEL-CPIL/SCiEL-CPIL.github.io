import time
import numpy as np

DIMENSIONS = [2, 4, 6]
N, M, P = 2, 4, 6
NM = N * M  # 8
MP = M * P  # 24
PN = P * N  # 12

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x6 matrix.
    
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Strassen coefficients for 2x2 by 2x2
    # Each term is (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict {(i, j): coeff}
    # V_coeffs: dict {(j, k): coeff}
    # W_coeffs: dict {(i, k): coeff}
    
    strassen_terms = [
        # M1 = (a+d)(e+h)
        ({(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1}),
        
        # M2 = (c+d)e
        ({(1,0): 1, (1,1): 1},
         {(0,0): 1},
         {(1,0): 1, (1,1): -1}),
        
        # M3 = a(f-h)
        ({(0,0): 1},
         {(0,1): 1, (1,1): -1},
         {(0,1): 1, (1,1): 1}),
        
        # M4 = d(g-e)
        ({(1,1): 1},
         {(1,0): 1, (0,0): -1},
         {(0,0): 1, (1,0): 1}),
        
        # M5 = (a+b)h
        ({(0,0): 1, (0,1): 1},
         {(1,1): 1},
         {(0,0): -1, (0,1): 1}),
        
        # M6 = (c-a)(e+f)
        ({(1,0): 1, (0,0): -1},
         {(0,0): 1, (0,1): 1},
         {(1,1): 1}),
        
        # M7 = (b-d)(g+h)
        ({(0,1): 1, (1,1): -1},
         {(1,0): 1, (1,1): 1},
         {(0,0): 1}),
    ]
    
    # Try to find a lower rank decomposition
    # First, try rank 42 (baseline Strassen)
    best_result = None
    best_rank = 42
    
    # Build the rank-42 decomposition
    U42 = [[0] * 42 for _ in range(NM)]
    V42 = [[0] * 42 for _ in range(MP)]
    W42 = [[0] * 42 for _ in range(PN)]
    
    col = 0
    # Blocks: (a_row_off, a_col_off, b_row_off, b_col_off, c_col_off)
    # A is 2x4: split into left (cols 0,1) and right (cols 2,3)
    # B is 4x6: split into top (rows 0,1) and bottom (rows 2,3), each with 3 col blocks of width 2
    # C is 2x6
    
    blocks = [
        # (a_row, a_col, b_row, b_col, c_col)
        (0, 0, 0, 0, 0),  # A_left, B_top, C_col0
        (0, 0, 0, 2, 2),  # A_left, B_top, C_col1
        (0, 0, 0, 4, 4),  # A_left, B_top, C_col2
        (0, 2, 2, 0, 0),  # A_right, B_bottom, C_col0
        (0, 2, 2, 2, 2),  # A_right, B_bottom, C_col1
        (0, 2, 2, 4, 4),  # A_right, B_bottom, C_col2
    ]
    
    for a_row, a_col, b_row, b_col, c_col in blocks:
        for u_dict, v_dict, w_dict in strassen_terms:
            if col >= 42:
                break
                
            for (i, j), val in u_dict.items():
                u_idx = (a_row + i) * M + (a_col + j)
                U42[u_idx][col] += val
                
            for (j, k), val in v_dict.items():
                v_idx = (b_row + j) * P + (b_col + k)
                V42[v_idx][col] += val
                
            for (i, k), val in w_dict.items():
                # W index: k*n + i (official trace convention)
                w_idx = (c_col + k) * N + (a_row + i)
                W42[w_idx][col] += val
                
            col += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    best_result = {"U": U42, "V": V42, "W": W42}
    best_rank = 42
    
    # Verify the rank-42 decomposition
    if _verify_decomposition(best_result, N, M, P, trials=20):
        pass  # Good
    else:
        # Fallback to schoolbook if Strassen fails
        best_result = _schoolbook_decomposition()
        best_rank = M
    
    # Try to find lower rank decompositions
    # Try rank 40, 38, 36, ... down to some minimum
    for target_rank in range(40, 24, -2):
        if time.time() > deadline - 10:
            break
        
        # Try to find a rank-target_rank decomposition
        result = _try_find_decomposition(target_rank, strassen_terms, deadline)
        if result is not None:
            if _verify_decomposition(result, N, M, P, trials=20):
                best_result = result
                best_rank = target_rank
                break
    
    return best_result

def _verify_decomposition(result, n, m, p, trials=10):
    """Verify a decomposition numerically with integer arithmetic."""
    U, V, W = result['U'], result['V'], result['W']
    rank = len(U[0])
    
    import random
    random.seed(42)
    
    for trial in range(trials):
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        
        # Expected C
        C_exp = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        # Computed C
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

def _schoolbook_decomposition():
    """Fallback: rank-m = 4 decomposition."""
    n, m, p = 2, 4, 6
    rank = m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for i in range(n):
            U[i * m + j][j] = 1
        for k in range(p):
            V[j * p + k][j] = 1
        for i in range(n):
            for k in range(p):
                W[k * n + i][j] = 1
    
    return {"U": U, "V": V, "W": W}

def _try_find_decomposition(target_rank, strassen_terms, deadline):
    """
    Try to find a rank-target_rank decomposition.
    This is a heuristic search.
    """
    n, m, p = 2, 4, 6
    
    # For now, just return None - we don't have a good search algorithm
    # In a more complete solution, we could use:
    # 1. Random initialization + local search
    # 2. Integer linear programming
    # 3. Tensor decomposition algorithms
    
    return None
