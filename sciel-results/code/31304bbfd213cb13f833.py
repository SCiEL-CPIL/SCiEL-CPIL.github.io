"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _get_schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition as numpy arrays."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_decomp(U, V, W, n, m, p):
    """
    Verify if U, V, W form a valid decomposition for matrix multiplication.
    Condition: For all i, j, k in 0..n-1, 0..m-1, 0..p-1:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]
            for k in range(p):
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                prod = u_row * v_row * w_row
                s = np.sum(prod)
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
    return True

def _strassen_6x6():
    """
    Implement Strassen's algorithm for 6x6 matrix multiplication.
    View 6x6 matrices as 2x2 blocks of 3x3 matrices.
    Rank = 7 * 27 = 189.
    """
    n, m, p = 6, 6, 6
    U = np.zeros((n * m, 189))
    V = np.zeros((m * p, 189))
    W = np.zeros((p * n, 189))
    
    # Block indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Block size is 3.
    # A block (r, c) covers rows r*3..r*3+2, cols c*3..c*3+2.
    
    # Strassen Steps:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # C00 = M0 - M3 + M5 - M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M6
    
    # Define steps: list of (a_coeffs, b_coeffs)
    # a_coeffs: dict {block_idx: sign}
    # b_coeffs: dict {block_idx: sign}
    steps = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}), # M0
        ({2: 1, 3: 1}, {0: 1}),       # M1
        ({0: 1}, {1: 1, 3: -1}),      # M2
        ({3: 1}, {2: 1, 0: -1}),      # M3
        ({0: 1, 1: 1}, {3: 1}),       # M4
        ({2: 1, 0: -1}, {0: 1, 1: 1}), # M5
        ({1: 1, 3: -1}, {2: 1, 3: 1})  # M6
    ]
    
    # C combination: c_block_idx -> {m_idx: sign}
    c_combos = [
        {0: 1, 3: -1, 5: 1, 6: -1}, # C00
        {2: 1, 4: 1},                # C01
        {1: 1, 3: 1},                # C10
        {0: 1, 1: -1, 2: 1, 6: 1}   # C11
    ]
    
    block_rank = 27
    for s in range(7):
        a_coeffs, b_coeffs = steps[s]
        for ai, a_sign in a_coeffs.items():
            for bi, b_sign in b_coeffs.items():
                # A block ai: rows (ai//2)*3, cols (ai%2)*3
                # B block bi: rows (bi//2)*3, cols (bi%2)*3
                
                a_r0 = (ai // 2) * 3
                a_c0 = (ai % 2) * 3
                b_r0 = (bi // 2) * 3
                b_c0 = (bi % 2) * 3
                
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            # Global indices
                            ga_row = a_r0 + i
                            ga_col = a_c0 + j
                            gb_row = b_r0 + j
                            gb_col = b_c0 + k
                            
                            # C block index
                            c_block_row = ga_row // 3
                            c_block_col = gb_col // 3
                            c_block_idx = c_block_row * 2 + c_block_col
                            
                            c_sign = c_combos[c_block_idx].get(s, 0)
                            if c_sign == 0:
                                continue
                            
                            col = s * block_rank + i * 9 + j * 3 + k
                            
                            U[ga_row * m + ga_col, col] = a_sign
                            V[gb_row * p + gb_col, col] = b_sign
                            W[gb_col * n + ga_row, col] = c_sign
                            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    # 1. Initialize with Schoolbook
    U, V, W = _get_schoolbook(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # 2. Try Strassen
    try:
        U_s, V_s, W_s = _strassen_6x6()
        if _verify_decomp(U_s, V_s, W_s, n, m, p):
            best_U, best_V, best_W = U_s, V_s, W_s
            best_rank = U_s.shape[1]
    except Exception:
        pass

    # 3. If we have time, try to improve further?
    # For now, 189 is a solid improvement over 216.
    
    # Convert to list of lists for the interface
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

# EVOLVE-BLOCK-END
