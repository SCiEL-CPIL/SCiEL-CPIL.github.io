"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    for i in range(n):
        for k in range(p):
            for j in range(m):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition():
    """Fallback schoolbook decomposition with rank n*m*p = 64."""
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2_terms():
    """
    Returns the 7 Strassen terms as a list of tuples (A_coeffs, B_coeffs, C_coeffs).
    A_coeffs: {(i,j): coeff}
    B_coeffs: {(i,j): coeff}
    C_coeffs: {(i,j): coeff}
    Indices are local 0..1 for 2x2 matrices.
    """
    terms = []
    
    # M1 = (a00 + a11)(b00 + b11)
    terms.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    ))
    
    # M2 = (a10 + a11)b00
    terms.append((
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1, (1,1): -1}
    ))
    
    # M3 = a00(b01 - b11)
    terms.append((
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1}
    ))
    
    # M4 = a11(b10 - b00)
    terms.append((
        {(1,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (1,0): 1}
    ))
    
    # M5 = (a00 + a01)b11
    terms.append((
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,0): -1, (0,1): 1}
    ))
    
    # M6 = (a10 - a00)(b00 + b01)
    terms.append((
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1}
    ))
    
    # M7 = (a01 - a11)(b10 + b11)
    terms.append((
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1}
    ))
    
    return terms

def _construct_strassen_56():
    """
    Construct rank-56 decomposition for 2x4x8.
    Splits A into two 2x2 blocks (cols 0-1, 2-3).
    Splits B into four 2x2 blocks for each of the two row groups (rows 0-1, 2-3).
    Specifically, for each column block of C (4 blocks of width 2), we compute:
    C_block = A_left * B_top_block + A_right * B_bot_block
    using 7 mults for each product.
    """
    n, m, p = DIMENSIONS # 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    terms = _strassen_2x2x2_terms()
    
    col = 0
    # k_block: 0..3, corresponds to output columns 2*k_block, 2*k_block+1
    for k_block in range(4):
        # A_left (cols 0,1) * B_top_block (rows 0,1, cols 2*k_block, 2*k_block+1)
        # A_right (cols 2,3) * B_bot_block (rows 2,3, cols 2*k_block, 2*k_block+1)
        
        # Part 1: A_left * B_top_block
        # A_left is A[:, 0:2]. B_top_block is B[0:2, 2*k_block:2*k_block+2]
        # Local indices:
        # A: i in 0..1, j in 0..1 (global j = j_local)
        # B: j in 0..1 (global j = j_local), k in 0..1 (global k = 2*k_block + k_local)
        # C: i in 0..1, k in 0..1 (global k = 2*k_block + k_local)
        
        for term in terms:
            A_coeffs, B_coeffs, C_coeffs = term
            
            for (li, lj), coeff_a in A_coeffs.items():
                global_i = li
                global_j = lj # 0 or 1
                U[global_i * m + global_j, col] += coeff_a
            
            for (lj, lk), coeff_b in B_coeffs.items():
                global_j = lj # 0 or 1
                global_k = 2 * k_block + lk
                V[global_j * p + global_k, col] += coeff_b
            
            for (li, lk), coeff_c in C_coeffs.items():
                global_i = li
                global_k = 2 * k_block + lk
                W[global_k * n + global_i, col] += coeff_c
            
            col += 1
            
        # Part 2: A_right * B_bot_block
        # A_right is A[:, 2:4]. B_bot_block is B[2:4, 2*k_block:2*k_block+2]
        # Local indices:
        # A: i in 0..1, j in 0..1 (global j = 2 + j_local)
        # B: j in 0..1 (global j = 2 + j_local), k in 0..1 (global k = 2*k_block + k_local)
        # C: i in 0..1, k in 0..1 (global k = 2*k_block + k_local)
        
        for term in terms:
            A_coeffs, B_coeffs, C_coeffs = term
            
            for (li, lj), coeff_a in A_coeffs.items():
                global_i = li
                global_j = 2 + lj # 2 or 3
                U[global_i * m + global_j, col] += coeff_a
            
            for (lj, lk), coeff_b in B_coeffs.items():
                global_j = 2 + lj # 2 or 3
                global_k = 2 * k_block + lk
                V[global_j * p + global_k, col] += coeff_b
            
            for (li, lk), coeff_c in C_coeffs.items():
                global_i = li
                global_k = 2 * k_block + lk
                W[global_k * n + global_i, col] += coeff_c
            
            col += 1

    # Convert to lists of integers
    U_list = [[int(x) for x in row] for row in U]
    V_list = [[int(x) for x in row] for row in V]
    W_list = [[int(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    n, m, p = DIMENSIONS
    best = _schoolbook_decomposition()
    best_rank = n * m * p
    
    # Try the rank-56 Strassen construction
    try:
        strassen_result = _construct_strassen_56()
        strassen_rank = len(strassen_result["U"][0])
        
        if strassen_rank < best_rank:
            if _verify(strassen_result["U"], strassen_result["V"], strassen_result["W"], n, m, p):
                best = strassen_result
                best_rank = strassen_rank
    except Exception:
        pass
    
    return best
# EVOLVE-BLOCK-END
