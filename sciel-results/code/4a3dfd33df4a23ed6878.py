"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

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

def _schoolbook():
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

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    Shapes: U (4,7), V (4,7), W (4,7).
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen formulas:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    #
    # c00 = m1 - m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 - m3 + m6
    
    # c=0 (m1)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # c=1 (m2)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # c=2 (m3)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # c=3 (m4)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1
    
    # c=4 (m5)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # c=5 (m6)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # c=6 (m7)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _construct_strassen_blocks():
    """
    Structural approach: Use Strassen's algorithm on the 2x2 block structure 
    of the matrix multiplication.
    
    A (2x4) is partitioned into 2x2 blocks: A = [A1, A2]
    B (4x8) is partitioned into 2x8 blocks: B = [B1; B2]
    C = A1*B1 + A2*B2
    
    B1 (2x8) is partitioned into four 2x2 blocks: B1 = [B11, B12, B13, B14]
    B2 (2x8) is partitioned into four 2x2 blocks: B2 = [B21, B22, B23, B24]
    
    A1*B1 = A1*B11 + A1*B12 + A1*B13 + A1*B14 (sum of four 2x2x2 products)
    A2*B2 = A2*B21 + A2*B22 + A2*B23 + A2*B24 (sum of four 2x2x2 products)
    
    Total rank = 8 * 7 = 56.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # For each A-block (2 of them) and B-block (4 of them)
    for a_block in range(2):  # a_block = 0 or 1, corresponds to j in {2*a_block, 2*a_block+1}
        for b_block in range(4):  # b_block = 0..3, corresponds to k in {2*b_block, 2*b_block+1}
            for c in range(7):
                # Fill U: U[i*m + j, col]
                for i in range(2):
                    for j in range(2):
                        U[i * m + (2 * a_block + j), col] = U_s[i * 2 + j, c]
                
                # Fill V: V[j*p + k, col]
                for j in range(2):
                    for k in range(2):
                        V[(2 * a_block + j) * p + (2 * b_block + k), col] = V_s[j * 2 + k, c]
                
                # Fill W: W[k*n + i, col]
                for k in range(2):
                    for i in range(2):
                        W[(2 * b_block + k) * n + i, col] = W_s[k * 2 + i, c]
                
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _optimize_decomposition(U_init, V_init, W_init, n, m, p, max_iter=50, tol=1e-8):
    """
    Try to reduce the rank by zeroing out columns and checking if the decomposition still works.
    This is a greedy approach: try to remove columns one by one, and if the decomposition still holds, keep it removed.
    """
    rank = len(U_init[0])
    U = np.array(U_init, dtype=float)
    V = np.array(V_init, dtype=float)
    W = np.array(W_init, dtype=float)
    
    active_cols = list(range(rank))
    current_rank = rank
    
    # Try to remove each column
    for c in range(rank):
        if c not in active_cols:
            continue
        
        # Temporarily remove column c
        new_active = [col for col in active_cols if col != c]
        
        # Check if the decomposition still holds with the remaining columns
        valid = True
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    val = 0.0
                    for col in new_active:
                        val += U[i * m + j, col] * V[j * p + k, col] * W[k * n + i, col]
                    if abs(val - 1.0) > tol:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            active_cols = new_active
            current_rank -= 1
    
    if current_rank == rank:
        return None
    
    # Extract the sub-matrices with only active columns
    U_opt = U[:, active_cols]
    V_opt = V[:, active_cols]
    W_opt = W[:, active_cols]
    
    return {"U": U_opt, "V": V_opt, "W": W_opt}

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try the Strassen-based construction
    try:
        strassen_result = _construct_strassen_blocks()
        strassen_rank = len(strassen_result["U"][0])
        
        if strassen_rank < best_rank:
            if _verify(strassen_result["U"].tolist(), strassen_result["V"].tolist(), strassen_result["W"].tolist(), n, m, p):
                best = strassen_result
                best_rank = strassen_rank
                
                # Try to optimize by removing redundant columns
                if time.time() < deadline:
                    optimized = _optimize_decomposition(strassen_result["U"], strassen_result["V"], strassen_result["W"], n, m, p)
                    if optimized is not None:
                        opt_rank = len(optimized["U"][0])
                        if opt_rank < best_rank:
                            if _verify(optimized["U"].tolist(), optimized["V"].tolist(), optimized["W"].tolist(), n, m, p):
                                best = optimized
                                best_rank = opt_rank
    except Exception:
        pass
    
    # Convert to lists for the return interface
    U_list = best["U"].tolist() if isinstance(best["U"], np.ndarray) else best["U"]
    V_list = best["V"].tolist() if isinstance(best["V"], np.ndarray) else best["V"]
    W_list = best["W"].tolist() if isinstance(best["W"], np.ndarray) else best["W"]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
