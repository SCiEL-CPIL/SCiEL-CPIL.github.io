"""Matrix multiplication 6x6x6 via Block-Strassen rank decomposition.

This program constructs a rank-189 exact tensor decomposition for 6x6 matrix
multiplication by partitioning the matrices into 3x3 blocks of 2x2 submatrices
and applying Strassen's rank-7 algorithm to each block product.

The previous implementation failed because the verification step incorrectly
simulated the Strassen products. This version verifies the 2x2 Strassen
decomposition by explicitly constructing its U, V, W factors and checking
against standard matrix multiplication. If verification passes, it constructs
the full 6x6x6 decomposition; otherwise, it falls back to schoolbook (rank 216).
"""
import time
import random

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _strassen22_tables():
    """
    Returns (M, W) for 2x2 Strassen decomposition.
    M: List of 7 tuples (u_vec, v_vec) where u_vec, v_vec are length-4 lists
       representing linear combinations of A and B elements respectively.
       Local 2x2 flat order: [00, 01, 10, 11] -> indices 0, 1, 2, 3.
    W: List of tuples (out_idx, m_idx, sign) for reconstructing C.
       out_idx is the local flat index in C [00, 01, 10, 11].
    """
    def vec(*entries):
        return list(entries)

    # Standard Strassen formulas for C = A * B
    # M0 = (A00 + A11) * (B00 + B11)
    # M1 = (A10 + A11) * B00
    # M2 = A00 * (B01 - B11)
    # M3 = A11 * (B10 - B00)
    # M4 = (A00 + A01) * B11
    # M5 = (A10 - A00) * (B00 + B01)
    # M6 = (A01 - A11) * (B10 + B11)
    
    # Reconstruction:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    M = [
        (vec(1, 0, 0, 1), vec(1, 0, 0, 1)),  # M0
        (vec(0, 0, 1, 1), vec(1, 0, 0, 0)),  # M1
        (vec(1, 0, 0, 0), vec(0, 1, 0, -1)), # M2
        (vec(0, 0, 0, 1), vec(0, 0, 1, -1)), # M3
        (vec(1, 1, 0, 0), vec(0, 0, 0, 1)),  # M4
        (vec(1, 0, -1, 0), vec(1, 1, 0, 0)), # M5
        (vec(0, 1, 0, -1), vec(0, 1, 1, 0)), # M6
    ]
    
    W = [
        # C00 (idx 0)
        (0, 0, 1), (0, 3, 1), (0, 4, -1), (0, 6, 1),
        # C01 (idx 1)
        (1, 2, 1), (1, 4, 1),
        # C10 (idx 2)
        (2, 1, 1), (2, 3, 1),
        # C11 (idx 3)
        (3, 0, 1), (3, 1, -1), (3, 2, 1), (3, 5, 1),
    ]
    return M, W

def _verify_strassen22(M, W, trials=50):
    """
    Verify the 2x2 Strassen decomposition by constructing U, V, W matrices
    and checking C = U * diag(M_vals) * V * W against A * B.
    """
    # Construct U (4x7), V (4x7), Wmat (4x7) for the 2x2 case
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    Wmat = [[0]*7 for _ in range(4)]
    
    for r, (u, v) in enumerate(M):
        for i in range(4):
            U[i][r] = u[i]
            V[i][r] = v[i]
            
    for out_idx, m_idx, sign in W:
        Wmat[out_idx][m_idx] = sign
        
    for _ in range(trials):
        A = [[random.randint(-10, 10) for _ in range(2)] for _ in range(2)]
        B = [[random.randint(-10, 10) for _ in range(2)] for _ in range(2)]
        
        # Compute expected C = A * B
        C_expected = [[0, 0], [0, 0]]
        for i in range(2):
            for k in range(2):
                for j in range(2):
                    C_expected[i][j] += A[i][k] * B[k][j]
                    
        # Compute C via Strassen factors
        # Flatten A and B
        a_flat = [A[0][0], A[0][1], A[1][0], A[1][1]]
        b_flat = [B[0][0], B[0][1], B[1][0], B[1][1]]
        
        # Compute M_vals
        m_vals = [0] * 7
        for r in range(7):
            sum_u = sum(U[i][r] * a_flat[i] for i in range(4))
            sum_v = sum(V[i][r] * b_flat[i] for i in range(4))
            m_vals[r] = sum_u * sum_v
            
        # Reconstruct C
        C_calc = [[0, 0], [0, 0]]
        for i in range(2):
            for j in range(2):
                c_val = 0
                for r in range(7):
                    c_val += Wmat[i * 2 + j][r] * m_vals[r]
                C_calc[i][j] = c_val
                
        if C_expected != C_calc:
            return False
            
    return True

def _build_block_strassen():
    """
    Construct U, V, W for 6x6x6 matrix multiplication using Block-Strassen.
    Rank = 9 (block positions) * 3 (sum indices) * 7 (Strassen terms) = 189.
    """
    n, m, p = DIMENSIONS
    M, W = _strassen22_tables()
    
    if not _verify_strassen22(M, W):
        return None
        
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    Wm = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Block indices: i_block, j_block, k_block in {0, 1, 2}
    # Block size: 2
    # A block (i_block, j_block): rows i_block*2..i_block*2+1, cols j_block*2..j_block*2+1
    # B block (j_block, k_block): rows j_block*2..j_block*2+1, cols k_block*2..k_block*2+1
    # C block (i_block, k_block): rows i_block*2..i_block*2+1, cols k_block*2..k_block*2+1
    
    for i_block in range(3):
        for k_block in range(3):
            for j_block in range(3):
                # Apply Strassen to A[i_block, j_block] * B[j_block, k_block]
                # and add to C[i_block, k_block]
                for s in range(7):
                    u_vec, v_vec = M[s]
                    
                    # Update U: A block (i_block, j_block)
                    for lr in range(2):
                        for lc in range(2):
                            local_idx = lr * 2 + lc
                            global_row = i_block * 2 + lr
                            global_col = j_block * 2 + lc
                            U[global_row * m + global_col][col] += u_vec[local_idx]
                            
                    # Update V: B block (j_block, k_block)
                    for lr in range(2):
                        for lc in range(2):
                            local_idx = lr * 2 + lc
                            global_row = j_block * 2 + lr
                            global_col = k_block * 2 + lc
                            V[global_row * p + global_col][col] += v_vec[local_idx]
                            
                    # Update W: C block (i_block, k_block)
                    # We need to map the Strassen W table to this block
                    # The W table from _strassen22_tables gives contributions to local C indices.
                    # We need to add these to the global Wm matrix.
                    # Note: The W table in _strassen22_tables is a list of (out_idx, m_idx, sign).
                    # Here, for a specific Strassen term s (which is m_idx), we need to find
                    # all (out_idx, sign) pairs where m_idx == s.
                    
                    for out_idx, m_idx, sign in W:
                        if m_idx == s:
                            # out_idx is local flat index in C block
                            lr = out_idx // 2
                            lc = out_idx % 2
                            global_row = i_block * 2 + lr
                            global_col = k_block * 2 + lc
                            # Wm index is (col * n + row) based on task description W[k*n+i,r]
                            # Wait, task says W[k*n+i, r]. So row index is k*n + i.
                            # Here i is global row of C, k is global col of C?
                            # Standard tensor T_{i,j,k} = A_{i,j} B_{j,k} C_{k,i} (trace convention?)
                            # Task: "output factor follows the official trace convention W[k*n+i,r]"
                            # Usually C = A B => C_{i,k} = sum_j A_{i,j} B_{j,k}.
                            # Tensor T_{i,j,k} = A_{i,j} B_{j,k} C_{k,i}? No, C is the output.
                            # The decomposition is T = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
                            # So W index is (k*n + i).
                            # In our loop:
                            # i_block, k_block define the C block.
                            # global_row of C is i_block*2 + lr. This is 'i' in C_{i,k}.
                            # global_col of C is k_block*2 + lc. This is 'k' in C_{i,k}.
                            # So W index is (global_col * n + global_row).
                            
                            Wm[global_col * n + global_row][col] += sign
                            
                    col += 1
                    
    if col != rank:
        return None
        
    return {"U": U, "V": V, "W": Wm}

def _build_schoolbook():
    """Fallback schoolbook decomposition with rank 216."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    Wm = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                # W index: k*n + i
                Wm[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": Wm}

def construct():
    deadline = time.time() + 120
    # Try Block-Strassen first
    try:
        result = _build_block_strassen()
        if result is not None and time.time() < deadline:
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _build_schoolbook()

# EVOLVE-BLOCK-END
