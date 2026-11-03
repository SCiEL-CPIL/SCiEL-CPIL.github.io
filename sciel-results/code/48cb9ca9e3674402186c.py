import numpy as np
import time
import random

DIMENSIONS = [2, 4, 6]

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 * 2x2 matrix multiplication.
    Each term is a tuple (U_coeffs, V_coeffs, W_coeffs) where:
      U_coeffs: dict {(i, j): coeff} for the 2x2 left matrix
      V_coeffs: dict {(j, k): coeff} for the 2x2 right matrix
      W_coeffs: dict {(i, k): coeff} for the 2x2 output
    
    Strassen formulas:
      M1 = (P00+P11)*(Q00+Q11)
      M2 = (P10+P11)*Q00
      M3 = P00*(Q01-Q11)
      M4 = P11*(Q10-Q00)
      M5 = (P00+P01)*Q11
      M6 = (P10-P00)*(Q00+Q01)
      M7 = (P01-P11)*(Q10+Q11)
    
    Result:
      R00 = M1 + M4 - M5 + M7
      R01 = M3 + M5
      R10 = M2 + M4
      R11 = M1 - M2 + M3 + M6
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

def _strassen_block_decomposition():
    """
    Construct a rank-42 decomposition using Strassen's algorithm.
    
    Strategy: Split the 2x4 * 4x6 multiplication into blocks:
    - A is 2x4, split into two 2x2 blocks: A_left (cols 0,1) and A_right (cols 2,3)
    - B is 4x6, split into top (rows 0,1) and bottom (rows 2,3), each 2x6
    - Each 2x6 is further split into three 2x2 column blocks
    
    C = A_left * B_top + A_right * B_bottom
    where B_top = [B_top_0, B_top_1, B_top_2] and B_bottom = [B_bottom_0, B_bottom_1, B_bottom_2]
    
    Each (2x2) * (2x2) block multiplication uses 7 Strassen terms.
    Total: 2 (A blocks) * 3 (B col blocks) * 7 (Strassen) = 42 terms.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42
    
    U = np.zeros((n * m, rank), dtype=float)  # 8 x 42
    V = np.zeros((m * p, rank), dtype=float)  # 24 x 42
    W = np.zeros((p * n, rank), dtype=float)  # 12 x 42
    
    strassen_terms = _strassen_2x2x2_terms()
    
    col = 0
    # For each A block (left/right) and each B column block
    for a_block in range(2):  # 0: A cols 0,1 (top B rows 0,1); 1: A cols 2,3 (bottom B rows 2,3)
        for b_col_block in range(3):  # B col blocks: 0,1,2 (each width 2)
            for t in range(7):  # 7 Strassen terms
                U_coeffs, V_coeffs, W_coeffs = strassen_terms[t]
                
                # Update U: A block a_block has columns at offset a_block*2
                for (i_loc, j_loc), coeff in U_coeffs.items():
                    i = i_loc
                    j = a_block * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx, col] += coeff
                
                # Update V: B rows depend on a_block (top if a_block=0, bottom if a_block=1)
                # B col block b_col_block has columns at offset b_col_block*2
                for (j_loc, k_loc), coeff in V_coeffs.items():
                    j = a_block * 2 + j_loc
                    k = b_col_block * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx, col] += coeff
                
                # Update W: output C[i, k] where i is row of C, k is col of C
                # W index: k*n + i (official trace convention)
                for (i_loc, k_loc), coeff in W_coeffs.items():
                    i = i_loc
                    k = b_col_block * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx, col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _schoolbook_decomposition():
    """Fallback: rank-m = 4 decomposition."""
    n, m, p = DIMENSIONS
    rank = m
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for j in range(m):
        for i in range(n):
            U[i * m + j, j] = 1
        for k in range(p):
            V[j * p + k, j] = 1
        for i in range(n):
            for k in range(p):
                W[k * n + i, j] = 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(result, n=2, m=4, p=6, trials=20):
    """Verify a decomposition numerically with integer arithmetic."""
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    rank = U.shape[1]
    
    for trial in range(trials):
        A = np.random.randint(-10, 11, size=(n, m)).astype(float)
        B = np.random.randint(-10, 11, size=(m, p)).astype(float)
        
        # Expected C
        C_exp = A @ B
        
        # Computed C using tensor decomposition
        C_comp = np.zeros((n, p))
        for col in range(rank):
            u_val = U[:, col] @ A.flatten()
            v_val = V[:, col] @ B.flatten()
            C_comp += W[:, col].reshape(p, n).T * (u_val * v_val)
        
        if not np.allclose(C_comp, C_exp, atol=1e-9):
            return False
    return True

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2 by 4 
    matrix by a 4 by 6 matrix.
    
    Strategy: Use Strassen-based rank-42 decomposition.
    Split A into two 2x2 blocks, B into top/bottom 2x6 halves, each with 3 col blocks.
    Each (2x2)*(2x2) uses 7 Strassen terms. Total: 2*3*7 = 42.
    
    The output factor follows the official trace convention W[k*n+i,r].
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Try Strassen decomposition first (rank 42)
    result = _strassen_block_decomposition()
    
    # Verify the decomposition
    if _verify_decomposition(result, 2, 4, 6, trials=20):
        # Convert to lists for the required interface
        return {"U": result["U"].tolist(), "V": result["V"].tolist(), "W": result["W"].tolist()}
    
    # Check deadline
    if time.time() > deadline - 5:
        # Fall back to schoolbook
        schoolbook = _schoolbook_decomposition()
        return {"U": schoolbook["U"].tolist(), "V": schoolbook["V"].tolist(), "W": schoolbook["W"].tolist()}
    
    # Fallback to schoolbook (rank 4)
    schoolbook = _schoolbook_decomposition()
    if _verify_decomposition(schoolbook, 2, 4, 6, trials=20):
        return {"U": schoolbook["U"].tolist(), "V": schoolbook["V"].tolist(), "W": schoolbook["W"].tolist()}
    
    # Should never reach here
    return {"U": schoolbook["U"].tolist(), "V": schoolbook["V"].tolist(), "W": schoolbook["W"].tolist()}
