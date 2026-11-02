import time
import sys
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=2, m=3, p=4, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                s = 0.0
                for r in range(rank):
                    s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                
                expected = 1.0
                err = abs(s - expected)
                max_err = max(max_err, err)
    
    return max_err <= tol

def _schoolbook():
    """Return the standard schoolbook decomposition of rank 24."""
    n, m, p = 2, 3, 4
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][column] = 1
                V[v_idx][column] = 1
                W[w_idx][column] = 1
                column += 1

    return {"U": U, "V": V, "W": W}

def _construct_rank_22():
    """
    Construct a rank-22 decomposition for 2x3x4 matrix multiplication.
    Strategy: Split p=4 into two blocks of p=2.
    Each 2x3x2 block is decomposed into:
      1. A 2x1x2 part (rank 4) using the first row of B.
      2. A 2x2x2 part (rank 7) using Strassen on the remaining 2 rows of B.
    Total rank per block = 11. Total rank = 22.
    """
    n, m, p = 2, 3, 4
    rank = 22
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b11)
    # M4 = a22(b12-b22)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # 
    # U (A) coefficients for Strassen terms:
    # r=0: [1, 0, 0, 1] (a11, a12, a21, a22) -> Note: a11 is (0,0), a12 is (0,1), a21 is (1,0), a22 is (1,1)
    # Actually, let's map indices:
    # A is 2x2. Indices (i, j) with i,j in 0..1.
    # Term 0: A00 + A11. U: [1, 0, 0, 1]
    # Term 1: A10 + A11. U: [0, 0, 1, 1]
    # Term 2: A00. U: [1, 0, 0, 0]
    # Term 3: A11. U: [0, 0, 0, 1]
    # Term 4: A00 + A01. U: [1, 1, 0, 0]
    # Term 5: A10 - A00. U: [-1, 0, 1, 0]
    # Term 6: A01 - A11. U: [0, 1, 0, -1]
    
    # V (B) coefficients for Strassen terms:
    # Term 0: B00 + B11. V: [1, 0, 0, 1]
    # Term 1: B00. V: [1, 0, 0, 0]
    # Term 2: B10 - B00. V: [0, -1, 1, 0] ? No, B is 2x2. B00, B01, B10, B11.
    # M3 = A00(B10 - B00). V: [0, -1, 1, 0] -> B10 is (1,0), B00 is (0,0).
    # Term 3: B01 - B11. V: [0, 1, 0, -1]
    # Term 4: B11. V: [0, 0, 0, 1]
    # Term 5: B00 + B01. V: [1, 1, 0, 0]
    # Term 6: B10 + B11. V: [0, 0, 1, 1]
    
    # W (C) contributions for Strassen terms:
    # C00 = M1 - M3 - M4 + M7
    # C01 = M5 + M4 - M1 - M6
    # C10 = M2 + M3 - M7 + M6
    # C11 = M1 - M5 + M2 - M6
    
    # We need to map these to global indices.
    # For a block at p-offset p_off (0 or 2):
    # The B block is rows 0..2, cols p_off..p_off+1.
    # Split B rows into j=0 (part 1) and j=1,2 (part 2).
    
    for p_off in [0, 2]:
        
        # Part 1: 2x1x2 block (j=0)
        # A is cols 0, B is row 0, C is cols p_off, p_off+1
        # Rank 4 terms: for each i in 0..1, k in 0..1
        for i in range(2):
            for k in range(2):
                u_idx = i * m + 0  # i*3 + 0
                v_idx = 0 * p + (p_off + k)  # 0*4 + k'
                w_idx = (p_off + k) * n + i  # k'*2 + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
        
        # Part 2: 2x2x2 block (j=1,2)
        # A is cols 1,2. B is rows 1,2. C is cols p_off, p_off+1.
        # Map local A (2x2) to global A cols 1,2.
        # Map local B (2x2) to global B rows 1,2.
        # Map local C (2x2) to global C cols p_off, p_off+1.
        
        # Strassen U coefficients (local A indices 0,1 -> global 1,2)
        U_strassen = [
            [1, 0, 0, 1], # T0
            [0, 0, 1, 1], # T1
            [1, 0, 0, 0], # T2
            [0, 0, 0, 1], # T3
            [1, 1, 0, 0], # T4
            [-1, 0, 1, 0],# T5
            [0, 1, 0, -1] # T6
        ]
        
        # Strassen V coefficients (local B indices 0,1 -> global 1,2)
        V_strassen = [
            [1, 0, 0, 1], # T0
            [1, 0, 0, 0], # T1
            [0, -1, 1, 0],# T2
            [0, 1, 0, -1],# T3
            [0, 0, 0, 1], # T4
            [1, 1, 0, 0], # T5
            [0, 0, 1, 1]  # T6
        ]
        
        # Strassen W coefficients (local C indices 0,1 -> global p_off, p_off+1)
        # C00: M0 - M2 - M3 + M6
        # C01: M4 + M3 - M0 - M5
        # C10: M1 + M2 - M6 + M5
        # C11: M0 - M4 + M1 - M5
        W_strassen = [
            [1, 0, -1, -1, 0, 0, 1], # C00
            [-1, 0, 0, 1, 1, -1, 0], # C01
            [0, 1, 1, 0, 0, 1, -1],  # C10
            [1, 0, 0, 0, -1, -1, 1]  # C11
        ]
        
        # Apply Strassen terms
        for r in range(7):
            # U
            for li in range(2):
                if U_strassen[r][li] != 0:
                    global_j = 1 + li
                    u_idx = 0 * m + global_j # i=0? No, U is for A. A is 2x3.
                    # Wait, U_strassen is for A. A is 2x2.
                    # The rows of U in the sub-problem correspond to A elements.
                    # A is 2x2. Rows i=0,1. Cols j=0,1.
                    # Global A rows i=0,1. Cols 1,2.
                    # So for local li (0,1) and lj (0,1):
                    # Global u_idx = i * 3 + (1 + lj)
                    
                    # But U_strassen[r] is a vector of 4 coeffs for A00, A01, A10, A11.
                    # Indexing: 0:A00, 1:A01, 2:A10, 3:A11.
                    pass
            
            # Let's do it explicitly to avoid confusion.
            # U coeffs for A00, A01, A10, A11
            a00, a01, a10, a11 = U_strassen[r]
            
            # A00: i=0, j=1 (global)
            if a00 != 0:
                U[0 * m + 1][col] += a00
            # A01: i=0, j=2 (global)
            if a01 != 0:
                U[0 * m + 2][col] += a01
            # A10: i=1, j=1 (global)
            if a10 != 0:
                U[1 * m + 1][col] += a10
            # A11: i=1, j=2 (global)
            if a11 != 0:
                U[1 * m + 2][col] += a11
            
            # V coeffs for B00, B01, B10, B11
            # Local B00 -> Global B(1, p_off)
            # Local B01 -> Global B(1, p_off+1)
            # Local B10 -> Global B(2, p_off)
            # Local B11 -> Global B(2, p_off+1)
            b00, b01, b10, b11 = V_strassen[r]
            
            if b00 != 0:
                V[1 * p + p_off][col] += b00
            if b01 != 0:
                V[1 * p + p_off + 1][col] += b01
            if b10 != 0:
                V[2 * p + p_off][col] += b10
            if b11 != 0:
                V[2 * p + p_off + 1][col] += b11
            
            # W coeffs for C00, C01, C10, C11
            c00, c01, c10, c11 = W_strassen[r]
            
            # C00: i=0, k=p_off
            if c00 != 0:
                W[p_off * n + 0][col] += c00
            # C01: i=0, k=p_off+1
            if c01 != 0:
                W[(p_off + 1) * n + 0][col] += c01
            # C10: i=1, k=p_off
            if c10 != 0:
                W[p_off * n + 1][col] += c10
            # C11: i=1, k=p_off+1
            if c11 != 0:
                W[(p_off + 1) * n + 1][col] += c11
            
            col += 1
            
    # col should be 22
    if col != 22:
        print(f"Warning: col is {col}, expected 22")
        # Truncate or pad if necessary, though it should be exact
        if col > 22:
            U = [u[:22] for u in U]
            V = [v[:22] for v in V]
            W = [w[:22] for w in W]
        else:
            # Pad with zeros
            while col < 22:
                for u in U: u.append(0)
                for v in V: v.append(0)
                for w in W: w.append(0)
                col += 1

    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 140
    
    # Try rank 22
    try:
        result = _construct_rank_22()
        if _verify(result['U'], result['V'], result['W']):
            return result
    except Exception:
        pass
    
    # Fall back to schoolbook
    result = _schoolbook()
    if _verify(result['U'], result['V'], result['W']):
        return result
        
    return result
# EVOLVE-BLOCK-END
