import numpy as np
import random
import time

DIMENSIONS = [3, 4, 6]

def _add_term(U, V, W, row_u, row_v, row_w, coeff):
    """Add a term to the decomposition. U, V, W are lists of lists.
    We append a new column."""
    U.append([0] * len(U[0]) if U else [0])
    # Actually, let's use a different approach: build columns incrementally
    pass

def _construct_strassen():
    """
    Construct a low-rank decomposition for 3x4x6 matrix multiplication.
    
    Strategy: Split the inner dimension 4 into 2+2.
    This gives two sub-problems: (3x2) * (2x6) each.
    
    For each 3x2x6 sub-problem:
      - Split 3 = 2+1 (rows of A)
      - Split 6 = 3+3 (cols of B)  
      - The 2x2x3 part: split 3=2+1, use Strassen for 2x2x2 (rank 7) + schoolbook for 2x2x1 (rank 4) = 11
      - The 1x2x3 part: schoolbook rank 6
      Total per sub-problem: 17, overall rank: 34
    """
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # We'll build the decomposition term by term
    # Each term is (u_row, v_row, w_row, coeff)
    # where U[u_row] * V[v_row] * W[w_row] contributes coeff to the tensor
    
    terms = []
    
    # Sub-problem 1: A[:, 0:2] * B[0:2, :]  (3x2 * 2x6)
    # Sub-problem 2: A[:, 2:4] * B[2:4, :]  (3x2 * 2x6)
    
    for sub in range(2):
        # sub=0: inner cols 0,1 ; sub=1: inner cols 2,3
        # For each sub-problem, we have a 3x2 * 2x6 multiplication
        
        # Split rows: top 2 rows (i=0,1), bottom 1 row (i=2)
        # Split cols: left 3 (k=0,1,2), right 3 (k=3,4,5)
        
        # Part A: 2x2x3 (top 2 rows of A, both cols, left 3 cols of B)
        #   Split 3 = 2+1: left 2 (k=0,1), right 1 (k=2)
        #   2x2x2: Strassen rank 7
        #   2x2x1: schoolbook rank 4
        
        # Part B: 1x2x3 (bottom row of A, both cols, left 3 cols of B)
        #   schoolbook rank 6
        
        # Part C: 2x2x3 (top 2 rows, both cols, right 3 cols)
        #   Same as Part A: 11
        
        # Part D: 1x2x3 (bottom row, both cols, right 3 cols)
        #   schoolbook rank 6
        
        # Total: 11 + 6 + 11 + 6 = 34 per sub-problem? No, that's 34 total for both.
        # Wait: 11+6 = 17 per sub, 17*2 = 34 total. Yes.
        
        # Let me define Strassen for 2x2x2 properly.
        # Each Strassen multiplication is ONE rank-1 term in the tensor decomposition.
        # For a 2x2x2 block with local indices (i,j,k) in {0,1}^3:
        #   U index: (row_offset + i) * m + (col_offset + j)
        #   V index: (col_offset + j) * p + (k_offset + k)
        #   W index: (k_offset + k) * n + (row_offset + i)
        #
        # Strassen formulas:
        # M1 = (a11+a22)*(b11+b22)
        # M2 = (a21+a22)*b11
        # M3 = a11*(b12-b22)
        # M4 = a22*(b21-b11)
        # M5 = (a11+a12)*b22
        # M6 = (a21-a11)*(b11+b12)
        # M7 = (a12-a22)*(b21+b22)
        #
        # Output:
        # C[0,0] = M1 - M4 + M6 - M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M3 + M2 - M5
        
        def add_strassen_2x2x2(row_offset, col_offset, k_offset):
            """
            Add 7 Strassen terms for a 2x2x2 block.
            Each term is a single column in U, V, W.
            """
            # For each Strassen multiplication, define:
            # - A coefficients: dict mapping (i,j) -> coeff
            # - B coefficients: dict mapping (j,k) -> coeff
            # - C coefficients: dict mapping (i,k) -> coeff (for the W vector)
            
            strassen_mults = [
                # M1 = (a11+a22)*(b11+b22), contributes +1 to C[0,0] and +1 to C[1,1]
                ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
                # M2 = (a21+a22)*b11, contributes +1 to C[1,0] and -1 to C[1,1]
                ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
                # M3 = a11*(b12-b22), contributes +1 to C[0,1] and -1 to C[1,1]
                ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): -1}),
                # M4 = a22*(b21-b11), contributes -1 to C[0,0] and +1 to C[1,0]
                ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): -1, (1,0): 1}),
                # M5 = (a11+a12)*b22, contributes +1 to C[0,1] and -1 to C[1,1]
                ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (1,1): -1}),
                # M6 = (a21-a11)*(b11+b12), contributes +1 to C[0,0]
                ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 1}),
                # M7 = (a12-a22)*(b21+b22), contributes -1 to C[0,0]
                ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): -1}),
            ]
            
            for a_coeffs, b_coeffs, c_coeffs in strassen_mults:
                # Create one column in U, V, W
                col = len(terms)
                terms.append(None)  # placeholder
                
                # Fill U
                for (i, j), coeff in a_coeffs.items():
                    u_idx = (row_offset + i) * m + (col_offset + j)
                    terms[col] = (u_idx, None, None, coeff)  # We'll build this differently
                
                # Actually, let me just build the U, V, W entries directly
                # Reset
                terms.pop()
                
                u_entries = {}
                for (i, j), coeff in a_coeffs.items():
                    u_idx = (row_offset + i) * m + (col_offset + j)
                    u_entries[u_idx] = coeff
                
                v_entries = {}
                for (j, k), coeff in b_coeffs.items():
                    v_idx = (col_offset + j) * p + (k_offset + k)
                    v_entries[v_idx] = coeff
                
                w_entries = {}
                for (i, k), coeff in c_coeffs.items():
                    w_idx = (k_offset + k) * n + (row_offset + i)
                    w_entries[w_idx] = coeff
                
                # Store as a dict of entries for this column
                terms.append({'U': u_entries, 'V': v_entries, 'W': w_entries})
        
        # Part A: 2x2x3 at (row_offset=0, col_offset=sub*2, k_offset=0)
        #   Split 3=2+1: 2x2x2 at k_offset=0, then 2x2x1 at k_offset=2
        add_strassen_2x2x2(0, sub*2, 0)  # 2x2x2, rank 7
        
        # 2x2x1: schoolbook, rank 4
        for ai in range(2):
            for aj in range(2):
                I = ai
                J = sub*2 + aj
                K = 2
                u_idx = I * m + J
                v_idx = J * p + K
                w_idx = K * n + I
                terms.append({'U': {u_idx: 1}, 'V': {v_idx: 1}, 'W': {w_idx: 1}})
        
        # Part B: 1x2x3 at (row_offset=2, col_offset=sub*2, k_offset=0)
        #   schoolbook, rank 6
        for aj in range(2):
            for ak in range(3):
                I = 2
                J = sub*2 + aj
                K = ak
                u_idx = I * m + J
                v_idx = J * p + K
                w_idx = K * n + I
                terms.append({'U': {u_idx: 1}, 'V': {v_idx: 1}, 'W': {w_idx: 1}})
        
        # Part C: 2x2x3 at (row_offset=0, col_offset=sub*2, k_offset=3)
        #   Split 3=2+1: 2x2x2 at k_offset=3, then 2x2x1 at k_offset=5
        add_strassen_2x2x2(0, sub*2, 3)  # 2x2x2, rank 7
        
        # 2x2x1: schoolbook, rank 4
        for ai in range(2):
            for aj in range(2):
                I = ai
                J = sub*2 + aj
                K = 5
                u_idx = I * m + J
                v_idx = J * p + K
                w_idx = K * n + I
                terms.append({'U': {u_idx: 1}, 'V': {v_idx: 1}, 'W': {w_idx: 1}})
        
        # Part D: 1x2x3 at (row_offset=2, col_offset=sub*2, k_offset=3)
        #   schoolbook, rank 6
        for aj in range(2):
            for ak in range(3):
                I = 2
                J = sub*2 + aj
                K = 3 + ak
                u_idx = I * m + J
                v_idx = J * p + K
                w_idx = K * n + I
                terms.append({'U': {u_idx: 1}, 'V': {v_idx: 1}, 'W': {w_idx: 1}})
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, term in enumerate(terms):
        for u_idx, coeff in term['U'].items():
            U[u_idx][col] = coeff
        for v_idx, coeff in term['V'].items():
            V[v_idx][col] = coeff
        for w_idx, coeff in term['W'].items():
            W[w_idx][col] = coeff
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(result, n, m, p):
    """Verify that the decomposition is correct."""
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    rank = U.shape[1]
    
    # Check: sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                total = np.dot(u_val, v_val * w_val)
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _construct_schoolbook():
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

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    result = _construct_strassen()
    
    # Verify the decomposition
    if _verify_decomposition(result, n, m, p):
        return result
    else:
        # Fallback to schoolbook if verification fails
        return _construct_schoolbook()
