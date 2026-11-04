"""Combined strategy: Strassen 2x2x8 + attempt to absorb 2x1x8 into the 28-term basis."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    
    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0],
        [0, 1, 0, -1] 
    ]
    S_V = [
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ]
    S_W = [
        [1, 0, 0, 1], 
        [0, 1, 0, -1],
        [0, 0, 1, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0],
        [0, 0, 0, 1], 
        [1, 0, 0, 0]  
    ]

    # Build the 28-term Strassen basis for the 2x2x8 part
    rank_base = 28
    U_base = [[0] * rank_base for _ in range(n * m)]
    V_base = [[0] * rank_base for _ in range(m * p)]
    W_base = [[0] * rank_base for _ in range(p * n)]
    
    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li*2 + la]
                    if u_val != 0:
                        U_base[li * m + la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la*2 + lk]
                    if v_val != 0:
                        V_base[la * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk*2 + li]
                    if w_val != 0:
                        W_base[(k0 + lk) * n + li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Attempt to absorb the a=2 part into the 28-term basis
    # We need U_ext (2 x 28) and V_ext (8 x 28) such that:
    # sum_r U_ext[i,r] * V_ext[k,r] * W_base[k*n+i, r] == 1 for all i in {0,1}, k in {0..7}
    
    # Strategy: Try diverse patterns for U_ext with half-integer entries, solve exactly for V_ext
    
    def _solve_exact_half_int(A_mat, b_vec):
        """Solve A x = b exactly, return x with half-integer entries if possible."""
        # Use Gaussian elimination with fractions
        n_rows = len(A_mat)
        n_cols = len(A_mat[0])
        
        # Build augmented matrix
        aug = [[Fraction(A_mat[i][j]) for j in range(n_cols)] + [Fraction(b_vec[i])] for i in range(n_rows)]
        
        # Gaussian elimination
        for col in range(n_cols):
            # Find pivot
            pivot = -1
            for row in range(col, n_rows):
                if aug[row][col] != 0:
                    pivot = row
                    break
            if pivot == -1:
                continue
            aug[col], aug[pivot] = aug[pivot], aug[col]
            
            # Eliminate
            for row in range(n_rows):
                if row != col and aug[row][col] != 0:
                    factor = aug[row][col] / aug[col][col]
                    for j in range(col, n_cols + 1):
                        aug[row][j] -= factor * aug[col][j]
        
        # Check consistency
        for row in range(n_rows):
            all_zero = all(aug[row][j] == 0 for j in range(n_cols))
            if all_zero and aug[row][n_cols] != 0:
                return None
        
        # Find a particular solution (set free variables to 0)
        x = [Fraction(0)] * n_cols
        for row in range(n_rows):
            # Find pivot column
            pivot_col = -1
            for j in range(n_cols):
                if aug[row][j] != 0:
                    pivot_col = j
                    break
            if pivot_col != -1:
                x[pivot_col] = aug[row][n_cols] / aug[row][pivot_col]
        
        # Check if all entries are half-integers
        for val in x:
            if (val * 2).denominator != 1:
                return None
        
        # Convert to floats for consistency
        return [float(val) for val in x]
    
    def _try_absorption(U_ext_pattern):
        """Given U_ext (2 x 28), try to find V_ext (8 x 28) with half-integer entries."""
        V_ext = [[0] * rank_base for _ in range(p)]
        
        for k in range(p):
            # Build 2 x rank_base matrix
            A_k = [[0] * rank_base for _ in range(n)]
            for i in range(n):
                row_idx = k * n + i
                for r in range(rank_base):
                    A_k[i][r] = U_ext_pattern[i][r] * W_base[row_idx][r]
            
            b_k = [1] * n
            
            v_k = _solve_exact_half_int(A_k, b_k)
            if v_k is None:
                return None
            
            V_ext[k] = v_k
        
        return V_ext
    
    # Generate diverse U_ext patterns
    def _generate_patterns():
        patterns = []
        
        # Pattern 1: All ones
        patterns.append([[1] * rank_base for _ in range(n)])
        
        # Pattern 2: Alternating 1, 0
        patterns.append([[1 if r % 2 == 0 else 0 for r in range(rank_base)] for _ in range(n)])
        
        # Pattern 3: First half ones, second half zeros
        patterns.append([[1 if r < rank_base // 2 else 0 for r in range(rank_base)] for _ in range(n)])
        
        # Pattern 4: Random sparse patterns
        np.random.seed(42)
        for trial in range(20):
            pattern = []
            for i in range(n):
                row = []
                for r in range(rank_base):
                    if np.random.rand() < 0.3:
                        row.append(1 if np.random.rand() < 0.5 else -1)
                    else:
                        row.append(0)
                pattern.append(row)
            patterns.append(pattern)
        
        return patterns
    
    patterns = _generate_patterns()
    
    for U_ext_pattern in patterns:
        if not _elapsed_ok():
            break
        
        V_ext = _try_absorption(U_ext_pattern)
        
        if V_ext is not None:
            # Construct candidate with rank 28
            U_final = [row[:] for row in U_base]
            V_final = [row[:] for row in V_base]
            W_final = [row[:] for row in W_base]
            
            for i in range(n):
                for r in range(rank_base):
                    U_final[i * m + 2][r] = U_ext_pattern[i][r]
                    
            for k in range(p):
                for r in range(rank_base):
                    V_final[2 * p + k][r] = V_ext[k][r]
                    
            candidate = {"U": U_final, "V": V_final, "W": W_final}
            
            if _validate(candidate):
                return candidate
    
    # Fallback: rank 44 decomposition
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Copy base (first 28 columns)
    for r in range(rank_base):
        for i in range(n):
            for a in range(2):
                U[i*m+a][r] = U_base[i*m+a][r]
        for a in range(2):
            for k in range(p):
                V[a*p+k][r] = V_base[a*p+k][r]
        for k in range(p):
            for i in range(n):
                W[k*n+i][r] = W_base[k*n+i][r]
                
    col = rank_base
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}
    if _validate(candidate):
        return candidate
    else:
        return _schoolbook()

# EVOLVE-BLOCK-END
