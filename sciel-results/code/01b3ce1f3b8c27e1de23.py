import time
import numpy as np
from fractions import Fraction
import sys

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 7
    schoolbook_rank = n * m * p  # 42

    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_exact(U, V, W, r, n, m, p):
        """Verify decomposition using Fractions."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    for c in range(r):
                        val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if val != 1:
                        return False
        return True

    def is_valid_entry(x):
        """Check if Fraction x is integer or half-integer."""
        return x.denominator in (1, 2)

    def solve_for_W_exact(U, V, r, n, m, p):
        """
        Given U and V (lists of Fractions), solve for W.
        Returns W as a list of lists of Fractions if all entries are valid (int or half-int), else None.
        """
        W = [[None] * r for _ in range(p * n)]
        
        # We need to solve for each (i, k) pair:
        # sum_{j=0}^{m-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all c? No.
        # The tensor decomposition is:
        # T_{i,j,k} = sum_{c=0}^{r-1} U_{i,j,c} V_{j,k,c} W_{k,i,c}
        # Here T_{i,j,k} = 1.
        # So for fixed i, k, we have a system of m equations in r unknowns (W_{k,i,0} ... W_{k,i,r-1}).
        # Equation j: sum_{c=0}^{r-1} (U_{i,j,c} * V_{j,k,c}) * W_{k,i,c} = 1
        
        for i in range(n):
            for k in range(p):
                # Build the m x r matrix A where A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                A = [[Fraction(0)] * r for _ in range(m)]
                for j in range(m):
                    for c in range(r):
                        A[j][c] = U[i * m + j][c] * V[j * p + k][c]
                
                # We need to find vector W_row (length r) such that A * W_row^T = [1, 1, ..., 1]^T
                # If r == m, we can solve directly.
                # If r > m, we have many solutions. We need one where all entries are int/half-int.
                # This is a Diophantine-like problem.
                
                # For simplicity and speed, we only attempt to solve if r == m (unique solution) 
                # or if we can find a simple solution. 
                # However, for rank reduction, r < n*m*p. 
                # If r > m, the system is underdetermined. 
                # A common strategy is to set r-m variables to 0 and solve the remaining m x m subsystem.
                
                # Let's try to find a solution by setting as many W entries to 0 as possible.
                # This is computationally expensive if done generally.
                
                # Strategy:
                # 1. If r == m: Solve m x m system.
                # 2. If r > m: Try to find a subset of m columns that form an invertible matrix, solve, and set others to 0.
                
                if r == m:
                    # Gaussian elimination with Fractions
                    # Augmented matrix [A | b]
                    b = [Fraction(1)] * m
                    mat = [A[j][:] + [b[j]] for j in range(m)]
                    
                    # Forward elimination
                    for col in range(m):
                        # Find pivot
                        pivot_row = None
                        for row in range(col, m):
                            if mat[row][col] != 0:
                                pivot_row = row
                                break
                        if pivot_row is None:
                            return None # Singular
                        if pivot_row != col:
                            mat[col], mat[pivot_row] = mat[pivot_row], mat[col]
                        
                        # Normalize pivot row
                        pivot_val = mat[col][col]
                        for c in range(col, m + 1):
                            mat[col][c] /= pivot_val
                            
                        # Eliminate other rows
                        for row in range(m):
                            if row != col and mat[row][col] != 0:
                                factor = mat[row][col]
                                for c in range(col, m + 1):
                                    mat[row][c] -= factor * mat[col][c]
                    
                    # Extract solution
                    W_row = [mat[j][m] for j in range(m)]
                    
                    # Check if all entries are valid
                    for val in W_row:
                        if not is_valid_entry(val):
                            return None
                    W[k * n + i] = W_row

                else:
                    # r != m. This is harder. 
                    # For now, we skip this to ensure speed, as random search with r=m is the most likely to yield exact solutions quickly.
                    # Actually, if r > m, we can't just solve m x m unless we pick columns.
                    # Let's just return None for r != m in this exact solver to keep it fast and reliable.
                    return None
                    
        return W

    def try_rank_exact(r, max_attempts):
        """Try to find a rank r decomposition using exact rational arithmetic."""
        if time.time() > deadline - 5:
            return None
            
        # We need to generate random U and V with small integer entries.
        # To make the system solvable with integer/half-integer W, 
        # the determinants of the submatrices should be small (1 or 2).
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
                
            # Generate random U and V with entries in {-1, 0, 1}
            # Using simple deterministic pseudo-random to avoid numpy overhead in inner loop?
            # No, numpy is fine for generation, we just need to convert to Fraction.
            
            U_np = np.random.randint(-1, 2, size=(n * m, r))
            V_np = np.random.randint(-1, 2, size=(m * p, r))
            
            # Convert to Fractions
            U = [[int(x) for x in row] for row in U_np]
            V = [[int(x) for x in row] for row in V_np]
            
            W = solve_for_W_exact(U, V, r, n, m, p)
            
            if W is not None:
                # Verify exactly
                if verify_exact(U, V, W, r, n, m, p):
                    return {"U": U, "V": V, "W": W}
                    
        return None

    best_result = None
    best_rank = schoolbook_rank

    # Try ranks from 41 down to 35. 
    # Rank 42 is guaranteed.
    # We prioritize higher ranks (closer to 42) as they are more likely to exist.
    
    # Note: The exact solver only works well if r == m (3) for the sub-problems? 
    # No, the sub-problem size is m x r. 
    # If r=41, m=3. The system is 3 x 41. 
    # My solver currently returns None if r != m. 
    # This is a limitation. 
    # However, for r=41, we have 41 variables and 3 equations. 
    # We can set 38 variables to 0 and solve the 3x3 subsystem.
    # Let's enhance the solver to handle r > m by picking a basis.
    
    def solve_for_W_general(U, V, r, n, m, p):
        """
        Solve for W when r >= m.
        For each (i, k), we have A (m x r) * w = 1.
        We look for a solution w with integer/half-integer entries.
        Strategy: Find m linearly independent columns in A.
        Let the indices be C. Solve A[:, C] * w_C = 1.
        Set w_j = 0 for j not in C.
        Check if w_C entries are valid.
        """
        W = [[None] * r for _ in range(p * n)]
        
        for i in range(n):
            for k in range(p):
                A = [[U[i * m + j][c] * V[j * p + k][c] for c in range(r)] for j in range(m)]
                
                # Find m independent columns
                # Simple approach: iterate through combinations of m columns? 
                # r is up to 41. C(41, 3) is ~ 10k. Might be too slow if done for every (i,k) and every attempt.
                # There are n*p = 14 (i,k) pairs.
                # 14 * 10k = 140k determinants per attempt. 
                # If we do 100 attempts, that's 14 million determinants. 
                # Determinant of 3x3 is cheap.
                
                found = False
                # Try all combinations of m columns
                # To speed up, we can just try the first m columns, then next m, etc.
                # Or use Gaussian elimination to find a basis.
                
                # Let's use a heuristic: Try the first m columns. If singular, try shifting.
                # Actually, let's just try a few specific column sets.
                
                # Set 1: First m columns
                cols = list(range(m))
                sub_A = [[A[j][c] for c in cols] for j in range(m)]
                
                # Check if invertible and solve
                # Determinant of 3x3
                a,b,c = sub_A[0]
                d,e,f = sub_A[1]
                g,h,i_val = sub_A[2]
                
                det = a*(e*i_val - f*h) - b*(d*i_val - f*g) + c*(d*h - e*g)
                
                if det != 0:
                    # Cramer's rule
                    # b = [1, 1, 1]
                    # w_c = det(A_c) / det
                    # A_c is A with col c replaced by b
                    
                    w_vals = []
                    valid = True
                    for idx in range(m):
                        # Replace column idx with [1,1,1]
                        new_col = [1, 1, 1]
                        # Build matrix
                        m0 = [A[0][cols[j]] if j != idx else 1 for j in range(m)]
                        m1 = [A[1][cols[j]] if j != idx else 1 for j in range(m)]
                        m2 = [A[2][cols[j]] if j != idx else 1 for j in range(m)]
                        
                        det_c = m0[0]*(m1[1]*m2[2] - m1[2]*m2[1]) - m0[1]*(m1[0]*m2[2] - m1[2]*m2[0]) + m0[2]*(m1[0]*m2[1] - m1[1]*m2[0])
                        
                        val = Fraction(det_c, det)
                        if not is_valid_entry(val):
                            valid = False
                            break
                        w_vals.append(val)
                    
                    if valid:
                        W_row = [0] * r
                        for idx, c_idx in enumerate(cols):
                            W_row[c_idx] = w_vals[idx]
                        W[k * n + i] = W_row
                        found = True
                        break
                
                # If first m columns didn't work, try last m columns?
                # Or random m columns?
                # For speed, we only try first m and last m.
                if not found and r > m:
                    cols = list(range(r - m, r))
                    sub_A = [[A[j][c] for c in cols] for j in range(m)]
                    a,b,c = sub_A[0]
                    d,e,f = sub_A[1]
                    g,h,i_val = sub_A[2]
                    det = a*(e*i_val - f*h) - b*(d*i_val - f*g) + c*(d*h - e*g)
                    if det != 0:
                        w_vals = []
                        valid = True
                        for idx in range(m):
                            m0 = [A[0][cols[j]] if j != idx else 1 for j in range(m)]
                            m1 = [A[1][cols[j]] if j != idx else 1 for j in range(m)]
                            m2 = [A[2][cols[j]] if j != idx else 1 for j in range(m)]
                            det_c = m0[0]*(m1[1]*m2[2] - m1[2]*m2[1]) - m0[1]*(m1[0]*m2[2] - m1[2]*m2[0]) + m0[2]*(m1[0]*m2[1] - m1[1]*m2[0])
                            val = Fraction(det_c, det)
                            if not is_valid_entry(val):
                                valid = False
                                break
                            w_vals.append(val)
                        if valid:
                            W_row = [0] * r
                            for idx, c_idx in enumerate(cols):
                                W_row[c_idx] = w_vals[idx]
                            W[k * n + i] = W_row
                            found = True
                            break
                
                if not found:
                    return None # Failed for this (i,k)
        
        return W

    def try_rank_general(r, max_attempts):
        if time.time() > deadline - 5:
            return None
            
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
                
            U_np = np.random.randint(-1, 2, size=(n * m, r))
            V_np = np.random.randint(-1, 2, size=(m * p, r))
            
            U = [[int(x) for x in row] for row in U_np]
            V = [[int(x) for x in row] for row in V_np]
            
            W = solve_for_W_general(U, V, r, n, m, p)
            
            if W is not None:
                if verify_exact(U, V, W, r, n, m, p):
                    return {"U": U, "V": V, "W": W}
                    
        return None

    # Search for better ranks
    # Start from 41 down to 35
    for r in range(41, 34, -1):
        if time.time() > deadline - 10:
            break
            
        # Fewer attempts for lower ranks as they are harder
        attempts = 50 if r > 38 else 20
        
        result = try_rank_general(r, attempts)
        if result is not None:
            best_result = result
            best_rank = r
            break

    if best_result is not None:
        # Convert Fractions to numbers for output
        def frac_to_num(f):
            if f.denominator == 1:
                return f.numerator
            else:
                return float(f)
                
        U_out = [[frac_to_num(x) for x in row] for row in best_result["U"]]
        V_out = [[frac_to_num(x) for x in row] for row in best_result["V"]]
        W_out = [[frac_to_num(x) for x in row] for row in best_result["W"]]
        
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        return make_schoolbook()
