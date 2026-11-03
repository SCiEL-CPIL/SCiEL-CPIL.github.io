import numpy as np
from fractions import Fraction
import time
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    def verify(U, V, W, r):
        """Exact verification of decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True
    
    def try_systematic_rank(r):
        """
        Systematic approach: 
        1. Choose U and V entries from a small set
        2. Solve for W using rational arithmetic
        3. Check if W entries are integers or half-integers
        """
        if time.time() > deadline:
            return None
        
        num_vars = p * n * r  # W has p*n*r entries
        num_eqs = n * m * p   # n*m*p equations
        
        # For each equation (i,j,k), we need:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        #
        # This is linear in W. We can write it as A @ w = b where:
        # A has shape (num_eqs, num_vars)
        # A[eq_idx, var_idx] = U[i*m+j, c] * V[j*p+k, c]
        
        # Strategy: Try different patterns for U and V
        # Use a grid of possible values: {-2, -1, 0, 1, 2}
        
        max_val = 2
        # To keep computation manageable, use structured patterns
        
        patterns_tried = 0
        max_patterns = 500
        
        for pattern_idx in range(max_patterns):
            if time.time() > deadline:
                return None
            patterns_tried += 1
            
            # Generate U and V
            # Use a mix of structured and random patterns
            if pattern_idx < 10:
                # Highly structured: identity-like patterns
                U_np = np.zeros((n * m, r), dtype=int)
                V_np = np.zeros((m * p, r), dtype=int)
                for c in range(r):
                    # Different patterns per column
                    if c < n * m:
                        U_np[c, c] = 1
                    if c < m * p:
                        V_np[c, c] = 1
            elif pattern_idx < 30:
                # Block patterns
                U_np = np.zeros((n * m, r), dtype=int)
                V_np = np.zeros((m * p, r), dtype=int)
                block_size = max(1, r // 4)
                for c in range(r):
                    block = c // block_size
                    # Assign blocks
                    for idx in range(n * m):
                        if (idx // block_size) == (block % (n * m // block_size + 1)):
                            U_np[idx, c] = 1
                    for idx in range(m * p):
                        if (idx // block_size) == (block % (m * p // block_size + 1)):
                            V_np[idx, c] = 1
            else:
                # Random with small values
                U_np = np.random.randint(-max_val, max_val + 1, size=(n * m, r))
                V_np = np.random.randint(-max_val, max_val + 1, size=(m * p, r))
            
            # Build the linear system A @ w = b
            # A[eq_idx, var_idx] where:
            # eq_idx = i * m * p + j * p + k
            # var_idx = (k * n + i) * r + c
            # A[eq_idx, var_idx] = U[i*m+j, c] * V[j*p+k, c]
            
            A = np.zeros((num_eqs, num_vars), dtype=float)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for c in range(r):
                            coeff = U_np[i * m + j, c] * V_np[j * p + k, c]
                            if coeff != 0:
                                var_idx = (k * n + i) * r + c
                                A[eq_idx, var_idx] = coeff
            
            b = np.ones(num_eqs)
            
            # Check if the system is consistent
            # Use least squares and check residual
            try:
                x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
            
            residual = np.abs(A @ x - b).max()
            if residual > 1e-8:
                continue
            
            # Try to round to nearest half-integer
            W_float = x.reshape(p * n, r)
            W_rounded = np.round(W_float * 2) / 2
            
            # Verify the rounded solution
            W_check = W_rounded
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r):
                            s += U_np[i * m + j, c] * V_np[j * p + k, c] * W_check[k * n + i, c]
                        max_err = max(max_err, abs(s - 1.0))
            
            if max_err > 1e-9:
                continue
            
            # Convert to exact fractions
            U = [[Fraction(int(U_np[i, c])) for c in range(r)] for i in range(n * m)]
            V = [[Fraction(int(V_np[i, c])) for c in range(r)] for i in range(m * p)]
            W = [[Fraction(W_rounded[i, c]) for c in range(r)] for i in range(p * n)]
            
            # Final exact verification
            if verify(U, V, W, r):
                return U, V, W
        
        return None
    
    def try_gaussian_elimination_rank(r):
        """
        Use exact rational Gaussian elimination.
        Fix U and V, solve for W exactly.
        """
        if time.time() > deadline:
            return None
        
        num_vars = p * n * r
        num_eqs = n * m * p
        
        # Try a few random U, V patterns
        for attempt in range(20):
            if time.time() > deadline:
                return None
            
            U_np = np.random.randint(-2, 3, size=(n * m, r))
            V_np = np.random.randint(-2, 3, size=(m * p, r))
            
            # Build A and b as Fraction matrices
            A = [[Fraction(0) for _ in range(num_vars)] for _ in range(num_eqs)]
            b = [Fraction(1) for _ in range(num_eqs)]
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for c in range(r):
                            coeff = Fraction(U_np[i * m + j, c] * V_np[j * p + k, c])
                            if coeff != 0:
                                var_idx = (k * n + i) * r + c
                                A[eq_idx][var_idx] = coeff
            
            # Gaussian elimination with partial pivoting
            # Work with augmented matrix
            aug = [row[:] + [b[i]] for i, row in enumerate(A)]
            
            try:
                # Forward elimination
                for col in range(min(num_eqs, num_vars)):
                    # Find pivot
                    pivot_row = -1
                    for row in range(col, num_eqs):
                        if aug[row][col] != 0:
                            pivot_row = row
                            break
                    
                    if pivot_row == -1:
                        continue
                    
                    # Swap rows
                    if pivot_row != col:
                        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
                    
                    # Eliminate below
                    for row in range(col + 1, num_eqs):
                        if aug[row][col] != 0:
                            factor = aug[row][col] / aug[col][col]
                            for j in range(col, num_vars + 1):
                                aug[row][j] -= factor * aug[col][j]
                
                # Back substitution to find a particular solution
                # Set free variables to 0
                x = [Fraction(0) for _ in range(num_vars)]
                
                for row in range(num_eqs - 1, -1, -1):
                    # Find the first non-zero column in this row
                    pivot_col = -1
                    for col in range(num_vars):
                        if aug[row][col] != 0:
                            pivot_col = col
                            break
                    
                    if pivot_col == -1:
                        if aug[row][num_vars] != 0:
                            return None  # Inconsistent
                        continue
                    
                    # x[pivot_col] = (b - sum of known terms) / a[pivot_col]
                    s = aug[row][num_vars]
                    for col in range(pivot_col + 1, num_vars):
                        s -= aug[row][col] * x[col]
                    x[pivot_col] = s / aug[row][pivot_col]
                
                # Check if solution is valid
                # Verify: A @ x should equal b
                for i in range(num_eqs):
                    s = Fraction(0)
                    for j in range(num_vars):
                        s += A[i][j] * x[j]
                    if s != b[i]:
                        break
                else:
                    # Solution is valid!
                    W_vals = [x[(k * n + i) * r + c] for k in range(p) for i in range(n) for c in range(r)]
                    
                    # Check if all values are integers or half-integers
                    valid = True
                    for val in W_vals:
                        # Check if val * 2 is an integer
                        if (val * 2).denominator != 1:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Construct exact solution
                    U = [[Fraction(int(U_np[i, c])) for c in range(r)] for i in range(n * m)]
                    V = [[Fraction(int(V_np[i, c])) for c in range(r)] for i in range(m * p)]
                    W = [[W_vals[i * r + c] for c in range(r)] for i in range(p * n)]
                    
                    if verify(U, V, W, r):
                        return U, V, W
                        
            except Exception:
                continue
        
        return None
    
    # Try ranks from lower bound upward
    # Theoretical lower bound: max(n*m, m*p, p*n) = 18
    # But for matrix multiplication, better bounds exist
    # Start from 18 and go up
    
    schoolbook_rank = n * m * p
    
    for r in range(18, schoolbook_rank + 1):
        if time.time() > deadline - 5:
            break
        
        # First try systematic approach
        result = try_systematic_rank(r)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        # Then try Gaussian elimination
        result = try_gaussian_elimination_rank(r)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = schoolbook_rank
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
# EVOLVE-BLOCK-END
