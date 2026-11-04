"""Find rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication."""
import numpy as np
from fractions import Fraction
import time
from itertools import product

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Build the target tensor T[i, j, k]
    # T[i,j,k] = 1 for all valid (i,j,k), representing the bilinear form
    # Actually, T[i,j,k] is the coefficient: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i,j,k in range
    T = np.ones((n, m, p), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = Fraction(1)
    
    def verify(U, V, W, rank):
        """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == T[i,j,k]"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                    if s != T[i, j, k]:
                        return False
        return True
    
    def to_lists(mat):
        return [[x for x in row] for row in mat]
    
    # Try to find a decomposition with rank < n*m*p = 40
    # Start with rank 20 and work up, or try specific known constructions
    
    # Strategy: Use the fact that for 2xn matrices, we can sometimes share computations
    # Let's try rank values from 20 to 39
    
    best_result = None
    
    # First, let's try a structured approach
    # For 2 x m x p, we can think of it as two independent rows of A
    # C[0,k] = sum_j A[0,j]*B[j,k]
    # C[1,k] = sum_j A[1,j]*B[j,k]
    
    # One idea: use rank-m decomposition for each row separately, but share B factors
    # This gives rank 2*m*p = 40 (schoolbook)
    
    # Better: try to find a rank-r decomposition using optimization
    # We'll use a greedy/greedy-search approach
    
    # Actually, let's try a different approach: 
    # For small cases, we can try to construct explicit decompositions
    
    # Known: For 2x2 x 2x2, Strassen gives rank 7
    # For 2x4 x 4x5, let's try to find if there's a known better rank
    
    # Let's try rank 30, 28, 26, etc. and see if we can find a solution
    
    # Approach: Use randomized initialization + alternating minimization with exact verification
    
    def try_find_decomposition(rank, max_iterations=1000, seed=42):
        """Try to find a rank-r decomposition using alternating least squares with exact arithmetic."""
        rng = np.random.RandomState(seed)
        
        for attempt in range(50):
            if time.time() > deadline - 10:
                return None
            
            # Initialize with random values in [-1, 1]
            U = rng.uniform(-1, 1, (n*m, rank))
            V = rng.uniform(-1, 1, (m*p, rank))
            W = rng.uniform(-1, 1, (p*n, rank))
            
            # Convert to Fractions for exact arithmetic
            U_frac = [[Fraction(int(x * 1000000), 1000000) for x in row] for row in U]
            V_frac = [[Fraction(int(x * 1000000), 1000000) for x in row] for row in V]
            W_frac = [[Fraction(int(x * 1000000), 1000000) for x in row] for row in W]
            
            # Alternating least squares: fix two matrices, solve for the third
            for iteration in range(max_iterations):
                if time.time() > deadline - 10:
                    return None
                
                # Solve for U: for each (i,j), U[i*m+j, :] should satisfy
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
                # This is a linear system in U[i*m+j, :]
                # For fixed (i,j), we need: sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
                # Let M[i,j,k,r] = V[j*p+k,r] * W[k*n+i,r]
                # Then sum_r U[i*m+j,r] * M[i,j,k,r] = 1 for all k
                # This is M[i,j,:] * U[i*m+j,:] = 1 (vector of length p)
                
                U_new = [[Fraction(0) for _ in range(rank)] for _ in range(n*m)]
                for i in range(n):
                    for j in range(m):
                        # Build system: for each k, sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                        # This is a linear system with p equations and rank unknowns
                        # If rank >= p, we can solve (may be underdetermined)
                        
                        # Collect coefficients
                        # A_sys[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        A_sys = [[V_frac[j*p+k][r] * W_frac[k*n+i][r] for r in range(rank)] for k in range(p)]
                        b_sys = [Fraction(1) for _ in range(p)]
                        
                        # Solve A_sys * x = b_sys for x = U[i*m+j, :]
                        # Use Gaussian elimination with exact arithmetic
                        x = solve_linear_system(A_sys, b_sys)
                        if x is not None:
                            U_new[i*m+j] = x
                
                # Solve for V: for each (j,k), V[j*p+k, :] should satisfy
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
                V_new = [[Fraction(0) for _ in range(rank)] for _ in range(m*p)]
                for j in range(m):
                    for k in range(p):
                        # For fixed (j,k), sum_r U[i*m+j,r] * W[k*n+i,r] * V[j*p+k,r] = 1 for all i
                        # A_sys[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        A_sys = [[U_new[i*m+j][r] * W_frac[k*n+i][r] for r in range(rank)] for i in range(n)]
                        b_sys = [Fraction(1) for _ in range(n)]
                        x = solve_linear_system(A_sys, b_sys)
                        if x is not None:
                            V_new[j*p+k] = x
                
                # Solve for W: for each (k,i), W[k*n+i, :] should satisfy
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
                W_new = [[Fraction(0) for _ in range(rank)] for _ in range(p*n)]
                for k in range(p):
                    for i in range(n):
                        # For fixed (k,i), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
                        # A_sys[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        A_sys = [[U_new[i*m+j][r] * V_new[j*p+k][r] for r in range(rank)] for j in range(m)]
                        b_sys = [Fraction(1) for _ in range(m)]
                        x = solve_linear_system(A_sys, b_sys)
                        if x is not None:
                            W_new[k*n+i] = x
                
                U_frac, V_frac, W_frac = U_new, V_new, W_new
                
                # Check if we have a valid decomposition
                if verify(U_frac, V_frac, W_frac, rank):
                    return U_frac, V_frac, W_frac
            
            # End of attempts
        return None
    
    def solve_linear_system(A, b):
        """Solve A*x = b using Gaussian elimination with exact arithmetic.
        A is list of lists (m x n), b is list of length m.
        Returns x (list of length n) or None if no solution."""
        m = len(A)
        n = len(A[0])
        
        # Augmented matrix
        aug = [A[i][:] + [b[i]] for i in range(m)]
        
        # Gaussian elimination
        row = 0
        for col in range(n):
            if row >= m:
                break
            # Find pivot
            pivot_row = None
            for r in range(row, m):
                if aug[r][col] != 0:
                    pivot_row = r
                    break
            if pivot_row is None:
                continue
            aug[row], aug[pivot_row] = aug[pivot_row], aug[row]
            
            # Normalize pivot row
            pivot = aug[row][col]
            aug[row] = [x / pivot for x in aug[row]]
            
            # Eliminate
            for r in range(m):
                if r != row and aug[r][col] != 0:
                    factor = aug[r][col]
                    aug[r] = [aug[r][c] - factor * aug[row][c] for c in range(n+1)]
            row += 1
        
        # Check for inconsistency
        for r in range(row, m):
            if all(aug[r][c] == 0 for c in range(n)) and aug[r][n] != 0:
                return None
        
        # Extract solution (underdetermined: set free variables to 0)
        x = [Fraction(0) for _ in range(n)]
        for r in range(row):
            # Find the leading column
            for c in range(n):
                if aug[r][c] != 0:
                    x[c] = aug[r][n]
                    break
        return x
    
    # Try ranks from 20 to 39
    for rank in [20, 22, 24, 26, 28, 30, 32, 34, 36, 38]:
        if time.time() > deadline - 30:
            break
        
        result = try_find_decomposition(rank, max_iterations=50, seed=rank)
        if result is not None:
            U, V, W = result
            best_result = {'U': to_lists(U), 'V': to_lists(V), 'W': to_lists(W)}
            # Verify once more
            if verify(U, V, W, rank):
                return best_result
    
    # If no better decomposition found, fall back to schoolbook
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
    return {'U': U, 'V': V, 'W': W}
