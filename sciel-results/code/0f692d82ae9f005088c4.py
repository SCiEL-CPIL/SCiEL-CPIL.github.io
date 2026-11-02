"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import random
import sys
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x6 matrix.
    
    Strategy: For a given rank r, randomly initialize U and V with small integer/half-integer entries,
    then solve for W via linear algebra. For each output index (k, i), we have a linear system
    in the r entries of W[k*n+i, :]. If the solution has integer/half-integer entries, we've found
    a valid decomposition.
    
    We search ranks from low to high, using multiple random trials.
    """
    n, m, p = DIMENSIONS  # 2, 2, 6
    
    def is_half_integer(x, tol=1e-9):
        """Check if x is an integer or half-integer."""
        x2 = 2 * x
        return abs(x2 - round(x2)) < tol
    
    def verify(U, V, W, n, m, p, r):
        """Verify the decomposition satisfies the tensor identity."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def solve_for_W(U, V, n, m, p, r):
        """
        Given U and V, solve for W.
        For each (i, k), we have m equations:
          For j in range(m): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        This is a system of m equations in r unknowns (W[k*n+i, 0..r-1]).
        """
        W = [[0.0] * r for _ in range(p * n)]
        for i in range(n):
            for k in range(p):
                # Build the linear system: A * w = b where w = W[k*n+i, :]
                # For each j: sum_c U[i*m+j, c] * V[j*p+k, c] * w[c] = 1
                A = np.zeros((m, r))
                b = np.ones(m)
                for j in range(m):
                    for c in range(r):
                        A[j, c] = U[i * m + j][c] * V[j * p + k][c]
                
                # Solve the underdetermined system A * w = b
                # Use least squares, then check if solution is half-integer
                if r > m:
                    # Underdetermined: find minimum norm solution
                    try:
                        w, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        return None
                else:
                    try:
                        w = np.linalg.solve(A, b)
                    except:
                        # Singular, try lstsq
                        try:
                            w, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        except:
                            return None
                
                # Check if w has half-integer entries
                for c in range(r):
                    if not is_half_integer(w[c]):
                        return None
                W[k * n + i] = [float(x) for x in w]
        
        return W
    
    def try_rank(r, num_trials=200, seed=42):
        """Try to find a rank-r decomposition with random U, V."""
        rng = random.Random(seed)
        
        # Candidate values for U and V entries
        candidates = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        
        for trial in range(num_trials):
            # Randomly initialize U and V
            U = [[rng.choice(candidates) for _ in range(r)] for _ in range(n * m)]
            V = [[rng.choice(candidates) for _ in range(r)] for _ in range(m * p)]
            
            # Solve for W
            W = solve_for_W(U, V, n, m, p, r)
            if W is None:
                continue
            
            # Verify the full decomposition
            if verify(U, V, W, n, m, p, r):
                # Convert to proper format (ensure half-integers are exact)
                U_out = [[(round(2*x)/2) for x in row] for row in U]
                V_out = [[(round(2*x)/2) for x in row] for row in V]
                W_out = [[(round(2*x)/2) for x in row] for row in W]
                return U_out, V_out, W_out
        
        return None
    
    # Search ranks from low to high
    # Lower bound: at least max(n, m, p) = 6, but realistically higher
    # Try ranks from 10 to 24
    for r in range(10, n * m * p + 1):
        result = try_rank(r, num_trials=300, seed=42 + r)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook solution with rank 24
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

# EVOLVE-BLOCK-END
