import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Target ranks to try, starting from a lower bound
    # Lower bound for 2x3x6 is likely 24
    target_ranks = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    
    start_time = time.time()
    deadline = 100.0  # Safety margin
    
    def solve_for_rank(r):
        # Shapes
        U_shape = (n * m, r)
        V_shape = (m * p, r)
        W_shape = (p * n, r)
        
        # Number of equations
        num_eqs = n * m * p
        
        # We fix U and V to random integer values (small)
        # and solve for W.
        # To increase chance of success, we try a few random initializations
        
        for attempt in range(50):
            if time.time() > deadline:
                return None
            
            # Initialize U and V with random integers in {-1, 0, 1}
            # Using numpy for speed, then convert to Fraction
            U_np = np.random.choice([-1, 0, 1], size=U_shape)
            V_np = np.random.choice([-1, 0, 1], size=V_shape)
            
            # Convert to Fraction
            U = [[Fraction(int(U_np[i, c])) for c in range(r)] for i in range(n * m)]
            V = [[Fraction(int(V_np[i, c])) for c in range(r)] for i in range(m * p)]
            
            # Set up linear system A x = b for W
            # x is flattened W of size (p * n * r)
            # A is (n * m * p) x (p * n * r)
            # b is (n * m * p) x 1, all ones
            
            # We can solve this using Gaussian elimination
            # But the matrix is sparse.
            # Let's build the matrix explicitly for small r
            
            # Variable index: var_idx = (k * n + i) * r + c
            # Equation index: eq_idx = i * m * p + j * p + k
            
            A = [[Fraction(0) for _ in range(p * n * r)] for _ in range(n * m * p)]
            b = [Fraction(1) for _ in range(n * m * p)]
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        # Coefficient for W[k*n + i, c] is U[i*m + j, c] * V[j*p + k, c]
                        for c in range(r):
                            coeff = U[i * m + j][c] * V[j * p + k][c]
                            if coeff != 0:
                                var_idx = (k * n + i) * r + c
                                A[eq_idx][var_idx] += coeff
            
            # Solve A x = b
            # Use Gaussian elimination
            # Since we only need one solution, we can use free variables = 0
            
            # Convert to list of lists for easier manipulation
            # A is list of lists of Fractions
            # Augmented matrix
            M = [row[:] + [b[i]] for i, row in enumerate(A)]
            
            # Gaussian elimination
            rows = len(M)
            cols = len(M[0]) - 1
            
            rank = 0
            for col in range(cols):
                if time.time() > deadline:
                    return None
                
                # Find pivot
                pivot_row = -1
                for row in range(rank, rows):
                    if M[row][col] != 0:
                        pivot_row = row
                        break
                
                if pivot_row == -1:
                    continue
                
                # Swap
                M[rank], M[pivot_row] = M[pivot_row], M[rank]
                
                # Normalize pivot row
                pivot_val = M[rank][col]
                for j in range(col, cols + 1):
                    M[rank][j] /= pivot_val
                
                # Eliminate
                for row in range(rows):
                    if row != rank and M[row][col] != 0:
                        factor = M[row][col]
                        for j in range(col, cols + 1):
                            M[row][j] -= factor * M[rank][j]
                
                rank += 1
            
            # Check for consistency
            for row in range(rank, rows):
                if M[row][cols] != 0:
                    return None # Inconsistent
            
            # Extract solution
            # Free variables are set to 0
            x = [Fraction(0) for _ in range(cols)]
            
            for i in range(rank):
                # Find the column where M[i][col] is 1
                # Since we normalized, the pivot is 1
                # We need to find which column is the pivot for row i
                # This is tricky if we didn't track it. 
                # Let's re-do the elimination tracking pivots.
                
                pass
            
            # Simpler: Use numpy to solve the linear system over floats, then round?
            # No, exactness is required.
            # Let's use a library or implement a proper solver.
            # Since the system is underdetermined, we can set free variables to 0.
            # We need to know which variables are free.
            
            # Let's try a different approach: 
            # If we fix U and V, and if the system is consistent, 
            # we can find a solution.
            # I will use a simple back-substitution if I track pivots.
            
            # For now, let's assume we can find a solution if the rank of A is full enough.
            # Given the randomness, it's likely solvable.
            
            # Let's try to reconstruct W from x
            # If x is found:
            # W[k*n + i, c] = x[(k*n + i)*r + c]
            
            # I will implement a proper Gaussian elimination with pivot tracking.
            
            # Re-implementing solver with pivot tracking
            M = [row[:] + [b[i]] for i, row in enumerate(A)]
            rows = len(M)
            cols = len(M[0]) - 1
            pivots = []
            
            rank = 0
            for col in range(cols):
                if time.time() > deadline:
                    return None
                
                pivot_row = -1
                for row in range(rank, rows):
                    if M[row][col] != 0:
                        pivot_row = row
                        break
                
                if pivot_row == -1:
                    continue
                
                M[rank], M[pivot_row] = M[pivot_row], M[rank]
                
                pivot_val = M[rank][col]
                for j in range(col, cols + 1):
                    M[rank][j] /= pivot_val
                
                for row in range(rows):
                    if row != rank and M[row][col] != 0:
                        factor = M[row][col]
                        for j in range(col, cols + 1):
                            M[row][j] -= factor * M[rank][j]
                
                pivots.append((rank, col))
                rank += 1
            
            for row in range(rank, rows):
                if M[row][cols] != 0:
                    return None
            
            x = [Fraction(0) for _ in range(cols)]
            for r_idx, c_idx in pivots:
                x[c_idx] = M[r_idx][cols]
            
            # Construct W
            W = [[Fraction(0) for _ in range(r)] for _ in range(p * n)]
            for i in range(p * n):
                for c in range(r):
                    W[i][c] = x[i * r + c]
            
            # Verify solution
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = Fraction(0)
                        for c in range(r):
                            s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                        if s != 1:
                            return None
            
            return U, V, W
        
        return None

    for r in target_ranks:
        if time.time() > deadline:
            break
        result = solve_for_rank(r)
        if result:
            U, V, W = result
            # Convert to lists of floats/ints if needed, but Fraction is fine
            # The interface expects "integer/half-integer entries"
            # Fractions are exact.
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = rank_schoolbook
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
