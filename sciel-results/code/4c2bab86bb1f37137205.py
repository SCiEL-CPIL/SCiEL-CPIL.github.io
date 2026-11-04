"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import random

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank_school = n * m * p
    
    # Function to check if a decomposition is valid
    def is_valid(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Condition: sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] == delta_{j,j'}
        # We can check this by constructing the tensor or checking specific indices.
        # For efficiency, we can check a subset or all.
        # Since dimensions are small, we can check all.
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        target = 1 if j == jp else 0
                        val = 0
                        for c in range(r):
                            val += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        if abs(val - target) > 1e-9:
                            return False
        return True

    def try_find_decomposition(rank):
        # Try to find U, V, W of given rank
        # We fix U and V to be random sparse 0/1 matrices and solve for W
        # Then check if W is close to half-integers and if the decomposition is exact.
        
        # Number of attempts
        attempts = 100
        
        for _ in range(attempts):
            # Initialize U and V with random 0s and 1s
            # Density should be low enough to allow sparsity but high enough to be solvable
            # Let's try density 0.5
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            
            for i in range(n*m):
                for c in range(rank):
                    if random.random() < 0.5:
                        U[i, c] = 1
            for i in range(m*p):
                for c in range(rank):
                    if random.random() < 0.5:
                        V[i, c] = 1
            
            # Build the system for W
            # We have equations for each (i, k, j, jp)
            # sum_c U[i*m+j, c] * V[jp*p+k, c] * W[k*n+i, c] = delta_{j,j'}
            # This is a linear system for W?
            # Wait, the equation is linear in W for fixed U, V.
            # Let A be the matrix of coefficients for W.
            # Variables are W[k*n+i, c] for all k, i, c.
            # Total variables: p * n * rank
            # Total equations: n * p * m * m
            
            num_vars = p * n * rank
            num_eqs = n * p * m * m
            
            A = np.zeros((num_eqs, num_vars))
            b = np.zeros(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            target = 1 if j == jp else 0
                            b[eq_idx] = target
                            for c in range(rank):
                                coeff = U[i*m + j, c] * V[jp*p + k, c]
                                var_idx = (k * n + i) * rank + c
                                A[eq_idx, var_idx] = coeff
                            eq_idx += 1
            
            # Solve A * W_flat = b
            # Use least squares
            try:
                W_flat, residuals, rank_solve, singular_values = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            # Check if the solution is exact
            # Residuals should be 0
            if len(residuals) > 0 and residuals[0] > 1e-9:
                continue
            if len(residuals) == 0:
                # Check if A is full rank or if the solution is unique?
                # If not unique, we might have a free variable. 
                # But if it satisfies the equation, it's valid.
                pass
            
            # Check if W entries are close to half-integers
            W_flat = W_flat.reshape(p * n, rank)
            W_check = W_flat * 2
            W_rounded = np.round(W_check)
            if not np.allclose(W_check, W_rounded, atol=1e-9):
                continue
            
            W_sol = W_rounded / 2.0
            
            # Verify the solution with the original U, V
            if is_valid(U, V, W_sol):
                # Convert to lists of floats (integers or half-integers)
                return U.tolist(), V.tolist(), W_sol.tolist()
        
        return None

    # Try ranks from schoolbook - 1 downwards
    for r in range(rank_school - 1, 0, -1):
        result = try_find_decomposition(r)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        # If we find one, we return. If not, we continue.
        # To save time, we might stop after a few lower ranks if not found quickly?
        # But we want the minimum.
        # Given 120s limit, and 100 attempts per rank, and 41 ranks, 
        # 41 * 100 * (cost of lstsq) should be fine.
        # Cost of lstsq: 
        # num_eqs = 2*7*3*3 = 126
        # num_vars = 7*2*r = 14r
        # For r=41, vars=574. 126x574 matrix. Very fast.
        
    # Fallback to schoolbook
    rank = rank_school
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
