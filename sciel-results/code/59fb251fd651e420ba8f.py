import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Default schoolbook solution
    def _construct_schoolbook():
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

    best_solution = _construct_schoolbook()
    
    # Search for better rank
    # Try ranks from 49 down to 40
    for rank in range(49, 39, -1):
        if time.time() > deadline - 5:
            break
            
        # Try several random initializations
        for attempt in range(50):
            if time.time() > deadline - 2:
                break
                
            # Random U and V with entries -1, 0, 1
            U = np.random.randint(-1, 2, (n * m, rank))
            V = np.random.randint(-1, 2, (m * p, rank))
            
            # Construct the linear system for W
            # Variables: W[k, i, r] for k in [0,p), i in [0,n), r in [0,rank)
            # Total variables: p * n * rank
            num_vars = p * n * rank
            num_eqs = n * m * p
            
            A = np.zeros((num_eqs, num_vars))
            b = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Equation: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
                        for r in range(rank):
                            coeff = U[i * m + j, r] * V[j * p + k, r]
                            if coeff != 0:
                                var_idx = (k * n + i) * rank + r
                                A[eq_idx, var_idx] = coeff
                        eq_idx += 1
            
            # Solve A W = b
            # Use least squares to find a solution, then check if it's exact
            # Since it's underdetermined, lstsq gives the minimum norm solution
            W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            
            # Check if the solution is exact
            # And if 2*W is close to integers
            if np.allclose(A @ W_flat, b, atol=1e-9):
                # Check if W entries are half-integers
                W_check = 2 * W_flat
                if np.allclose(W_check, np.round(W_check), atol=1e-9):
                    # Construct the solution matrices
                    U_sol = U.tolist()
                    V_sol = V.tolist()
                    W_sol = [[0] * rank for _ in range(p * n)]
                    for k in range(p):
                        for i in range(n):
                            for r in range(rank):
                                val = W_flat[(k * n + i) * rank + r]
                                # Round to nearest half-integer
                                val_int = int(np.round(val * 2)) / 2.0
                                W_sol[k * n + i][r] = val_int
                    
                    # Verify the solution exactly
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s_val = 0.0
                                for r in range(rank):
                                    s_val += U_sol[i * m + j][r] * V_sol[j * p + k][r] * W_sol[k * n + i][r]
                                if abs(s_val - 1.0) > 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if valid:
                        return {"U": U_sol, "V": V_sol, "W": W_sol}

    return best_solution

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
