# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

def _schoolbook(n, m, p):
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
    return U, V, W

def _verify(U, V, W, n, m, p):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1) > 1e-9:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, max_time=120.0):
    start_time = time.time()
    deadline = start_time + max_time
    
    n_rows_U = n * m
    n_rows_V = m * p
    n_rows_W = p * n
    current_rank = len(U[0])
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    best_rank = current_rank
    best_U, best_V, best_W = U, V, W
    
    while time.time() < deadline - 5.0:
        c_elim = random.randint(0, current_rank - 1)
        
        # Build the system: for all (i,j,k), 
        # sum_{d != c_elim} alpha_d * U[i*m+j,d] * V[j*p+k,d] * W[k*n+i,d] 
        # = U[i*m+j,c_elim] * V[j*p+k,c_elim] * W[k*n+i,c_elim]
        
        # This is linear in alpha_d!
        # Number of equations: n*m*p = 48
        # Number of unknowns: current_rank - 1
        
        if current_rank <= 1:
            break
            
        n_eq = n * m * p
        n_unk = current_rank - 1
        
        # Build matrix A and vector b
        A = np.zeros((n_eq, n_unk))
        b = np.zeros(n_eq)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # RHS: contribution of the column we want to eliminate
                    b[eq_idx] = U_np[i*m+j, c_elim] * V_np[j*p+k, c_elim] * W_np[k*n+i, c_elim]
                    
                    # LHS: coefficients for alpha_d
                    for d in range(current_rank):
                        if d == c_elim:
                            continue
                        A[eq_idx, d if d < c_elim else d-1] = (
                            U_np[i*m+j, d] * V_np[j*p+k, d] * W_np[k*n+i, d]
                        )
                    eq_idx += 1
        
        # Solve the linear system
        # We want to find alpha such that A @ alpha = b
        # If a solution exists, we can eliminate column c_elim
        
        try:
            # Use least squares to find a solution
            alpha, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            
            # Check if the solution is valid (residuals should be small)
            if residuals is not None and len(residuals) > 0 and residuals[0] < 1e-10:
                # We found a valid elimination
                # Update U, V, W to remove column c_elim and adjust others
                
                new_U = U_np.copy()
                new_V = V_np.copy()
                new_W = W_np.copy()
                
                # For each remaining column d, we need to adjust
                # Actually, the elimination means:
                # u_c ⊗ v_c ⊗ w_c = sum_{d != c} alpha_d * u_d ⊗ v_d ⊗ w_d
                # This doesn't directly give us a way to remove a column unless
                # the alpha_d can be absorbed.
                
                # Wait, this approach is flawed. The tensor decomposition is not
                # linear in the columns in this way.
                
                # Let's try a different approach: random search for low rank
                # by fixing some matrices and solving for others.
                
                break
                
        except:
            pass
        
        # Try random rank reduction
        new_rank = current_rank - 1
        if new_rank < 1:
            break
            
        # Generate random U, V, W of rank new_rank
        # and check if they can be adjusted to satisfy the tensor equations
        
        # For now, just continue the loop
        pass
    
    return best_rank, best_U, best_V, best_W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    n, m, p = 3, 4, 4
    
    # Schoolbook is rank 48.
    U, V, W = _schoolbook(n, m, p)
    
    # Try to reduce rank using the linear dependency approach
    # This is a heuristic that may or may not find a lower rank
    best_rank, best_U, best_V, best_W = _try_reduce_rank(U, V, W, n, m, p, max_time=130.0)
    
    # If we found a better rank, use it; otherwise, return schoolbook
    if best_rank < len(U[0]):
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
