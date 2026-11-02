"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
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
    
    best_result = make_schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank
    # For a given rank r, try random U, V and solve for W
    # The system: for each (i,j,k), sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1
    
    # This is a linear system in W. Let's set it up.
    # Number of equations: n*m*p = 60
    # Number of unknowns: p*n*r = 20*r
    
    # For r=60, 20*60=1200 unknowns, 60 equations - underdetermined, always solvable
    # For r=30, 20*30=600 unknowns, 60 equations - still underdetermined
    # For r=3, 20*3=60 unknowns, 60 equations - square system
    
    # But the constraint is that U and V are fixed, and we need the product U[i*m+j,c]*V[j*p+k,c] to be such that the system is consistent.
    
    # Actually, for any U, V, the system is linear in W and has a solution if the equations are consistent.
    # The system is: A * W_flat = b, where A is 60 x (20*r) and b is 60-vector of 1s.
    # A has a solution if b is in the column space of A.
    
    # Let's try: for random U, V, check if the system is consistent, and if so, find W.
    # Then check if W has the required structure (it doesn't need to, just needs to satisfy the equations).
    
    # Actually, W just needs to be a matrix of shape (p*n, r) = (20, r).
    
    # Let's try rank 59, 58, etc.
    
    rng = np.random.default_rng(42)
    
    for r in range(best_rank - 1, max(20, best_rank - 20), -1):  # Try r from 59 down to 40
        if time.time() > deadline - 10:
            break
        
        # Try several random U, V for this rank
        found = False
        for attempt in range(100):
            if time.time() > deadline - 10:
                break
            
            U_rand = rng.standard_normal((n * m, r))
            V_rand = rng.standard_normal((m * p, r))
            
            # Build the linear system for W
            # Equations: for each (i,j,k), sum_c W[k*n+i, c] * U_rand[i*m+j, c] * V_rand[j*p+k, c] = 1
            # Unknowns: W is (p*n, r) = (20, r), so 20*r unknowns
            
            num_eq = n * m * p
            num_unk = p * n * r
            
            # Build A matrix: A[(i,j,k), (k',i',c)] = U_rand[i*m+j, c] * V_rand[j*p+k, c] if k'=k and i'=i else 0
            # Actually, for equation (i,j,k), it involves W[k*n+i, c] for all c.
            # So A[(i,j,k), (k*n+i)*r + c] = U_rand[i*m+j, c] * V_rand[j*p+k, c]
            
            A = np.zeros((num_eq, num_unk))
            b = np.ones(num_eq)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(r):
                            w_idx = (k * n + i) * r + c
                            A[eq_idx, w_idx] = U_rand[i * m + j, c] * V_rand[j * p + k, c]
                        eq_idx += 1
            
            # Solve A * W_flat = b
            try:
                # Use least squares
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                # Check if the solution is valid (residuals should be small)
                if len(residuals) > 0 and residuals[0] < 1e-10:
                    W_rand = W_flat.reshape(p * n, r)
                    # Verify
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = sum(W_rand[k * n + i, c] * U_rand[i * m + j, c] * V_rand[j * p + k, c] for c in range(r))
                                if abs(val - 1) > 1e-8:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if valid:
                        # Found a valid decomposition
                        best_result = {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_rand.tolist()}
                        best_rank = r
                        found = True
                        break
            except:
                pass
        
        if found:
            # Try even lower ranks
            continue
        else:
            # If we can't find a decomposition for this rank, try lower
            continue
    
    return best_result
# EVOLVE-BLOCK-END
