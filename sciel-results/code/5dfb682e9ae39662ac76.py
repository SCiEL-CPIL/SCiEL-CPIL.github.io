"""Matrix multiplication 3x4x5 with known low-rank decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 3x4 matrix by a 4x5 matrix.
    
    The output factor W follows the trace convention W[k*n+i, r].
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Schoolbook decomposition of rank n*m*p = 60
    rank_baseline = n * m * p
    U_base = [[0] * rank_baseline for _ in range(n * m)]
    V_base = [[0] * rank_baseline for _ in range(m * p)]
    W_base = [[0] * rank_baseline for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_base[i * m + j][col] = 1
                V_base[j * p + k][col] = 1
                W_base[k * n + i][col] = 1
                col += 1
    
    best_solution = None
    best_rank = rank_baseline
    
    # Strategy: For a given rank R, try random U, V with small integer entries.
    # Then check if there exists W with half-integer entries such that
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    #
    # Key insight: For fixed U and V, define for each c a matrix A_c of size (n*p):
    # A_c[(i,k)] = sum_j U[i*m+j,c] * V[j*p+k,c]
    # This is NOT directly useful because the tensor equation is trilinear.
    #
    # Alternative: The equation sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # can be viewed as: for each (i,k), the vector of values {V[j*p+k,c]*U[i*m+j,c]} 
    # weighted by W[k*n+i,c] must sum to 1 for each j... no, that's not right either.
    #
    # Let's use a different approach: Fix U and V. Then the equation is linear in W
    # IF we consider it correctly. Actually, for fixed U and V, the equation
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # is NOT linear in W because W[k*n+i,c] multiplies a product that depends on c.
    #
    # However, we can reformulate: Let B_c[i,j,k] = U[i*m+j,c] * V[j*p+k,c].
    # Then we need sum_c B_c[i,j,k] * W[k*n+i,c] = 1.
    # This IS linear in W! The unknowns are W[k*n+i,c] for all k,i,c.
    # The equations are indexed by (i,j,k).
    #
    # So for fixed U, V, we have a linear system in the (p*n)*R unknowns W[k*n+i,c].
    # Number of equations: n*m*p = 60.
    # Number of unknowns: p*n*R = 15*R.
    # For R >= 4, we have more unknowns than equations, so solutions exist.
    #
    # We need W entries to be half-integers (multiples of 0.5).
    # Strategy: Solve the linear system, then check if solution has half-integer entries.
    # Or: Set up the system with the constraint that W entries are multiples of 0.5.
    
    # For efficiency, we'll use numpy to solve the linear system.
    # Build the linear system: for each (i,j,k), equation is:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # Unknowns: W[k*n+i,c] for k in range(p), i in range(n), c in range(R)
    # Map unknown index: idx = (k*n + i)*R + c
    
    def try_rank(R, num_trials=200, seed_offset=0):
        """Try to find a valid decomposition of rank R."""
        np.random.seed(42 + seed_offset)
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Random U, V with entries from {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(n * m, R))
            V = np.random.randint(-1, 2, size=(m * p, R))
            
            # Build linear system A * W_vec = b
            # A has shape (n*m*p, p*n*R)
            # b has shape (n*m*p,)
            num_eqs = n * m * p
            num_unk = p * n * R
            
            A = np.zeros((num_eqs, num_unk), dtype=float)
            b = np.ones(num_eqs, dtype=float)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Equation: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                        for c in range(R):
                            coeff = U[i * m + j, c] * V[j * p + k, c]
                            if coeff != 0:
                                unk_idx = (k * n + i) * R + c
                                A[eq_idx, unk_idx] = coeff
                        eq_idx += 1
            
            # Solve the linear system. We want a solution with half-integer entries.
            # First, try to find ANY solution, then check if it's half-integer.
            # Use least squares or check if system is consistent.
            
            # Check if the system is consistent and find a solution
            try:
                # Use lstsq for a minimum-norm solution
                W_vec, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            # Check if the solution actually satisfies the equations
            A_W = A @ W_vec
            if not np.allclose(A_W, b, atol=1e-10):
                continue
            
            # Check if W entries are close to half-integers
            # Multiply by 2 and check if close to integers
            W_2x = W_vec * 2
            W_rounded = np.round(W_2x)
            if not np.allclose(W_2x, W_rounded, atol=1e-8):
                continue
            
            # Check if the rounded solution also works
            W_check = W_rounded / 2.0
            A_W_check = A @ W_check
            if not np.allclose(A_W_check, b, atol=1e-10):
                continue
            
            # Valid solution found!
            W_mat = W_check.reshape((p * n, R))
            
            # Verify the full tensor decomposition
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(R):
                            s += U[i * m + j, c] * V[j * p + k, c] * W_mat[k * n + i, c]
                        if abs(s - 1.0) > 1e-10:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W_mat.tolist()
                }
        
        return None
    
    # Try ranks from 59 down to 45
    # Known result: rank(3,4,5) = 46, so we expect to find something at 46
    for R in range(rank_baseline - 1, 44, -1):
        if time.time() > deadline - 5:
            break
        
        # More trials for higher ranks (easier to find), fewer for lower
        if R > 55:
            trials = 100
        elif R > 50:
            trials = 300
        else:
            trials = 500
        
        result = try_rank(R, num_trials=trials, seed_offset=R)
        
        if result is not None:
            best_solution = result
            best_rank = R
            # Continue searching for lower rank
            if R <= 46:
                break
    
    if best_solution is not None:
        return best_solution
    else:
        return {"U": U_base, "V": V_base, "W": W_base}
# EVOLVE-BLOCK-END
