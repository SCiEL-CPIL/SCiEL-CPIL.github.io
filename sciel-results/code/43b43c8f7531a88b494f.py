import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    # Schoolbook rank
    rank_schoolbook = n * m * p
    best_rank = rank_schoolbook
    best_solution = None

    def get_schoolbook():
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

    # Initialize with schoolbook
    best_solution = get_schoolbook()

    # We want to find a rank R < 120 such that there exist U, V, W with integer/half-integer entries
    # satisfying the tensor decomposition.
    
    # Strategy:
    # 1. Try to find a solution for R = 119, 118, ... using a randomized approach or structured approach.
    # 2. A known lower bound for 4x5x6 is around 100-110? 
    #    Actually, the lower bound for 4x4x4 is 25 (Coppersmith-Winograd gives ~25.22, but exact is 25? No, 25 is for 4x4x4? 
    #    Wait, 4x4x4 rank is 25? No, 4x4x4 rank is at most 64 (schoolbook). 
    #    Strassen gives 7 for 2x2x2.
    #    For 4x4x4, one can partition into 2x2x2 blocks.
    #    If we partition 4x4 into four 2x2 blocks, we get 8 blocks of 2x2x2.
    #    Rank <= 8 * 7 = 56.
    #    So 4x4x4 has rank <= 56.
    #    For 4x5x6:
    #    We can view it as a block matrix.
    #    Let's try to construct a solution using a "tensor product" or "sub-tensor" approach.
    
    # Let's try a specific construction for R < 120.
    # If we can't find one quickly, we return schoolbook.
    
    # Random search for R in [119, 110]
    # The system is:
    # sum_r U[i*m+j, r] * V[jp*p+k, r] * W[k*n+i, r] = delta_{j, jp}
    
    # This is a multilinear system.
    # We can fix U and V and solve for W if the system is linear in W.
    # The equations are linear in W for fixed U, V.
    # Number of equations: n * m * m * p = 4 * 5 * 5 * 6 = 600.
    # Number of unknowns for W: p * n * R = 6 * 4 * R = 24 * R.
    
    # If R is large enough, we might have a solution.
    # For R=119, 24*119 = 2856 unknowns, 600 equations. Highly underdetermined.
    # We can find a solution W for random U, V.
    # The issue is that W must be half-integers (or integers).
    # If we pick U, V with integer entries, the solution W might be rational.
    # We can scale W to make it integer if the denominators are small.
    # But the requirement is "integer/half-integer". This usually means entries in Z or Z + 1/2.
    # This implies that 2*W must be integer.
    
    # Let's try R = 119.
    # We will randomly initialize U and V with small integers (e.g., -1, 0, 1).
    # Then solve for W.
    # Check if 2*W is close to integer.
    
    # To speed up, we can use the fact that the target tensor is sparse (only 4*5*6 = 120 non-zeros).
    
    # Let's iterate R downwards from 119.
    
    for R in range(119, 109, -1):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Try a few random instances
        found = False
        for attempt in range(10):
            if time.time() - start_time > time_limit - 5:
                break
                
            # Random U, V
            # Using small integers to keep numbers small
            U = np.random.randint(-1, 2, size=(n*m, R))
            V = np.random.randint(-1, 2, size=(m*p, R))
            
            # Build linear system for W
            # Variables: W[k*n+i, r] for k in [0,p), i in [0,n), r in [0,R)
            # Total vars: p*n*R
            num_vars = p * n * R
            num_eqs = n * m * m * p
            
            # We construct A such that A @ W_flat = b
            # A[eq_idx, var_idx] = U[i*m+j, r] * V[jp*p+k, r]
            # where eq_idx corresponds to (i, j, jp, k)
            # and var_idx corresponds to (k, i, r)
            
            # We can build A more efficiently.
            # For each r, the contribution to A is from U[:, r] and V[:, r].
            
            A = np.zeros((num_eqs, num_vars))
            
            # Precompute indices for speed
            # eq_idx(i, j, jp, k) = ((i*m + j)*m + jp)*p + k
            # var_idx(k, i, r) = (k*n + i)*R + r
            
            for r in range(R):
                u_col = U[:, r]
                v_col = V[:, r]
                
                for i in range(n):
                    for k in range(p):
                        var_idx_base = (k * n + i) * R + r
                        
                        # For this (i, k, r), the variable is W[k, i, r].
                        # It appears in equations for all j, jp.
                        # Coefficient is U[i, j, r] * V[jp, k, r].
                        
                        # u_vals[j] = U[i*m + j, r]
                        u_vals = u_col[i*m : (i+1)*m]
                        # v_vals[jp] = V[jp*p + k, r]
                        v_vals = v_col[k::p]
                        
                        # block[j, jp] = u_vals[j] * v_vals[jp]
                        block = np.outer(u_vals, v_vals)
                        
                        # Fill A
                        # eq_idx = i*m*m*p + j*m*p + jp*p + k
                        # We can loop j, jp
                        for j in range(m):
                            for jp in range(m):
                                eq_idx = i * m * m * p + j * m * p + jp * p + k
                                A[eq_idx, var_idx_base] = block[j, jp]
            
            # Build b
            b = np.zeros(num_eqs)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * m * p + j * m * p + j * p + k
                        b[eq_idx] = 1.0
            
            # Solve A W = b
            # Since it's underdetermined, use lstsq to find a minimum norm solution?
            # Or just any solution.
            # We want a solution that is "nice" (half-integers).
            # Lstsq gives the minimum norm solution in real space.
            # It might not be half-integer.
            
            try:
                W_flat, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
                
            # Check if W_flat is close to half-integers
            W_scaled = W_flat * 2
            W_rounded = np.round(W_scaled)
            
            if np.allclose(W_flat, W_rounded / 2.0, atol=1e-5):
                W_int = W_rounded.astype(int)
                
                # Verify
                U_check = U.reshape(n, m, R)
                V_check = V.reshape(m, p, R)
                W_check = W_int.reshape(p, n, R)
                
                T = np.einsum('ijr,jkr,kir->ijjk', U_check, V_check, W_check)
                
                target = np.zeros((n, m, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            target[i, j, j, k] = 1.0
                
                if np.allclose(T, target, atol=1e-5):
                    # Found a valid solution with rank R
                    best_rank = R
                    best_solution = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W_int.reshape(p * n, R).tolist()
                    }
                    found = True
                    break
        
        if found:
            # We found a better rank. We could try to go lower, but let's stop if we have a good one.
            # Actually, the loop continues to try lower R.
            pass

    # If we didn't find anything better, return schoolbook
    if best_rank == rank_schoolbook:
        return best_solution
    else:
        return best_solution

# Fallback if time runs out or no better solution found
# The function above already handles this by initializing best_solution to schoolbook.
