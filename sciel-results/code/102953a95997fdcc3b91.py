"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Trivial decomposition: rank = n*m*p = 84
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    # Try to find a better decomposition using structured search
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    
    # Strategy: Use alternating minimization with exact rational reconstruction
    # For each candidate rank R, try to find U, V, W such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # We'll use a greedy approach: build up the decomposition one rank term at a time
    # Each term is a rank-1 tensor u_r ⊗ v_r ⊗ w_r
    
    # First, let's try a smarter initial guess based on the matrix structure
    
    # For 3x4 * 4x7, we can try block decompositions
    # The key insight: C[i,k] = sum_j A[i,j] * B[j,k]
    # We can group terms by j
    
    # Try ranks from 28 to 40, but limit attempts per rank
    max_attempts_per_rank = 50
    start_rank = 28
    end_rank = min(40, best_rank)
    
    for rank in range(start_rank, end_rank + 1):
        if time.time() > deadline:
            break
        
        for attempt in range(max_attempts_per_rank):
            if time.time() > deadline:
                break
            
            # Generate random sparse support patterns
            # Each column of U, V, W gets 1-3 non-zero entries
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            for r in range(rank):
                # Choose non-zero positions
                nz_u = np.random.choice(n * m, size=np.random.randint(1, 4), replace=False)
                nz_v = np.random.choice(m * p, size=np.random.randint(1, 4), replace=False)
                nz_w = np.random.choice(p * n, size=np.random.randint(1, 4), replace=False)
                
                for idx in nz_u:
                    U[idx, r] = 1
                for idx in nz_v:
                    V[idx, r] = 1
                for idx in nz_w:
                    W[idx, r] = 1
            
            # Now try to solve for scaling factors
            # For each (i,k), we need: sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # Let's fix U and V, and solve for W
            # For each i,k: sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r]) = 1
            # Let M[i,k,r] = sum_j U[i*m+j, r] * V[j*p+k, r]
            # Then for each i,k: M[i,k,:] @ W[k*n+i, :] = 1
            # This is a system of n*p equations with n*p*rank unknowns (but each equation
            # only involves rank variables for one (i,k) pair)
            
            # Actually, for each i,k, we have one equation:
            # sum_r M[i,k,r] * W[k*n+i, r] = 1
            # This is underdetermined if rank > 1, so we need additional constraints
            
            # Better approach: Use least squares to find W that minimizes error
            # Then check if the solution is exact
            
            # Build the full system
            # Variables: W[k*n+i, r] for all i,k,r
            # Equations: for each i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # This is a linear system in W
            # Number of equations: n*m*p = 84
            # Number of unknowns: p*n*rank = 21*rank
            
            # For rank >= 4, this is underdetermined, so we can find a solution
            # if the system is consistent
            
            # Let's build the matrix A where A[eq, var] = coefficient
            # eq index: (i,j,k) -> i*m*p + j*p + k
            # var index: (i,k,r) -> (k*n+i)*rank + r
            
            A = np.zeros((n * m * p, p * n * rank))
            b = np.ones(n * m * p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for r in range(rank):
                            var_idx = (k * n + i) * rank + r
                            A[eq_idx, var_idx] = U[i * m + j, r] * V[j * p + k, r]
            
            # Solve least squares
            try:
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            # Check if solution is exact
            if not np.allclose(A @ W_flat, b, atol=1e-10):
                continue
            
            # Reshape W
            W_sol = W_flat.reshape(p * n, rank)
            
            # Check if entries are close to integers or half-integers
            # and convert to exact fractions
            valid = True
            U_frac = [[Fraction(0) for _ in range(rank)] for _ in range(n * m)]
            V_frac = [[Fraction(0) for _ in range(rank)] for _ in range(m * p)]
            W_frac = [[Fraction(0) for _ in range(rank)] for _ in range(p * n)]
            
            for i in range(n * m):
                for r in range(rank):
                    val = U[i, r]
                    # Try to round to nearest integer or half-integer
                    rounded = round(val * 2) / 2
                    if abs(val - rounded) > 1e-8:
                        valid = False
                        break
                    U_frac[i][r] = Fraction(rounded).limit_denominator(100)
                if not valid:
                    break
            
            if not valid:
                continue
            
            for i in range(m * p):
                for r in range(rank):
                    val = V[i, r]
                    rounded = round(val * 2) / 2
                    if abs(val - rounded) > 1e-8:
                        valid = False
                        break
                    V_frac[i][r] = Fraction(rounded).limit_denominator(100)
                if not valid:
                    break
            
            if not valid:
                continue
            
            for i in range(p * n):
                for r in range(rank):
                    val = W_sol[i, r]
                    rounded = round(val * 2) / 2
                    if abs(val - rounded) > 1e-8:
                        valid = False
                        break
                    W_frac[i][r] = Fraction(rounded).limit_denominator(100)
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Verify the decomposition exactly
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = Fraction(0)
                        for r in range(rank):
                            s += U_frac[i * m + j][r] * V_frac[j * p + k][r] * W_frac[k * n + i][r]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = rank
                best_U = [[float(x) for x in row] for row in U_frac]
                best_V = [[float(x) for x in row] for row in V_frac]
                best_W = [[float(x) for x in row] for row in W_frac]
                break
        
        if best_rank == rank:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
