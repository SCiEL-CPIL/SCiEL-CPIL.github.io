"""Rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction
from itertools import product, combinations
import random

DIMENSIONS = [4, 5, 6]
DEADLINE = 145.0  # safety margin before the 150s kill


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS  # 4, 5, 6
    nm, mp, pn = n * m, m * p, p * n  # 20, 30, 24
    
    # Start with schoolbook rank 120 as fallback
    rank_fallback = nm * mp * pn // n  # Actually n*m*p = 120
    rank_fallback = n * m * p
    U_fb = [[0] * rank_fallback for _ in range(nm)]
    V_fb = [[0] * rank_fallback for _ in range(mp)]
    W_fb = [[0] * rank_fallback for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_fallback
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    def time_left():
        return DEADLINE - (time.time() - t0)
    
    # Subspace method: try to find lower rank decompositions
    # For each candidate rank r, try to find U, V, W such that
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta[i,j,k] for all (i,j,k)
    # This is a system of nonlinear equations, but we can use the subspace approach.
    
    # Strategy: Fix sparse patterns for U and V, then solve for W
    # Or use random initialization with iterative refinement
    
    # Try ranks from a reasonable lower bound up to fallback
    # Lower bound for matrix multiplication rank is at least max(n*m, m*p, p*n) = 30
    # But for 4x5*5x6, known results suggest rank around 60-80
    
    random.seed(42)
    
    for target_rank in [60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115]:
        if time_left() < 5:
            break
        if target_rank >= best_rank:
            continue
        
        # Try multiple random initializations
        for attempt in range(20):
            if time_left() < 2:
                break
            
            # Random sparse initialization for U and V
            # Each column of U should have support in exactly one (i,j) pair
            # Each column of V should have support in exactly one (j,k) pair
            # This ensures the tensor structure is preserved
            
            U_try = np.zeros((nm, target_rank), dtype=float)
            V_try = np.zeros((mp, target_rank), dtype=float)
            
            for r in range(target_rank):
                # Randomly assign which (i,j) this rank component acts on
                ij = random.randint(0, nm - 1)
                i, j = divmod(ij, m)
                U_try[ij, r] = random.choice([1, -1])
                
                # Randomly assign which (j,k) this rank component acts on
                jk = random.randint(0, mp - 1)
                jv, k = divmod(jk, p)
                V_try[jk, r] = random.choice([1, -1])
            
            # Now solve for W: for each (k,i) and rank r,
            # W[k*n+i, r] = sum over j of delta[i,j,k] / (U[i*m+j,r] * V[j*p+k,r])
            # But this only works if the system is consistent
            
            # Actually, let's set up the linear system properly
            # The tensor equation is: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 if consistent
            # Wait, the tensor is T[i,j,k] = 1 for all valid (i,j,k) in the multiplication
            # No, the matrix multiplication tensor has T[i,j,k] = 1 iff the output (i,k) depends on input (j)
            # Actually for matrix multiplication, the tensor is:
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # So the tensor T has T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
            # And the decomposition is: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            
            # For fixed U and V, W is determined by:
            # For each (i,k), W[k*n+i, r] appears in equations for all j
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for each j
            
            # This is a linear system in W for each (i,k)
            # But W[k*n+i, r] is the same across all j for fixed (i,k)
            # So for fixed (i,k), we have m equations in target_rank unknowns W[k*n+i, 0..target_rank-1]
            
            # Let's build the system: for each (i,k), A_{j,r} = U[i*m+j,r] * V[j*p+k,r]
            # Then A @ w_{i,k} = ones(m) where w_{i,k} is W[k*n+i, :]
            
            W_try = np.zeros((pn, target_rank), dtype=float)
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build m x target_rank matrix
                    A_mat = np.zeros((m, target_rank))
                    for j in range(m):
                        for r in range(target_rank):
                            A_mat[j, r] = U_try[i * m + j, r] * V_try[j * p + k, r]
                    
                    # Solve A_mat @ w = ones(m)
                    # This is an underdetermined system (m equations, target_rank unknowns)
                    # Use least squares or find a particular solution
                    try:
                        w_sol, residuals, rank_A, sv = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                        # Check if solution is valid (residuals should be small)
                        if np.max(np.abs(A_mat @ w_sol - np.ones(m))) > 1e-10:
                            valid = False
                            break
                        W_try[k * n + i, :] = w_sol
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Verify the decomposition
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U_try[i * m + j, r] * V_try[j * p + k, r] * W_try[k * n + i, r] 
                                  for r in range(target_rank))
                        max_err = max(max_err, abs(val - 1))
            
            if max_err < 1e-10:
                # Convert to fractions for exactness
                U_frac = [[Fraction(U_try[i, r]).limit_denominator(1000000) for r in range(target_rank)] 
                         for i in range(nm)]
                V_frac = [[Fraction(V_try[i, r]).limit_denominator(1000000) for r in range(target_rank)] 
                         for i in range(mp)]
                W_frac = [[Fraction(W_try[i, r]).limit_denominator(1000000) for r in range(target_rank)] 
                         for i in range(pn)]
                
                # Verify with fractions
                max_err_frac = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U_frac[i * m + j][r] * V_frac[j * p + k][r] * W_frac[k * n + i][r] 
                                     for r in range(target_rank))
                            max_err_frac = max(max_err_frac, abs(val - 1))
                
                if max_err_frac == 0:
                    best_rank = target_rank
                    best_U = U_frac
                    best_V = V_frac
                    best_W = W_frac
                    break  # Found valid decomposition for this rank, try lower
            elif max_err < 1e-6:
                # Close but not exact, try to polish
                pass
    
    # Return the best found
    if best_rank < rank_fallback:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U_fb, "V": V_fb, "W": W_fb}
