import numpy as np
import time
from math import gcd
from functools import reduce

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    # The known rank for 2x3 * 3x6 is 24 (from literature on small matrix multiplication)
    # Actually, let's think: rank <= n*m*p = 36 (schoolbook)
    # Known bounds: rank(2x3 x 3x6) is at most 24? 
    # Actually, the trivial bound is 36. But we can do better.
    # Let's try to find a rank-minimal decomposition.
    
    # First, let's establish a valid baseline using a structured approach.
    # For 2x3 * 3x6, we can use the following:
    # C = A * B where A is 2x3, B is 3x6
    
    # Let's try a systematic search for low-rank decompositions.
    # We'll use a greedy/iterative refinement approach.
    
    def compute_decomposition(U, V, W):
        """Compute C[i,k] = sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]"""
        C = np.zeros((n, p), dtype=float)
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for c in range(U.shape[1]):
                        C[i, k] += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
        return C
    
    def is_valid(U, V, W, atol=1e-9):
        """Check if the decomposition gives C = ones(n,p)"""
        C = compute_decomposition(U, V, W)
        return np.allclose(C, np.ones((n, p)), atol=atol)
    
    # Try to construct a valid decomposition of minimal rank
    # Start with a known construction
    
    # Strategy 1: Try rank from 16 to 36, searching for valid decompositions
    # Use a smarter initialization based on Kronecker-like structures
    
    # For 2x3 * 3x6, one approach:
    # Think of it as computing 12 entries, each is a sum of 3 products.
    # Total scalar products needed: 12 * 3 = 36 (schoolbook)
    # But we can share computations.
    
    # Let's try a random search with structured initialization
    
    best_result = None
    best_rank = None
    
    # First, let's get a valid baseline
    # Use the schoolbook method as a guaranteed valid solution
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook), dtype=float)
    V = np.zeros((m * p, rank_schoolbook), dtype=float)
    W = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    if is_valid(U, V, W):
        best_rank = rank_schoolbook
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now try to find lower rank solutions
    # We'll search from lower ranks up to the best known
    
    # For each rank, try multiple random initializations
    # Use {-1, 0, 1} for U and V, then solve for W
    
    def solve_for_W(U, V):
        """Given U and V, solve for W such that C = ones"""
        R = U.shape[1]
        
        # S[i, k, c] = sum_j U[i*m+j, c] * V[j*p+k, c]
        U_reshaped = U.reshape(n, m, R)
        V_reshaped = V.reshape(m, p, R)
        
        S = np.zeros((n, p, R), dtype=float)
        for i in range(n):
            for k in range(p):
                for c in range(R):
                    S[i, k, c] = np.sum(U_reshaped[i, :, c] * V_reshaped[:, k, c])
        
        # We need W[k*n+i, c] such that sum_c S[i, k, c] * W[k*n+i, c] = 1
        # This is a linear system for each (i, k)
        
        W = np.zeros((p * n, R), dtype=float)
        for i in range(n):
            for k in range(p):
                s = S[i, k, :]
                # Solve s @ w_row = 1
                # If s is all zeros, no solution
                if np.allclose(s, 0, atol=1e-12):
                    return None
                # Use least squares or pseudo-inverse
                # Prefer exact solutions when possible
                # Check if we can get a simple rational solution
                w_row, residuals, rank_s, _ = np.linalg.lstsq(s.reshape(-1, 1), np.ones(1), rcond=1e-12)
                w_row = w_row.flatten()
                W[k * n + i, :] = w_row
        
        return W
    
    # Search for lower rank solutions
    for R in range(16, best_rank):
        if time.time() > deadline - 5:
            break
        
        found = False
        max_attempts = 5000 if R <= 24 else 2000
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Generate random U and V with entries in {-1, 0, 1}
            # Use sparsity to make the problem more tractable
            U = np.random.choice([-1, 0, 1], size=(n * m, R), p=[0.2, 0.4, 0.4])
            V = np.random.choice([-1, 0, 1], size=(m * p, R), p=[0.2, 0.4, 0.4])
            
            # Ensure U and V don't have all-zero columns
            # (all-zero columns in U or V make that rank component useless)
            U_nonzero = np.any(U != 0, axis=0)
            V_nonzero = np.any(V != 0, axis=0)
            
            # Remove zero columns
            valid_cols = U_nonzero & V_nonzero
            if np.sum(valid_cols) < min(R, n * p):
                continue
            
            U = U[:, valid_cols]
            V = V[:, valid_cols]
            R_eff = U.shape[1]
            
            if R_eff < R:
                continue
            
            # Solve for W
            W = solve_for_W(U, V)
            if W is None:
                continue
            
            # Check if the decomposition is valid
            if is_valid(U, V, W, atol=1e-8):
                # Convert to half-integers if possible
                # Check if all entries are multiples of 0.5
                all_half = True
                for mat in [U, V, W]:
                    if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-9):
                        all_half = False
                        break
                
                if all_half:
                    found = True
                    best_rank = R
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
        
        if found:
            # Found a solution at this rank, try lower ranks
            continue
    
    # If we found a better solution, return it
    if best_result is not None:
        return best_result
    
    # Fallback: return schoolbook
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
