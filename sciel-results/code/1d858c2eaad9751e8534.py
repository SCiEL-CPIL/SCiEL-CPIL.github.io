"""Matrix multiplication 2x3x4 rank-minimal decomposition."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U,V,W exactly decompose the multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c]
    """
    rank = U.shape[1]
    # Build multiplication tensor T[i,j,k] = 1 if we need C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor equation: sum_c U[i*m+j,c] * V[j*p+k,c] = W[k*n+i,c] for each (i,j,k)
    # Actually the decomposition is: T_{i,j,k} = sum_c U_{i*m+j,c} * V_{j*p+k,c} * W_{k*n+i,c}
    # Wait, let me re-read the convention.
    # W[k*n+i, r] - so W is indexed by (k, i) where k is row of B (0..p-1), i is col of A (0..n-1)
    # The tensor for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The multiplication tensor T has T[i,j,k] = 1 for all valid (i,j,k)
    # Rank-r decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_round_and_verify(U, V, W, n, m, p):
    """Try to round to nearest 0.5 and verify."""
    U_r = [[round(v * 2) / 2 for v in row] for row in U]
    V_r = [[round(v * 2) / 2 for v in row] for row in V]
    W_r = [[round(v * 2) / 2 for v in row] for row in W]
    if _verify(U_r, V_r, W_r, n, m, p, tol=1e-9):
        return U_r, V_r, W_r
    # Try rounding to integers
    U_i = [[round(v) for v in row] for row in U]
    V_i = [[round(v) for v in row] for row in V]
    W_i = [[round(v) for v in row] for row in W]
    if _verify(U_i, V_i, W_i, n, m, p, tol=1e-9):
        return U_i, V_i, W_i
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Known lower bound: for 2x3x4 matrix multiplication,
    # The rank is known to be 16 (from literature on matrix multiplication ranks)
    # Let's search from low rank upward.
    
    # The schoolbook rank is n*m*p = 24
    # We know from Strassen-type results that for small matrices,
    # the rank can be less than n*m*p.
    # For 2x3x4, let's try ranks from 12 to 24.
    
    # Actually, let me think about this more carefully.
    # The tensor T has dimensions (n*m) x (m*p) x (p*n) = 6 x 12 x 8
    # We need rank-r decomposition where:
    # T[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # for all i in [0,2), j in [0,3), k in [0,4)
    
    # That's 2*3*4 = 24 equations, and we have r*(6+12+8) = 26r unknowns.
    # For r=1: 26 unknowns, 24 equations - might work but unlikely
    # For r=2: 52 unknowns, 24 equations - likely solvable
    
    # Let's use a random search approach with gradient-free optimization
    # to find low-rank decompositions with half-integer entries.
    
    best_result = None
    
    # Start with schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try to find lower rank decompositions
    # Use random initialization and local search
    
    for target_rank in range(12, rank_sb + 1):
        if time.time() > deadline - 10:
            break
        
        found = False
        max_attempts = 200 if target_rank < 20 else 50
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            if found:
                break
            
            # Random initialization with half-integer values
            # Use small set of possible values: -1, -0.5, 0, 0.5, 1, 1.5, 2
            possible_vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
            
            U = [[random.choice(possible_vals) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice(possible_vals) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice(possible_vals) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Local search: randomly perturb one entry at a time
            best_err = float('inf')
            for iteration in range(500):
                if time.time() > deadline - 3:
                    break
                
                # Compute current error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(target_rank):
                                val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                            err += (val - 1.0) ** 2
                
                if err < 1e-12:
                    found = True
                    break
                
                if err < best_err:
                    best_err = err
                    if err < 1e-9:
                        found = True
                        break
                
                # Random perturbation
                choice = random.random()
                if choice < 0.33:
                    row = random.randrange(n * m)
                    col_idx = random.randrange(target_rank)
                    U[row][col_idx] = random.choice(possible_vals)
                elif choice < 0.66:
                    row = random.randrange(m * p)
                    col_idx = random.randrange(target_rank)
                    V[row][col_idx] = random.choice(possible_vals)
                else:
                    row = random.randrange(p * n)
                    col_idx = random.randrange(target_rank)
                    W[row][col_idx] = random.choice(possible_vals)
            
            if found:
                # Verify
                if _verify(U, V, W, n, m, p, tol=1e-9):
                    best_result = {"U": U, "V": V, "W": W}
                    break
        
        if found:
            break
    
    return best_result
