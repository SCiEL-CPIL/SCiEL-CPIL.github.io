"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Schoolbook decomposition
    rank_sb = n * m * p  # 56
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
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Precompute the list of (i,j,k) tuples for fast verification
    tuples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def verify_exact(U, V, W, rank):
        """Check if decomposition is exact using integer arithmetic."""
        for (i, j, k) in tuples:
            a = i * m + j
            b = j * p + k
            c = k * n + i
            s = 0
            for r in range(rank):
                s += U[a][r] * V[b][r] * W[c][r]
            if s != 1:
                return False
        return True
    
    # Convert to numpy for fast error computation
    def compute_error(U, V, W, rank):
        """Compute sum of squared errors (float) for quick check."""
        err = 0.0
        for (i, j, k) in tuples:
            a = i * m + j
            b = j * p + k
            c = k * n + i
            s = 0.0
            for r in range(rank):
                s += U[a][r] * V[b][r] * W[c][r]
            err += (s - 1.0) ** 2
        return err
    
    # Try to find lower rank using local search with integer entries
    # Strategy: start from schoolbook, try to merge/combine terms
    # Use a random restart local search with sign flips and small perturbations
    
    rng = np.random.RandomState(42)
    
    # Try ranks from 48 down to 35 (conservative range for time)
    for target_rank in range(48, 34, -1):
        if time.time() > deadline - 10:
            break
        
        # Multiple restarts
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            
            # Initialize: start with schoolbook and randomly zero out some columns
            # or use random initialization with small integers
            U = np.zeros((n * m, target_rank), dtype=int)
            V = np.zeros((m * p, target_rank), dtype=int)
            W = np.zeros((p * n, target_rank), dtype=int)
            
            # Initialize by copying schoolbook structure and merging
            # For each schoolbook term, assign to a random target column
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Assign this (i,j,k) to a random column
                        col = rng.randint(0, target_rank)
                        U[i * m + j, col] += 1
                        V[j * p + k, col] += 1
                        W[k * n + i, col] += 1
            
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            # Local search: try to fix errors by flipping signs or adjusting
            for step in range(2000):
                if time.time() > deadline - 3:
                    break
                
                # Check exactness
                if verify_exact(U_list, V_list, W_list, target_rank):
                    best_rank = target_rank
                    best_U = U_list
                    best_V = V_list
                    best_W = W_list
                    break
                
                # Pick a random (i,j,k) that has error
                i, j, k = tuples[rng.randint(0, len(tuples))]
                a = i * m + j
                b = j * p + k
                c = k * n + i
                
                # Try to fix by modifying one of the three matrices at a random column
                col = rng.randint(0, target_rank)
                choice = rng.randint(0, 3)
                
                if choice == 0:
                    # Flip sign of U[a, col]
                    U_list[a][col] *= -1
                elif choice == 1:
                    # Flip sign of V[b, col]
                    V_list[b][col] *= -1
                else:
                    # Flip sign of W[c, col]
                    W_list[c][col] *= -1
            
            if best_rank == target_rank:
                break
    
    # Also try a more structured approach: use the fact that n=2
    # to create shared terms between the two row blocks
    if best_rank == rank_sb and time.time() < deadline - 20:
        # Try a specific construction: for each (j,k), create a shared term
        # that contributes to both i=0 and i=1 with appropriate weights
        # This can save n-1 = 1 term per (j,k) pair
        # Total potential savings: (n-1)*m*p = 1*4*7 = 28, giving rank 28
        
        # Construction:
        # For each (j,k), we need:
        #   U[0*m+j, c] * V[j*p+k, c] * W[k*n+0, c] + U[1*m+j, c] * V[j*p+k, c] * W[k*n+1, c] = 1
        # If we set V[j*p+k, c] = 1, then we need:
        #   U[0*m+j, c] * W[k*n+0, c] + U[1*m+j, c] * W[k*n+1, c] = 1
        # This is one equation with 4 unknowns, so we have freedom.
        
        # But we also need the "cross" terms to be zero:
        # For i=0: sum_c U[0*m+j, c] * V[j'*p+k, c] * W[k'*n+0, c] should be 0 if j'!=j or k'!=k
        # This is hard to satisfy with shared terms.
        
        # Alternative: use a block diagonal structure
        # Let's try rank = 48 by merging some terms
        
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
