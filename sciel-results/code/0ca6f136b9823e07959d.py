"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """Find rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Index helpers
    def idx_u(i, j):
        return i * m + j
    
    def idx_v(j, k):
        return j * p + k
    
    def idx_w(k, i):
        return k * n + i
    
    # Schoolbook decomposition as baseline
    schoolbook_rank = n * m * p  # 72
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[idx_u(i, j), col] = 1.0
                V_sb[idx_v(j, k), col] = 1.0
                W_sb[idx_w(k, i), col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = schoolbook_rank
    
    def verify(U, V, W):
        """Verify that U, V, W exactly decompose the all-ones tensor."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[idx_u(i, j), r] * V[idx_v(j, k), r] * W[idx_w(k, i), r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def is_valid_entries(U, V, W):
        """Check that all entries are integer or half-integer."""
        for mat in [U, V, W]:
            for val in mat.flatten():
                v = round(val * 2)
                if abs(val * 2 - v) > 1e-9:
                    return False
        return True
    
    # Try structured decompositions with various ranks
    # Strategy: Use a "block" structure where we partition ranks into groups
    
    # For rank r, try to find U, V, W with entries in {0, 1, 1/2}
    # such that the tensor equation holds.
    
    # Key insight: For the all-ones tensor, we can use a "covering" approach.
    # Each rank-1 term covers a subset of (i,j,k) triples.
    # We need the sum of indicators to be exactly 1 for each triple.
    
    # Let's try a systematic approach:
    # For each candidate rank from lower_bound to schoolbook_rank:
    #   Try to construct a valid decomposition using random/greedy methods.
    
    lower_bound = max(n * m, m * p, p * n)  # 24
    
    # Try ranks from lower_bound upward
    for target_rank in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline - 30:
            break
        
        # Method 1: Structured "factor" decomposition
        # Try to express the all-ones tensor as a product of structured factors
        
        # For n=3, m=3, p=8, try rank = 24, 25, 26, ...
        # Use a "grid" structure: divide ranks into blocks
        
        # Simple structured attempt: 
        # If we can find matrices A (n x r1), B (m x r2), C (p x r3)
        # such that the outer product structure works...
        
        # Actually, let's try a direct random search with local optimization.
        
        # Initialize with random binary/half-integer values
        found = False
        
        for attempt in range(50):
            if time.time() > deadline - 20:
                break
            
            # Random initialization with entries in {0, 1, 1/2}
            U = np.random.choice([0.0, 0.5, 1.0], size=(n * m, target_rank))
            V = np.random.choice([0.0, 0.5, 1.0], size=(m * p, target_rank))
            W = np.random.choice([0.0, 0.5, 1.0], size=(p * n, target_rank))
            
            # Local search: try flipping individual entries to reduce error
            for iteration in range(200):
                if time.time() > deadline - 10:
                    break
                
                # Compute current error
                errors = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(target_rank):
                                s += U[idx_u(i, j), r] * V[idx_v(j, k), r] * W[idx_w(k, i), r]
                            errors.append((abs(s - 1.0), i, j, k))
                
                if all(e[0] < 1e-9 for e in errors):
                    if verify(U, V, W) and is_valid_entries(U, V, W):
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_rank = target_rank
                        found = True
                        break
                
                # Find the worst error
                errors.sort(reverse=True)
                if not errors or errors[0][0] < 1e-9:
                    break
                    
                _, i, j, k = errors[0]
                
                # Try flipping entries in U, V, W that affect this (i,j,k)
                improved = False
                
                # Try flipping U[idx_u(i,j), r] for various r
                for r in range(target_rank):
                    old_val = U[idx_u(i, j), r]
                    for new_val in [0.0, 0.5, 1.0]:
                        if new_val == old_val:
                            continue
                        U[idx_u(i, j), r] = new_val
                        
                        # Check if this helps
                        s = 0.0
                        for rr in range(target_rank):
                            s += U[idx_u(i, j), rr] * V[idx_v(j, k), rr] * W[idx_w(k, i), rr]
                        new_err = abs(s - 1.0)
                        
                        if new_err < errors[0][0] - 1e-12:
                            improved = True
                            break
                        else:
                            U[idx_u(i, j), r] = old_val
                    if improved:
                        break
                
                if not improved:
                    # Try V
                    for r in range(target_rank):
                        old_val = V[idx_v(j, k), r]
                        for new_val in [0.0, 0.5, 1.0]:
                            if new_val == old_val:
                                continue
                            V[idx_v(j, k), r] = new_val
                            
                            s = 0.0
                            for rr in range(target_rank):
                                s += U[idx_u(i, j), rr] * V[idx_v(j, k), rr] * W[idx_w(k, i), rr]
                            new_err = abs(s - 1.0)
                            
                            if new_err < errors[0][0] - 1e-12:
                                improved = True
                                break
                            else:
                                V[idx_v(j, k), r] = old_val
                        if improved:
                            break
                
                if not improved:
                    # Try W
                    for r in range(target_rank):
                        old_val = W[idx_w(k, i), r]
                        for new_val in [0.0, 0.5, 1.0]:
                            if new_val == old_val:
                                continue
                            W[idx_w(k, i), r] = new_val
                            
                            s = 0.0
                            for rr in range(target_rank):
                                s += U[idx_u(i, j), rr] * V[idx_v(j, k), rr] * W[idx_w(k, i), rr]
                            new_err = abs(s - 1.0)
                            
                            if new_err < errors[0][0] - 1e-12:
                                improved = True
                                break
                            else:
                                W[idx_w(k, i), r] = old_val
                        if improved:
                            break
                
                if not improved:
                    break
            
            if found:
                break
        
        if found:
            break
    
    # If we found a better decomposition, use it; otherwise use schoolbook
    if best_rank < schoolbook_rank:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
