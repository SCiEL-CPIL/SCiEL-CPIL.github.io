"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]
DEADLINE = 140  # seconds, with safety margin

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try to find lower-rank decompositions using structured approaches
    for target_rank in range(rank_schoolbook - 1, 39, -1):
        if time.time() - start_time > DEADLINE:
            break
        
        result = _try_decomposition(n, m, p, target_rank)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break
    
    return best_result

def _try_decomposition(n, m, p, target_rank):
    """Try to find a decomposition of the given rank using structured patterns."""
    start_time = time.time()
    
    # Approach: Use the fact that matrix multiplication tensor has structure.
    # Try Kronecker-like decompositions.
    
    # For 3x4x5: split m=4 into 2+2
    # This gives two sub-tensors: 3x2x5 and 3x2x5
    # Each 3x2x5 has rank 30, total 60 (no improvement)
    
    # Try: split p=5 into 2+3
    # 3x4x2 (rank 24) + 3x4x3 (rank 36) = 60 (no improvement)
    
    # Try Strassen-like: for the 2x2x2 core, rank 7 instead of 8
    # If we can embed Strassen into the 3x4x5 problem...
    
    # One approach: view 3x4x5 as containing a 2x2x2 subproblem
    # Use Strassen for that part, saving 1 rank
    
    # Let's try a more systematic approach: 
    # Use random initialization with half-integers and verify
    
    # Actually, let's try a specific structured decomposition:
    # For the 4x5 multiplication part, use the fact that 4=2*2
    
    # Try: decompose using block structure
    # A (3x4) = [A1 A2] where A1, A2 are 3x2
    # B (4x5) = [B1; B2] where B1, B2 are 2x5
    # C = A1*B1 + A2*B2
    
    # Each 3x2x5 multiplication has rank 30
    # But we can potentially share some structure
    
    # For now, try a randomized search with structured constraints
    # Use values in {0, 1/2, 1} and verify
    
    import random
    random.seed(42)
    
    max_attempts = 100
    for attempt in range(max_attempts):
        if time.time() - start_time > 30:  # Time limit per rank
            break
        
        # Initialize with random half-integers
        U = np.random.choice([0, 0.5, 1], size=(n*m, target_rank))
        V = np.random.choice([0, 0.5, 1], size=(m*p, target_rank))
        W = np.random.choice([0, 0.5, 1], size=(p*n, target_rank))
        
        # Verify: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Should equal 1 for all valid (i,j,k)
        
        # This is too slow for direct verification with large tensors
        # Instead, check a subset of constraints
        
        # Check all (i,j,k) tuples
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r in range(target_rank):
                        actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(actual - expected) > 1e-9:
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
                "W": W.tolist()
            }
    
    return None
# EVOLVE-BLOCK-END
