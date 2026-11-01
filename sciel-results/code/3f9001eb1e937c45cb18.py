"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    # Check dimensions
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    # For each (i, k) output, check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] 
    # equals the standard multiplication tensor
    # The multiplication tensor T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # Actually, the tensor for matmul is: T[i,j,k] = 1 (it's the identity in the middle)
    # The decomposition requires: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, let me re-read the convention. W[k*n+i, r]
    # So W is indexed by (k, i) where k is from B's column and i is from A's row
    # The tensor T[i, j, k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For standard matmul: C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (it's the full tensor of 1s in the multiplication sense)
    # Actually no. The multiplication tensor is T[i,j,k] = 1 if we're multiplying A[i,j] by B[j,k]
    # to get C[i,k]. So T[i,j,k] = 1 for all valid i, j, k.
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                # T[i,j,k] should be 1
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second budget
    
    n, m, p = DIMENSIONS
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition using greedy merging
    # Start with schoolbook and try to merge pairs of columns
    
    U = [row[:] for row in best["U"]]
    V = [row[:] for row in best["V"]]
    W = [row[:] for row in best["W"]]
    rank = best_rank
    
    # Greedy: try to merge two columns if their product can be represented
    # Column c represents a term: U_col * V_col * W_col
    # We can merge columns c1 and c2 if there exist vectors such that
    # the sum of the two rank-1 tensors can be represented as a single rank-1 tensor
    # This is generally not possible, but we can try specific patterns
    
    # Alternative structural approach: exploit the middle dimension
    # For 2x4x8, consider that the middle dimension is 4.
    # We can use the fact that 4 = 2*2 and try to use a 2x2 Strassen-like trick
    # on the middle dimension, but that typically increases rank for the other dims.
    
    # Let me try a different structural approach: use the fact that we can
    # potentially combine operations when the middle dimension has structure.
    
    # For now, let's try a simple greedy: randomly perturb and check if rank can be reduced
    # This is unlikely to work, but let's at least have the structure in place.
    
    # Better idea: try to find if there's a known lower bound or construction
    # For 2x4x8 matmul, the border rank might be lower than 64.
    # Let's try a random search for a lower-rank decomposition.
    
    np.random.seed(42)
    
    for trial_rank in range(48, 64):  # Try ranks from 48 to 63
        if time.time() > deadline - 30.0:
            break
        # Try to find a decomposition of this rank
        # Use random initialization and check
        for attempt in range(50):
            if time.time() > deadline - 30.0:
                break
            U_trial = np.random.randn(n * m, trial_rank)
            V_trial = np.random.randn(m * p, trial_rank)
            W_trial = np.random.randn(p * n, trial_rank)
            
            # Scale to make entries potentially integer/half-integer
            # For now, just check if the decomposition is valid
            valid = True
            for i in range(n):
                if not valid:
                    break
                for k in range(p):
                    if not valid:
                        break
                    for j in range(m):
                        val = 0.0
                        for c in range(trial_rank):
                            val += U_trial[i * m + j, c] * V_trial[j * p + k, c] * W_trial[k * n + i, c]
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
            
            if valid:
                # Convert to list of lists with integer/half-integer entries
                # This is a placeholder - in practice we'd need to find exact entries
                pass
    
    # Return the best found (which is the schoolbook if no improvement)
    return best
# EVOLVE-BLOCK-END
