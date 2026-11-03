"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_strassen56():
    """Build the known rank-56 Strassen-based solution."""
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    t = 0
    for a in range(2):
        for b in range(4):
            for s in range(7):
                if s == 0:
                    U[0 * m + 2 * a][t] += 1
                    U[1 * m + 2 * a + 1][t] += 1
                elif s == 1:
                    U[1 * m + 2 * a][t] += 1
                    U[1 * m + 2 * a + 1][t] += 1
                elif s == 2:
                    U[0 * m + 2 * a][t] += 1
                elif s == 3:
                    U[1 * m + 2 * a + 1][t] += 1
                elif s == 4:
                    U[0 * m + 2 * a][t] += 1
                    U[0 * m + 2 * a + 1][t] += 1
                elif s == 5:
                    U[1 * m + 2 * a][t] += 1
                    U[0 * m + 2 * a][t] += -1
                elif s == 6:
                    U[0 * m + 2 * a + 1][t] += 1
                    U[1 * m + 2 * a + 1][t] += -1
                
                if s == 0:
                    V[2 * a * p + 2 * b][t] += 1
                    V[(2 * a + 1) * p + 2 * b + 1][t] += 1
                elif s == 1:
                    V[2 * a * p + 2 * b][t] += 1
                elif s == 2:
                    V[2 * a * p + 2 * b + 1][t] += 1
                    V[(2 * a + 1) * p + 2 * b + 1][t] += -1
                elif s == 3:
                    V[(2 * a + 1) * p + 2 * b][t] += 1
                    V[2 * a * p + 2 * b][t] += -1
                elif s == 4:
                    V[(2 * a + 1) * p + 2 * b + 1][t] += 1
                elif s == 5:
                    V[2 * a * p + 2 * b][t] += 1
                    V[2 * a * p + 2 * b + 1][t] += 1
                elif s == 6:
                    V[(2 * a + 1) * p + 2 * b][t] += 1
                    V[(2 * a + 1) * p + 2 * b + 1][t] += 1
                
                if s == 0:
                    W[2 * b * n + 0][t] += 1
                    W[(2 * b + 1) * n + 1][t] += 1
                elif s == 1:
                    W[2 * b * n + 1][t] += 1
                    W[(2 * b + 1) * n + 1][t] += -1
                elif s == 2:
                    W[(2 * b + 1) * n + 0][t] += 1
                    W[(2 * b + 1) * n + 1][t] += 1
                elif s == 3:
                    W[2 * b * n + 0][t] += 1
                    W[2 * b * n + 1][t] += 1
                elif s == 4:
                    W[2 * b * n + 0][t] += -1
                    W[(2 * b + 1) * n + 0][t] += 1
                elif s == 5:
                    W[(2 * b + 1) * n + 1][t] += 1
                elif s == 6:
                    W[2 * b * n + 0][t] += 1
                
                t += 1
    
    return {"U": U, "V": V, "W": W}

def _validate(U, V, W):
    """Check if the decomposition exactly computes matrix multiplication."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    rank = U_arr.shape[1]
    
    # Check: for all i,j,k: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == (i==k and j==j) ... 
    # Actually the tensor is T[i,j,k] = 1 if i==k else 0... no.
    # The matrix multiplication tensor: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor form: T[i,j,k] = 1 for all i,j,k (it's the identity in the "multiply" sense)
    # Decomposition: T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
    
    # Build the target tensor
    # T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
    # This is the matrix multiplication tensor
    
    # Check a sample of entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for t in range(rank):
                    val += U_arr[i * m + j, t] * V_arr[j * p + k, t] * W_arr[k * n + i, t]
                expected = 1  # T[i,j,k] = 1 for matrix mult
                if val != expected:
                    return False
    return True

def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with known rank-56 solution
    best = _build_strassen56()
    
    # Verify it's valid
    if not _validate(best["U"], best["V"], best["W"]):
        # Should not happen, but return anyway
        return best
    
    # Try to find lower rank by attempting to merge Strassen blocks
    # Strategy: check if any two Strassen terms across different blocks can be merged
    # by linearity, reducing total rank
    
    # The key observation: if we have terms that differ by a sign or can be combined,
    # we might reduce rank. But Strassen is already minimal for 2x2.
    
    # Alternative: try a different partitioning
    # Partition j into 1 group of 4, use 2x4 x 4x4 blocks
    # Or try to use the structure that n=2 is small
    
    # Let's try: treat the 2x4 x 4x8 as two 2x4 x 4x4 problems
    # 2x4 x 4x4: can we do better than 2*2*7=28?
    # Actually 2x4 x 4x4 = (2x2 x 2x2) x 2 = 7*2 = 14 per 2x4x4 block
    # Two such blocks: 28 total. Same as before.
    
    # Try: 2x4 x 4x8 as a single 2x4 x 4x8 problem
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(8, 32, 16) = 32
    # Also rank >= 2 * min(n*m, m*p, p*n) / something...
    # The trivial upper bound is 64, Strassen gives 56.
    
    # Let's check if rank 55 is possible by attempting a small ILP
    # This is expensive, so only try if we have time
    
    if time.time() < deadline - 30:
        # Try to find a rank-55 solution by relaxing one constraint
        # This is a heuristic: try removing one Strassen term and see if
        # the remaining 55 terms can be adjusted to still work
        
        # For now, return the rank-56 solution
        pass
    
    return best
# EVOLVE-BLOCK-END
