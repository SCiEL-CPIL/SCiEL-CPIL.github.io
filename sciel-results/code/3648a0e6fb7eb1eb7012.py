"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import itertools
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def _check_decomposition(U, V, W, n, m, p):
    """Verify that U @ V.T @ W decomposes the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i, j, k] = 1 if and only if it corresponds to multiplication.
    T[i, j, k] = sum_c U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
    Should equal 1 for all i,j,k (for the standard matmul tensor).
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[i*m + j][c]) * Fraction(V[j*p + k][c]) * Fraction(W[k*n + i][c])
                if total != 1:
                    return False
    return True

def _count_nonzero_cols(U, V, W):
    """Count the effective rank (number of columns that are not all-zero in at least one matrix)."""
    # The rank is the number of columns in U (and V, W)
    return len(U[0]) if U else 0

def _to_fraction_list(matrix):
    """Convert numpy matrix to list of lists of Fractions."""
    return [[Fraction(str(x)) for x in row] for row in matrix]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = start_time + 145.0  # 5s safety margin
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find a lower-rank decomposition using structured search
    # Strategy: Try small integer/half-integer values in a structured way
    # For n=3, m=4, p=7, schoolbook is 84.
    # Known lower bounds for matmul rank:
    # - Trivial lower bound: n*p (output size) = 21, m*p (V rows) = 28, n*m (U rows) = 12
    # - The true rank is somewhere between these and 84
    
    # Let's try a randomized search with restarts, looking for rank in a reasonable range
    # Start from rank_school and try to reduce
    # Use a simple iterative refinement: random initialization, then gradient-like search
    
    # For speed, use numpy and check periodically
    # Given the constraints (exact decomposition), random search is very hard.
    # Instead, let's try a known technique: exploit structure via tensor contraction
    
    # Actually, for small dimensions, let's try a more systematic approach:
    # Build the tensor T and try to find a low-rank decomposition
    # But T is 3*4*7 = 84 entries, each should be 1
    
    # A better approach: use the fact that matmul tensor has structure
    # T[i,j,k] = delta (it's all 1s in the sense that each (i,j,k) tuple appears once)
    # Actually the matmul tensor is T[i,j,k] = 1 for all i,j,k (it's the identity in the tensor sense)
    # No, that's not right. The matmul tensor maps (A, B) -> C where C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T[i,k,j] = 1 (it selects the j-th product)
    
    # Let me re-examine: we need U, V, W such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # This is equivalent to: for each (i,j,k), there's a rank-1 component that is 1 at that position
    # Schoolbook achieves this with exactly one component per (i,j,k)
    
    # To get lower rank, we need some components to cover multiple (i,j,k) positions
    # This is possible if the components have a special structure
    
    # Let's try a simple optimization: use random real matrices, then round to half-integers
    # and verify. This is a heuristic.
    
    np.random.seed(42)
    
    # Try several ranks from low to high
    min_possible = max(n*m, m*p, p*n)  # = 28
    max_try = min(rank_school - 1, 60)
    
    found_lower = False
    
    # Try rank = 40 to 60 with random search
    for target_rank in range(40, max_try + 1):
        if time.time() > DEADLINE - 30:
            break
        if found_lower:
            break
            
        for attempt in range(50):
            if time.time() > DEADLINE - 15:
                    break
                    
            # Random initialization with small integer values
            U = np.random.randint(-2, 3, size=(n*m, target_rank))
            V = np.random.randint(-2, 3, size=(m*p, target_rank))
            W = np.random.randint(-2, 3, size=(p*n, target_rank))
            
            # Quick check: compute the full tensor and see how close we are
            # T_reconstructed[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # This is expensive for each (i,j,k), so let's use numpy vectorization
            
            # Build index arrays
            success = True
            # Sample a subset of (i,j,k) for speed
            indices = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        indices.append((i, j, k))
            
            # Check all
            for (i, j, k) in indices:
                total = 0
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                for c in range(target_rank):
                    total += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                if total != 1:
                    success = False
                    break
            
            if success:
                found_lower = True
                best_rank = target_rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
        
        if found_lower:
            break
    
    # Also try half-integer values (multiply by 2, use odd integers, then divide)
    if not found_lower and time.time() < DEADLINE - 20:
        for target_rank in range(40, max_try + 1):
            if time.time() > DEADLINE - 10:
                break
            if found_lower:
                break
                
            for attempt in range(30):
                if time.time() > DEADLINE - 5:
                    break
                    
                # Random odd integers (representing half-integers)
                U = np.random.randint(-4, 5, size=(n*m, target_rank)) * 2 + 1  # odd integers
                V = np.random.randint(-4, 5, size=(m*p, target_rank)) * 2 + 1
                W = np.random.randint(-4, 5, size=(p*n, target_rank)) * 2 + 1
                
                indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
                success = True
                for (i, j, k) in indices:
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    total = 0
                    for c in range(target_rank):
                        total += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                    # total should be 8 (since each entry is doubled, 1 * 8 = 8 when we divide by 8 later)
                    # Actually: if U,V,W are odd ints, their product is odd, sum of odd*rank terms...
                    # This heuristic is flawed. Let's just check if total/8 is close to 1
                    if total != 8:  # 2*2*2 = 8 scaling
                        success = False
                        break
                
                if success:
                    found_lower = True
                    best_rank = target_rank
                    # Convert back to half-integers
                    U_frac = [[U[i][c] / 2 for c in range(target_rank)] for i in range(n*m)]
                    V_frac = [[V[i][c] / 2 for c in range(target_rank)] for i in range(m*p)]
                    W_frac = [[W[i][c] / 2 for c in range(target_rank)] for i in range(p*n)]
                    best_result = {"U": U_frac, "V": V_frac, "W": W_frac}
                    break
        
    if time.time() > DEADLINE:
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return best_result
# EVOLVE-BLOCK-END
