"""
Matrix multiplication 2x5 x 5x5: rank-minimal exact tensor decomposition.

The tensor T has indices (i, j, k) for i in range(2), j in range(5), k in range(5).
We need U (10 x r), V (25 x r), W (10 x r) such that:
  sum_c U[i*5+j, c] * V[j*5+k, c] * W[k*2+i, c] = 1  for all valid (i,j,k)

Strategy:
1. Use known lower bounds and known constructions for small matrix multiplication.
2. For 2x5x5, the schoolbook rank is 50.
3. We can try to use Strassen-like ideas or known results.

Known results for small matrix multiplication:
- 2x2x2: rank 7 (Strassen)
- 2x2xn: various results
- For 2xm x mxm, there are known bounds.

A practical approach:
1. Start with schoolbook rank 50.
2. Try to find lower rank decompositions using:
   a. Known algebraic constructions
   b. Randomized search with structured initialization
   c. Linear algebra approaches

For 2x5x5, let's think about what's possible:
- The tensor has 50 entries (2*5*5), all equal to 1.
- We need sum_c U[i*5+j, c] * V[j*5+k, c] * W[k*2+i, c] = 1

One approach: use the fact that we can factor the problem.
For example, if we could find a rank-r decomposition where each "column" c
contributes to multiple (i,j,k) triples, we could reduce the rank.

Let me try a more systematic approach:
1. Fix the structure of U, V, W based on known patterns.
2. Use randomized search to find valid decompositions.

Actually, let me think about this differently. The key insight is that for matrix
multiplication, the tensor has a specific structure. We can try to exploit this.

For 2x5x5:
- U has 10 rows (i*5+j for i in {0,1}, j in {0,...,4})
- V has 25 rows (j*5+k for j in {0,...,4}, k in {0,...,4})
- W has 10 rows (k*2+i for k in {0,...,4}, i in {0,1})

The constraint is: for each (i,j,k), sum_c U[i*5+j,c] * V[j*5+k,c] * W[k*2+i,c] = 1

Let me try a constructive approach:
1. Start with a base decomposition (schoolbook, rank 50).
2. Try to merge or combine terms to reduce rank.

Or use a different strategy:
1. Use the fact that we can have U, V, W with small integer/half-integer entries.
2. Try ranks from 50 downward, using randomized search.

Given the time constraint, I'll implement:
1. A fast validation function using NumPy.
2. A randomized search that tries various rank values.
3. Keep the best (lowest rank) valid decomposition found.

Let me also consider known results:
- For 2x2x2, Strassen gives rank 7.
- For 2x3x3, the best known rank is 17 (vs 18 schoolbook).
- For 2x4x4, the best known rank might be around 28-30 (vs 32 schoolbook).
- For 2x5x5, let's estimate: the schoolbook is 50. A reasonable target might be 40-45.

Let me implement a search that tries to find decompositions with rank < 50.
"""

import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]

def validate_fast(U, V, W, n, m, p):
    """Validate that U, V, W reconstruct the matrix multiplication tensor exactly."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    
    r = U.shape[1]
    
    # For each (i, j, k), check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def schoolbook_decomposition(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def try_rank_randomized(r, n, m, p, timeout, max_attempts=5000):
    """Try to find a rank-r decomposition using randomized search."""
    start_time = time.time()
    
    # We need to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    
    # Strategy: fix some matrices randomly, solve for others.
    # Or use a more structured approach.
    
    # Let's try: initialize U, V, W with random small values, then iteratively refine.
    # Or: use the fact that we can set some entries to 0 and solve.
    
    # A simpler approach: try random sparse matrices with small integer/half-integer entries.
    
    values = [-2, -1, -0.5, 0.5, 1, 2]
    
    for attempt in range(max_attempts):
        if time.time() - start_time > timeout:
            break
        
        # Random sparse initialization
        density = 0.1 + 0.2 * random.random()
        
        U = np.zeros((n * m, r), dtype=np.float64)
        V = np.zeros((m * p, r), dtype=np.float64)
        W = np.zeros((p * n, r), dtype=np.float64)
        
        for row in range(n * m):
            for col in range(r):
                if random.random() < density:
                    U[row, col] = random.choice(values)
        
        for row in range(m * p):
            for col in range(r):
                if random.random() < density:
                    V[row, col] = random.choice(values)
        
        for row in range(p * n):
            for col in range(r):
                if random.random() < density:
                    W[row, col] = random.choice(values)
        
        if validate_fast(U, V, W, n, m, p):
            # Convert to list of lists with clean values
            U_list = [[_clean_val(U[i, c]) for c in range(r)] for i in range(n * m)]
            V_list = [[_clean_val(V[i, c]) for c in range(r)] for i in range(m * p)]
            W_list = [[_clean_val(W[i, c]) for c in range(r)] for i in range(p * n)]
            return U_list, V_list, W_list
    
    return None

def _clean_val(x):
    """Clean a float value to integer or half-integer if possible."""
    if abs(x - round(x)) < 1e-9:
        return int(round(x))
    elif abs(2 * x - round(2 * x)) < 1e-9:
        return round(2 * x) / 2.0
    return x

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = schoolbook_decomposition(n, m, p)
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Try ranks from 49 down to some lower bound
    # Lower bound: at least max(n*m, m*p, p*n) = 25, but realistically higher.
    
    # Let's try a range of ranks
    for r in range(best_rank - 1, 24, -1):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        # Allocate more time for lower ranks (harder to find)
        remaining = deadline - elapsed - 5
        if remaining <= 0:
            break
        
        # Try with a timeout proportional to remaining time
        timeout_for_rank = min(3.0, remaining * 0.5)
        if timeout_for_rank < 0.5:
            break
        
        result = try_rank_randomized(r, n, m, p, timeout_for_rank, max_attempts=3000)
        
        if result is not None:
            U_r, V_r, W_r = result
            best_result = {"U": U_r, "V": V_r, "W": W_r}
            best_rank = r
            # Continue trying lower ranks
            continue
    
    # Final validation
    elapsed = time.time() - start_time
    if elapsed < deadline - 2:
        U_arr = np.array(best_result["U"])
        V_arr = np.array(best_result["V"])
        W_arr = np.array(best_result["W"])
        if not validate_fast(U_arr, V_arr, W_arr, n, m, p):
            # Fallback to schoolbook
            best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return best_result
