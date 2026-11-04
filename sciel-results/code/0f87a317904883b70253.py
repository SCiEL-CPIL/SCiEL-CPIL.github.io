"""
Rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.

Strategy:
- The tensor T_{i,j,k} for (A_{i,j} * B_{j,k}) has entries indexed by (i,j,k) with i in [0,3), j in [0,3), k in [0,7).
- A rank-r decomposition gives U (n*m x r), V (m*p x r), W (p*n x r) such that:
  For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
- Schoolbook has rank n*m*p = 63. We need to find lower rank.

Known results:
- For 2x2*2x2, min rank is 7 (Strassen).
- For 3x3*3x3, min rank is 23 (Landsberg).
- For 3x3*3x7, the lower bound is max(n*m, m*p, p*n, ceil(n*m*p / max(n,m,p))) etc.
  n*m = 9, m*p = 21, p*n = 21.
  Simple lower bound: rank >= max(n*m, m*p, p*n) = 21.
  Another bound: rank >= ceil(n*m*p / max(n,m,p)) = ceil(63/7) = 9. Not helpful.
  
  Actually, a standard lower bound for matrix multiplication M_{n,p}^m is:
  rank >= max(n*p, m*n, m*p) ... wait, let me think again.
  
  The tensor is in (F^n)^* ⊗ (F^m)^* ⊗ (F^p)^* ⊗ F^n ⊗ F^m ⊗ F^p... no.
  
  Let's use the standard formulation. We multiply A (n x m) by B (m x p) to get C (n x p).
  C_{i,k} = sum_j A_{i,j} B_{j,k}.
  
  The multiplication tensor T ∈ (F^n ⊗ F^m)^* ⊗ (F^m ⊗ F^p)^* ⊗ (F^n ⊗ F^p)
  has T((e_i ⊗ e_j), (e_j' ⊗ e_k), (e_i' ⊗ e_k')) = 1 if i=i', j=j'=j', k=k'.
  
  Wait, the task says: W[k*n+i, r]. So the output index is (k, i) where k is the column index of C and i is the row index. So C[i,k] is stored at position k*n + i in W. This is column-major for C (n rows, p columns).
  
  So the tensor equation is:
  For all i in [0,n), k in [0,p):
    C[i,k] = sum_j A[i,j] * B[j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
  Actually more precisely:
  sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal delta_{j,j} ... no.
  
  Let me re-read: "exactly decompose the matrix multiplication tensor"
  
  The standard matrix multiplication tensor M_{n,m,p} has:
  M_{n,m,p} = sum_{i,j,k} e_i^* ⊗ e_j^* ⊗ e_j ⊗ e_k ⊗ e_i ⊗ e_k^* ... this is getting confused.
  
  Let's use the direct formulation. We want:
  For each (i, j, k) with i in [0,n), j in [0,m), k in [0,p):
    The coefficient of A[i,j] * B[j,k] in C[i,k] is 1.
  
  In tensor decomposition form:
  sum_{c=0}^{r-1} U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c] = 1  for all i,j,k.
  
  This is a system of n*m*p = 63 equations in r * (n*m + m*p + p*n) = r * (9 + 21 + 21) = 51r unknowns, but with the constraint that each "term" c contributes a rank-1 tensor.
  
  For small dimensions, we can try to find a decomposition by solving for integer/half-integer entries.
  
  Known: for 3x3 * 3x3, rank 23 is optimal. For 3x3 * 3x7, let's think about lower bounds.
  
  Lower bound from slice rank or border rank... The simple lower bound is:
  rank >= max(n*m, m*p, p*n) = max(9, 21, 21) = 21.
  
  Can we achieve rank 21? Let's try a structured approach.
  
  Actually, for M_{n,m,p}, a lower bound is:
  rank >= max(n*m, m*p, p*n) when one dimension is 1... no that's not right either.
  
  Let me just try to construct a low-rank decomposition. 
  
  One approach: use the fact that 3x3*3x7 can be related to 3x3*3x3 blocks.
  
  Actually, let's try a direct search with the constraint that entries are in {0, ±1/2, ±1, ±3/2, ±2} or similar small rational values.
  
  Given the time constraint (150s), let's try:
  1. First verify schoolbook (rank 63) works.
  2. Try to find a lower rank using a heuristic search.
  
  For 3x3 * 3x7, I recall that the minimum rank might be around 23-25. Let me try to construct one.
  
  Actually, let's use a different approach. We can formulate this as: find matrices U, V, W of minimal rank r such that the tensor equation holds.
  
  For small cases, we can try random initialization and gradient descent, but we need exact integer/half-integer solutions.
  
  Let me try a structured construction. For M_{3,3,7}:
  
  One idea: decompose the 7 columns into 3+3+1, use 3x3*3x3 decompositions for the first two parts and a 3x3*3x1 for the last.
  
  3x3*3x1: rank is 3 (since it's just 3 dot products).
  3x3*3x3: rank 23.
  Total: 23 + 23 + 3 = 49. Not better than 63 by much, but better.
  
  Can we do better? What about 3+4 split? 3x3*3x4 and 3x3*3x3.
  3x3*3x4: rank? Lower bound max(9, 12, 12) = 12. 
  Hmm, I don't know the exact min rank for 3x3*3x4.
  
  Let me try a computational approach: use a SAT/SMT-like search or integer programming for small rank values.
  
  Given the 150s budget, let's try:
  1. Start with rank = 21 (theoretical lower bound).
  2. Try to find a solution with entries in {0, ±1} or {0, ±1/2, ±1}.
  3. If not found, increase rank.
  
  Actually, let's use a smarter approach. We can use the following:
  
  For the tensor equation:
  sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
  
  This is a system where for each (i,j,k), we have a constraint. Let's denote:
  a_{i,j,c} = U[i*m+j, c]
  b_{j,k,c} = V[j*p+k, c]  
  c_{k,i,c} = W[k*n+i, c]
  
  Then: sum_c a_{i,j,c} * b_{j,k,c} * c_{k,i,c} = 1 for all i,j,k.
  
  This is a 3-indexed system. For fixed i,k: sum_c sum_j a_{i,j,c} * b_{j,k,c} * c_{k,i,c} = 1.
  
  Let's try to find a solution with small entries. I'll use a backtracking/branch-and-bound approach.
  
  Given the complexity, let me try a different tactic: use known results.
  
  For M_{3,3,3}, min rank is 23. For M_{3,3,7}, a lower bound is 21.
  
  Let me try to construct a rank-21 or rank-22 solution by hand or by search.
  
  Actually, I'll implement a search that tries to find a valid decomposition for ranks starting from 21 upward, using a heuristic that fixes some structure.
  
  Here's my plan:
  1. Try rank values from 21 to 35.
  2. For each rank, try to find U, V, W with entries in {0, ±1/2, ±1} that satisfy the tensor equation.
  3. Use a constraint satisfaction approach.
  
  Given the time limit, I'll implement a simplified version that tries a few structured patterns.
  
  Let me try a specific construction. For M_{3,3,7}, consider:
  
  We can write the 7 columns as a combination. Let's try:
  
  Actually, let me just implement a search that tries small integer/half-integer values and checks if the tensor equation is satisfied. I'll use a randomized approach with local search.
  
  Given the 150s budget, let's be practical:
  1. Return schoolbook (rank 63) as fallback.
  2. Try to find a better rank using a limited search.
  
  Let me code a search that tries to minimize rank by solving the system.
"""

import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _verify_float(U, V, W, n, m, p, tol=1e-9):
    """Verify with floating point tolerance."""
    r = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                if abs(s - 1) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = 21
    lower_bound = max(nm, mp, pn)
    
    # Candidate values for entries
    # Try small rational values
    candidate_vals = [0, Fraction(1,2), 1, Fraction(3,2), 2, Fraction(-1,2), -1, Fraction(-3,2), -2]
    
    # For each rank from lower_bound upward, try to find a solution
    # Use a structured approach: fix W to have a specific pattern, then solve for U and V
    
    # Strategy: For each rank r, try to find U, V, W
    # We can use the fact that the system is:
    # For all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Let's try a specific construction for small ranks.
    # For M_{3,3,3}, rank 23 is known. Let's try to extend to M_{3,3,7}.
    
    # Try ranks from lower_bound to min(lower_bound + 15, best_rank - 1)
    max_search_rank = min(lower_bound + 20, best_rank - 1)
    
    for r in range(lower_bound, max_search_rank + 1):
        if time.time() > deadline:
            break
        
        # Try a randomized local search
        # Initialize with small random values from candidate set
        import random
        random.seed(42 + r)
        
        found = False
        max_iters = 5000 if r <= 25 else 2000
        
        for attempt in range(3):  # 3 attempts per rank
            if time.time() > deadline:
                break
                
            # Initialize
            U = [[random.choice(candidate_vals) for _ in range(r)] for _ in range(nm)]
            V = [[random.choice(candidate_vals) for _ in range(r)] for _ in range(mp)]
            W = [[random.choice(candidate_vals) for _ in range(r)] for _ in range(pn)]
            
            # Local search: flip individual entries to reduce error
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                    
                # Compute current error
                errors = {}
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(r))
                            err = abs(s - 1)
                            errors[(i,j,k)] = s
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-15:
                    found = True
                    break
                
                # Try flipping a random entry
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    idx = random.randrange(nm)
                    col = random.randrange(r)
                    old = U[idx][col]
                    new = random.choice([v for v in candidate_vals if v != old])
                    U[idx][col] = new
                    # Check if this reduces error
                    new_max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(r))
                                if abs(s - 1) > new_max_err:
                                    new_max_err = abs(s - 1)
                    if new_max_err >= max_err:
                        U[idx][col] = old
                elif mat == 'V':
                    idx = random.randrange(mp)
                    col = random.randrange(r)
                    old = V[idx][col]
                    new = random.choice([v for v in candidate_vals if v != old])
                    V[idx][col] = new
                    new_max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(r))
                                if abs(s - 1) > new_max_err:
                                    new_max_err = abs(s - 1)
                    if new_max_err >= max_err:
                        V[idx][col] = old
                else:  # W
                    idx = random.randrange(pn)
                    col = random.randrange(r)
                    old = W[idx][col]
                    new = random.choice([v for v in candidate_vals if v != old])
                    W[idx][col] = new
                    new_max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(r))
                                if abs(s - 1) > new_max_err:
                                    new_max_err = abs(s - 1)
                    if new_max_err >= max_err:
                        W[idx][col] = old
            
            if found:
                # Convert to int or Fraction
                U_out = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in U]
                V_out = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in V]
                W_out = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in W]
                
                # Verify
                if _verify_float(U_out, V_out, W_out, n, m, p):
                    best_result = {"U": U_out, "V": V_out, "W": W_out}
                    best_rank = r
                    break
        
        if found:
            break
    
    return best_result

