"""
Rank-minimal exact tensor decomposition for 3x4 * 4x7 matrix multiplication.

We use a structured ansatz that generalizes the schoolbook decomposition.
The key insight: for multiplying an n x m matrix by an m x p matrix,
we can sometimes find decompositions with rank < n*m*p by exploiting
linear dependencies.

For 3x4 * 4x7: n=3, m=4, p=7, schoolbook rank = 84.

Known results:
- The rank of matrix multiplication is bounded below by max(n*m, m*p, p*n) = max(12, 28, 21) = 28
- Upper bounds: n*m*p = 84 (schoolbook)
- For 2x2: rank is 7 (not 8)
- For 3x3: rank is at most 23 (Landsberg), at least 19
- For small dimensions, we can try to find exact decompositions using
  structured patterns.

Strategy:
1. Start with schoolbook (rank 84) as fallback.
2. Try to find lower-rank decompositions using:
   a. Kronecker product structure (if dimensions factor)
   b. Randomized search with integer/half-integer entries
   c. Exploiting the specific dimensions 3, 4, 7

For 3x4*4x7, note that 4 = 2*2, so we might use a Kronecker-like structure.
However, 3 and 7 are prime, limiting such approaches.

Let's try a more systematic approach:
- The tensor T[i,j,k] = delta_{jk} (where i in 0..2, j in 0..3, k in 0..6)
  represents the multiplication tensor.
- We seek rank r such that T = sum_{c=0}^{r-1} U[:,c] ⊗ V[:,c] ⊗ W[:,c]
  where U is (12 x r), V is (28 x r), W is (21 x r).

The constraint is: for all (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{jk}

This is a system of polynomial equations. For small ranks, we can try to solve.

Given the time constraint, let's try:
1. Check if rank 28 (the lower bound) is achievable - unlikely for these dims.
2. Try ranks between 40 and 84 using structured patterns.
3. Use the fact that entries must be integer or half-integer.

A practical approach: use a bilinear algorithm structure.
For each "column" c, we have U_c (12-dim), V_c (28-dim), W_c (21-dim).

The condition is:
sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = 1 if j==k else 0

Let's try a pattern where we group by j. For each j in 0..3, we need:
sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = delta_{jk}

If we set V_c[j*p+k] = 0 unless k is in some subset, we can decouple.

Actually, let's try a simpler structured approach:
Use the fact that 4 = 2+2. Split j into j1 (0,1) and j2 (2,3).

For j1 in {0,1}: need sum over c where V_c[j1*p+k] != 0
For j2 in {2,3}: need sum over c where V_c[j2*p+k] != 0

If we make these disjoint in c, we get two independent 3x2*2x7 problems.
Each would need rank at most 3*2*7 = 42, total 84. No gain.

Let's try a different approach: use random search with a time limit.
We'll try to find a decomposition of rank r by solving the polynomial system
using a simple iterative method or by random initialization + local optimization.

Given the complexity, let's implement a practical search:
1. For a target rank r, initialize U, V, W randomly with small integer/half-integer values.
2. Use a simple gradient-free method or just check if a random structured pattern works.

Actually, for the time available, let's try a known technique:
Use the "bilinear" structure where each term c corresponds to a pair (j, k)
but we allow sharing across different (i, j, k) tuples.

Let me try a concrete construction:
For each j in 0..3 and k in 0..6, we need the product to contribute to delta_{jk}.
Schoolbook uses one term per (j,k) pair with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.

Can we combine terms? If two terms share the same (j,k) but different i, we might merge.
But W depends on i, so we can't easily merge.

Let's try: for fixed j,k, we need sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = delta_{jk} for all i.
If j==k, this is 1 for all i. If j!=k, this is 0 for all i.

For j==k: sum_c U_c[i*m+j] * V_c[j*p+j] * W_c[j*n+i] = 1 for all i in 0..2.
For j!=k: sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = 0 for all i in 0..2.

This is a system of 4*7*3 = 84 equations (one for each (i,j,k) triple).

Let's try rank 50 as a target and see if we can find a solution via random search.
Given the time constraint, I'll implement a simple randomized search with a deadline.
"""

import numpy as np
import time
import random

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s margin
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank decompositions
    # Start from lower bounds and go up
    # Lower bound: max(n*m, m*p, p*n) = 28
    # Try some candidate ranks
    
    def check_decomposition(U, V, W, rank):
        """Check if the decomposition is exact."""
        # For each (i, j, k), compute sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Should equal 1 if j==k else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    expected = 1 if j == k else 0
                    if abs(s - expected) > 1e-9:
                        return False
        return True
    
    def try_rank(rank, num_trials=1000):
        """Try to find a decomposition of given rank using random search."""
        nonlocal best_rank, best_U, best_V, best_W
        if rank >= best_rank:
            return False
        if time.time() > deadline - 5:
            return False
        
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            # Initialize with small random integer/half-integer values
            scale = 1
            U = [[random.choice([-scale, 0, scale]) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.choice([-scale, 0, scale]) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.choice([-scale, 0, scale]) for _ in range(rank)] for _ in range(p*n)]
            
            # Try a few iterations of local improvement
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                # Pick a random column and try to adjust
                c = random.randint(0, rank-1)
                # Try modifying one entry in U, V, or W
                mat_choice = random.choice(['U', 'V', 'W'])
                if mat_choice == 'U':
                    row = random.randint(0, n*m-1)
                    old_val = U[row][c]
                    new_val = old_val + random.choice([-1, 1]) * 0.5
                    U[row][c] = new_val
                elif mat_choice == 'V':
                    row = random.randint(0, m*p-1)
                    old_val = V[row][c]
                    new_val = old_val + random.choice([-1, 1]) * 0.5
                    V[row][c] = new_val
                else:
                    row = random.randint(0, p*n-1)
                    old_val = W[row][c]
                    new_val = old_val + random.choice([-1, 1]) * 0.5
                    W[row][c] = new_val
                
                # Check if improved
                if check_decomposition(U, V, W, rank):
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    return True
            
            # Direct check
            if check_decomposition(U, V, W, rank):
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                return True
        
        return False
    
    # Try ranks from lower to higher
    # Given the difficulty, start from 60 and go down
    for r in range(70, 30, -1):
        if time.time() > deadline - 10:
            break
        if try_rank(r, num_trials=500):
            break
    
    # Convert to proper format (ensure integer/half-integer)
    def clean_matrix(M):
        """Ensure entries are integer or half-integer."""
        cleaned = []
        for row in M:
            new_row = []
            for val in row:
                # Round to nearest half-integer
                val_rounded = round(val * 2) / 2
                new_row.append(val_rounded)
            cleaned.append(new_row)
        return cleaned
    
    best_U = clean_matrix(best_U)
    best_V = clean_matrix(best_V)
    best_W = clean_matrix(best_W)
    
    # Final verification
    if not check_decomposition(best_U, best_V, best_W, best_rank):
        # Fall back to schoolbook
        best_rank = rank_school
        best_U, best_V, best_W = U_school, V_school, W_school
    
    return {"U": best_U, "V": best_V, "W": best_W}
