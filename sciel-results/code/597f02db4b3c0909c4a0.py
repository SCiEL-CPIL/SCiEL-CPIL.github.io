"""
Rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.

We use a structured search over integer/half-integer factor matrices.
The known upper bound is 18 (schoolbook). We search for lower ranks.

Approach:
1. Start with the schoolbook rank-18 decomposition.
2. Use a randomized local search / integer linear programming approach to try to reduce rank.
3. Since the problem is small (2*3=6, 3*3=9, 3*2=6), we can do exhaustive or near-exhaustive search.

Key insight: The matrix multiplication tensor T has dimensions 6x9x6 (flattened).
We want to find rank r such that T = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
where u_c ∈ R^6, v_c ∈ R^9, w_c ∈ R^6.

The entries must be integer or half-integer (i.e., in (1/2)Z).

Let's use a systematic approach:
- For small ranks r, we can try to solve the system of polynomial equations.
- The tensor has 6*9*6 = 324 entries, each of which is a sum of r products.
- This is a system of 324 polynomial equations in 6r + 9r + 6r = 21r variables.

For r < 18, this is a hard problem. Let's try a different approach:
Use the fact that we can fix some structure. 

Actually, let me think about this more carefully. The matrix multiplication tensor for (2,3,3) is:
T[i, j, k] = 1 if and only if... wait, no.

The matrix multiplication tensor M_{n,m,p} is defined by:
M[i, j, k] = δ_{i,j} * δ_{j,k} ... no, that's not right either.

For multiplying an (n×m) matrix A by an (m×p) matrix B to get an (n×p) matrix C:
C[i,k] = sum_j A[i,j] * B[j,k]

The tensor T is indexed by (i, j, k) where i ∈ {0,...,n-1}, j ∈ {0,...,m-1}, k ∈ {0,...,p-1}.
T[i,j,k] = 1 if the term A[i,j]*B[j,k] appears in C[i,k], which is always true for valid indices.
So T[i,j,k] = 1 for all valid (i,j,k).

Wait, that's the identity tensor. The matrix multiplication tensor is actually:
T[i, j, k] = 1 for all i, j, k (since every product term appears exactly once).

So the tensor is all-ones of shape (n, m, p) = (2, 3, 3).

The rank of the all-ones tensor of shape (n,m,p) is known to be at most n*m*p (schoolbook).
Can we do better?

For the all-ones tensor, we want to write:
1 = sum_{c=1}^{r} u_{i,c} * v_{j,c} * w_{k,c} for all i, j, k.

This is equivalent to finding rank-r decompositions of the tensor.

Known results:
- For (2,2,2), the all-ones tensor has rank 7 (Strassen).
- For (2,3,3), what is the minimal rank?

Let me try to construct a lower-rank decomposition using Strassen-like ideas.

Actually, let me just implement a search that tries to find the minimal rank by:
1. Starting from r = 1 and going up.
2. For each r, try to find integer/half-integer solutions.

Given the complexity, let me use a practical approach:
- Use the known Strassen decomposition for the 2x2x2 part.
- Extend it to 2x3x3.

Let me think about this differently. We can view the 2x3 * 3x3 multiplication as:
- Split the 3 columns of A into 2+1, and the 3 rows of B into 2+1.
- Use Strassen for the 2x2 * 2x2 part (rank 7).
- Handle the cross terms.

But this might not give a lower total rank.

Let me just implement a brute-force search for small ranks with the constraint that entries are in {0, 1/2, 1, 3/2, 2, ...} but bounded.

Given the time constraint, let me implement a randomized search with verification.
"""

import numpy as np
from itertools import product
import random
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140 second deadline, 10s safety margin
    
    # The tensor is all-ones of shape (n, m, p)
    # T[i, j, k] = 1 for all i in range(n), j in range(m), k in range(p)
    
    # Schoolbook rank is n*m*p = 18
    best_rank = n * m * p
    best_decomp = None
    
    # Generate the schoolbook decomposition as a fallback
    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_decomp = make_schoolbook()
    
    # Try to find lower rank decompositions
    # The entries must be half-integers: x/2 for integer x
    # So we can scale by 2: work with integers, then divide by 2 at the end
    
    # For a rank-r decomposition, we have:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Let's try ranks from 13 to 17 (since 18 is schoolbook, and we know 2x2x2 has rank 7 < 8)
    
    # Actually, let me try a structured approach based on Strassen for 2x2x2
    
    # For 2x2x2, Strassen's decomposition uses 7 multiplications:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # For 2x3x3, let's try to extend this idea.
    
    # Let's try a direct search for ranks 14-17 with half-integer entries in a small range.
    
    # The search space is huge, so let's use a smarter approach.
    
    # Fix the structure: assume the decomposition has some symmetry or pattern.
    
    # Alternative: use integer programming via scipy or a simple local search.
    
    # Let's try a randomized local search:
    # 1. Start with a random rank-r decomposition.
    # 2. Compute the error.
    # 3. Try to reduce the error by adjusting entries.
    
    # But the constraint is exact equality, so this is tricky.
    
    # Let me try a different approach: enumerate possible decompositions with small integer entries.
    
    # For rank r, each factor has r columns. The entries are half-integers.
    # Let's restrict entries to {-2, -1, -0.5, 0, 0.5, 1, 2} for simplicity.
    
    # This is still a huge search space. Let's use a heuristic.
    
    # Actually, let me just try to verify if rank 17, 16, 15, 14, 13 are possible
    # by trying to find solutions to the polynomial system.
    
    # Given the complexity, let me implement a simple approach:
    # Try to find a rank-r decomposition by solving a linear system after fixing some variables.
    
    # For now, let's return the schoolbook decomposition if we can't find better quickly.
    
    # Try a few random searches for lower ranks
    for target_rank in range(17, 12, -1):
        if time.time() > deadline - 10:
            break
        
        # Try to find a decomposition of this rank
        # Use a structured initialization based on Strassen-like patterns
        
        # For 2x3x3, let's try to build on the 2x2x2 Strassen decomposition
        # by adding terms for the extra column/row.
        
        # This is complex, so let's try a simple randomized approach with verification.
        
        # Generate random half-integer matrices
        for trial in range(100):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with small half-integer values
            vals = [0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2]
            
            U = [[random.choice(vals) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice(vals) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice(vals) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Check if this decomposition works
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(target_rank):
                            s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                best_rank = target_rank
                best_decomp = {"U": U, "V": V, "W": W}
                break
        
        if best_rank < n * m * p:
            break
    
    return best_decomp
# EVOLVE-BLOCK-END
