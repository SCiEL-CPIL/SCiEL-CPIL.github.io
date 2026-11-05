"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

# The task: matrix multiplication 3x3 by 3x6
# n=3, m=3, p=6
# We need to find a rank-minimal tensor decomposition.
# The matrix multiplication tensor T has dimensions (n*m, m*p, p*n) = (9, 18, 18)
# The tensor entry T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k, else 0.
# A rank-r decomposition gives matrices U (9 x r), V (18 x r), W (18 x r) such that
# T[a,b,c] = sum_{t=0}^{r-1} U[a,t] * V[b,t] * W[c,t]

# The schoolbook rank is n*m*p = 3*3*6 = 54.
# Known lower bounds: max(n*m, m*p, p*n) = max(9, 18, 18) = 18.
# For 3x3 by 3x6, the optimal rank is not known a priori, but we can try to find
# a decomposition with rank less than 54.

# One approach: use the structure of the problem.
# A 3x3 by 3x6 multiplication can be seen as three independent 3x3 by 3x2 multiplications
# (treating the 6 columns as three pairs), plus cross terms... actually it's a single 3x3 by 3x6.
#
# Let's think about this differently. The standard matrix multiplication of (n x m) * (m x p)
# has a tensor of size (nm x mp x pn). 
#
# For our case: n=3, m=3, p=6.
# 
# A known technique: if p is a multiple of n, we can sometimes use structured decompositions.
# Here p=6=2*3.
#
# Let's try a direct search approach with rank r, starting from a reasonable lower bound.
# We'll use the fact that we need integer/half-integer entries.
#
# Actually, let's think about what rank is achievable. 
# The trivial bound is 54. 
# Using the "border rank" or other techniques might help, but for exact decomposition we need
# the actual rank.
#
# Let's try a randomized/optimization approach:
# 1. Start with a random initialization of U, V, W with small integer/half-integer entries.
# 2. Use gradient descent or coordinate descent to minimize the error.
# 3. Round to nearest half-integer and verify.
#
# But this might be too slow. Let's think of a smarter approach.
#
# Alternative: exploit the structure. The matrix multiplication tensor for (n x m) * (m x p)
# can be decomposed using the fact that it's a "structured" tensor.
#
# For n=3, m=3, p=6:
# The output is a 3x6 matrix. Each entry C[i,k] = sum_j A[i,j] * B[j,k].
# 
# We can write this as a sum of rank-1 terms. Each rank-1 term corresponds to a "path" 
# through the tensor.
#
# Let's try to find a decomposition with rank r by solving a system of equations.
# For each (i,j,k) with i in [0,n), j in [0,m), k in [0,p), we need:
#   sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
# For all other (a,b,c), the sum should be 0.
#
# This is a nonlinear system. Let's try a different approach.
#
# We can use the fact that the tensor is "multi-linear" in each mode. Fix all but one mode
# and solve linear equations.
#
# Actually, let's just try to verify if rank 18 is achievable (the lower bound).
# If not, try 19, 20, etc.
#
# Given time constraints, let's use a practical approach:
# 1. Try the schoolbook decomposition (rank 54) as a baseline.
# 2. Try to find a lower rank using a structured approach.
#
# For a 3x3 by 3x6 multiplication, we can view it as:
# C = A * B where A is 3x3 and B is 3x6.
# 
# One idea: decompose B into two 3x3 matrices B1 and B2 (columns 0-2 and 3-5).
# Then C = A*B1 concatenated with A*B2.
# Each A*B3 is a 3x3 multiplication, which has rank 23 (Strassen-like) or 27 (schoolbook).
# But this doesn't directly help because the two multiplications share A.
#
# Actually, the tensor for (3x3)*(3x6) is related to two (3x3)*(3x3) tensors that share
# the first factor. This is a "block" structure.
#
# Let me try a direct numerical approach with a time limit.
#
# Given the complexity, let's return the schoolbook decomposition for now, but try a 
# quick optimization if time permits.

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    # Check all tensor entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                if abs(s - 1.0) > 1e-9:
                    return False
    # Check that other entries are 0
    # We need to check all (a,b,c) where the tensor should be 0
    # The tensor is nonzero only when a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # Let's build a set of valid indices
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
    
    # For efficiency, only check a subset if the tensor is large
    # But for n=3,m=3,p=6, the tensor is 9x18x18 = 2916 entries, which is manageable
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a,b,c) not in valid:
                    s = 0.0
                    for t in range(rank):
                        s += U[a][t] * V[b][t] * W[c][t]
                    if abs(s) > 1e-9:
                        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Schoolbook decomposition as baseline
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
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try to find a lower rank decomposition
    # Use a structured approach: try ranks from lower bound up
    lower_bound = max(n*m, m*p, p*n)
    
    # For each candidate rank, try to find a decomposition
    # We'll use a simple randomized approach with local optimization
    
    for r in range(lower_bound, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations
        for trial in range(10):
            if time.time() > deadline - 5:
                break
            
            # Initialize with small random values
            random.seed(42 + r * 100 + trial)
            U = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(r)] for _ in range(n*m)]
            V = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(r)] for _ in range(m*p)]
            W = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(r)] for _ in range(p*n)]
            
            # Verify
            if _verify(U, V, W, n, m, p):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = r
                break
        
        if best_rank == r:
            break
    
    return best_result
# EVOLVE-BLOCK-END
