import numpy as np
from fractions import Fraction
import time

# DIMENSIONS for the task: 3x5 matrix times 5x7 matrix
# A is 3x5 (n=3, m=5), B is 5x7 (m=5, p=7), C is 3x7 (n=3, p=7)
# n, m, p = 3, 5, 7
# Schoolbook rank = 3*5*7 = 105
# We want to find a lower rank exact tensor decomposition.

# The tensor T is defined by T[i, j, k] = 1 for all i, j, k in the multiplication.
# We need to find U (3*5 x R), V (5*7 x R), W (7*3 x R) such that:
# sum_{c=0}^{R-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,0..2; j,0..4; k,0..6

# Using Strassen-like ideas or other algebraic decompositions.
# However, for rectangular matrices, it's more complex.
# One approach is to use a known decomposition or search for one.

# For now, let's try to construct a better decomposition than schoolbook.
# A simple improvement: use the fact that we can combine some terms.

# Actually, the rank of matrix multiplication for (n,m,p) is not known in general.
# For (3,5,7), the best known upper bound might be lower than 105.

# Let's use a structured ansatz. We can try to use a "split" decomposition.
# Or we can use a linear programming approach to find a sparse decomposition.

# Since the problem requires exact decomposition with rational (half-integer) entries,
# we can set up a system of linear equations.

# For a given rank R, we have R^3 variables (the non-zero entries of U, V, W).
# We need to satisfy n*m*p = 105 equations.

# This is a non-linear system (bilinear in U and V, linear in W, etc.), 
# but for each column c, the contribution is outer(u_c, v_c) * w_c^T.
# The sum over c must equal the all-ones tensor.

# This is equivalent to finding rank-R decomposition of a 3-way tensor.

# For small dimensions, we can try random initializations and use gradient-based methods
# on a relaxation, but exactness is hard.

# Alternatively, we can use the fact that the tensor is all-ones.
# The all-ones tensor has a special structure.

# Let's try a specific structured ansatz.
# For example, use a decomposition based on the fact that:
# sum_{i,j,k} A_{ij} B_{jk} = sum_{i,j,k} (sum_{a} U_{ij,a}) (sum_{b} V_{jk,a}) (sum_{c} W_{ki,a})
# with U_{ij,a} * V_{jk,a} * W_{ki,a} = 1 for each a.

# A simple idea: use a decomposition where each column has a specific pattern.

# Given the complexity, let's start with the schoolbook decomposition and see if we can 
# find a small improvement through a local search or a known identity.

# Actually, for the all-ones tensor, the rank is known in some cases.
# For n=m=p=1, rank=1.
# For n=m=p=2, rank=7 (Strassen).

# For (3,5,7), we can try to use a block decomposition.
# Split the dimensions into blocks and apply Strassen-like methods.

# However, implementing a general Strassen-like method for arbitrary (n,m,p) is complex.

# Let's try a different approach: use a known low-rank decomposition for the all-ones tensor.
# The all-ones tensor of size n*m*p has a decomposition of rank min(n,m,p)^2 in some cases?
# Not exactly.

# Actually, the rank of the all-ones tensor is not well-studied for rectangular cases.

# Given the constraints, let's implement a search for a sparse decomposition.
# We'll try rank R from 104 downwards and try to find a solution.

# But the time limit is 120s, so we need to be efficient.

# Instead, let's use a structured ansatz based on the following idea:
# We can use a decomposition that exploits the linearity.

# For now, let's return the schoolbook decomposition as a baseline, and then 
# try to improve it with a small perturbation or a known identity.

# Actually, the task says to return a decomposition that exactly decomposes the tensor.
# The schoolbook decomposition is valid. Can we do better?

# Let's try a specific decomposition for (3,5,7).

# One idea: use a "diagonal" decomposition with some off-diagonal terms.

# Given the complexity, let's implement a simple heuristic:
# 1. Start with schoolbook (rank 105).
# 2. Try to find a rank-104 decomposition by solving a system of linear equations 
#    for a structured ansatz.

# For a rank-R decomposition, we can fix the structure of U and V and solve for W.
# Or fix U and W and solve for V.

# Let's try the following: for a given R, initialize U and V with random ±1 values,
# and solve for W from the linear system.

# The system is: for each (i,j,k), sum_{c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
# This is linear in W if U and V are fixed.

# We can set up a linear system A * w = b, where w is the vectorized W,
# A is a (n*m*p) x (p*n*R) matrix, and b is the all-ones vector.

# But this is a large system. For R=104, the system has 105 equations and 7*3*104=2184 unknowns.
# This is underdetermined, so there are infinitely many solutions.

# We can find a solution using least squares or by setting some variables to 0.

# Let's try this approach:
# For R from 104 down to some minimum, try to find a solution.

# Steps for a given R:
# 1. Randomly initialize U (15 x R) and V (35 x R) with small rational numbers.
# 2. Solve for W (21 x R) from the linear system.
# 3. Check if the solution is exact (within a small tolerance).
# 4. If yes, return the decomposition.

# But the solution W must also be rational (half-integer).

# Let's implement this.

def construct():
    n, m, p = 3, 5, 7
    target = n * m * p  # 105
    
    # Start from schoolbook rank and try to reduce
    best_result = None
    best_rank = None
    
    deadline = time.time() + 120  # 120 seconds
    
    # Try ranks from 104 down to, say, 90
    for R in range(104, 89, -1):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Initialize U and V with random ±1
            np.random.seed(attempt * 1000 + R)
            U = np.random.choice([-1, 1], size=(n*m, R)).astype(float)
            V = np.random.choice([-1, 1], size=(m*p, R)).astype(float)
            
            # Build the linear system for W
            # For each (i,j,k), the equation is:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # Let W_flat be a vector of length p*n*R
            # The coefficient for W_flat[k*n+i*R + c] is U[i*m+j, c] * V[j*p+k, c]
            
            A = np.zeros((target, p*n*R))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = (i * m * p + j * p + k)  # Index in the flattened tensor
                        for c in range(R):
                            col_idx = c * (p * n) + (k * n + i)
                            A[idx, col_idx] = U[i * m + j, c] * V[j * p + k, c]
            
            b = np.ones(target)
            
            # Solve the linear system using least squares
            # Since the system is underdetermined, we use a pseudoinverse
            # But this might be slow for large systems.
            try:
                W_flat, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                # Check if the solution is good
                if residuals.size > 0 and residuals[0] < 1e-8:
                    W = W_flat.reshape(R, p, n).transpose(1, 2, 0)  # (p, n, R) -> (p*n, R)
                    W = W.reshape(p * n, R)
                    
                    # Verify the decomposition
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0
                                for c in range(R):
                                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                                max_err = max(max_err, abs(val - 1))
                    
                    if max_err < 1e-8:
                        # Convert to rational if needed
                        # For now, return as float
                        U_list = U.tolist()
                        V_list = V.tolist()
                        W_list = W.tolist()
                        
                        if best_result is None or R < best_rank:
                            best_rank = R
                            best_result = {'U': U_list, 'V': V_list, 'W': W_list}
            except:
                continue
        
        if best_result is not None and best_rank <= R:
            # We found a solution for this rank, so we can stop
            break
    
    if best_result is not None:
        return best_result
    
    # Fall back to schoolbook
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
