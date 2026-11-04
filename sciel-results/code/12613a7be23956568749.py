"""
Rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.

The known lower bound for C(n,m,p) is max(n*m, m*p, n*p).
For n=4, m=5, p=6: lower bound = max(20, 30, 24) = 30.
The schoolbook rank is n*m*p = 120.

We use a systematic search to find a rank-30 decomposition.
The approach:
1. Start with the schoolbook decomposition (rank 120).
2. Use a greedy/iterative compression to merge terms where possible.
3. Verify exactness via the tensor identity.

Actually, let me think about this differently. The matrix multiplication tensor T[i,j,k] for (n,m,p)
is the identity in the sense that T[i,j,k] = 1 if the indices correspond to (i,j) in A, (j,k) in B,
and (i,k) in C. 

For a rank-r decomposition, we need U (n*m x r), V (m*p x r), W (p*n x r) such that
for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta(i,j,k)

where delta(i,j,k) = 1 if the triple (i,j,k) is "valid" (which it always is in the standard tensor),
and 0 otherwise. Wait, no. The matrix multiplication tensor has entries:
T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).

Actually, the tensor is defined such that:
(A * B)[i,k] = sum_j A[i,j] * B[j,k]

The tensor T has T[i,j,k] = 1 for all valid indices, and the decomposition means:
sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.

So we need to find rank-r matrices U, V, W with this property.

For the 4x5x6 case, let me try a constructive approach based on known results.
The best known upper bound for C(4,5,6) might be achievable at rank 30 or slightly above.

Let me try a different approach: use the fact that we can decompose the multiplication
by partitioning the middle dimension.

Actually, let me just implement a search that tries to find a low-rank decomposition
using random initialization and iterative refinement, with exact verification.

Given the time constraint, let me try a structured approach:
- Use the "border rank" or known algebraic constructions.

For now, let me implement a robust search that:
1. Tries rank 30 (the theoretical lower bound)
2. If not found, tries higher ranks
3. Uses random initialization with integer/half-integer constraints

Let me implement this carefully.
"""

import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline, 20s safety margin
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Start with schoolbook rank = n*m*p = 120
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    # Convert to list format for return
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # Try to find lower rank decompositions
    # Strategy: use random projections and iterative refinement
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(U.shape[1]))
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True
    
    def try_random_rank(r, max_iters=500):
        """Try to find a rank-r decomposition via random initialization and refinement."""
        if r < max(n*m, m*p, n*p):
            return None, None, None
        
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values in {-1, 0, 1} or {-0.5, 0, 0.5, 1}
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, r))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, r))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, r))
            
            # Iterative refinement: for each index (i,j,k), adjust the column that
            # best explains the residual
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Compute current values
                vals = {}
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            vals[(i,j,k)] = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                
                # Check if we're close to solution
                max_err = max(abs(vals[(i,j,k)] - 1.0) for i in range(n) for j in range(m) for k in range(p))
                if max_err < 1e-6:
                    return U, V, W
                
                # Find the worst error
                worst_idx = max(vals.keys(), key=lambda idx: abs(vals[idx] - 1.0))
                i, j, k = worst_idx
                residual = 1.0 - vals[(i,j,k)]
                
                # Try adjusting one entry
                # Find the column c that minimizes the change in total error
                best_c = None
                best_delta = None
                for c in range(r):
                    # Try flipping or adjusting U[i*m+j, c]
                    for new_val in [-1, -0.5, 0, 0.5, 1]:
                        old_val = U[i*m+j, c]
                        if new_val == old_val:
                            continue
                        # Estimate impact
                        U[i*m+j, c] = new_val
                        # Recompute only affected entries (those with same i,j or same j,k or same k,i)
                        # This is expensive, so just do a quick check
                        err = 0
                        for kk in range(p):
                            for ii in range(n):
                                for jj in range(m):
                                    if ii == i and jj == j:
                                        v = sum(U[ii*m+jj, cc] * V[jj*p+kk, cc] * W[kk*n+ii, cc] for cc in range(r))
                                        err += abs(v - 1.0)
                        U[i*m+j, c] = old_val
                        if best_delta is None or err < best_delta:
                            best_delta = err
                            best_c = (c, 'U', i*m+j, new_val)
                
                if best_c is not None:
                    c, mat, idx, new_val = best_c
                    if mat == 'U':
                        U[idx, c] = new_val
                    elif mat == 'V':
                        V[idx, c] = new_val
                    else:
                        W[idx, c] = new_val
        
        return None, None, None
    
    # Try ranks from lower bound upward
    lower_bound = max(n*m, m*p, n*p)
    
    for r in range(lower_bound, min(lower_bound + 20, rank_school + 1)):
        if time.time() > deadline - 10:
            break
        
        U_try, V_try, W_try = try_random_rank(r)
        if U_try is not None and verify(U_try, V_try, W_try):
            best_rank = r
            best_U = U_try
            best_V = V_try
            best_W = W_try
            break
    
    # If we found a better rank, use it; otherwise use schoolbook
    if best_rank < rank_school:
        U_out = to_list(best_U)
        V_out = to_list(best_V)
        W_out = to_list(best_W)
    else:
        U_out = to_list(best_U)
        V_out = to_list(best_V)
        W_out = to_list(best_W)
    
    return {"U": U_out, "V": V_out, "W": W_out}
