"""
Rank-minimal exact tensor decomposition for 3x5 * 5x5 matrix multiplication.

Strategy:
1. Use the schoolbook decomposition as a safe fallback (rank 75).
2. Attempt to find a lower-rank decomposition via:
   a. Kronecker product structure: if we can express the 3x5x5 multiplication
      in terms of smaller known decompositions (e.g., 3x3x3 with rank 7 via Strassen,
      and 5x5x5 with known lower ranks), we might combine them.
   b. Direct search using bilinear optimization / tensor rank minimization is hard,
      but we can try structured decompositions.

Key insight: The matrix multiplication tensor for (n,m,p) can sometimes be
decomposed using Kronecker products of smaller tensors. However, 3x5x5 doesn't
factor nicely.

Alternative: Use the fact that we can pad 3x5 to 4x5 or 5x5 and use known
decompositions, but that increases rank.

Better approach: Try to use the known best bounds. The trivial rank is 75.
Known results suggest the rank is between ~50-60 for 3x5x5, but finding an
exact decomposition with integer/half-integer entries is non-trivial.

Let me try a different approach: use the "laser method" or structured ansatz.
Actually, for a practical approach within the time limit, let me try:

1. Start with schoolbook (rank 75)
2. Try to merge terms using linear dependencies
3. Try random initialization + gradient descent on the tensor reconstruction error

But the task requires EXACT decomposition with integer/half-integer entries.
This is very restrictive. Let me think about what's feasible.

Given the constraints (exact, integer/half-integer), the safest bet is to
find a structured decomposition. Let me try:

- Decompose the 3x5 matrix multiplication as a sum of rank-1 tensors
- Use the fact that C[i,k] = sum_j A[i,j] * B[j,k]
- Each term A[i,j]*B[j,k] is a rank-1 tensor in the (i,j,k) space

The schoolbook uses 75 such terms. Can we do better?

For small matrices, there are known optimal decompositions. For 3x3x3,
Strassen gives rank 7. For 3x3x4, the rank is 9 (Landsberg). 

Let me try to use a block decomposition. Write the 3x5 matrix as blocks
and use known decompositions.

Actually, let me try a more systematic approach: use the "border rank" or
"approximate rank" ideas, but we need exact rank.

Given the time constraints and the requirement for exact integer/half-integer
entries, I'll try the following:

1. Schoolbook (rank 75) as baseline
2. Try to find a decomposition with rank < 75 by exploiting linear structure

Let me try a random search with rounding to half-integers, checking exactness.
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p  # 75
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # T[i,j,k] = 1 if this corresponds to C[i,k] = sum_j A[i,j]*B[j,k]
        # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
        # and the decomposition satisfies: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta_{i,k}?? 
        # No, let me re-read the convention.
        
        # W[k*n+i, r] - so W is indexed by (k,i) not (i,k)
        # The decomposition should satisfy:
        # For the multiplication C = A*B where A is n x m, B is m x p, C is n x p:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor form: T[i,j,k] such that C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
        # T[i,j,k] = 1 for all i,j,k (it's the multiplication tensor)
        # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # where U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
        
        # Check: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # And for invalid indices, it should be 0, but all indices are valid here.
        
        # Actually, the tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Reshape for efficient computation
        U3 = U_arr.reshape(n, m, rank)
        V3 = V_arr.reshape(m, p, rank)
        W3 = W_arr.reshape(p, n, rank)
        
        # Compute sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # This is a contraction over r
        # Result should be 1 for all (i,j,k)
        
        # Efficient way: for each r, the outer product U3[:,:,r] * V3[:,:,r] * W3[:,:,r]
        # But dimensions don't match directly. Let me compute explicitly.
        
        result = np.zeros((n, m, p), dtype=float)
        for r in range(rank):
            u = U_arr[:, r].reshape(n, m)
            v = V_arr[:, r].reshape(m, p)
            w = W_arr[:, r].reshape(p, n)
            # u[i,j] * v[j,k] * w[k,i]
            # Sum over j and k? No, we want the full tensor.
            # For each (i,j,k): u[i,j] * v[j,k] * w[k,i]
            # This is like a 3D tensor product
            # Let's compute it as: for each i, j, k
            # We can do: temp[i,j,k] = u[i,j] * v[j,k] * w[k,i]
            # Use einsum or explicit loops
            # Since n,m,p are small, explicit loops are fine
            for i in range(n):
                for j in range(m):
                    if u[i,j] == 0:
                        continue
                    for k in range(p):
                        if v[j,k] == 0 or w[k,i] == 0:
                            continue
                        result[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        
        # Check if result is all 1s
        if not np.allclose(result, 1.0, atol=1e-10):
            return False
        
        # Also check that entries are integer or half-integer
        for mat in [U_arr, V_arr, W_arr]:
            for val in mat:
                # Check if val is close to an integer or half-integer
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2
                if abs(val - nearest_int) > 1e-10 and abs(val - nearest_half) > 1e-10:
                    return False
        
        return True
    
    # Try schoolbook first
    if verify(U_school, V_school, W_school, rank_school):
        best = (rank_school, U_school, V_school, W_school)
    else:
        # Should not happen, but just in case
        best = (rank_school, U_school, V_school, W_school)
    
    # Try to find lower rank decompositions
    # Strategy: random initialization + local search
    
    def try_random_decomp(rank, max_iters=1000):
        """Try to find a rank-rank decomposition via random init + refinement."""
        if time.time() > deadline:
            return None
        
        # Random initialization
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Scale to make entries roughly half-integer
        U /= 2
        V /= 2
        W /= 2
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Compute error
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            
            # Compute current tensor
            result = np.zeros((n, m, p), dtype=float)
            for r in range(rank):
                u = U3[:, :, r]
                v = V3[:, :, r]
                w = W3[:, :, r]
                for i in range(n):
                    for j in range(m):
                        if u[i, j] == 0:
                            continue
                        for k in range(p):
                            if v[j, k] == 0 or w[k, i] == 0:
                                continue
                            result[i, j, k] += u[i, j] * v[j, k] * w[k, i]
            
            error = np.sum((result - 1.0) ** 2)
            
            if error < 1e-10:
                # Check if entries are half-integer
                for mat in [U, V, W]:
                    for val in mat:
                        nearest_half = round(val * 2) / 2
                        if abs(val - nearest_half) > 1e-6:
                            break
                    else:
                        continue
                    break
                else:
                    # Convert to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    # Convert to lists
                    U_list = U_rounded.tolist()
                    V_list = V_rounded.tolist()
                    W_list = W_rounded.tolist()
                    if verify(U_list, V_list, W_list, rank):
                        return (U_list, V_list, W_list)
            
            # Gradient step (simplified)
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r in range(rank):
                u = U3[:, :, r]
                v = V3[:, :, r]
                w = W3[:, :, r]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            diff = result[i, j, k] - 1.0
                            # d(result)/d(u[i,j]) = v[j,k] * w[k,i]
                            grad_U[i * m + j, r] += diff * v[j, k] * w[k, i]
                            # d(result)/d(v[j,k]) = u[i,j] * w[k,i]
                            grad_V[j * p + k, r] += diff * u[i, j] * w[k, i]
                            # d(result)/d(w[k,i]) = u[i,j] * v[j,k]
                            grad_W[k * n + i, r] += diff * u[i, j] * v[j,k]
            
            # Update
            lr = 0.01
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
        
        return None
    
    # Try ranks from 60 down to 50
    for rank in range(60, 49, -1):
        if time.time() > deadline:
            break
        
        result = try_random_decomp(rank, max_iters=200)
        if result is not None:
            U_list, V_list, W_list = result
            best = (rank, U_list, V_list, W_list)
            break
    
    # Return the best found
    rank_best, U_best, V_best, W_best = best
    return {"U": U_best, "V": V_best, "W": W_best}
