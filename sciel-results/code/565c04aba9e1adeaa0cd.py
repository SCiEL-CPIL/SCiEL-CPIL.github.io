"""
Rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.

Strategy:
1. Use a known upper bound on the rank (from literature or constructive bounds).
2. For 3x3 * 3x7, the schoolbook rank is 3*3*7 = 63.
3. Try to find a lower rank via random initialization + gradient descent on the
   tensor factorization, then round to half-integers and verify.
4. If rounding fails, use exact rational reconstruction or return the best valid
   decomposition found.

For m=3, n=3, p=7:
  The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
  must equal 1 if the multiplication is correct (i.e., (A*B)[i,k] = sum_j A[i,j]*B[j,k]).

Actually, the matrix multiplication tensor for C = A * B where A is n x m and B is m x p:
  C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]

So the tensor is T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
Wait, that's not right. The tensor is indexed such that:
  T[i, j, k] = 1 if we want C[i,k] to include A[i,j]*B[j,k].

Actually the matrix multiplication tensor T has T[i, j, k] = 1 for all i,j,k,
and the multiplication is: C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k] = sum_j A[i,j] * B[j,k].

So T is the all-ones tensor of shape (n, m, p).

A rank-r decomposition means:
  T[i,j,k] = sum_{c=0}^{r-1} U[i,j,c] * V[j,k,c] * W[k,i,c]

Wait, let me re-read the interface:
  U: shape (n*m, rank) - indexed by (i, j)
  V: shape (m*p, rank) - indexed by (j, k)
  W: shape (p*n, rank) - indexed by (k, i)  [trace convention W[k*n+i, r]]

And the decomposition is:
  sum_c U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c] = T[i,j,k] = 1

So we need: for all i in [0,3), j in [0,3), k in [0,7):
  sum_c U[3*j + i, c] * V[3*k + j, c] * W[7*i + k, c] = 1

Note: U is indexed by (i*m + j) = (i*3 + j), V by (j*p + k) = (j*7 + k), W by (k*n + i) = (k*3 + i).

Let me verify: n=3, m=3, p=7.
U shape: (n*m, rank) = (9, rank), indexed by i*m+j where i in [0,3), j in [0,3).
V shape: (m*p, rank) = (21, rank), indexed by j*p+k where j in [0,3), k in [0,7).
W shape: (p*n, rank) = (21, rank), indexed by k*n+i where k in [0,7), i in [0,3).

The condition is:
  For all i in {0,1,2}, j in {0,1,2}, k in {0,...,6}:
    sum_{c=0}^{r-1} U[3*j + i, c] * V[7*k + j, c] * W[3*i + k, c] = 1

This is a system of 3*3*7 = 63 equations.

Known results:
- For 3x3 * 3x3, the minimal rank is 23 (Strassen-like).
- For 3x3 * 3x7, we can potentially do better than 63.

Let me think about a constructive approach. One idea: use the fact that we can factor the p=7 dimension.

Actually, let me try a different approach: use random search with rounding.

For each candidate rank r, we can:
1. Initialize U, V, W randomly.
2. Minimize ||sum_c U V W - T||^2 using gradient descent or alternating least squares.
3. Round the entries to half-integers.
4. Check if the rounded solution is exact.

But this is hard because the objective is non-convex and rounding is difficult.

Alternative: use the fact that for small problems, we can use exact methods.

Let me try a constructive approach for rank < 63.

One idea: use the decomposition of 3x3 matrix multiplication and extend to 3x7.

For 3x3 * 3x3, rank 23 is known. For 3x3 * 3x7, we can think of it as 3x3 * 3x3 + 3x3 * 3x3 + 3x3 * 3x1.

The 3x3 * 3x3 part has rank 23, and the 3x3 * 3x1 part (matrix-vector) has rank 3*3*1 = 9 for schoolbook, but can be done in rank 9 (since it's just dot products).

Wait, matrix-vector multiplication A (3x3) * v (3x1) = w (3x1): w_i = sum_j A[i,j] v_j. This is a sum of 3 dot products, each of rank 1 (in the tensor sense, each term A[i,j]*v[j] is a rank-1 tensor in the appropriate sense). Actually, the tensor for matrix-vector is T[i,j,k] where k is fixed... no, for matrix-vector, the output is 3x1, so it's not a tensor in the same sense.

Let me reconsider. For 3x3 * 3x7:
  C[i,k] = sum_j A[i,j] B[j,k] for i in [0,3), k in [0,7).

This can be viewed as 7 independent matrix-vector multiplications (one for each column k of B), or 3 independent vector-matrix multiplications.

Each matrix-vector multiplication (3x3 matrix times 3-vector) can be done with rank 9 in schoolbook, but the tensor for this is different.

Actually, I think the cleanest approach is to use the known rank for 3x3*3x3 and combine.

But let me just try a practical approach: use alternating least squares or random restarts to find a low-rank decomposition, then round.

Given the time constraints, let me implement a practical search:

1. Start with rank = 63 (schoolbook).
2. Try to reduce rank by using random initializations and ALS.
3. Round to half-integers and verify.

For the rounding: after finding a real-valued decomposition, round each entry to the nearest half-integer (i.e., multiply by 2, round to integer, divide by 2). Then check if the rounded solution exactly satisfies the equations.

Let me implement this.
"""

import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s margin before 150s limit
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def to_half_integer(x):
        """Round to nearest half-integer."""
        return round(2 * x) / 2.0
    
    def try_rank(r, max_iters=200, n_restarts=5):
        """Try to find a rank-r decomposition using ALS."""
        if r >= best_rank:
            return None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            
            # Random initialization
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # ALS iterations
            for it in range(max_iters):
                if time.time() > deadline - 15:
                    break
                
                # Update U: for each (i,j), solve for U[i*m+j, :]
                for i in range(n):
                    for j in range(m):
                        # U[i*m+j, c] * sum_k V[j*p+k, c] * W[k*n+i, c] = 1
                        # Let a_c = sum_k V[j*p+k, c] * W[k*n+i, c]
                        # Then sum_c U[i*m+j, c] * a_c = 1
                        # This is underdetermined if r > 1. We need to minimize ||U[i*m+j] * a - 1||
                        # Actually, we want: for each k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                        # This is a system: for each k, sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
                        # Let B[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        # Then B @ U[i*m+j, :]^T = 1 (vector of ones, length p)
                        B = np.zeros((p, r))
                        for k in range(p):
                            for c in range(r):
                                B[k, c] = V[j * p + k, c] * W[k * n + i, c]
                        # Solve B @ x = ones(p), where x = U[i*m+j, :]
                        try:
                            x, _, _, _ = np.linalg.lstsq(B, np.ones(p), rcond=None)
                            U[i * m + j, :] = x
                        except:
                            pass
                
                # Update V: for each (j,k), solve for V[j*p+k, :]
                for j in range(m):
                    for k in range(p):
                        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
                        # Let A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        # Then A @ V[j*p+k, :]^T = ones(n)
                        A = np.zeros((n, r))
                        for i in range(n):
                            for c in range(r):
                                A[i, c] = U[i * m + j, c] * W[k * n + i, c]
                        try:
                            x, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            V[j * p + k, :] = x
                        except:
                            pass
                
                # Update W: for each (k,i), solve for W[k*n+i, :]
                for k in range(p):
                    for i in range(n):
                        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
                        # Let C[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        # Then C @ W[k*n+i, :]^T = ones(m)
                        C = np.zeros((m, r))
                        for j in range(m):
                            for c in range(r):
                                C[j, c] = U[i * m + j, c] * V[j * p + k, c]
                        try:
                            x, _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                            W[k * n + i, :] = x
                        except:
                            pass
            
            # Round to half-integers
            U_rounded = np.array([[to_half_integer(U[i, c]) for c in range(r)] for i in range(n * m)])
            V_rounded = np.array([[to_half_integer(V[i, c]) for c in range(r)] for i in range(m * p)])
            W_rounded = np.array([[to_half_integer(W[i, c]) for c in range(r)] for i in range(p * n)])
            
            # Verify
            if verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        
        return None
    
    # Try ranks from low to high
    # Lower bound: at least max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But realistically, for 3x3*3x7, the rank is probably around 30-50
    for r in range(25, best_rank + 1):
        if time.time() > deadline - 20:
            break
        
        result = try_rank(r, max_iters=100, n_restarts=3)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END
