"""Tensor rank minimization for matrix multiplication 2x3 * 3x10."""
DIMENSIONS = [2, 3, 10]
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The tensor T[i,j,k] = delta_{i,k} * delta_{j,j} ... wait, matrix multiplication
    # C = A * B where A is n x m, B is m x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The multiplication tensor M[i,j,k] = 1 if this term contributes to C[i,k]
    # In the standard formulation, the tensor is M[i, j, k] where:
    #   i indexes rows of A (and C), j indexes columns of A / rows of B, k indexes columns of B (and C)
    #   M[i, j, k] = 1 (always, for the schoolbook)
    #   Actually, the tensor for matrix multiplication is defined such that:
    #   <M, A \otimes B> = C
    #   M[i, j, k] = 1 for all i, j, k? No.
    #   The multiplication tensor has entries M[i, j, k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
    #   Since every term A[i,j]*B[j,k] contributes to C[i,k], M[i,j,k] = 1 for all i,j,k.
    #   Wait, that's the "generic" tensor. The rank is the minimum number of rank-1 tensors to sum to M.
    #   
    #   A rank-1 tensor is u \otimes v \otimes w where (u \otimes v \otimes w)[i,j,k] = u[i]*v[j]*w[k].
    #   We want sum_r U[r][i] * V[r][j] * W[r][k] = M[i,j,k] = 1 for all i,j,k.
    #   Wait, that's not right either. The tensor for matrix multiplication is:
    #   M[i, j, k] = 1 if j is the "middle" index. Actually, the standard definition is:
    #   The multiplication tensor \mu_{n,m,p} has entries \mu_{i,j,k} = 1 if the entry (i,k) of the product
    #   involves A_{i,j} and B_{j,k}. Since it always does, \mu_{i,j,k} = 1 for all i,j,k.
    #   BUT, the convention in the problem statement says W[k*n+i, r]. This suggests the output factor W
    #   is indexed by (k, i) where k is the column of C and i is the row of C.
    #   
    #   Let's re-read: "The output factor follows the official trace convention W[k*n+i,r]."
    #   This implies the third factor W is indexed by (k, i) -> W[k*n+i, r].
    #   The first factor U is indexed by (i, j) -> U[i*m+j, r] (rows of A).
    #   The second factor V is indexed by (j, k) -> V[j*p+k, r] (rows of B, which are cols of A).
    #   
    #   The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k}? No.
    #   It should equal 1 if the term A[i,j]*B[j,k] is part of the sum for C[i,k].
    #   Since C[i,k] = sum_j A[i,j]*B[j,k], the tensor is T[i,j,k] = 1 for all i,j,k.
    #   Wait, if T[i,j,k] = 1 for all i,j,k, then the rank is 1? No, because we can't write
    #   1 = sum_r u_r[i] v_r[j] w_r[k] for all i,j,k with a single term unless u, v, w are constant 1s.
    #   If u[i]=1, v[j]=1, w[k]=1, then sum = 1. So rank 1?
    #   That can't be right. Matrix multiplication rank is not 1.
    #   
    #   Ah, the tensor is NOT all ones. The tensor for matrix multiplication is:
    #   T[i, j, k] = 1 if j is the shared index. But in the standard "multiplication tensor",
    #   the entries are T[i, j, k] = 1 if the product A[i,j] * B[j,k] contributes to C[i,k].
    #   This is ALWAYS true. So why is the rank not 1?
    #   
    #   Because the tensor is defined on the space of matrices A (n x m), B (m x p), and C (n x p).
    #   The bilinear map is (A, B) -> C. The tensor is in A* \otimes B* \otimes C*.
    #   The entry T_{i,j;k} (where i,j are indices of A, k is index of B... wait)
    #   
    #   Let's use the standard definition:
    #   The multiplication tensor \mu is in V^* \otimes W^* \otimes X^* where V is space of A, W is space of B, X is space of C.
    #   \mu(a, b, c) = sum_{i,j,k} a_{i,j} b_{j,k} c^*_{i,k} (where c^* is the dual basis).
    #   So the tensor entries are \mu_{i,j; j,k; i,k} = 1.
    #   In index notation: \mu_{(i,j), (j,k), (i,k)} = 1.
    #   This means for a fixed rank-1 decomposition term u \otimes v \otimes w,
    #   we have sum_r u_r[i,j] v_r[j,k] w_r[i,k] = 1 if the indices match, 0 otherwise.
    #   Wait, no. The sum must equal the tensor.
    #   The tensor T has T[i,j, j',k, i',k'] = 1 if i=i', j=j', k=k'.
    #   Actually, T is a 6-index tensor? No, it's a 3-tensor with multi-indices.
    #   T[a, b, c] where a=(i,j), b=(j',k), c=(i',k').
    #   T[a, b, c] = 1 if i=i', j=j', k=k'.
    #   So T[(i,j), (j,k), (i,k)] = 1, and 0 otherwise.
    #   
    #   So the condition is: sum_r U[a, r] * V[b, r] * W[c, r] = T[a, b, c].
    #   Where a = i*m + j, b = j*p + k, c = k*n + i.
    #   T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(i,k) for some i,j,k.
    #   Specifically, for a given a=(i,j) and b=(j',k), T[a,b,c] is non-zero only if j=j'.
    #   And then c must be (i,k).
    #   So T[(i,j), (j,k), (i,k)] = 1.
    #   
    #   Therefore, the equation is:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  (for all i, j, k)
    #   And for any other combination of indices (where the middle index j doesn't match, etc.),
    #   the sum must be 0. But since U, V, W are only defined on the valid indices,
    #   and the tensor is only non-zero on the "diagonal" in the middle index,
    #   we only need to satisfy the equation for all i, j, k.
    #   
    #   So the problem is: find rank r and matrices U (nm x r), V (mp x r), W (pn x r)
    #   such that for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_{r'} U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] = 1
    #
    #   This is the standard matrix multiplication tensor rank problem.
    #   For 2x3 * 3x10, the schoolbook rank is 2*3*10 = 60.
    #   We want to find a lower rank.
    
    # Strategy: Use a known low-rank decomposition if possible, or search.
    # For small dimensions, we can try to find a rank-r decomposition by solving a system of equations.
    # The number of equations is n*m*p = 60.
    # The number of variables is r*(n*m + m*p + p*n) = r*(6 + 30 + 20) = 56*r.
    # For r=60, it's trivial. For lower r, it's a non-linear system.
    
    # Let's try to find a rank-r decomposition by initializing U, V, W randomly and
    # using gradient descent to minimize the error.
    
    def get_error(U, V, W, r):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r_ in range(r):
                        val += U[i*m+j, r_] * V[j*p+k, r_] * W[k*n+i, r_]
                    error += (val - 1.0) ** 2
        return error
    
    def get_gradient(U, V, W, r):
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r_ in range(r):
                        val += U[i*m+j, r_] * V[j*p+k, r_] * W[k*n+i, r_]
                    err = val - 1.0
                    for r_ in range(r):
                        dU[i*m+j, r_] += 2 * err * V[j*p+k, r_] * W[k*n+i, r_]
                        dV[j*p+k, r_] += 2 * err * U[i*m+j, r_] * W[k*n+i, r_]
                        dW[k*n+i, r_] += 2 * err * U[i*m+j, r_] * V[j*p+k, r_]
        return dU, dV, dW
    
    # Try ranks from low to high
    for r in range(1, 61):
        best_error = float('inf')
        best_U = best_V = best_W = None
        for trial in range(10):
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            lr = 0.01
            for step in range(1000):
                dU, dV, dW = get_gradient(U, V, W, r)
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                if step % 100 == 0:
                    err = get_error(U, V, W, r)
                    if err < best_error:
                        best_error = err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if err < 1e-6:
                        break
            if best_error < 1e-6:
                # Found a valid decomposition
                # Round to integers/half-integers
                U_int = np.round(best_U * 2) / 2
                V_int = np.round(best_V * 2) / 2
                W_int = np.round(best_W * 2) / 2
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r_ in range(r):
                                val += U_int[i*m+j, r_] * V_int[j*p+k, r_] * W_int[k*n+i, r_]
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
        if best_error < 1e-3:
            # Close, try rounding
            U_int = np.round(best_U * 2) / 2
            V_int = np.round(best_V * 2) / 2
            W_int = np.round(best_W * 2) / 2
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r_ in range(r):
                            val += U_int[i*m+j, r_] * V_int[j*p+k, r_] * W_int[k*n+i, r_]
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
    
    # Fallback: schoolbook
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
# EVOLVE-BLOCK-END
