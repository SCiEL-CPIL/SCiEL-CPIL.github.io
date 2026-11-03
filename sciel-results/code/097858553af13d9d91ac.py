"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from numpy.linalg import norm

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0  # seconds, but we'll use a safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 60
    
    # Try to find a lower rank decomposition using ALS (Alternating Least Squares)
    # We'll try ranks from 50 down to 40 (or as low as feasible)
    # The matrix multiplication tensor T[i,j,k] = e_{i,j} * e_{j,k} -> e_{i,k}
    # We need U (n*m x r), V (m*p x r), W (p*n x r) such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) is a valid triple, else 0
    
    # Actually, the tensor is: T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p], and the decomposition
    # must satisfy: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta_{i,j,k}^tensor
    # Wait, let me re-read the problem.
    
    # Matrix multiplication: A (n x m) * B (m x p) = C (n x p)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 for all i,j,k (it's the identity tensor in the sense that
    # T[i,j,k] * A[i,j] * B[j,k] summed over j gives C[i,k])
    
    # Actually, the standard tensor decomposition for matrix multiplication:
    # We want rank-r decomposition such that:
    # C[i,k] = sum_{c=1}^{r} (sum_j U[i*m+j, c] * A[i,j]) * (sum_l V[j*p+l, c] * B[l,k])
    # Wait, that's not quite right either. Let me think more carefully.
    
    # The matrix multiplication tensor T in {0,1}^{n*m*p} where T[i,j,k] = 1 for all (i,j,k).
    # A rank-r decomposition is: T[i,j,k] = sum_{c=1}^r U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But wait, that would mean for each (i,j,k), the sum equals 1. But that's not how matrix multiplication works.
    
    # Let me reconsider. The matrix multiplication tensor is defined as:
    # T[i, j, k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    # Actually, T is a 3-tensor where T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    # No wait, that's not right either.
    
    # The correct formulation: We have three matrices U (n*m x r), V (m*p x r), W (p*n x r).
    # The decomposition satisfies: for all i, j, k:
    #   sum_c U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c] = 1
    # This is because the "tensor" of matrix multiplication has all entries equal to 1.
    
    # Wait, I think I'm overcomplicating this. Let me look at the seed code:
    # U[i*m+j][column] = 1, V[j*p+k][column] = 1, W[k*n+i][column] = 1
    # So for each (i,j,k), there's a column where U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    # And the product U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] summed over c gives 1 for each (i,j,k).
    
    # So the constraint is: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a bilinear (actually trilinear) constraint. We can use ALS to minimize:
    #   sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
    
    # Let's implement ALS:
    # 1. Initialize U, V, W randomly
    # 2. Iterate:
    #    - Fix V, W, solve for U: for each (i,j), minimize sum_k (U[i*m+j,:] dot (V[j*p+k,:] * W[k*n+i,:]) - 1)^2
    #      This is a least squares problem: U[i*m+j,:] * (V[j*p+k,:] * W[k*n+i,:]) = 1 for each k
    #      Actually, for fixed V, W and fixed (i,j), we want:
    #      sum_k (sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] - 1)^2
    #      Let a_k,c = V[j*p+k,c] * W[k*n+i,c]. Then we want U[i*m+j,:] such that
    #      sum_k (U[i*m+j,:] dot a_k - 1)^2 is minimized.
    #      This is: minimize ||A @ U[i*m+j,:] - 1||^2 where A[k,c] = a_k,c
    #      Solution: U[i*m+j,:] = (A^T A + lambda*I)^{-1} A^T 1
    
    # Let's implement this:
    
    def time_left():
        return deadline - time.time()
    
    best_rank = schoolbook_rank
    best_result = None
    
    # Try decreasing ranks
    for rank in range(55, 39, -1):
        if time_left() < 10:
            break
        
        # Try multiple random initializations
        max_iters = 100
        for trial in range(3):
            if time_left() < 5:
                break
            
            np.random.seed(42 + trial)
            random.seed(42 + trial)
            
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            for it in range(max_iters):
                if time_left() < 3:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # Build A matrix: A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        A = np.zeros((p, rank))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        # Solve: minimize ||A @ u - 1||^2
                        # Normal equations: A^T A u = A^T 1
                        AtA = A.T @ A
                        At1 = A.T @ np.ones(p)
                        # Add small regularization for stability
                        AtA += 1e-8 * np.eye(rank)
                        try:
                            U[i * m + j, :] = np.linalg.solve(AtA, At1)
                        except:
                            U[i * m + j, :] = np.linalg.lstsq(AtA, At1, rcond=None)[0]
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        # Build A matrix: A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        A = np.zeros((n, rank))
                        for i in range(n):
                            A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        AtA = A.T @ A
                        At1 = A.T @ np.ones(n)
                        AtA += 1e-8 * np.eye(rank)
                        try:
                            V[j * p + k, :] = np.linalg.solve(AtA, At1)
                        except:
                            V[j * p + k, :] = np.linalg.lstsq(AtA, At1, rcond=None)[0]
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        # Build A matrix: A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        A = np.zeros((m, rank))
                        for j in range(m):
                            A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        AtA = A.T @ A
                        At1 = A.T @ np.ones(m)
                        AtA += 1e-8 * np.eye(rank)
                        try:
                            W[k * n + i, :] = np.linalg.solve(AtA, At1)
                        except:
                            W[k * n + i, :] = np.linalg.lstsq(AtA, At1, rcond=None)[0]
                
                # Check convergence
                if it % 10 == 0:
                    # Compute error
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                                err += (val - 1.0) ** 2
                    if err < 1e-6:
                        break
            
            # Final check
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        err += (val - 1.0) ** 2
            
            if err < 1e-4:
                # Check if entries are close to integer/half-integer
                # The problem says integer/half-integer entries
                # Let's round to nearest 0.5
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Check if rounded version still works
                err_rounded = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_rounded[i * m + j, :], V_rounded[j * p + k, :] * W_rounded[k * n + i, :])
                            err_rounded += (val - 1.0) ** 2
                
                if err_rounded < 1e-6:
                    # Convert to lists
                    U_list = U_rounded.tolist()
                    V_list = V_rounded.tolist()
                    W_list = W_rounded.tolist()
                    best_rank = rank
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    else:
        # Fall back to schoolbook
        rank = schoolbook_rank
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
