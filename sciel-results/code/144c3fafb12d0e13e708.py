"""Matrix multiplication 2x5x6 rank-minimal decomposition search."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x5x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    N, M, P = n, m, p
    total = N * M * P  # 60
    
    # The matrix multiplication tensor T is defined as:
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # where i in [0,n), j in [0,m), k in [0,p)
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Decomposition: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Schoolbook rank is n*m*p = 60
    # We try to find a lower rank decomposition.
    
    # Known lower bounds: For n x m x p matrix multiplication,
    # the rank is at least max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # Also, the rank is at least n*m*p / (some factor) but that's not tight.
    # 
    # For 2x5x6, let's try various approaches:
    # 1. Direct construction with structured matrices
    # 2. Random search + local optimization
    
    # First, let's try a structured approach based on the tensor structure.
    # The tensor has 60 non-zero entries (one for each (i,j,k) triple).
    # Each rank-1 term can cover multiple entries if the structure allows.
    
    # Let's try to find a decomposition by solving a system of equations.
    # For a given rank r, we need to find U (10 x r), V (30 x r), W (12 x r)
    # such that for all a, b, c: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # This is a non-linear system. Let's try a different approach:
    # Use the fact that for each fixed i, the sub-tensor T_i is an m x p matrix
    # multiplication tensor (5x6), and for each fixed k, similar structure.
    
    # Actually, let's think about it differently. The 2x5x6 multiplication can be
    # viewed as two independent 5x6 multiplications (one for each row of the 2x5 matrix).
    # So the rank should be at least 2 * rank(5x6 multiplication).
    # 
    # What's the rank of 5x6 matrix multiplication?
    # The schoolbook rank is 5*6 = 30.
    # Strassen's algorithm gives rank 7 for 2x2, 19 for 4x4, etc.
    # For 5x6, we can use block decomposition.
    
    # Let's try a direct approach: assume the decomposition has a specific structure.
    # For 2x5x6, let's try rank values from the lower bound upward.
    
    # Lower bound: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # But we can do better with structured decompositions.
    
    # Let's try to construct a decomposition using the following insight:
    # The 2x5x6 multiplication tensor can be decomposed by considering
    # the 2x5 matrix A and 5x6 matrix B. The product C = A*B is 2x6.
    
    # For each output entry C[i,k], we have C[i,k] = sum_j A[i,j]*B[j,k]
    # This is a bilinear form. The tensor is sum over (i,j,k) of e_i ⊗ e_j ⊗ e_k
    # where e_i is the i-th standard basis vector in R^2, etc.
    
    # Let's try a random search approach with local refinement.
    
    def validate(U, V, W, tol=1e-9):
        """Check if the decomposition is exact."""
        # Compute the decomposed tensor
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Check against the true tensor
        n, m, p = 2, 5, 6
        rank = U.shape[1]
        
        # Build the true tensor as a 3D array
        T_true = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_true[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Compute the decomposed tensor
        # For efficiency, compute in chunks
        max_err = 0.0
        for a in range(n*m):
            for b in range(m*p):
                # T[a,b,:] = sum_r U[a,r]*V[b,r]*W[:,r]
                # = V[b,:] @ diag(U[a,:]) @ W.T
                # More efficiently: (U[a,:] * V[b,:]) @ W.T
                vec = U[a, :] * V[b, :]  # shape (rank,)
                row = vec @ W.T  # shape (p*n,)
                err = np.abs(row - T_true[a, b, :]).max()
                if err > max_err:
                    max_err = err
                if max_err > tol:
                    return False, max_err
        return max_err <= tol, max_err
    
    def try_rank(r, max_iters=5000, seed=42):
        """Try to find a rank-r decomposition using random initialization + local search."""
        rng = np.random.RandomState(seed)
        
        n, m, p = 2, 5, 6
        nm, mp, pn = n*m, m*p, p*n  # 10, 30, 12
        
        # Initialize with random matrices
        best_U = None
        best_V = None
        best_W = None
        best_err = float('inf')
        
        for attempt in range(3):  # Few attempts to save time
            U = rng.randn(nm, r)
            V = rng.randn(mp, r)
            W = rng.randn(pn, r)
            
            # Local optimization: gradient descent on the loss
            # Loss = sum_{a,b,c} (sum_r U[a,r]*V[b,r]*W[c,r] - T[a,b,c])^2
            # This is expensive to compute directly, so use a sampled approach.
            
            lr = 0.01
            for it in range(max_iters // 3):
                if time.time() > deadline:
                    break
                
                # Sample a subset of (a,b,c) triples
                sample_size = 200
                a_idx = rng.randint(0, nm, sample_size)
                b_idx = rng.randint(0, mp, sample_size)
                c_idx = rng.randint(0, pn, sample_size)
                
                # Compute predictions
                preds = np.zeros(sample_size)
                for s in range(sample_size):
                    a, b, c = a_idx[s], b_idx[s], c_idx[s]
                    preds[s] = np.dot(U[a, :] * V[b, :], W[c, :])
                
                # True values
                targets = np.zeros(sample_size)
                for s in range(sample_size):
                    a, b, c = a_idx[s], b_idx[s], c_idx[s]
                    # Check if T[a,b,c] = 1
                    # a = i*m+j, b = j'*p+k, c = k'*n+i'
                    # T=1 iff i=i', j=j', k=k'
                    i = a // m
                    j = a % m
                    jp = b // p
                    k = b % p
                    kp = c // n
                    ii = c % n
                    if i == ii and j == jp and k == kp:
                        targets[s] = 1.0
                
                # Compute gradients
                # dLoss/dU[a,r] = 2 * (pred - target) * V[b,r] * W[c,r]
                # We'll accumulate gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for s in range(sample_size):
                    a, b, c = a_idx[s], b_idx[s], c_idx[s]
                    err = preds[s] - targets[s]
                    if abs(err) < 1e-10:
                        continue
                    # grad_U[a, :] += 2 * err * V[b, :] * W[c, :]
                    grad_U[a, :] += 2 * err * V[b, :] * W[c, :]
                    grad_V[b, :] += 2 * err * U[a, :] * W[c, :]
                    grad_W[c, :] += 2 * err * U[a, :] * V[b, :]
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Compute error on a validation set
                if it % 50 == 0:
                    valid_err = 0.0
                    for a in range(nm):
                        for b in range(mp):
                            vec = U[a, :] * V[b, :]
                            row = vec @ W.T
                            # Compute true row
                            true_row = np.zeros(pn)
                            i = a // m
                            j = a % m
                            for k in range(p):
                                true_row[k*n+i] = 1.0 if (j == b // p and k == b % p) else 0.0
                            valid_err = max(valid_err, np.abs(row - true_row).max())
                    if valid_err < best_err:
                        best_err = valid_err
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                    if best_err < 1e-9:
                        return best_U, best_V, best_W
                if time.time() > deadline:
                    break
        
        return best_U, best_V, best_W, best_err
    
    # Try various ranks starting from a reasonable lower bound
    # Theoretical lower bound for 2x5x6 is at least 30 (from max dimension product)
    # But with structure, we might do better. Let's try from 30 upward.
    
    # Actually, let's first check if the schoolbook decomposition works (rank 60)
    # and then try to reduce.
    
    # For time efficiency, let's try a structured approach first.
    # The 2x5x6 multiplication can be seen as 2 independent 5x6 multiplications.
    # If we can find a rank-r decomposition for 5x6, then 2r is a decomposition for 2x5x6.
    # But we might do better by sharing some structure.
    
    # Let's try random search for various ranks.
    for r in [30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 55, 60]:
        if time.time() > deadline - 20:
            break
        
        result = try_rank(r, max_iters=2000, seed=r)
        if len(result) == 3:
            U, V, W = result
            valid, err = validate(U, V, W)
            if valid:
                # Convert to list format
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # If we couldn't find a better decomposition, return the schoolbook one
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}
