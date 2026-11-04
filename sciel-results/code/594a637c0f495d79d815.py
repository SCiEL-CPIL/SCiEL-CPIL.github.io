"""Rewritten program for 3x3 by 3x5 matrix multiplication tensor decomposition."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix
    by a 3x5 matrix.
    
    Returns dict with keys 'U', 'V', 'W' where:
      U: shape (n*m, rank) = (9, rank)
      V: shape (m*p, rank) = (15, rank)
      W: shape (p*n, rank) = (15, rank)
    
    The decomposition satisfies: for all i,j,k,
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 15
    pn = p * n  # 15
    nmp = n * m * p  # 45
    
    # Schoolbook rank is 45. We try to find lower rank decompositions.
    # Known results: 3x3x3 has rank 23 (Strassen-like), but for 3x3x5,
    # the minimal rank is not well-known. We'll try to find a good decomposition.
    
    # First, let's try to use a structured approach.
    # The matrix multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want to write T as sum of rank-1 tensors: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
    
    # Indexing: U has rows indexed by (i,j) with index i*m+j
    #            V has rows indexed by (j,k) with index j*p+k
    #            W has rows indexed by (k,i) with index k*n+i
    
    # For a rank-1 term r, we have:
    #   U[i*m+j, r] = a_{i,j}^{(r)}
    #   V[j*p+k, r] = b_{j,k}^{(r)}
    #   W[k*n+i, r] = c_{k,i}^{(r)}
    # And we need: sum_r a_{i,j}^{(r)} * b_{j,k}^{(r)} * c_{k,i}^{(r)} = 1 for all i,j,k
    
    # This is equivalent to: for each r, define matrices A_r (n x m), B_r (m x p), C_r (p x n)
    # such that sum_r (A_r * B_r * C_r)_{i,k} ... wait, let me re-derive.
    
    # Actually, the condition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Let A_r[i,j] = U[i*m+j, r], B_r[j,k] = V[j*p+k, r], C_r[k,i] = W[k*n+i, r]
    # Then sum_r A_r[i,j] * B_r[j,k] * C_r[k,i] = 1 for all i,j,k
    
    # This is a sum of "cyclic" products. Note that A_r * B_r is an n x p matrix,
    # and we're taking its (i,k) entry times C_r[k,i].
    # So sum_r trace((A_r * B_r) * C_r^T) ... no, that's not quite right.
    
    # Let's think of it as: for each r, define M_r = A_r @ B_r (n x p matrix)
    # Then the condition is: sum_r M_r[i,k] * C_r[k,i] = 1 for all i,k
    # But C_r[k,i] is just the (k,i) entry of C_r, so this is:
    # sum_r sum_k M_r[i,k] * C_r[k,i] = 1 ... no, the sum is over r, and for each r we have
    # a single term M_r[i,k] * C_r[k,i].
    
    # Actually, for each fixed i,j,k, we need:
    # sum_r A_r[i,j] * B_r[j,k] * C_r[k,i] = 1
    
    # This is a system of 45 equations (one for each (i,j,k) triple) with 3*rank*9 + 3*rank*15 + 3*rank*15 = rank*(27+45+45) = 117*rank unknowns.
    # But the structure is nonlinear (products of three terms).
    
    # Let's try a different approach: use known lower bounds and try to find a decomposition.
    # For 3x3x3, the minimal rank is 23 (over reals). For 3x3x5, it's at least... 
    # The tensor rank of 3x3x5 is at least max(9, 15, 15) = 15? No, that's not a valid lower bound.
    # Actually, the rank is at least the dimension of the space spanned by the slices.
    
    # Let's just try to find a decomposition using optimization.
    # We'll use a simple approach: start with random initialization and use gradient descent
    # on the loss function L = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
    
    deadline = time.time() + 140  # Leave 10 seconds margin
    
    # Try different ranks from low to high
    for rank in range(15, 46):
        if time.time() > deadline:
            break
            
        best_loss = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        # Multiple random restarts
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + restart)
            
            # Initialize with random values
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # Gradient descent
            lr = 0.01
            n_steps = 500
            for step in range(n_steps):
                if time.time() > deadline:
                    break
                
                # Compute gradients
                # Loss = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
                # dL/dU[i*m+j,r] = 2 * sum_k (pred[i,j,k] - 1) * V[j*p+k,r] * W[k*n+i,r]
                # where pred[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
                
                # Efficient computation:
                # For each r, define:
                #   A_r = U[:, r].reshape(n, m)
                #   B_r = V[:, r].reshape(m, p)
                #   C_r = W[:, r].reshape(p, n)
                # Then pred[i,j,k] = sum_r A_r[i,j] * B_r[j,k] * C_r[k,i]
                
                # Compute predictions
                pred = np.zeros((n, m, p))
                for r in range(rank):
                    A_r = U[:, r].reshape(n, m)
                    B_r = V[:, r].reshape(m, p)
                    C_r = W[:, r].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                pred[i, j, k] += A_r[i, j] * B_r[j, k] * C_r[k, i]
                
                # Compute loss
                loss = np.sum((pred - 1)**2)
                if loss < best_loss:
                    best_loss = loss
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if loss < 1e-10:
                    break
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for r in range(rank):
                    A_r = U[:, r].reshape(n, m)
                    B_r = V[:, r].reshape(m, p)
                    C_r = W[:, r].reshape(p, n)
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                err = pred[i, j, k] - 1
                                dU[i * m + j, r] += 2 * err * B_r[j, k] * C_r[k, i]
                                dV[j * p + k, r] += 2 * err * A_r[i, j] * C_r[k, i]
                                dW[k * n + i, r] += 2 * err * A_r[i, j] * B_r[j, k]
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Adaptive learning rate
                if step % 100 == 0 and step > 0:
                    lr *= 0.9
        
        # Check if we found a good solution
        if best_loss < 1e-10 and best_U is not None:
            # Verify the solution
            pred = np.zeros((n, m, p))
            for r in range(rank):
                A_r = best_U[:, r].reshape(n, m)
                B_r = best_V[:, r].reshape(m, p)
                C_r = best_W[:, r].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            pred[i, j, k] += A_r[i, j] * B_r[j, k] * C_r[k, i]
            
            if np.allclose(pred, 1, atol=1e-9):
                # Convert to lists
                U_list = [[float(best_U[i, r]) for r in range(rank)] for i in range(nm)]
                V_list = [[float(best_V[i, r]) for r in range(rank)] for i in range(mp)]
                W_list = [[float(best_W[i, r]) for r in range(rank)] for i in range(pn)]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition
    rank = nmp
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
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
