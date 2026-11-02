import time
import random
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72

    # Try to find a lower rank decomposition using known algorithms
    # For 3x3x8, the trivial upper bound is 72.
    # Let's try rank 72 first, then attempt to reduce.
    
    # We'll use a random initialization + alternating least squares approach
    # to try to find a rank-r decomposition.
    
    def random_decomposition(r):
        """Generate a random rank-r decomposition."""
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        # Scale to have unit Frobenius norm per column
        U /= np.sqrt(np.sum(U**2, axis=0))
        V /= np.sqrt(np.sum(V**2, axis=0))
        W /= np.sqrt(np.sum(W**2, axis=0))
        return U, V, W

    def compute_tensor(U, V, W):
        """Compute the tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        and compare with the canonical 3x3x8 tensor T[i,j,k] = 1 if all indices valid."""
        # The target tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # Actually, the matrix multiplication tensor is:
        # T[i,j,k] = 1 for all valid i,j,k (since (AB)[i,k] = sum_j A[i,j]*B[j,k])
        # The decomposition must satisfy:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        
        target = np.ones((n, m, p))
        
        # Compute the decomposed tensor
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        tensor = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    tensor[i,j,k] = s
        return tensor, target

    def compute_error(U, V, W):
        """Compute the Frobenius error between decomposed tensor and target."""
        target = np.ones((n, m, p))
        tensor = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i,j,k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
        return np.sqrt(np.sum((tensor - target)**2))

    def als_decomposition(U, V, W, r, num_iters=100, tol=1e-10):
        """Alternating Least Squares to refine a rank-r decomposition."""
        target = np.ones((n, m, p))
        
        for it in range(num_iters):
            if time.time() > deadline:
                break
            
            # Update U: for each row i*m+j, minimize sum over k of (target[i,j,k] - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])^2
            # This is a linear system in U[i*m+j,:]
            for i in range(n):
                for j in range(m):
                    # A[c1,c2] = sum_k V[j*p+k,c1]*W[k*n+i,c2]
                    # b[c] = sum_k target[i,j,k]*V[j*p+k,c]*W[k*n+i,c]
                    A = np.zeros((r, r))
                    b = np.zeros(r)
                    for k in range(p):
                        A += np.outer(V[j*p+k, :], W[k*n+i, :])
                        b += target[i,j,k] * V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U[i*m+j, :] = np.linalg.solve(A + 1e-12*np.eye(r), b)
                    except:
                        pass
            
            # Update V: for each row j*p+k
            for j in range(m):
                for k in range(p):
                    A = np.zeros((r, r))
                    b = np.zeros(r)
                    for i in range(n):
                        A += np.outer(U[i*m+j, :], W[k*n+i, :])
                        b += target[i,j,k] * U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.solve(A + 1e-12*np.eye(r), b)
                    except:
                        pass
            
            # Update W: for each row k*n+i
            for k in range(p):
                for i in range(n):
                    A = np.zeros((r, r))
                    b = np.zeros(r)
                    for j in range(m):
                        A += np.outer(U[i*m+j, :], V[j*p+k, :])
                        b += target[i,j,k] * U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.solve(A + 1e-12*np.eye(r), b)
                    except:
                        pass
            
            if it % 10 == 0:
                err = compute_error(U, V, W)
                if err < tol:
                    break
        
        return compute_error(U, V, W)

    # Try various ranks from low to high
    best_U, best_V, best_W = None, None, None
    best_rank = rank_schoolbook

    # First, get the schoolbook solution as baseline
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    # Try to find lower rank decompositions
    # For 3x3x8, try ranks below 72
    # Known: R(3,3,3) <= 23, so R(3,3,8) might be lower than 72
    # Let's try some candidate ranks
    
    candidate_ranks = [60, 64, 68, 70, 71, 72]
    
    for r in candidate_ranks:
        if time.time() > deadline - 20:
            break
        
        best_err = float('inf')
        best_U_r, best_V_r, best_W_r = None, None, None
        
        # Multiple random restarts
        for restart in range(50):
            if time.time() > deadline - 15:
                break
            
            U, V, W = random_decomposition(r)
            err = als_decomposition(U, V, W, r, num_iters=50, tol=1e-8)
            
            if err < best_err:
                best_err = err
                best_U_r, best_V_r, best_W_r = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-8:
                break
        
        if best_err < 1e-8 and best_U_r is not None:
            best_U, best_V, best_W = best_U_r, best_V_r, best_W_r
            best_rank = r
            break
    
    # Convert to list format
    def to_list(arr):
        return [[float(arr[i, j]) for j in range(arr.shape[1])] for i in range(arr.shape[0])]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
