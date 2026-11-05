import numpy as np
import time
import random

DIMENSIONS = [3, 3, 6]

def _error(U, V, W, n, m, p):
    """Compute the squared error of the tensor decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', U, V, W)
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                err += (T[a, b, c] - 1) ** 2
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                i2 = c % n
                k2 = c // n
                if not (i == i2 and j == j2 and k == k2):
                    err += T[a, b, c] ** 2
    return err

def _build_strassen_3x3():
    """
    Build a rank-7 decomposition for 3x3 matrix multiplication.
    Returns U, V, W each of shape (9, 7) for the 3x3 case.
    
    We use a known rank-7 algorithm for 3x3 matrices.
    The standard schoolbook is rank 27. Strassen for 2x2 is rank 7.
    For 3x3, there are known algorithms with rank 19 or lower, but let's use
    a simple approach: treat 3x3 as block matrices and use Strassen-like ideas,
    or just use a known rank-19 algorithm.
    
    Actually, for simplicity and correctness, let's use the schoolbook for 3x3 (rank 27)
    and then try to find a better one. But the task asks for rank-minimal.
    
    Let me think differently. The 3x3x6 tensor can be viewed as:
    For each of the 6 columns of the output, we're computing a 3x3 matrix product
    with a different right-hand side. So it's like doing 6 separate 3x3 multiplications
    but they share the left matrix A.
    
    Actually, the tensor is T[i,j,k] where:
    - i indexes rows of A (0..2)
    - j indexes columns of A / rows of B (0..2)  
    - k indexes columns of B (0..5)
    
    The multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
    
    So for each k (0..5), we compute C[:,k] = A @ B[:,k]
    
    This is equivalent to: for each k, do a 3x3 matrix-vector multiplication.
    
    We can think of this as: the tensor is the "sum" (in a tensor sense) of 6 separate
    3x3x1 multiplication tensors, but they all share the A factor.
    
    A better approach: use the fact that we can compute all 6 columns at once.
    The rank of matrix multiplication M_{n,m,p} is at most n*m*p (schoolbook).
    For 3x3x6, schoolbook is 54.
    
    Let me just implement a proper optimization-based search.
    """
    pass

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # Schoolbook baseline: rank = n*m*p = 54
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find lower rank decompositions using optimization
    # The key insight: we can use alternating least squares or gradient-based methods
    
    def try_rank(r, max_iters=2000, lr=0.01, tol=1e-10):
        """Try to find a rank-r decomposition using alternating minimization."""
        if time.time() > deadline - 10:
            return None
        
        # Initialize with random values
        U_r = np.random.randn(n * m, r) * 0.1
        V_r = np.random.randn(m * p, r) * 0.1
        W_r = np.random.randn(p * n, r) * 0.1
        
        best_err = float('inf')
        best_U_r, best_V_r, best_W_r = U_r.copy(), V_r.copy(), W_r.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Compute current error
            T = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        err += (T[a, b, c] - 1) ** 2
            for a in range(n * m):
                for b in range(m * p):
                    for c in range(p * n):
                        i = a // m
                        j = a % m
                        j2 = b // p
                        k = b % p
                        i2 = c % n
                        k2 = c // n
                        if not (i == i2 and j == j2 and k == k2):
                            err += T[a, b, c] ** 2
            
            if err < best_err:
                best_err = err
                best_U_r, best_V_r, best_W_r = U_r.copy(), V_r.copy(), W_r.copy()
            
            if err < tol:
                break
            
            # Alternating least squares: update one factor at a time
            # For each factor, we solve a linear system
            
            # Update U_r: for each column r_idx, minimize ||T - sum U[:,r] V[r,:] W[r,:]||
            # This is complex, so use simple gradient step instead
            
            # Compute gradients approximately
            # dU = -2 * dT/dU where dT is the residual tensor
            
            # Residual tensor
            R = np.zeros((n * m, m * p, p * n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        R[a, b, c] = T[a, b, c] - 1
            for a in range(n * m):
                for b in range(m * p):
                    for c in range(p * n):
                        i = a // m
                        j = a % m
                        j2 = b // p
                        k = b % p
                        i2 = c % n
                        k2 = c // n
                        if not (i == i2 and j == j2 and k == k2):
                            R[a, b, c] = T[a, b, c]
            
            # Gradient for U: dU[a, r] = sum_{b,c} R[a,b,c] * V[b,r] * W[c,r]
            grad_U = np.einsum('abc,br,cr->ar', R, V_r, W_r)
            grad_V = np.einsum('abc,ar,cr->br', R, U_r, W_r)
            grad_W = np.einsum('abc,ar,br->cr', R, U_r, V_r)
            
            # Gradient descent step
            U_r -= lr * grad_U
            V_r -= lr * grad_V
            W_r -= lr * grad_W
        
        if best_err < tol:
            return best_U_r, best_V_r, best_W_r, best_err
        return None
    
    # Try various ranks
    for r in range(20, 54):
        if time.time() > deadline - 20:
            break
        
        # Multiple random restarts
        for restart in range(3):
            if time.time() > deadline - 15:
                break
            
            result = try_rank(r, max_iters=1000, lr=0.005)
            if result is not None:
                U_r, V_r, W_r, err = result
                if err < 1e-8:
                    best_rank = r
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    break  # Found valid decomposition for this rank, try lower
        
        if best_rank == r:
            # Found it, try lower ranks
            continue
    
    # Final verification
    if best_rank < rank_school:
        err = _error(best_U, best_V, best_W, n, m, p)
        if err < 1e-6:
            return {'U': best_U.tolist(), 'V': best_V.tolist(), 'W': best_W.tolist()}
    
    # Fall back to schoolbook
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
