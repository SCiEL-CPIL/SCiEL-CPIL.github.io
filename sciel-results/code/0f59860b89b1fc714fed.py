"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Start with schoolbook
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    # Try to find lower rank decompositions
    # Known: 3x4x5 matrix multiplication tensor
    # Lower bound: max(n*p, m*p, n*m) = max(15, 20, 12) = 20
    # Upper bound: 60 (schoolbook)
    # Try ranks from lower bound up to schoolbook
    
    # The tensor T[i,j,k] = delta[i,j,k] where output[k*n+i] gets A[i,j]*B[j,k]
    # We need U, V, W such that for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if this is the "identity" term
    # Actually, the matrix multiplication tensor is:
    # M[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
    # and we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = M[i,j,k] = 1
    
    # Wait, let me reconsider. The matrix multiplication tensor T is defined as:
    # T[i,j,k] = 1 for all valid i,j,k
    # A decomposition of rank r means: T[i,j,k] = sum_{c=1}^{r} U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # For schoolbook: U[i*m+j, (i,j,k)] = 1, V[j*p+k, (i,j,k)] = 1, W[k*n+i, (i,j,k)] = 1
    # So sum over c gives 1 for each (i,j,k). Correct.
    
    # Let's try to find lower rank solutions using random initialization and optimization
    # or structured approaches.
    
    # One known approach: try rank = n*m*p - (something)
    # For 3x4x5, the minimum rank is known to be at most 60, but can it be lower?
    # Actually, for matrix multiplication, the tensor rank is generally n*m*p for small sizes,
    # but there might be savings.
    
    # Let's try a direct approach: assume rank r, and try to solve for U, V, W
    # This is a hard problem, but we can try random restarts with gradient descent.
    
    # Alternative: use the fact that for small matrices, we can try all possible structures.
    
    # Let's first verify the schoolbook solution works, then try to improve.
    
    def verify(U, V, W, rank):
        """Check if U, V, W form a valid decomposition."""
        # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1) > 1e-6:
                        return False
        return True
    
    def verify_numpy(U, V, W, rank):
        """Faster verification using numpy."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :rank], V[j*p+k, :rank] * W[k*n+i, :rank])
                    if abs(s - 1) > 1e-6:
                        return False
        return True
    
    # Try to find lower rank solutions
    # Use random initialization with some structure
    best_score = best_rank
    
    # Try ranks from 20 up to 59
    for r in range(20, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try several random initializations
        max_attempts = 5
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values
            np.random.seed(attempt * 12345 + r)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Try gradient descent to minimize the error
            # Error = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
            
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U[i*m+j, :r], V[j*p+k, :r] * W[k*n+i, :r])
                            err = s - 1
                            # d/dU[i*m+j,c] = err * V[j*p+k,c] * W[k*n+i,c]
                            grad_U[i*m+j, :] += err * V[j*p+k, :r] * W[k*n+i, :r]
                            grad_V[j*p+k, :] += err * U[i*m+j, :r] * W[k*n+i, :r]
                            grad_W[k*n+i, :] += err * U[i*m+j, :r] * V[j*p+k, :r]
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if step % 50 == 0:
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = np.dot(U[i*m+j, :r], V[j*p+k, :r] * W[k*n+i, :r])
                                max_err = max(max_err, abs(s - 1))
                    if max_err < 1e-4:
                        break
            
            # Check if we found a valid solution
            if verify_numpy(U, V, W, r):
                best_rank = r
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                break
        
        if best_rank <= r:
            break
    
    # Convert to lists with half-integer rounding if needed
    def to_list(arr):
        return [[float(x) for x in row] for row in arr]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
