"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]
N, M, P = DIMENSIONS

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s kill limit
    
    n, m, p = N, M, P
    u_size = n * m  # 12
    v_size = m * p  # 24
    w_size = p * n  # 18
    
    # Start with schoolbook rank 72
    rank_schoolbook = n * m * p
    best_rank = rank_schoolbook
    best_U = np.zeros((u_size, rank_schoolbook))
    best_V = np.zeros((v_size, rank_schoolbook))
    best_W = np.zeros((w_size, rank_schoolbook))
    
    # Initialize schoolbook decomposition
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m + j, col] = 1.0
                best_V[j*p + k, col] = 1.0
                best_W[k*n + i, col] = 1.0
                col += 1
    
    # Verify schoolbook is correct
    def verify(U, V, W, tol=1e-10):
        """Verify that sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] equals the matmul tensor."""
        rank = U.shape[1]
        # Compute the tensor as sum of outer products
        # For each (i,j,k), tensor[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # But we need to map indices properly:
        # U is indexed by (i*m + j), V by (j*p + k), W by (k*n + i)
        
        # Let's verify by checking a few random matmuls
        for trial in range(20):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            # Compute C from decomposition: C[i,k] = sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            C_comp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    total = 0.0
                    for j in range(m):
                        u_val = U[i*m + j, :]
                        v_val = V[j*p + k, :]
                        w_val = W[k*n + i, :]
                        total += np.dot(np.dot(u_val, v_val), w_val)  # sum_c u_c * v_c * w_c
                    C_comp[i, k] = total
            
            if not np.allclose(C_comp, C_expected, atol=tol, rtol=tol):
                return False
        return True
    
    if not verify(best_U, best_V, best_W):
        # Should not happen, but just in case
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to improve using bilinear algorithm search / tensor decomposition
    # We'll use a randomized approach: start from schoolbook and try to merge terms
    # by finding linear dependencies
    
    # Alternative approach: use known algorithms for small matrix multiplication
    # For 3x4 by 4x6, we can try various block decompositions
    
    # Let's try a structured search: reduce rank by finding zero-sum patterns
    # We'll use a simple approach: try to find rank-1 terms that can be removed
    # by adjusting other terms (this is essentially finding a better bilinear algorithm)
    
    # Use a local search / optimization approach
    # Parametrize U, V, W with rank r and optimize to match the tensor
    
    # We'll try decreasing ranks from 72 down to a lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    
    # For efficiency, we'll use a few random restarts with lower ranks
    
    current_time = time.time()
    if current_time > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try various ranks
    # The minimum possible rank is at least max(n*m, m*p, p*n) = 24
    # But it's likely higher. Let's try from 72 down to maybe 50 in steps
    
    # Use a simple iterative solver: fix one matrix, solve for others
    # This is a non-convex problem, so we'll use random initialization and local search
    
    def try_rank(r, num_restarts=10, tol=1e-8):
        """Try to find a decomposition of rank r."""
        for restart in range(num_restarts):
            start_t = time.time()
            if start_t > deadline:
                return None
            
            # Initialize randomly
            U = np.random.randn(u_size, r)
            V = np.random.randn(v_size, r)
            W = np.random.randn(w_size, r)
            
            # Scale to avoid numerical issues
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            # Iteratively refine using alternating least squares
            for iteration in range(200):
                if time.time() > deadline:
                    return None
                
                # Update U: for each (i,j), solve for U[i*m+j, :] 
                # such that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta_{j,k}
                # This is a linear system for each (i,j)
                
                new_U = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        # We need: sum_k (sum_c U[i*m+j,c] * W[k*n+i,c]) * V[j*p+k,c] = delta_{j,k} for all k
                        # Let x_c = U[i*m+j,c] * W[k*n+i,c] ... this is coupled
                        # 
                        # Actually, the condition is:
                        # For all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 if j==k else 0
                        # 
                        # This is nonlinear. Let's use a different approach.
                        # 
                        # Instead, let's verify and use gradient-based optimization
                        pass
                
                # Check current solution
                C_comp = np.zeros((n, m, p))
                for c in range(r):
                    u_vec = U[:, c]
                    v_vec = V[:, c]
                    w_vec = W[:, c]
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                C_comp[i, j, k] += u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]
                
                # Target tensor: T[i,j,k] = 1 if j==k else 0
                T_target = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        T_target[i, j, j] = 1.0
                
                err = np.linalg.norm(C_comp - T_target)
                if err < tol:
                    # Verify with matmul
                    if verify(U, V, W, tol=tol):
                        return U, V, W
                
                # Simple gradient step (this is very expensive, so skip for now)
                # Instead, just return None if we can't find a solution
                pass
            
            return None
        
        return None
    
    # Given the complexity of the optimization, let's try a more structured approach
    # Use block decomposition with Strassen-like ideas
    
    # For 3x4 * 4x6, we can split into blocks
    # A is 3x4, B is 4x6
    
    # Let's try to use the fact that we can do 2x2 blocks with Strassen (rank 7 instead of 8)
    
    # Split A into blocks of size 2x2 and 1x2
    # Split B into blocks of size 2x2 and 2x4
    
    # Actually, let's just return the schoolbook solution if we can't do better
    # The search for improved rank is non-trivial
    
    current_time = time.time()
    if current_time > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try a few specific structured decompositions
    
    # Decomposition 1: Use block diagonal structure
    # Since the matmul tensor has the property that T[i,j,k] = delta_{j,k},
    # we can exploit this with a block approach
    
    # For now, return the schoolbook solution as it's guaranteed correct
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
