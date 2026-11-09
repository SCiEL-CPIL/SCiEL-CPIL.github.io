"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.linalg import null_space

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Start with schoolbook decomposition
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_schoolbook
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    # Try to find lower rank decompositions using structured approaches
    # For matrix multiplication, we can try various rank values
    
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # Also rank >= n*m*p / min(n*m, m*p, p*n) ... various bounds
    
    # Let's try ranks from a reasonable lower bound upward
    # Theoretical lower bound for 3x5 * 5x7: at least max(15, 35, 21) = 35
    # But we can do better with structured decompositions
    
    # Try various rank values
    min_possible_rank = 35  # max(n*m, m*p, p*n)
    
    # We'll try to find decompositions of rank r for r in range(min_possible_rank, rank_schoolbook)
    # Use a structured approach: try to find low-rank approximations that are exact
    
    # For exact decomposition, we need:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # This is a tensor decomposition problem. Let's use a different approach.
    # We can formulate this as finding matrices U, V, W such that the tensor equation holds.
    
    # Alternative: use the fact that matrix multiplication can be done with fewer multiplications
    # by exploiting linearity. For example, we can group terms.
    
    # Let's try a systematic search for low-rank decompositions using random initialization
    # and then verifying if the decomposition is exact.
    
    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-10):
        """Verify if U, V, W form an exact decomposition of matrix multiplication."""
        n, m, p = DIMENSIONS
        # The decomposition should satisfy:
        # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] 
        # represents the coefficient of A[i,j]*B[j,k] in C[i,k]
        
        # Actually, the tensor T[i,j,k] = 1 for the multiplication tensor
        # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Let's verify by checking that the computed tensor matches the multiplication tensor
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1.0  # The multiplication tensor has all ones
        
        # Compute the decomposed tensor
        T_decomp = np.zeros((n, m, p))
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        
        for c in range(len(U_mat[0])):
            # U[:, c] has shape (n*m,), V[:, c] has shape (m*p,), W[:, c] has shape (p*n,)
            # Reshape to (n, m), (m, p), (p, n)
            Uc = U_arr[:, c].reshape(n, m)
            Vc = V_arr[:, c].reshape(m, p)
            Wc = W_arr[:, c].reshape(p, n)
            
            # The contribution to T[i,j,k] is Uc[i,j] * Vc[j,k] * Wc[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_decomp[i, j, k] += Uc[i, j] * Vc[j, k] * Wc[k, i]
        
        return np.allclose(T_decomp, T, atol=tol)
    
    # Try to find lower rank decompositions
    # Use a structured approach based on known algorithms
    
    # For 3x5 * 5x7, let's try ranks from 35 upward
    for target_rank in range(min_possible_rank, rank_schoolbook + 1):
        if time.time() > deadline:
            break
        
        # Try random initializations and local optimization
        # This is a non-convex problem, so we'll use multiple restarts
        
        found = False
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with random values
            np.random.seed(42 + restart + target_rank)
            U_rand = np.random.randn(n * m, target_rank) * 0.1
            V_rand = np.random.randn(m * p, target_rank) * 0.1
            W_rand = np.random.randn(p * n, target_rank) * 0.1
            
            # Try to optimize using gradient-free methods or just check if close
            # For exact decomposition, we need to solve a system of equations
            
            # The tensor equation: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # This gives n*m*p = 105 equations, with 3*target_rank variables
            # Actually (n*m + m*p + p*n) * target_rank variables
            
            # Let's try a different approach: use the structure of matrix multiplication
            # We can decompose using block methods
            
            # For now, let's just try to find if there's a known decomposition
            # with rank less than schoolbook
            
        # If we found a better decomposition, update best
        if found:
            best_rank = target_rank
            break
    
    # Return the best decomposition found
    if best_rank < rank_schoolbook:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
