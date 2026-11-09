"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 4, 8
    deadline = time.time() + 115  # 115 second deadline (5s margin)
    
    # Schoolbook decomposition rank = n*m*p = 64
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    def verify(U_mat, V_mat, W_mat, n, m, p):
        """Verify that the decomposition exactly computes matrix multiplication"""
        rank_val = len(U_mat[0])
        
        # Build coefficient matrix: coeff[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
        # We need coeff[i,k,j] = 1 for all i,k,j
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    coeff = np.sum(U_arr[i*m+k, :] * V_arr[k*p+j, :] * W_arr[j*n+i, :])
                    if abs(coeff - 1.0) > 1e-10:
                        return False
        return True
    
    # Verify schoolbook
    if not verify(U, V, W, n, m, p):
        return {"U": U, "V": W, "W": W}
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Strategy: For each candidate rank, try to find a decomposition by solving a nonlinear system
    # or by using known constructions
    
    # Known results: The tensor rank of 2x4x8 matrix multiplication
    # Upper bounds from literature suggest ranks around 40-50 are achievable
    # Let's try to construct a decomposition using block structure
    
    # Approach: Use the fact that 2x4x8 can be seen as 2*(2x2x4) or similar
    # Actually, let's try a more systematic approach using random initialization and optimization
    
    # For now, let's try a few specific constructions
    
    # Construction 1: Try to merge some schoolbook terms
    # The key insight is that for matrix multiplication, we have:
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # The tensor T[i,k,j] = 1 for all valid (i,k,j)
    
    # Let's try to find a decomposition of rank < 64 by solving the system
    # sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1 for all i,k,j
    
    # This is a system of n*m*p = 64 equations
    # With r terms, we have (n*m + m*p + p*n)*r = (8 + 32 + 16)*r = 56*r unknowns
    
    # For r = 50, we have 2800 unknowns and 64 equations - very underdetermined
    # We can try to find a solution by fixing some variables and solving for others
    
    # Let's try a greedy approach: start with schoolbook and try to eliminate terms
    
    # Actually, let's try a different approach: use the structure of small matrix multiplication
    # For 2x2x2, Strassen's algorithm gives rank 7 instead of 8
    
    # For 2x4x8, we can potentially use a similar idea by breaking into blocks
    
    # Let's try to construct a decomposition using 2x2 blocks with Strassen-like multiplication
    
    # A is 2x4, B is 4x8
    # Split A into [A_left, A_right] where each is 2x2
    # Split B into [B_top; B_bottom] where each is 2x4 (wait, B is 4x8, so top is 2x4, bottom is 2x4)
    # Actually B is 4x8, so if we split the rows: B = [B1; B2] where B1 is 2x4, B2 is 2x4
    # But we need B to be 4x8, so B1 is 2x8, B2 is 2x8
    
    # C = A_left * B1 + A_right * B2
    # Where A_left is 2x2, B1 is 2x8, A_right is 2x2, B2 is 2x8
    
    # Each 2x2 * 2x8 multiplication can be done with rank at most 7*8/2 = 28? No...
    # Actually, 2x2 * 2x8: the tensor has 2*2*8 = 32 entries
    # Using Strassen for the 2x2 part: rank 7 for the 2x2x2 core, then extend to 8 columns
    # So 2x2 * 2x8 has rank at most 7 * 4 = 28? Not exactly...
    
    # Let's think differently. The 2x2 * 2x8 multiplication:
    # C[i,j] = sum_k A[i,k] * B[k,j] for i in {0,1}, j in {0,...,7}, k in {0,1}
    # This is equivalent to 4 separate 2x2 * 2x1 multiplications (one for each pair of columns)
    # Each 2x2 * 2x1 has tensor rank at most 4 (schoolbook) or 3 (using Strassen-like)
    
    # Actually, for 2x2 * 2x1: C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j] for j in {0}
    # This is just 4 terms: (i=0,k=0), (i=0,k=1), (i=1,k=0), (i=1,k=1)
    # Can we do it in rank 3? 
    # T[0,0,0]=1, T[0,1,0]=1, T[1,0,0]=1, T[1,1,0]=1
    # We need sum_r U[0,0,r]*V[0,0,r]*W[0,r] = 1
    # sum_r U[0,1,r]*V[1,0,r]*W[0,r] = 1
    # sum_r U[1,0,r]*V[0,0,r]*W[1,r] = 1
    # sum_r U[1,1,r]*V[1,0,r]*W[1,r] = 1
    # This is a 4x3 system (if rank=3), which might be solvable
    
    # Let's try to find a rank 3 decomposition for 2x2*2x1
    # Actually, I recall that the border rank of 2x2*2x1 is 3, but the tensor rank might be 4
    
    # Let's just try to solve the full problem numerically for small ranks
    
    # For rank r, we need to find U (8xr), V (32xr), W (16xr) such that:
    # sum_{r'=0}^{r-1} U[i*m+k, r'] * V[k*p+j, r'] * W[j*n+i, r'] = 1 for all i,k,j
    
    # Let's try to solve this using a simple iterative method
    
    # Initialize with random values
    np.random.seed(42)
    
    # Try ranks from 55 down to 40
    for target_rank in range(55, 39, -1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline - 20:
                break
            
            # Initialize U, V, W with random values
            U_try = np.random.randn(n*m, target_rank)
            V_try = np.random.randn(m*p, target_rank)
            W_try = np.random.randn(p*n, target_rank)
            
            # Normalize to avoid overflow
            U_try /= np.sqrt(np.sum(U_try**2))
            V_try /= np.sqrt(np.sum(V_try**2))
            W_try /= np.sqrt(np.sum(W_try**2))
            
            # Try to optimize using gradient descent on the constraint
            # Objective: minimize sum_{i,k,j} (sum_r U[i*m+k,r]*V[k*p+j,r]*W[j*n+i,r] - 1)^2
            
            lr = 0.01
            for iteration in range(100):
                if time.time() > deadline - 20:
                    break
                
                # Compute gradients
                grad_U = np.zeros_like(U_try)
                grad_V = np.zeros_like(V_try)
                grad_W = np.zeros_like(W_try)
                
                for i in range(n):
                    for k in range(m):
                        for j in range(p):
                            # Compute current coefficient
                            coeff = np.sum(U_try[i*m+k, :] * V_try[k*p+j, :] * W_try[j*n+i, :])
                            error = coeff - 1.0
                            
                            # Gradient with respect to U
                            grad_U[i*m+k, :] += 2 * error * V_try[k*p+j, :] * W_try[j*n+i, :]
                            # Gradient with respect to V
                            grad_V[k*p+j, :] += 2 * error * U_try[i*m+k, :] * W_try[j*n+i, :]
                            # Gradient with respect to W
                            grad_W[j*n+i, :] += 2 * error * U_try[i*m+k, :] * V_try[k*p+j, :]
                
                # Update
                U_try -= lr * grad_U
                V_try -= lr * grad_V
                W_try -= lr * grad_W
                
                # Check convergence
                max_error = 0
                for i in range(n):
                    for k in range(m):
                        for j in range(p):
                            coeff = np.sum(U_try[i*m+k, :] * V_try[k*p+j, :] * W_try[j*n+i, :])
                            max_error = max(max_error, abs(coeff - 1.0))
                
                if max_error < 1e-8:
                    # Found a solution! Round to nearest half-integer
                    U_rounded = np.round(U_try * 2) / 2
                    V_rounded = np.round(V_try * 2) / 2
                    W_rounded = np.round(W_try * 2) / 2
                    
                    # Verify the rounded solution
                    if verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
                        if target_rank < best_rank:
                            best_rank = target_rank
                            best_U = U_rounded.tolist()
                            best_V = V_rounded.tolist()
                            best_W = W_rounded.tolist()
                    break
                if iteration % 20 == 0:
                    # Adaptive learning rate
                    if max_error > 1.0:
                        lr *= 0.5
                    elif max_error < 0.1:
                        lr *= 1.5
                    lr = min(max(lr, 0.001), 0.1)
    
    # Convert to lists if needed
    if isinstance(best_U, np.ndarray):
        best_U = best_U.tolist()
    if isinstance(best_V, np.ndarray):
        best_V = best_V.tolist()
    if isinstance(best_W, np.ndarray):
        best_W = best_W.tolist()
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
