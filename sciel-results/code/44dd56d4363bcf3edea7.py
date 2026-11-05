"""Program for 3x4x5 matrix multiplication tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    n, m, p = DIMENSIONS
    rank_full = n * m * p
    
    # Start with the known valid schoolbook decomposition
    U = [[0] * rank_full for _ in range(n * m)]
    V = [[0] * rank_full for _ in range(m * p)]
    W = [[0] * rank_full for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_full
    
    # Try to find a lower rank decomposition using random search
    # Strategy: start with a random low-rank initialization and try to fit the tensor
    # The multiplication tensor T has T[i,j,k] = 1 if we're computing (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # In our indexing: T has indices (i*m+j, j*p+k, k*n+i) -> value 1
    # Actually, the tensor is: T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a, b, c] = 1 iff the indices are consistent (i, j, k all match)
    
    # Let's build the target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try random low-rank approximations
    # For rank r, we have U (nm x r), V (mp x r), W (pn x r)
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # We'll try ranks from a lower bound up to full rank
    # Lower bound: max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # Also, for rectangular matrix multiplication, lower bounds exist
    # Let's try ranks from 20 up to 59
    
    np.random.seed(42)
    
    for target_rank in range(20, rank_full):
        if time.time() > deadline - 10.0:
            break
        
        # Try multiple random initializations for this rank
        max_attempts = 50 if target_rank < 40 else 20
        for attempt in range(max_attempts):
            if time.time() > deadline - 5.0:
                break
            
            # Initialize with random values
            U_r = np.random.randn(n * m, target_rank)
            V_r = np.random.randn(m * p, target_rank)
            W_r = np.random.randn(p * n, target_rank)
            
            # Normalize to avoid scale issues
            U_r /= np.linalg.norm(U_r, axis=0, keepdims=True) + 1e-10
            V_r /= np.linalg.norm(V_r, axis=0, keepdims=True) + 1e-10
            W_r /= np.linalg.norm(W_r, axis=0, keepdims=True) + 1e-10
            
            # Use alternating least squares / gradient descent to fit the tensor
            # We want to minimize ||T - T_approx||_F^2 where T_approx[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            
            # Use a simple iterative approach: fix two factors, solve for the third
            # This is like ALS for CP decomposition
            
            for iteration in range(100):
                if time.time() > deadline - 2.0:
                    break
                
                # Solve for U: for each a, U[a,:] should minimize sum_b,c T[a,b,c]*V[b,:]^T*W[c,:]
                # Actually, for fixed V and W, U[a,r] = sum_{b,c} T[a,b,c] * V[b,r] * W[c,r]
                # But this is not quite right for ALS. Let me think more carefully.
                #
                # T_approx = sum_r U[:,r] * V[:,r]^T * W[:,r]^T (as a tensor)
                # For fixed V and W, the optimal U is:
                # U[a,r] = (sum_{b,c} T[a,b,c] * V[b,r] * W[c,r]) / (sum_{b,c} V[b,r]^2 * W[c,r]^2)
                # This is the closed-form solution for the least squares problem.
                
                # Compute for U
                denom_U = np.sum(V_r ** 2, axis=0) * np.sum(W_r ** 2, axis=0)
                denom_U = np.where(denom_U > 1e-10, denom_U, 1.0)
                # U[a,r] = sum_{b,c} T[a,b,c] * V[b,r] * W[c,r]
                # = (T @ (V[:,r] * W[:,r]))[a] for each r
                # More efficiently: reshape T and use matrix multiplication
                # T has shape (nm, mp, pn)
                # For each r, we need: sum_{b,c} T[a,b,c] * V[b,r] * W[c,r]
                # = (T.reshape(nm, -1) * (V * W).reshape(mp*pn)) summed over b,c
                # Actually: sum_b sum_c T[a,b,c] * V[b,r] * W[c,r]
                # Let's compute this as: for each r, (V[:,r] * W[:,r]) gives a vector of length mp*pn
                # Then T.reshape(nm, mp*pn) @ (V[:,r] * W[:,r])
                
                # Vectorized computation
                # For all r at once:
                # TW = V_r * W_r expanded... this is tricky.
                # Let's do it differently.
                # U_new[a,r] = sum_b T[a,b,:] @ (V_r[b,r] * W_r[:,r]) ... no
                # U_new[a,r] = sum_{b,c} T[a,b,c] * V_r[b,r] * W_r[c,r]
                
                # Compute using einsum or manual loop
                # For each r: U[:,r] = (T.reshape(nm, -1) @ (V_r[:,r] * W_r[:,r]).reshape(-1))
                # Wait, that's not right either. T[a,b,c] * V_r[b,r] * W_r[c,r]
                # = sum_c (T[a,b,c] * W_r[c,r]) * V_r[b,r]
                # = (sum_c T[a,b,c] * W_r[c,r]) * V_r[b,r]
                # For fixed a, r: sum_b [V_r[b,r] * (sum_c T[a,b,c] * W_r[c,r])]
                
                # Let's use np.einsum for clarity and correctness
                # U_new = np.einsum('abc,br,cr->ar', T, V_r, W_r)
                U_new = np.einsum('abc,br,cr->ar', T, V_r, W_r)
                U_r = U_new / denom_U[np.newaxis, :]
                
                # Solve for V: V[b,r] = sum_{a,c} T[a,b,c] * U[a,r] * W[c,r] / denom
                denom_V = np.sum(U_r ** 2, axis=0) * np.sum(W_r ** 2, axis=0)
                denom_V = np.where(denom_V > 1e-10, denom_V, 1.0)
                V_new = np.einsum('abc,ar,cr->br', T, U_r, W_r)
                V_r = V_new / denom_V[np.newaxis, :]
                
                # Solve for W: W[c,r] = sum_{a,b} T[a,b,c] * U[a,r] * V[b,r] / denom
                denom_W = np.sum(U_r ** 2, axis=0) * np.sum(V_r ** 2, axis=0)
                denom_W = np.where(denom_W > 1e-10, denom_W, 1.0)
                W_new = np.einsum('abc,ar,br->cr', T, U_r, V_r)
                W_r = W_new / denom_W[np.newaxis, :]
            
            # Check if the decomposition is exact
            # Compute T_approx = sum_r U_r[:,r] * V_r[:,r]^T * W_r[:,r]^T
            T_approx = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            
            # Check max absolute error
            max_err = np.max(np.abs(T - T_approx))
            
            if max_err < 1e-9:
                # We found an exact decomposition!
                # Convert to lists
                U_list = [[float(x) for x in row] for row in U_r]
                V_list = [[float(x) for x in row] for row in V_r]
                W_list = [[float(x) for x in row] for row in W_r]
                best_U, best_V, best_W = U_list, V_list, W_list
                best_rank = target_rank
                return {"U": best_U, "V": best_V, "W": best_W}
    
    # If no lower rank found, return the schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
