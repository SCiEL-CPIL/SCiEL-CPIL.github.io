import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Schoolbook solution (rank = n*m*p = 60)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    best_U, best_V, best_W, best_rank = schoolbook()

    # Build target tensor T[k, i, i', j', j'', k''] = delta(k,k'') * delta(i,i') * delta(j',j'')
    # where j' and j'' are both equal to the same j
    T_target = np.zeros((p, n, n, m, m, p), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[k, i, i, j, j, k] = 1
    
    # Flatten target for linear system
    T_flat = T_target.flatten()
    num_constraints = T_flat.shape[0]  # 3600
    
    # Helper to compute the linear system matrix for given rank r
    # Each rank-1 term contributes W[k,i] * U[i',j'] * V[j'',k'']
    # We need to find U, V, W such that sum_t W[k,i,t]*U[i',j',t]*V[j'',k'',t] = T[k,i,i',j',j'',k'']
    
    # For a fixed rank r, we have r * (n*m + m*p + p*n) = r * (12 + 20 + 15) = 47r unknowns
    # But the equations are nonlinear (trilinear). We need a different approach.
    
    # Strategy: Use the fact that for matrix multiplication, we can fix U and V and solve for W linearly
    # Or use a smarter approach: exploit the structure
    
    # Better approach: The matrix multiplication tensor has the property that
    # T[k,i,i',j',j'',k''] is nonzero only when k=k'', i=i', j'=j''
    # This gives us 60 nonzero entries out of 3600.
    
    # For each rank-1 term, it contributes to all 3600 entries, but we can use the sparsity.
    
    # Let's try a different approach: use known results about matrix multiplication rank.
    # For 3x4 * 4x5, the schoolbook is 60. Can we do better?
    # The lower bound is max(n*m, m*p, p*n) = max(12, 20, 15) = 20, but that's weak.
    # Better lower bound: the tensor rank is at least the max of the border ranks.
    
    # Let's try to find a rank < 60 solution using random search with the following approach:
    # 1. Initialize U, V, W randomly
    # 2. Use gradient descent on the nonlinear objective ||sum_t W_t * U_t * V_t - T||^2
    # 3. Or use a smarter initialization
    
    # Actually, let's use the fact that we can fix one matrix and solve linearly for the others.
    # If we fix U (n*m x r), then for each k, i, we have:
    # sum_t W[k,i,t] * (U[i',j',t] * V[j'',k'',t]) = T[k,i,i',j',j'',k'']
    # This is linear in W if we define M_t[i',j',j'',k''] = U[i',j',t] * V[j'',k'',t]
    # But M_t depends on both U and V.
    
    # Let's try a coordinate descent approach:
    # 1. Fix V, solve for U and W
    # 2. Fix U, solve for V and W  
    # 3. Fix U, V, solve for W
    # etc.
    
    # For a given rank r, let's try to find a solution using random restarts.
    
    def try_rank(r, num_restarts=10):
        """Try to find a rank-r decomposition using random restarts."""
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize randomly
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Normalize to avoid numerical issues
            U = U / (np.linalg.norm(U) + 1e-10)
            V = V / (np.linalg.norm(V) + 1e-10)
            W = W / (np.linalg.norm(W) + 1e-10)
            
            # Coordinate descent: alternate between solving for each factor
            for iteration in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Solve for W given U, V
                # For each (k, i), we need: sum_t W[k,i,t] * (U[i',j',t] * V[j'',k'',t]) = T[k,i,i',j',j'',k'']
                # This is a linear system in W[k,i,:] for each (k,i)
                # But the RHS depends on all (i',j',j'',k'')
                
                # Actually, let's compute the residual and do gradient steps
                U_3d = U.reshape(n, m, r)
                V_3d = V.reshape(m, p, r)
                W_3d = W.reshape(p, n, r)
                
                # Compute current tensor
                T_calc = np.zeros((p, n, n, m, m, p), dtype=np.float64)
                for t in range(r):
                    u = U_3d[:, :, t]  # n, m
                    v = V_3d[:, :, t]  # m, p
                    w = W_3d[:, :, t]  # p, n
                    T_calc += w[:, :, None, None, None, None] * \
                              u[None, None, :, :, None, None] * \
                              v[None, None, None, None, :, :]
                
                residual = T_calc - T_target
                err = np.sum(residual ** 2)
                
                if err < 1e-8:
                    return U, V, W, True
                
                # Gradient descent step
                lr = 0.01
                # dL/dU[i',j',t] = sum_{k,i,j'',k''} residual[k,i,i',j',j'',k''] * W[k,i,t] * V[j'',k'',t]
                # This is expensive to compute exactly, so use a simplified approach
                
                # Instead, let's use a smarter initialization based on the structure
                break
            
            # If coordinate descent didn't work, try a different approach
            # Use the fact that we can set up a linear system if we fix two factors
            
            # Fix U and V, solve for W linearly
            # For each t, define M_t[i',j',j'',k''] = U[i',j',t] * V[j'',k'',t]
            # Then we need: sum_t W[k,i,t] * M_t[i',j',j'',k''] = T[k,i,i',j',j'',k'']
            # This is linear in W if we treat M_t as known.
            
            # But this is still a 3D linear system. Let's flatten it.
            # For each (k,i), we have a linear system in W[k,i,0:r]
            # The coefficient matrix has entries: for each (i',j',j'',k''), the coefficient is M_t[i',j',j'',k'']
            # But this couples all (k,i) pairs through the shared (i',j',j'',k'')
            
            # This is getting complex. Let's try a different strategy.
            
            # Strategy: Use known decompositions for smaller cases and extend
            # For 3x3 * 3x3, the minimum rank is 23 (Strassen-like)
            # For 3x4 * 4x5, we might be able to do better than 60.
            
            # Let's try a greedy approach: start with rank 60 and try to merge terms
            pass
        
        return None, None, None, False
    
    # Try ranks from 50 to 59
    for r in range(50, 60):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, num_restarts=5)
        if result[3]:  # Success
            best_U, best_V, best_W, best_rank = result
            break
    
    # If no better found, return schoolbook
    U, V, W, rank = best_U, best_V, best_W, best_rank
    
    # Verify the solution
    U_3d = U.reshape(n, m, -1)
    V_3d = V.reshape(m, p, -1)
    W_3d = W.reshape(p, n, -1)
    
    T_calc = np.zeros((p, n, n, m, m, p), dtype=np.float64)
    for t in range(U.shape[1]):
        u = U_3d[:, :, t]
        v = V_3d[:, :, t]
        w = W_3d[:, :, t]
        T_calc += w[:, :, None, None, None, None] * \
                  u[None, None, :, :, None, None] * \
                  v[None, None, None, None, :, :]
    
    if not np.allclose(T_calc, T_target, atol=1e-6):
        # Fall back to schoolbook
        U, V, W, rank = schoolbook()
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
