"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k'*n+i' where i'=i, k'=k
    # i.e., T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find a decomposition of rank r
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # such that sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] = T
    
    best_rank = n * m * p  # schoolbook
    best_result = None
    
    # Schoolbook as fallback
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try lower ranks with optimization
    np.random.seed(42)
    
    for target_rank in [20, 25, 30, 35]:
        if time.time() > deadline - 30:
            break
            
        success = False
        for attempt in range(3):
            if time.time() > deadline - 20:
                break
                
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Normalize to avoid scale issues
            for c in range(target_rank):
                norm = max(np.linalg.norm(U[:,c]), np.linalg.norm(V[:,c]), np.linalg.norm(W[:,c]), 1e-10)
                U[:,c] /= norm
                V[:,c] /= norm
                W[:,c] /= norm
            
            # Gradient descent
            lr = 0.01
            for step in range(2000):
                if time.time() > deadline - 15:
                    break
                    
                # Compute residual: R = T - sum_c U_c ⊗ V_c ⊗ W_c
                # Gradient wrt U: for each a, grad_U[a,c] = sum_{b,c_idx} T[a,b,c_idx] V[b,c] W[c_idx,c] - sum_{b,c_idx} (sum_c2 U[:,c2]⊗V[:,c2]⊗W[:,c2])[a,b,c_idx] V[b,c] W[c_idx,c]
                # This is expensive. Let's use a simpler approach.
                
                # Compute current decomposition
                # R[a,b,c_idx] = T[a,b,c_idx] - sum_c2 U[a,c2]*V[b,c2]*W[c_idx,c2]
                
                # For efficiency, compute in chunks
                # grad_U[a,c] = sum_{b,c_idx} T[a,b,c_idx] * V[b,c] * W[c_idx,c] - sum_{b,c_idx} R[a,b,c_idx] * V[b,c] * W[c_idx,c]
                #             = sum_{b,c_idx} (T[a,b,c_idx] - R[a,b,c_idx]) * V[b,c] * W[c_idx,c]
                # Wait, that's circular. Let me think again.
                
                # The objective is min ||T - sum_c U_c⊗V_c⊗W_c||^2
                # d/dU[a,c] = -2 * sum_{b,c_idx} (T[a,b,c_idx] - sum_c2 U[a,c2]V[b,c2]W[c_idx,c2]) * V[b,c] * W[c_idx,c]
                
                # This is O(n*m * m*p * p*n * r) per step which is 8*20*10*r = 1600r per step.
                # For r=20, that's 32000 per step, 2000 steps = 64M operations. Should be fast with numpy.
                
                # Let's compute the residual and gradients
                # Residual: R = T - np.einsum('ac,bc,dc->abd', U, V, W)
                # This might be memory intensive. Let's do it differently.
                
                # Actually, let's compute the residual efficiently
                # R[a,b,d] = T[a,b,d] - sum_c U[a,c]*V[b,c]*W[d,c]
                
                # For gradient computation, we need:
                # grad_U[a,c] = -2 * sum_{b,d} R[a,b,d] * V[b,c] * W[d,c]
                # grad_V[b,c] = -2 * sum_{a,d} R[a,b,d] * U[a,c] * W[d,c]
                # grad_W[d,c] = -2 * sum_{a,b} R[a,b,d] * U[a,c] * V[b,c]
                
                # Let's compute R first
                # R = T - np.einsum('ac,bc,dc->abd', U, V, W)
                # Shape: (8, 20, 10) - manageable
                
                R = T - np.einsum('ac,bc,dc->abd', U, V, W)
                err = np.linalg.norm(R)
                
                if err < 1e-8:
                    success = True
                    break
                    
                # Compute gradients
                # grad_U[a,c] = -2 * sum_{b,d} R[a,b,d] * V[b,c] * W[d,c]
                # = -2 * einsum('abd,bc,dc->ac', R, V, W)
                grad_U = -2 * np.einsum('abd,bc,dc->ac', R, V, W)
                grad_V = -2 * np.einsum('abd,ac,dc->bc', R, U, W)
                grad_W = -2 * np.einsum('abd,ac,bc->dc', R, U, V)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Renormalize to prevent divergence
                for c in range(target_rank):
                    nu = np.linalg.norm(U[:,c])
                    nv = np.linalg.norm(V[:,c])
                    nw = np.linalg.norm(W[:,c])
                    if nu < 1e-10 or nv < 1e-10 or nw < 1e-10:
                        break
                    # Keep product roughly constant
                    # Just normalize each to unit norm
                    U[:,c] /= nu
                    V[:,c] /= nv
                    W[:,c] /= nw
                    # But this changes the scale... let's not renormalize, just clip
                    # Actually, let's skip renormalization and use adaptive LR
                # Reduce learning rate if error increased
                # (simplified: just use fixed small LR)
            
            if success:
                # Verify
                R_check = T - np.einsum('ac,bc,dc->abd', U, V, W)
                if np.linalg.norm(R_check) < 1e-6:
                    # Round to nearest half-integer or integer
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    R_rounded = T - np.einsum('ac,bc,dc->abd', U_rounded, V_rounded, W_rounded)
                    if np.linalg.norm(R_rounded) < 1e-6:
                        best_rank = target_rank
                        best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                        break
                    else:
                        # Use unrounded if close enough
                        best_rank = target_rank
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        break
        
        if success:
            break
    
    return best_result
# EVOLVE-BLOCK-END
