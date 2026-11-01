import numpy as np
import time
import random

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to find rank 15 decomposition
    rank_target = 15
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin
    
    # Target tensor
    # Flatten indices
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0

    best_solution = None
    
    # Randomized search with gradient descent
    for attempt in range(20): # Try a few random initializations
        if time.time() > deadline:
            break
            
        # Initialize U, V, W with small random values
        # Symmetry breaking: fix first column of U to 1s? No, just random.
        U = np.random.randn(n * m, rank_target) * 0.1
        V = np.random.randn(m * p, rank_target) * 0.1
        W = np.random.randn(p * n, rank_target) * 0.1
        
        # Gradient Descent
        lr = 0.01
        for step in range(1000):
            if time.time() > deadline:
                break
                
            # Compute current tensor
            # T = sum_r U[:, r] * V[:, r] * W[:, r]
            # This is slow if done naively. Use einsum or loops.
            # Given small sizes, loops are fine.
            T_current = np.zeros_like(target)
            for r in range(rank_target):
                # Outer product of three vectors
                # U_col = U[:, r], V_col = V[:, r], W_col = W[:, r]
                # T += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                T_current += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
            error = T_current - target
            if np.linalg.norm(error) < 1e-6:
                break
            
            # Compute gradients
            # dT/dU[:, r] = V[:, r] * W[:, r] (outer product)
            # grad_U[i, r] = sum_{j,k} error[i, j, k] * V[j, r] * W[k, r]
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r in range(rank_target):
                # grad_U[:, r] = einsum('ijk,j,k->i', error, V[:, r], W[:, r])
                grad_U[:, r] = np.einsum('ijk,j,k->i', error, V[:, r], W[:, r])
                # grad_V[:, r] = einsum('ijk,i,k->j', error, U[:, r], W[:, r])
                grad_V[:, r] = np.einsum('ijk,i,k->j', error, U[:, r], W[:, r])
                # grad_W[:, r] = einsum('ijk,i,j->k', error, U[:, r], V[:, r])
                grad_W[:, r] = np.einsum('ijk,i,j->k', error, U[:, r], V[:, r])
                
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Optional: normalize to prevent drift
            # for r in range(rank_target):
            #     norm_u = np.linalg.norm(U[:, r])
            #     norm_v = np.linalg.norm(V[:, r])
            #     norm_w = np.linalg.norm(W[:, r])
            #     if norm_u > 1e-6:
            #         U[:, r] /= norm_u
            #         V[:, r] *= norm_u
            #     if norm_v > 1e-6:
            #         V[:, r] /= norm_v
            #         W[:, r] *= norm_v

        # Check if converged
        T_final = np.zeros_like(target)
        for r in range(rank_target):
            T_final += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        
        if np.linalg.norm(T_final - target) < 1e-4:
            # Try to snap to half-integers
            U_snapped = np.round(U * 2) / 2
            V_snapped = np.round(V * 2) / 2
            W_snapped = np.round(W * 2) / 2
            
            # Verify snapped solution
            T_snapped = np.zeros_like(target)
            for r in range(rank_target):
                T_snapped += np.einsum('i,j,k->ijk', U_snapped[:, r], V_snapped[:, r], W_snapped[:, r])
            
            if np.allclose(T_snapped, target, atol=1e-9):
                # Convert to lists of integers/half-integers
                # The problem asks for integer/half-integer entries.
                # Representing as floats is fine if they are exact halves.
                # But the output format is lists.
                
                def to_list(matrix):
                    return [[float(val) for val in row] for row in matrix]
                
                best_solution = {
                    "U": to_list(U_snapped),
                    "V": to_list(V_snapped),
                    "W": to_list(W_snapped)
                }
                break # Found a valid rank 15 decomposition

    if best_solution:
        return best_solution
    else:
        return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
