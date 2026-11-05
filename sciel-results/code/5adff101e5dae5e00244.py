import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Define the target tensor T
    shape = (n * m, m * p, p * n)
    T_target = np.zeros(shape, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0

    # Precompute error terms
    # E = sum (T - T_t)^2
    # We maintain T_curr
    
    # Search for a decomposition with rank < 24
    values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    start_time = time.time()
    max_time = 110.0
    
    for rank in range(14, 24):
        if time.time() - start_time > max_time:
            break
            
        for restart in range(10):
            if time.time() - start_time > max_time:
                break
            
            U = np.random.choice(values, size=(n * m, rank))
            V = np.random.choice(values, size=(m * p, rank))
            W = np.random.choice(values, size=(p * n, rank))
            
            # Compute initial T and Error
            T_curr = np.zeros(shape, dtype=np.float64)
            for r in range(rank):
                T_curr += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
            diff = T_curr - T_target
            current_err = np.sum(diff ** 2)
            
            if current_err < 1e-9:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
            
            max_steps = 2000
            for step in range(max_steps):
                if time.time() - start_time > max_time:
                    break
                    
                mat_idx = np.random.randint(0, 3)
                if mat_idx == 0:
                    mat = U
                    rows, cols = n * m, rank
                    other1 = V
                    other2 = W
                elif mat_idx == 1:
                    mat = V
                    rows, cols = m * p, rank
                    other1 = U
                    other2 = W
                else:
                    mat = W
                    rows, cols = p * n, rank
                    other1 = U
                    other2 = V
                
                i = np.random.randint(0, rows)
                j = np.random.randint(0, cols)
                
                old_val = mat[i, j]
                new_val = np.random.choice(values)
                
                if new_val == old_val:
                    continue
                
                delta_val = new_val - old_val
                
                # Update T_curr
                # T_curr += delta_val * (e_i \otimes other1[:, j] \otimes other2[:, j])
                # Note: The position of i depends on which matrix is changed.
                # If mat is U, i is the first index.
                # If mat is V, i is the second index.
                # If mat is W, i is the third index.
                
                vec1 = other1[:, j]
                vec2 = other2[:, j]
                
                # Create the slice to add
                # We need to add delta_val * vec1[:, None, None] * vec2[None, :, None] etc depending on position
                # But we must place it at index i.
                
                # Efficient update:
                # The change is delta_val * (unit_vector_i \otimes vec1 \otimes vec2)
                # We can just update the specific slice of T_curr
                
                # Let's extract the current slice of T_curr for this rank component j?
                # No, T_curr is the sum.
                # We just add the delta to T_curr.
                
                # Construct the delta tensor
                delta_T = np.zeros(shape, dtype=np.float64)
                
                if mat_idx == 0:
                    # U changed at i
                    # delta_T[i, :, :] += delta_val * vec1[:, None] * vec2[None, :]
                    # vec1 is V[:, j], vec2 is W[:, j]
                    delta_T[i, :, :] += delta_val * (vec1[:, None] * vec2[None, :])
                elif mat_idx == 1:
                    # V changed at i
                    # delta_T[:, i, :] += delta_val * vec1[None, :] * vec2[:, None]
                    # vec1 is U[:, j], vec2 is W[:, j]
                    delta_T[:, i, :] += delta_val * (vec1[None, :] * vec2[:, None])
                else:
                    # W changed at i
                    # delta_T[:, :, i] += delta_val * vec1[:, None] * vec2[None, :]
                    # vec1 is U[:, j], vec2 is V[:, j]
                    delta_T[:, :, i] += delta_val * (vec1[:, None] * vec2[None, :])
                
                # Apply change
                T_curr += delta_T
                mat[i, j] = new_val
                
                # Update error
                # New diff = (T_curr + delta_T) - T_target = diff + delta_T
                # New err = sum (diff + delta_T)^2 = sum(diff^2) + 2 sum(diff*delta_T) + sum(delta_T^2)
                # We have diff (old), delta_T, and current_err (old).
                
                # Optimization: We can compute the change in error
                # But summing over the whole tensor is expensive.
                # However, delta_T is sparse!
                # It only affects a slice.
                
                # Let's just recompute err on the affected slice?
                # Or just recompute full err if it's fast enough?
                # shape is 6x12x8 = 576.
                # np.sum(diff**2) is very fast.
                
                diff += delta_T
                new_err = np.sum(diff ** 2)
                
                if new_err < current_err:
                    current_err = new_err
                    if current_err < 1e-9:
                        return {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                else:
                    # Revert
                    mat[i, j] = old_val
                    T_curr -= delta_T
                    diff -= delta_T
                    
    # Fallback to seed (rank 24)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
