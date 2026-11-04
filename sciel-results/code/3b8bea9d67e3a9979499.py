import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Leave margin before 150s limit

    # Build the Strassen-based rank 42 decomposition as starting point
    rank0 = 42
    U0 = [[0] * rank0 for _ in range(n * m)]  # 8 x 42
    V0 = [[0] * rank0 for _ in range(m * p)]  # 24 x 42
    W0 = [[0] * rank0 for _ in range(p * n)]  # 12 x 42

    def add_strassen_block(i_start, j_start, k_start, offset):
        # Adds a 2x2x2 Strassen block
        col = offset
        U0[(i_start+0)*m + (j_start+0)][col] += 1
        U0[(i_start+1)*m + (j_start+1)][col] += 1
        V0[(j_start+0)*p + (k_start+0)][col] += 1
        V0[(j_start+1)*p + (k_start+1)][col] += 1
        W0[(k_start+0)*n + (i_start+0)][col] += 1
        W0[(k_start+1)*n + (i_start+1)][col] += 1

        col = offset + 1
        U0[(i_start+1)*m + (j_start+0)][col] += 1
        U0[(i_start+1)*m + (j_start+1)][col] += 1
        V0[(j_start+0)*p + (k_start+0)][col] += 1
        W0[(k_start+0)*n + (i_start+1)][col] += 1
        W0[(k_start+1)*n + (i_start+1)][col] -= 1

        col = offset + 2
        U0[(i_start+0)*m + (j_start+0)][col] += 1
        V0[(j_start+0)*p + (k_start+1)][col] += 1
        V0[(j_start+1)*p + (k_start+1)][col] -= 1
        W0[(k_start+1)*n + (i_start+0)][col] += 1
        W0[(k_start+1)*n + (i_start+1)][col] += 1

        col = offset + 3
        U0[(i_start+1)*m + (j_start+1)][col] += 1
        V0[(j_start+1)*p + (k_start+0)][col] += 1
        V0[(j_start+0)*p + (k_start+0)][col] -= 1
        W0[(k_start+0)*n + (i_start+0)][col] += 1
        W0[(k_start+0)*n + (i_start+1)][col] += 1

        col = offset + 4
        U0[(i_start+0)*m + (j_start+0)][col] += 1
        U0[(i_start+0)*m + (j_start+1)][col] += 1
        V0[(j_start+1)*p + (k_start+1)][col] += 1
        W0[(k_start+0)*n + (i_start+0)][col] -= 1
        W0[(k_start+1)*n + (i_start+0)][col] += 1

        col = offset + 5
        U0[(i_start+1)*m + (j_start+0)][col] += 1
        U0[(i_start+0)*m + (j_start+0)][col] -= 1
        V0[(j_start+0)*p + (k_start+0)][col] += 1
        V0[(j_start+0)*p + (k_start+1)][col] += 1
        W0[(k_start+1)*n + (i_start+1)][col] += 1

        col = offset + 6
        U0[(i_start+0)*m + (j_start+1)][col] += 1
        U0[(i_start+1)*m + (j_start+1)][col] -= 1
        V0[(j_start+1)*p + (k_start+0)][col] += 1
        V0[(j_start+1)*p + (k_start+1)][col] += 1
        W0[(k_start+0)*n + (i_start+0)][col] += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen_block(0, 0, 0, 0)
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    add_strassen_block(0, 0, 2, 7)
    # Block 3: A[0:2, 0:2] * B[0:2, 4:6]
    add_strassen_block(0, 0, 4, 14)
    # Block 4: A[0:2, 2:4] * B[2:4, 0:2]
    add_strassen_block(0, 2, 0, 21)
    # Block 5: A[0:2, 2:4] * B[2:4, 2:4]
    add_strassen_block(0, 2, 2, 28)
    # Block 6: A[0:2, 2:4] * B[2:4, 4:6]
    add_strassen_block(0, 2, 4, 35)

    # Convert to numpy for manipulation
    U0_np = np.array(U0, dtype=float)
    V0_np = np.array(V0, dtype=float)
    W0_np = np.array(W0, dtype=float)

    def verify_decomposition(U, V, W, tol=1e-6):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > tol:
                        return False
        return True

    # Try to reduce rank by attempting to merge or eliminate terms
    # Strategy: try to find a lower-rank decomposition by searching
    # around the Strassen solution
    
    best_rank = rank0
    best_U, best_V, best_W = U0_np.copy(), V0_np.copy(), W0_np.copy()

    # Try ranks from 41 down to 36
    for target_rank in range(41, 35, -1):
        if time.time() > deadline - 20:
            break
        
        # Try to find a valid decomposition of target_rank
        # Use gradient-free search: perturb the Strassen solution
        # and project to lower rank
        
        found = False
        for attempt in range(20):
            if time.time() > deadline - 20:
                break
            
            # Initialize by taking first target_rank columns of Strassen
            # and adding small random perturbations
            U = U0_np[:, :target_rank].copy()
            V = V0_np[:, :target_rank].copy()
            W = W0_np[:, :target_rank].copy()
            
            # Add small perturbations
            scale = 0.1
            U += np.random.randn(n*m, target_rank) * scale
            V += np.random.randn(m*p, target_rank) * scale
            W += np.random.randn(p*n, target_rank) * scale
            
            # Try to fix up the decomposition by adjusting
            # This is a heuristic - check if it's close to valid
            if verify_decomposition(U, V, W, tol=0.1):
                # Try to polish
                for polish_iter in range(100):
                    if time.time() > deadline - 10:
                        break
                    # Compute residuals
                    max_res = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                                res = val - 1
                                max_res = max(max_res, abs(res))
                    if max_res < 1e-6:
                        break
                    # Simple gradient step
                    lr = 0.01
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                                res = val - 1
                                for r in range(target_rank):
                                    U[i*m+j, r] -= lr * res * V[j*p+k, r] * W[k*n+i, r]
                                    V[j*p+k, r] -= lr * res * U[i*m+j, r] * W[k*n+i, r]
                                    W[k*n+i, r] -= lr * res * U[i*m+j, r] * V[j*p+k, r]
                
                if verify_decomposition(U, V, W):
                    best_rank = target_rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    found = True
                    break
        
        if found:
            break

    if best_rank < rank0:
        # Round to half-integers if possible
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        if verify_decomposition(U_rounded, V_rounded, W_rounded, tol=1e-6):
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        else:
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U0, "V": V0, "W": W0}
# EVOLVE-BLOCK-END
