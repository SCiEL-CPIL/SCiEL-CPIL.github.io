import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # True tensor for matrix multiplication
    # T[i, j, k] is 1 if (i, k) is the output index for input (i, j) and (j, k)
    # Actually, the tensor is defined such that C_{ik} = sum_j A_{ij} B_{jk}
    # The decomposition is T_{ikj} = sum_r U_{ir} V_{jr} W_{kr} ?
    # The contract says: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # Output W[k*n+i, r]
    # So the tensor element T[i, j, k] (where i is row of A, j is col of A/row of B, k is col of B)
    # corresponds to output index (i, k).
    # The factorization is: T[i, j, k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    
    # Let's construct the target tensor T of shape (n*m, m*p, p*n) ?
    # No, the standard tensor for matmul has indices (i, j, k) -> (i, k).
    # The factors are:
    # U has index (i, j)
    # V has index (j, k)
    # W has index (k, i)  <-- Note: W is indexed by output (k, i) in the contract?
    # Contract: W[k*n+i, r]. So W row index is k*n + i.
    # The term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # Summing over r gives the coefficient for the operation A_ij * B_jk contributing to C_ik.
    # So the target tensor T is 1 at (i, j, k) for all i, j, k.
    # Wait, the tensor T_{i,j,k} is 1 for all i,j,k in the standard matmul tensor?
    # Yes, because every A_ij and B_jk pair contributes to C_ik.
    # So the target tensor is all ones of shape (n*m, m*p, p*n) ?
    # Let's check dimensions.
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R)
    # Product: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This sum must equal 1 for all i, j, k.
    # So the target tensor T has shape (n*m, m*p, p*n) and T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i.
    # Note that the indices are coupled.
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # For a given (a, b, c), is there a unique (i, j, k)?
    # j = a % m, i = a // m
    # k = b % p, j' = b // p. We must have j == j'.
    # i' = c % n, k' = c // n. We must have k == k' and i == i'.
    # If these consistency checks pass, T[a,b,c] = 1, else 0?
    # Actually, for matrix multiplication, the tensor is sparse in the sense that not all (a,b,c) triples are valid.
    # But the factorization must reproduce the *non-zero* entries of the matmul operation.
    # The matmul operation is defined for all i, j, k.
    # So for every valid i, j, k, the sum must be 1.
    # For invalid (a,b,c) triples (where indices don't align), the value doesn't matter?
    # No, the tensor is defined on the full space.
    # However, the "Matrix Multiplication Tensor" usually refers to the tensor $T$ where $T_{i,j,k} = 1$ if the indices are consistent, 0 otherwise?
    # Actually, the standard tensor is just the set of basis vectors.
    # Let's just define the target values for all valid (i, j, k) tuples.
    # There are n*m*p = 27 such tuples.
    # We need to find U, V, W such that for all i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Target constraints
    targets = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                targets.append((ui, vj, wk))
                
    # Try to find a rank 23 solution
    best_solution = None
    
    # Fallback: Rank 27
    def get_fallback():
        rank = 27
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for idx, (ui, vj, wk) in enumerate(targets):
            U[ui, idx] = 1.0
            V[vj, idx] = 1.0
            W[wk, idx] = 1.0
        return U, V, W

    def snap_and_verify(U, V, W, rank):
        # Snap to nearest 0.5
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Check exactness
        for ui, vj, wk in targets:
            val = np.dot(U_snap[ui], V_snap[vj]) # This is not right, it's sum over r of U*V*W
            # val = sum_r U[ui, r] * V[vj, r] * W[wk, r]
            val = np.sum(U_snap[ui] * V_snap[vj] * W_snap[wk])
            if abs(val - 1.0) > 1e-9:
                return False
        
        # Prune zero columns
        mask = np.any(U_snap != 0, axis=0) | np.any(V_snap != 0, axis=0) | np.any(W_snap != 0, axis=0)
        U_final = U_snap[:, mask]
        V_final = V_snap[:, mask]
        W_final = W_snap[:, mask]
        
        # Convert to lists
        return {
            "U": U_final.tolist(),
            "V": V_final.tolist(),
            "W": W_final.tolist()
        }

    def optimize(U, V, W, steps, lr):
        # Simple gradient descent with L1 sparsity
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Precompute gradients? No, just loop.
        for step in range(steps):
            if time.time() > deadline:
                break
                
            # Calculate current values
            # Error = sum over targets (sum_r U V W - 1)^2
            # Grad U[ui, r] += 2 * (val - 1) * V[vj, r] * W[wk, r]
            # Plus L1 penalty: sign(U)
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for ui, vj, wk in targets:
                u_row = U[ui]
                v_row = V[vj]
                w_row = W[wk]
                
                val = np.sum(u_row * v_row * w_row)
                err = val - 1.0
                
                # Gradient of (val-1)^2 w.r.t U[ui, r] is 2 * err * V[vj, r] * W[wk, r]
                g_u = 2 * err * v_row * w_row
                g_v = 2 * err * u_row * w_row
                g_w = 2 * err * u_row * v_row
                
                grad_U[ui] += g_u
                grad_V[vj] += g_v
                grad_W[wk] += g_w
            
            # Add L1 penalty for sparsity
            lambda_l1 = 0.01
            grad_U += lambda_l1 * np.sign(U)
            grad_V += lambda_l1 * np.sign(V)
            grad_W += lambda_l1 * np.sign(W)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
        return U, V, W

    # Try random restarts with rank 23
    for restart in range(10):
        if time.time() > deadline:
            break
            
        rank = 23
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Optimize
        U_opt, V_opt, W_opt = optimize(U, V, W, steps=500, lr=0.01)
        
        # Check error
        max_err = 0
        for ui, vj, wk in targets:
            val = np.sum(U_opt[ui] * V_opt[vj] * W_opt[wk])
            max_err = max(max_err, abs(val - 1.0))
            
        if max_err < 1e-5:
            result = snap_and_verify(U_opt, V_opt, W_opt, rank)
            if result:
                return result
                
    # If rank 23 failed, try rank 24, 25, 26? Or just return fallback.
    # Given time constraints, fallback is safe.
    U_f, V_f, W_f = get_fallback()
    return {
        "U": U_f.tolist(),
        "V": V_f.tolist(),
        "W": W_f.tolist()
    }
