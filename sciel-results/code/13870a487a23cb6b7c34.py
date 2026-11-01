import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Target tensor: T[i*m+j, j*p+k, k*n+i] = 1
    # U shape: (n*m, r) = (6, r)
    # V shape: (m*p, r) = (12, r)  
    # W shape: (p*n, r) = (8, r)
    
    # Precompute constraint indices
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                constraints.append((u_idx, v_idx, w_idx))
    
    def compute_error(U, V, W):
        """Compute squared error of decomposition."""
        err = 0.0
        for u_idx, v_idx, w_idx in constraints:
            val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
            err += (val - 1.0) ** 2
        return err
    
    def solve_W_given_UV(U, V, r):
        """Given U and V, solve for W using least squares."""
        # For each (a,b,c) constraint: sum_k U[a,k]*V[b,k]*W[c,k] = T[a,b,c]
        # This is linear in W. Build system: A @ W_flat = b
        # W_flat has shape (p*n*r,)
        
        num_eqs = len(constraints)
        num_vars = p * n * r
        
        # Build A and b
        A = np.zeros((num_eqs, num_vars))
        b = np.zeros(num_eqs)
        
        for eq_i, (u_idx, v_idx, w_idx) in enumerate(constraints):
            b[eq_i] = 1.0
            for k in range(r):
                col_idx = w_idx * r + k
                A[eq_i, col_idx] = U[u_idx, k] * V[v_idx, k]
        
        # Solve least squares
        W_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        W = W_flat.reshape(p * n, r)
        return W
    
    def try_rank(r, num_restarts=200, max_steps=300):
        """Try to find rank-r decomposition using fix-U-V-solve-W strategy."""
        best_result = None
        best_err = float('inf')
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization for U and V
            # Use small values to avoid overflow
            U = np.random.randn(n * m, r) * 0.3
            V = np.random.randn(m * p, r) * 0.3
            
            # Solve for W
            W = solve_W_given_UV(U, V, r)
            
            # Compute initial error
            err = compute_error(U, V, W)
            
            if err < 1e-10:
                return U, V, W
            
            # Gradient descent on U and V only
            lr = 0.05
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                
                # Compute gradients for U and V
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                
                for u_idx, v_idx, w_idx in constraints:
                    # val = sum_k U[u,k] * V[v,k] * W[w,k]
                    val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                    diff = val - 1.0
                    
                    # dE/dU[u,k] = 2 * diff * V[v,k] * W[w,k]
                    dU[u_idx] += 2 * diff * V[v_idx] * W[w_idx]
                    # dE/dV[v,k] = 2 * diff * U[u,k] * W[w,k]
                    dV[v_idx] += 2 * diff * U[u_idx] * W[w_idx]
                
                # Update U and V
                U -= lr * dU
                V -= lr * dV
                
                # Re-solve for W
                W = solve_W_given_UV(U, V, r)
                
                # Check error
                err = compute_error(U, V, W)
                
                if err < best_err:
                    best_err = err
                    best_result = (U.copy(), V.copy(), W.copy())
                
                if err < 1e-10:
                    # Found exact solution!
                    return U, V, W
                
                # Adaptive learning rate
                if step % 50 == 0 and step > 0:
                    lr *= 0.8
        
        # Return best found
        if best_result is not None:
            return best_result
        return None
    
    # Try to find lower rank solutions
    # Theoretical lower bound for 2x3 x 3x4 is around 20-22
    for target_rank in [20, 21, 22, 23]:
        if time.time() > deadline - 10:
            break
        
        result = try_rank(target_rank)
        if result is not None:
            U, V, W = result
            # Round to half-integers if possible
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Check if rounded version is still exact
            if compute_error(U_rounded, V_rounded, W_rounded) < 1e-9:
                return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
            else:
                # Return unrounded if exact
                if compute_error(U, V, W) < 1e-9:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook rank 24
    rank_seed = n * m * p
    U_seed = np.zeros((n * m, rank_seed), dtype=np.float64)
    V_seed = np.zeros((m * p, rank_seed), dtype=np.float64)
    W_seed = np.zeros((p * n, rank_seed), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j, col] = 1
                V_seed[j * p + k, col] = 1
                W_seed[k * n + i, col] = 1
                col += 1
    
    return {"U": U_seed.tolist(), "V": V_seed.tolist(), "W": W_seed.tolist()}
