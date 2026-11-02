import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    deadline = time.time() + 140  # safety margin
    
    # Define indices
    # U shape: (n*m, rank) -> (9, R)
    # V shape: (m*p, rank) -> (18, R)
    # W shape: (p*n, rank) -> (18, R)
    
    # Pre-compute the list of (i, j, k) tuples
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # Schoolbook rank is n*m*p = 54
    schoolbook_rank = n * m * p
    
    def schoolbook_decomposition():
        R = schoolbook_rank
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find a lower rank decomposition
    # Theoretical lower bound for 3x3x3 is ~19-23. 
    # If 3x3x6 is two 3x3x3, rank ~ 2*23 = 46.
    # Let's try ranks 42 to 53.
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Use Adam optimizer
    def adam_step(params, grads, state, beta1=0.9, beta2=0.999, eps=1e-8):
        m = {}
        v = {}
        t = state['t'] + 1
        for p in params:
            if p not in m:
                m[p] = np.zeros_like(params[p])
                v[p] = np.zeros_like(params[p])
            m[p] = beta1 * m[p] + (1 - beta1) * grads[p]
            v[p] = beta2 * v[p] + (1 - beta2) * grads[p] ** 2
            m_hat = m[p] / (1 - beta1 ** t)
            v_hat = v[p] / (1 - beta2 ** t)
            params[p] -= state['lr'] * m_hat / (np.sqrt(v_hat) + eps)
        state['m'] = m
        state['v'] = v
        state['t'] = t
        return params

    def try_rank(R, max_iters=2000, lr=0.01):
        if time.time() > deadline:
            return None
            
        # Initialize
        # Small random initialization
        np.random.seed(0)
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        params = {'U': U, 'V': V, 'W': W}
        grads = {'U': np.zeros_like(U), 'V': np.zeros_like(V), 'W': np.zeros_like(W)}
        state = {'t': 0, 'm': {}, 'v': {}, 'lr': lr}
        
        # Pre-compute index arrays for vectorization
        idx_i = np.array([i for i, j, k in indices])
        idx_j = np.array([j for i, j, k in indices])
        idx_k = np.array([k for i, j, k in indices])
        
        u_idx = idx_i * m + idx_j
        v_idx = idx_j * p + idx_k
        w_idx = idx_k * n + idx_i
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Compute tensor values T[i,j,k]
            # T_val[r] for each rank r is U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
            # We want sum_r T_val[r] = 1 for all 54 entries
            
            # Efficient calculation
            # Shape: (54, R)
            # U_vals: (54, R) -> U[u_idx, :]
            # V_vals: (54, R) -> V[v_idx, :]
            # W_vals: (54, R) -> W[w_idx, :]
            
            U_vals = U[u_idx, :]
            V_vals = V[v_idx, :]
            W_vals = W[w_idx, :]
            
            T_vals = U_vals * V_vals * W_vals
            Sum_T = np.sum(T_vals, axis=1)  # Shape (54,)
            
            Error = Sum_T - 1.0
            Loss = np.sum(Error ** 2)
            
            if Loss < 1e-10:
                return U, V, W
                
            # Compute Gradients
            # dL/dU[u_idx, r] = sum_{i,j,k: u_idx(i,j)=idx} V[v_idx, r] W[w_idx, r] Error[i,j,k]
            # Since each (i,j) appears p=6 times in indices, we can aggregate.
            
            # Initialize grads to 0
            grads['U'][:] = 0.0
            grads['V'][:] = 0.0
            grads['W'][:] = 0.0
            
            # We need to scatter the gradients back
            # grad_U[u, r] += V[v, r] * W[w, r] * Error[entry]
            
            # Vectorized update:
            # For each entry e in 0..53:
            # u_e, v_e, w_e are the indices.
            # We want to add V[v_e, :] * W[w_e, :] * Error[e] to grads['U'][u_e, :]
            
            # Using np.add.at
            np.add.at(grads['U'], u_idx, V_vals * W_vals * Error[:, np.newaxis])
            np.add.at(grads['V'], v_idx, U_vals * W_vals * Error[:, np.newaxis])
            np.add.at(grads['W'], w_idx, U_vals * V_vals * Error[:, np.newaxis])
            
            # Adam step
            params = adam_step(params, grads, state)
            U, V, W = params['U'], params['V'], params['W']
            
            # Decay learning rate
            if it % 500 == 0 and it > 0:
                state['lr'] *= 0.5
            
        # Check final loss
        U_vals = U[u_idx, :]
        V_vals = V[v_idx, :]
        W_vals = W[w_idx, :]
        T_vals = U_vals * V_vals * W_vals
        Sum_T = np.sum(T_vals, axis=1)
        Loss = np.sum((Sum_T - 1.0) ** 2)
        
        if Loss < 1e-8:
            return U, V, W
        else:
            return None

    # Try ranks from 42 up to 53
    for R in range(42, schoolbook_rank):
        if time.time() > deadline:
            break
            
        print(f"Trying rank {R}...")
        result = try_rank(R)
        
        if result is not None:
            U, V, W = result
            # Round to half-integers? The problem asks for integer/half-integer entries.
            # Numerical solutions might not be exactly half-integers.
            # We can try to round to nearest 0.5 and check validity.
            
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Verify rounded solution
            U_vals = U_r[u_idx, :]
            V_vals = V_r[v_idx, :]
            W_vals = W_r[w_idx, :]
            T_vals = U_vals * V_vals * W_vals
            Sum_T = np.sum(T_vals, axis=1)
            Error = np.max(np.abs(Sum_T - 1.0))
            
            if Error < 1e-5:
                print(f"Found valid half-integer decomposition with rank {R}")
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            else:
                # If rounding fails, maybe the solution is not close to half-integers?
                # The problem states "integer/half-integer entries".
                # If we can't find one, we should probably return the schoolbook which has integers.
                pass
        
    # Fallback to schoolbook
    return schoolbook_decomposition()

# EVOLVE-BLOCK-END
