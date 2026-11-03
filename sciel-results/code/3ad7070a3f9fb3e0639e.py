"""Rank-minimal exact tensor decomposition for 4x4 x 4x5 matrix multiplication.
Uses structured initialization based on the tensor's inherent block structure,
combined with Adam optimizer and careful convergence checking."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 130  # safety margin before 150s limit
    
    # Schoolbook fallback: rank 80
    best_rank = n * m * p  # 80
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1
    
    # Count non-zero entries in target
    nnz = np.count_nonzero(target)
    
    random.seed(42)
    np.random.seed(42)
    
    # Try ranks from lower values upward
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(16, 20, 20) = 20
    # But we need more due to structure. Let's try a range.
    ranks_to_try = [40, 44, 48, 52, 56, 60, 64, 68, 72, 76]
    
    for r in ranks_to_try:
        if time.time() > deadline - 15:
            break
        if r >= best_rank:
            continue
        
        # Number of trials based on available time
        max_trials = 6 if r <= 56 else 4
        
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                break
            
            # Structure-aware initialization
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            # Initialize based on the structure of the tensor
            # Each rank component should correspond to a (i,j,k) triple
            # but we can combine multiple triples per component
            
            # Strategy: assign each rank component to handle a subset of the
            # non-zero entries, with some overlap for flexibility
            triples = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        triples.append((i, j, k))
            
            # Shuffle and assign
            np.random.shuffle(triples)
            
            for rr in range(r):
                # Assign 1-3 triples to this rank component
                num_triples = min(3, len(triples) - rr * 3 + 1) if rr * 3 < len(triples) else 1
                num_triples = max(1, num_triples)
                
                for t in range(num_triples):
                    if rr * 3 + t >= len(triples):
                        break
                    i, j, k = triples[rr * 3 + t]
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    
                    # Add with random signs to allow cancellation
                    sign = np.random.choice([-1, 1])
                    U[a, rr] += sign
                    V[b, rr] += sign
                    W[c, rr] += sign
                
                # Add small random perturbations to break symmetry
                if np.random.random() < 0.3:
                    idx_u = np.random.randint(0, n*m)
                    idx_v = np.random.randint(0, m*p)
                    idx_w = np.random.randint(0, p*n)
                    U[idx_u, rr] += 0.5 * np.random.choice([-1, 1])
                    V[idx_v, rr] += 0.5 * np.random.choice([-1, 1])
                    W[idx_w, rr] += 0.5 * np.random.choice([-1, 1])
            
            # Adam optimizer
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            lr = 0.1
            
            mU = np.zeros_like(U)
            vU = np.zeros_like(U)
            mV = np.zeros_like(V)
            vV = np.zeros_like(V)
            mW = np.zeros_like(W)
            vW = np.zeros_like(W)
            
            t = 0
            max_iter = 500
            min_res = np.inf
            stagnation_count = 0
            converged = False
            
            for step in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                t += 1
                
                # Compute current tensor estimate
                est = np.einsum('ar,br,cr->abc', U, V, W)
                residual = est - target
                max_res = np.max(np.abs(residual))
                
                if max_res < 1e-12:
                    converged = True
                    break
                
                if max_res < min_res:
                    min_res = max_res
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                
                if stagnation_count > 50:
                    # Perturb to escape local minimum
                    U += np.random.randn(n*m, r) * 0.1
                    V += np.random.randn(m*p, r) * 0.1
                    W += np.random.randn(p*n, r) * 0.1
                    mU = np.zeros_like(U)
                    vU = np.zeros_like(U)
                    mV = np.zeros_like(V)
                    vV = np.zeros_like(V)
                    mW = np.zeros_like(W)
                    vW = np.zeros_like(W)
                    t = 0
                    stagnation_count = 0
                    continue
                
                # Compute gradients
                dU = np.einsum('abc,br,cr->ar', residual, V, W)
                dV = np.einsum('abc,ar,cr->br', residual, U, W)
                dW = np.einsum('abc,ar,br->cr', residual, U, V)
                
                # Adam update
                mU = beta1 * mU + (1 - beta1) * dU
                vU = beta2 * vU + (1 - beta2) * dU**2
                mU_hat = mU / (1 - beta1**t)
                vU_hat = vU / (1 - beta2**t)
                
                mV = beta1 * mV + (1 - beta1) * dV
                vV = beta2 * vV + (1 - beta2) * dV**2
                mV_hat = mV / (1 - beta1**t)
                vV_hat = vV / (1 - beta2**t)
                
                mW = beta1 * mW + (1 - beta1) * dW
                vW = beta2 * vW + (1 - beta2) * dW**2
                mW_hat = mW / (1 - beta1**t)
                vW_hat = vW / (1 - beta2**t)
                
                U = U - lr * mU_hat / (np.sqrt(vU_hat) + eps)
                V = V - lr * mV_hat / (np.sqrt(vV_hat) + eps)
                W = W - lr * mW_hat / (np.sqrt(vW_hat) + eps)
                
                # Decay learning rate
                lr = 0.1 / (1.0 + 0.005 * step)
            
            # Check final result
            est_final = np.einsum('ar,br,cr->abc', U, V, W)
            final_res = np.max(np.abs(est_final - target))
            
            if final_res < 1e-8:
                # Try to clean up to integers/half-integers
                best_clean = None
                best_clean_err = np.inf
                
                for factor in [1, 2, 4, 8, 16, 32]:
                    U_clean = np.round(U * factor) / factor
                    V_clean = np.round(V * factor) / factor
                    W_clean = np.round(W * factor) / factor
                    
                    est_clean = np.einsum('ar,br,cr->abc', U_clean, V_clean, W_clean)
                    err = np.max(np.abs(est_clean - target))
                    
                    if err < 1e-10 and err < best_clean_err:
                        best_clean = (U_clean, V_clean, W_clean)
                        best_clean_err = err
                
                if best_clean is not None:
                    best_rank = r
                    U_c, V_c, W_c = best_clean
                    best_result = {"U": U_c.tolist(), "V": V_c.tolist(), "W": W_c.tolist()}
                    break
                elif final_res < 1e-6:
                    # Use the float version if close enough
                    best_rank = r
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
        
        if best_rank < n * m * p:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
