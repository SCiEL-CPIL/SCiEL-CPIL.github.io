import numpy as np
import time
import random

N = 256

def get_aps():
    ap_indices = []
    for d in range(1, N + 1):
        for a in range(1, N + 1):
            max_l = (N - a) // d + 1
            for l in range(1, max_l + 1):
                indices = [a + i * d for i in range(l)]
                ap_indices.append(indices)
    return ap_indices

ap_indices = get_aps()
num_aps = len(ap_indices)

# Build sparse representation of AP constraints
# For each position, list of (ap_idx, coefficient) where coefficient is always 1
pos_to_ap = [[] for _ in range(N)]
for ap_idx, indices in enumerate(ap_indices):
    for idx in indices:
        pos_to_ap[idx - 1].append(ap_idx)

def construct():
    deadline = time.time() + 55.0
    rng = random.Random(12345)
    
    best_seq = None
    best_score = float('inf')
    
    # Multiple restarts with gradient-based optimization
    max_restarts = 30
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Initialize with random continuous values
        x = np.random.uniform(-1.0, 1.0, size=N)
        
        # Compute initial AP sums
        S = np.zeros(num_aps, dtype=np.float64)
        for ap_idx, indices in enumerate(ap_indices):
            s_indices = np.array(indices) - 1
            S[ap_idx] = np.sum(x[s_indices])
        
        # Gradient descent with soft-max surrogate
        # Loss = sum(exp(2*|S_i|) - 1) approximates max|S|
        # Gradient w.r.t. x_j = sum_{i: j in AP_i} 2*exp(2*|S_i|)*sign(S_i)
        
        lr = 0.05
        momentum = np.zeros(N, dtype=np.float64)
        beta1 = 0.9
        beta2 = 0.999
        v = np.zeros(N, dtype=np.float64)
        m = np.zeros(N, dtype=np.float64)
        t = 0
        
        max_iters = 5000
        no_improve = 0
        max_no_improve = 500
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Compute gradient of soft-max surrogate
            # d/dx_j [sum_i exp(2|S_i|)] = sum_{i: j in AP_i} 2*exp(2|S_i|)*sign(S_i)
            
            # Use a stable computation: for large |S|, exp(2|S|) overflows
            # Clip |S| to avoid overflow
            abs_S = np.abs(S)
            # Use a scaled version to avoid overflow
            scale = 0.5  # exp(2*0.5*|S|) = exp(|S|), still can overflow for large S
            # Better: use tanh-based approximation or clip
            # Let's use: f(S) = sum (1 - exp(-|S|^2)) which is bounded and differentiable
            # Or simply use sum of squares as a simpler surrogate
            
            # Let's use sum of squares: L = sum(S_i^2)
            # Gradient: dL/dx_j = 2 * sum_{i: j in AP_i} S_i
            
            grad = np.zeros(N, dtype=np.float64)
            for j in range(N):
                s_val = 0.0
                for ap_idx in pos_to_ap[j]:
                    s_val += S[ap_idx]
                grad[j] = 2.0 * s_val
            
            # Adam optimizer
            t += 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad * grad)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            x -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            
            # Update AP sums incrementally
            # For each position, compute delta
            delta = np.zeros(N, dtype=np.float64)
            for j in range(N):
                delta[j] = grad[j] * (-lr * m_hat[j] / (np.sqrt(v_hat[j]) + 1e-8))
            
            # Actually, let's just recompute S from x for correctness
            # This is O(num_aps * avg_length) which might be slow
            # Better: incremental update
            
            # Recompute S incrementally
            for j in range(N):
                if abs(delta[j]) > 1e-10:
                    for ap_idx in pos_to_ap[j]:
                        S[ap_idx] += delta[j]
            
            # Discretize and evaluate
            if it % 50 == 0:
                s = np.where(x >= 0, 1, -1).astype(np.int8)
                
                # Compute max abs AP sum for this discrete assignment
                S_disc = np.zeros(num_aps, dtype=np.int16)
                for ap_idx, indices in enumerate(ap_indices):
                    s_indices = np.array(indices) - 1
                    S_disc[ap_idx] = np.sum(s[s_indices])
                
                current_max = int(np.max(np.abs(S_disc)))
                
                if current_max < best_score:
                    best_score = current_max
                    best_seq = s.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                
                if no_improve > max_no_improve:
                    break
            
            # Project x to [-1, 1] occasionally to keep it bounded
            if it % 100 == 0:
                x = np.clip(x, -1.0, 1.0)
        
        # Final discretization
        s = np.where(x >= 0, 1, -1).astype(np.int8)
        S_disc = np.zeros(num_aps, dtype=np.int16)
        for ap_idx, indices in enumerate(ap_indices):
            s_indices = np.array(indices) - 1
            S_disc[ap_idx] = np.sum(s[s_indices])
        current_max = int(np.max(np.abs(S_disc)))
        
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
    
    # Local search refinement on the best solution found
    if best_seq is not None and time.time() < deadline - 5.0:
        s = best_seq.copy()
        S = np.zeros(num_aps, dtype=np.int16)
        for ap_idx, indices in enumerate(ap_indices):
            s_indices = np.array(indices) - 1
            S[ap_idx] = np.sum(s[s_indices])
        
        hist = np.zeros(N + 1, dtype=np.int32)
        abs_S = np.abs(S)
        np.add.at(hist, abs_S, 1)
        current_max = int(np.max(abs_S))
        
        local_best = current_max
        local_best_seq = s.copy()
        
        while time.time() < deadline - 2.0:
            j = rng.randint(0, N - 1)
            old_val = s[j]
            delta = -2 * old_val
            
            max_new_abs = 0
            for ap_idx in pos_to_ap[j]:
                old_sum = S[ap_idx]
                new_sum = old_sum + delta
                old_abs = abs(old_sum)
                new_abs = abs(new_sum)
                hist[old_abs] -= 1
                hist[new_abs] += 1
                S[ap_idx] = new_sum
                if new_abs > max_new_abs:
                    max_new_abs = new_abs
            
            if hist[current_max] == 0:
                while current_max > 0 and hist[current_max] == 0:
                    current_max -= 1
                if max_new_abs > current_max:
                    current_max = max_new_abs
            else:
                if max_new_abs > current_max:
                    current_max = max_new_abs
            
            if current_max < local_best:
                s[j] = -old_val
                local_best = current_max
                local_best_seq = s.copy()
            else:
                for ap_idx in pos_to_ap[j]:
                    new_sum = S[ap_idx]
                    old_sum = new_sum - delta
                    new_abs = abs(new_sum)
                    old_abs = abs(old_sum)
                    hist[new_abs] -= 1
                    hist[old_abs] += 1
                    S[ap_idx] = old_sum
        
        if local_best < best_score:
            best_score = local_best
            best_seq = local_best_seq
    
    return [int(x) for x in best_seq]
