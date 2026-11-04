"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Start with a multi-harmonic initial guess
    # This distributes energy across frequencies to suppress individual shift peaks
    h = np.full(N, 0.5)
    
    # Add harmonic components to break symmetry
    np.random.seed(42)
    for freq in [1, 2, 3, 5, 7]:
        amp = 0.08 / np.sqrt(freq)
        phase = np.random.uniform(0, 2 * np.pi)
        h += amp * np.cos(2 * np.pi * freq * np.arange(N) / N + phase)
    
    # Project to [0, 1] with mean 0.5
    h = np.clip(h, 0.0, 1.0)
    h = h - np.mean(h) + 0.5
    h = np.clip(h, 0.0, 1.0)
    
    def compute_overlap(h):
        """Compute max circular overlap of h with shifted 1-h."""
        # Overlap at shift k: sum(h[i] * (1-h[(i+k)%N])) for all i
        # = N*mean(h) - sum(h[i]*h[(i+k)%N])
        # Since mean(h) = 0.5, N*mean(h) = N/2
        # So overlap(k) = (N/2 - autocorr(k)) / (N/2) = 1 - 2*autocorr(k)/(N/2)
        # Actually: overlap = sum(h_i * (1-h_{i+k})) / (N/2)
        # = (N/2 - sum(h_i * h_{i+k})) / (N/2)
        
        overlaps = np.zeros(N)
        for k in range(N):
            shifted = np.roll(h, -k)
            overlaps[k] = np.sum(h * (1.0 - shifted)) / (N / 2.0)
        return overlaps
    
    def objective(h):
        """Smooth surrogate for max overlap using soft-max."""
        overlaps = np.zeros(N)
        for k in range(N):
            shifted = np.roll(h, -k)
            overlaps[k] = np.sum(h * (1.0 - shifted)) / (N / 2.0)
        # Soft-max with temperature
        T = 0.05
        scaled = overlaps / T
        max_val = np.max(scaled)
        log_sum_exp = max_val + np.log(np.sum(np.exp(scaled - max_val)))
        return log_sum_exp * T
    
    def project(h):
        """Project h to [0,1] with mean 0.5."""
        h = np.clip(h, 0.0, 1.0)
        # Adjust mean to exactly 0.5
        mean_diff = np.mean(h) - 0.5
        if mean_diff > 0:
            # Need to decrease some values
            # Decrease values closest to 1.0 first
            order = np.argsort(1.0 - h)
            h = h.copy()
            excess = mean_diff * N
            for i in order:
                reduce = min(h[i] - 0.0, excess)
                h[i] -= reduce
                excess -= reduce
                if excess <= 0:
                    break
        elif mean_diff < 0:
            # Need to increase some values
            order = np.argsort(h)
            h = h.copy()
            deficit = -mean_diff * N
            for i in order:
                increase = min(1.0 - h[i], deficit)
                h[i] += increase
                deficit -= increase
                if deficit <= 0:
                    break
        return h
    
    # Gradient descent
    best_h = h.copy()
    best_score = float('inf')
    
    lr = 0.01
    for iteration in range(5000):
        if time.time() > deadline - 10.0:
            break
        
        # Compute gradient of soft-max objective
        # d/dh_j [sum_k h_i (1-h_{i+k})] = for each k: contribution to h_j
        # Overlap_k = sum_i h_i (1 - h_{i+k}) / (N/2)
        # d(overlap_k)/d(h_j) = (1/N/2) * [(1 - h_{j+k}) - sum_i h_i * d(h_{i+k})/d(h_j)]
        # d(h_{i+k})/d(h_j) = 1 if j = i+k for some i, i.e., i = j-k
        # So d(overlap_k)/d(h_j) = (2/N) * [(1 - h_{j+k}) - h_{j-k}]
        
        grad = np.zeros(N)
        for k in range(N):
            # Gradient of overlap_k w.r.t. h
            g_k = np.zeros(N)
            for j in range(N):
                g_k[j] = (2.0 / N) * ((1.0 - h[(j + k) % N]) - h[(j - k) % N])
            grad += g_k
        
        # Soft-max gradient: weight each overlap by its softmax probability
        overlaps = np.zeros(N)
        for k in range(N):
            shifted = np.roll(h, -k)
            overlaps[k] = np.sum(h * (1.0 - shifted)) / (N / 2.0)
        
        T = 0.05
        scaled = overlaps / T
        max_val = np.max(scaled)
        exp_vals = np.exp(scaled - max_val)
        probs = exp_vals / np.sum(exp_vals)
        
        # Weighted sum of gradients
        for k in range(N):
            shifted_grad = np.roll(grad, k)  # need to be careful with indexing
            # Actually, let's just compute the full gradient properly
            pass
        
        # Simpler: use finite differences for gradient (slower but correct)
        # Or just use the analytical gradient with proper weighting
        full_grad = np.zeros(N)
        for k in range(N):
            g_k = (2.0 / N) * ((1.0 - np.roll(h, -k)) - np.roll(h, k))
            full_grad += probs[k] * g_k
        
        # Take gradient step
        h = h - lr * full_grad
        h = project(h)
        
        # Check score
        score = objective(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
        
        # Adaptive learning rate
        if iteration % 500 == 0:
            lr *= 0.8
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5.0:
        h = best_h.copy()
        for iteration in range(2000):
            if time.time() > deadline - 2.0:
                break
            
            full_grad = np.zeros(N)
            overlaps = np.zeros(N)
            for k in range(N):
                shifted = np.roll(h, -k)
                overlaps[k] = np.sum(h * (1.0 - shifted)) / (N / 2.0)
                g_k = (2.0 / N) * ((1.0 - np.roll(h, -k)) - np.roll(h, k))
                # Use hard max for polish (focus on worst shift)
                full_grad += g_k / N  # average, then we'll focus on max
            
            # Focus on the worst overlap
            worst_k = np.argmax(overlaps)
            g_worst = (2.0 / N) * ((1.0 - np.roll(h, -worst_k)) - np.roll(h, worst_k))
            
            h = h - 0.001 * g_worst
            h = project(h)
    
    return best_h
# EVOLVE-BLOCK-END
