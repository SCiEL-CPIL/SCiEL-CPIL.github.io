import numpy as np
import time

N = 100

def construct():
    # Start with constant
    h = np.full(N, 0.5)
    
    # Try to improve using gradient descent on max overlap
    # Time limit check
    start_time = time.time()
    time_limit = 10.0 # 10 seconds for optimization
    
    best_h = h.copy()
    best_score = np.inf
    
    def get_score(h):
        # Calculate S(k) for all k
        # S(k) = sum h_i (1 - h_{i+k})
        # Using FFT for speed?
        # h is length N.
        # Convolution of h and (1-h) reversed?
        # Simpler: loop over k if N is small, but N=100 is small.
        max_s = 0
        for k in range(N):
            # Circular shift
            shifted = np.roll(h, -k)
            s = np.sum(h * (1 - shifted))
            if s > max_s:
                max_s = s
        return max_s

    current_score = get_score(h)
    best_score = current_score
    
    # Simple random perturbation and accept if better
    # Or gradient step
    lr = 0.01
    for step in range(500):
        if time.time() - start_time > time_limit:
            break
            
        # Find k with max overlap
        max_s = -1
        k_star = 0
        for k in range(N):
            shifted = np.roll(h, -k)
            s = np.sum(h * (1 - shifted))
            if s > max_s:
                max_s = s
                k_star = k
        
        # Gradient of S(k_star) wrt h
        # dS/dh_j = 1 - h_{j+k} - h_{j-k}
        grad = np.zeros(N)
        for j in range(N):
            h_next = h[(j + k_star) % N]
            h_prev = h[(j - k_star) % N]
            grad[j] = 1 - h_next - h_prev
        
        # Update h
        h_new = h - lr * grad
        # Project to [0, 1] and sum constraint
        h_new = np.clip(h_new, 0, 1)
        # Adjust sum
        sum_h = np.sum(h_new)
        if sum_h > 0:
            h_new = (h_new - np.mean(h_new)) + 0.5
            h_new = np.clip(h_new, 0, 1)
            # Re-adjust sum exactly
            diff = 50 - np.sum(h_new)
            h_new += diff / N
            h_new = np.clip(h_new, 0, 1)
            
        new_score = get_score(h_new)
        if new_score < best_score:
            best_score = new_score
            best_h = h_new.copy()
            h = h_new
        else:
            # Decay learning rate
            lr *= 0.95
            if lr < 1e-6:
                break
            
    return best_h
