import numpy as np
import time

N = 100
DEADLINE = 145.0  # 5s margin

def construct():
    start_time = time.time()
    
    # Helper: compute overlap O(k) = 1 - C(k) where C(k) is autocorrelation
    def compute_O(h):
        """Compute overlap values O(k) = 1 - (2/N) * sum(h[i]*h[(i-k)%N])"""
        H = np.fft.fft(h, 2*N)
        C = np.fft.ifft(H * np.conj(H)).real[:N]
        C *= (2.0 / N)
        return 1.0 - C
    
    def softmax_max(O, tau=0.01):
        """Softmax smoothing of max(O) for differentiability"""
        O_max = np.max(O)
        return O_max + tau * np.log(np.sum(np.exp((O - O_max) / tau)))
    
    def subgrad_softmax(O, tau=0.01):
        """Subgradient of softmax max"""
        O_max = np.max(O)
        weights = np.exp((O - O_max) / tau)
        weights /= np.sum(weights)
        return weights
    
    def project(h):
        """Project h onto {x in [0,1], sum(x)=N/2} using water-filling"""
        target_sum = N / 2.0
        for _ in range(50):
            # Water-filling: find lambda such that sum(clip(h - lambda, 0, 1)) = target_sum
            # Binary search on lambda
            lo, hi = -1.0, 1.0
            for _ in range(30):
                mid = (lo + hi) / 2
                projected = np.clip(h - mid, 0, 1)
                s = np.sum(projected)
                if s > target_sum:
                    lo = mid
                else:
                    hi = mid
            h = np.clip(h - (lo + hi) / 2, 0, 1)
            # Fine-tune to ensure sum is exactly target_sum
            diff = target_sum - np.sum(h)
            if abs(diff) < 1e-10:
                break
            # Distribute difference to elements not at bounds
            mask = (h > 1e-8) & (h < 1 - 1e-8)
            if np.sum(mask) > 0:
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0, 1)
            else:
                # All at bounds, adjust slightly
                h += diff / N
                h = np.clip(h, 0, 1)
        return h
    
    def optimize(h_init, max_iters=2000):
        """Subgradient descent with softmax smoothing and projection"""
        h = h_init.copy()
        best_h = h.copy()
        best_val = np.inf
        
        # Adaptive learning rate
        lr = 0.01
        tau = 0.05  # Start with larger smoothing
        
        for it in range(max_iters):
            if time.time() > DEADLINE:
                break
            
            O = compute_O(h)
            current_max = np.max(O)
            
            if current_max < best_val:
                best_val = current_max
                best_h = h.copy()
            
            # Get subgradient
            w = subgrad_softmax(O, tau)
            
            # Subgradient of C(k) w.r.t. h[i]
            # C(k) = (2/N) * sum_j h[j] * h[(j-k)%N]
            # dC(k)/dh[i] = (2/N) * sum_k w[k] * (h[(i-k)%N] + h[(i+k)%N])
            # So dO(k)/dh[i] = -dC(k)/dh[i]
            # grad_O[i] = -(2/N) * sum_k w[k] * (h[(i-k)%N] + h[(i+k)%N])
            
            # Compute gradient efficiently
            grad_O = np.zeros(N)
            for k in range(N):
                if w[k] > 1e-6:
                    grad_O -= (2.0 / N) * w[k] * (np.roll(h, -k) + np.roll(h, k))
            
            # Update
            h_new = h - lr * grad_O
            h_new = project(h_new)
            
            # Check if improvement
            O_new = compute_O(h_new)
            if np.max(O_new) < current_max + 1e-4:
                h = h_new
            else:
                # Reduce learning rate
                lr *= 0.95
            
            # Decrease smoothing over time
            tau = max(0.005, tau * 0.995)
            
            # Occasionally restart from best with fresh noise
            if it % 200 == 199 and it > 100:
                if np.random.random() < 0.3:
                    h = best_h + np.random.normal(0, 0.01, N)
                    h = project(h)
                    lr = 0.01
                    tau = 0.03
        
        return best_h, best_val
    
    # Multiple random restarts
    best_h = np.full(N, 0.5)
    best_val = np.max(compute_O(best_h))
    
    num_restarts = 15
    for restart in range(num_restarts):
        if time.time() > DEADLINE - 10:
            break
        
        # Random initialization
        if restart == 0:
            h_init = np.full(N, 0.5)
        else:
            h_init = np.random.uniform(0.3, 0.7, N)
            h_init = project(h_init)
        
        h_opt, val_opt = optimize(h_init, max_iters=1500)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    return best_h
