import numpy as np
import time
from scipy.optimize import linprog

N = 100

def _overlap_max(h):
    """Compute max cyclic overlap of h with 1-h, scaled by step width."""
    w = 2.0 / N
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    overlaps = N / 2.0 - autocorr
    return w * np.max(overlaps)

def construct():
    deadline = time.time() + 110
    
    # Step 1: Solve LP to find optimal step heights minimizing max overlap
    # Decision variables: h_0, ..., h_{N-1}, t
    # Minimize t
    # Subject to:
    #   w * (N/2 - sum_i h_i * h_{i+k}) <= t  for all k
    #   0 <= h_i <= 1
    #   sum h_i = N/2
    
    # Since the constraint involves products h_i * h_{i+k}, this is NOT linear.
    # We need to reformulate.
    
    # Actually, overlap_k = w * (N/2 - autocorr_k) where autocorr_k = sum_i h_i * h_{i+k}
    # This is quadratic in h. So LP won't work directly.
    
    # Let me think differently. The inspiration mentioned "linear program with exact piecewise-linear overlap constraints"
    # but that's for a different formulation.
    
    # Let's try a different approach: use the fact that for binary h (0 or 1), 
    # autocorr_k = number of positions i where h_i = 1 and h_{i+k} = 1
    # For binary h with sum = N/2, we have exactly N/2 ones.
    
    # Actually, let's just use gradient-based optimization but with a better initial guess
    # and more robust projection.
    
    # Alternative: use scipy.optimize.minimize with SLSQP or trust-constr
    
    from scipy.optimize import minimize
    
    def objective(h_vec):
        h = h_vec[:N]
        return _overlap_max(h)
    
    # Initial guess: try a few structured patterns
    best_h = np.full(N, 0.5)
    best_val = _overlap_max(best_h)
    
    # Try various initializations
    inits = []
    
    # Uniform
    inits.append(np.full(N, 0.5))
    
    # Alternating 0 and 1
    inits.append(np.tile([0.0, 1.0], N // 2).astype(float))
    
    # Sine/cosine patterns
    for freq in [1, 2, 3, 4, 5]:
        inits.append(0.5 + 0.4 * np.sin(2 * np.pi * freq * np.arange(N) / N))
        inits.append(0.5 + 0.4 * np.cos(2 * np.pi * freq * np.arange(N) / N))
    
    # Random initializations
    np.random.seed(42)
    for _ in range(20):
        h_rand = np.random.rand(N)
        h_rand = np.clip(h_rand, 0, 1)
        # Project to sum = N/2
        for _ in range(50):
            diff = N / 2.0 - np.sum(h_rand)
            h_rand += diff / N
            h_rand = np.clip(h_rand, 0, 1)
        inits.append(h_rand)
    
    # Evaluate all initializations
    for h_init in inits:
        if time.time() > deadline - 30:
            break
        h_p = h_init.copy()
        # Ensure exact sum
        for _ in range(100):
            diff = N / 2.0 - np.sum(h_p)
            if abs(diff) < 1e-12:
                break
            h_p += diff / N
            h_p = np.clip(h_p, 0, 1)
        
        val = _overlap_max(h_p)
        if val < best_val:
            best_val = val
            best_h = h_p.copy()
    
    # Now do local optimization from best initial guess
    # Use smooth surrogate gradient descent
    def _smooth_overlap_grad(h, beta=100.0):
        w = 2.0 / N
        H = np.fft.fft(h)
        autocorr = np.fft.ifft(H * np.conj(H)).real
        overlaps = w * (N / 2.0 - autocorr)
        
        max_ov = np.max(overlaps)
        shifted = beta * (overlaps - max_ov)
        exp_terms = np.exp(shifted)
        probs = exp_terms / np.sum(exp_terms)
        
        P_fft = np.fft.fft(probs)
        h_fft = np.fft.fft(h)
        term1 = np.fft.ifft(P_fft * h_fft).real
        P_flip = np.flip(probs)
        P_flip[0] = probs[0]
        P_flip_fft = np.fft.fft(P_flip)
        term2 = np.fft.ifft(P_flip_fft * h_fft).real
        grad = -w * (term1 + term2)
        
        return grad
    
    # Gradient descent from best_h
    h_curr = best_h.copy()
    step = 0.01
    for beta in [50, 100, 200, 500]:
        if time.time() > deadline - 5:
            break
        for it in range(500):
            if time.time() > deadline - 2:
                break
            grad = _smooth_overlap_grad(h_curr, beta)
            if np.max(np.abs(grad)) < 1e-6:
                break
            new_h = h_curr - step * grad
            # Project
            new_h = np.clip(new_h, 0, 1)
            for _ in range(10):
                diff = N / 2.0 - np.sum(new_h)
                if abs(diff) < 1e-12:
                    break
                new_h += diff / N
                new_h = np.clip(new_h, 0, 1)
            
            new_val = _overlap_max(new_h)
            curr_val = _overlap_max(h_curr)
            
            if new_val < curr_val:
                h_curr = new_h
                if new_val < best_val:
                    best_val = new_val
                    best_h = h_curr.copy()
            else:
                step *= 0.5
                if step < 1e-7:
                    break
    
    # Final projection to ensure exact sum
    for _ in range(100):
        diff = N / 2.0 - np.sum(best_h)
        if abs(diff) < 1e-12:
            break
        best_h += diff / N
        best_h = np.clip(best_h, 0, 1)
    
    return best_h
