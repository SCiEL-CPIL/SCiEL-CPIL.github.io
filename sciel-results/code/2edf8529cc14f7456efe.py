"""Rewritten program for the third autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize

N = 400


def construct():
    """
    Construct step heights h of length N=400 to minimize:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
    
    Approach: Model the problem using the PSD Toeplitz structure of the autoconvolution.
    
    Key insights:
    1. The value is scale-invariant, so we fix sum(h) = 1.
    2. We want to minimize max_k (h*h)[k] where (h*h) = np.convolve(h, h).
    3. The autoconvolution entries are: ac[k] = sum_i h[i]*h[k-i] (with valid indices)
    4. We use a smooth soft-max approximation and projected gradient descent.
    
    The lower bound analysis:
    - ac[0] = h[0]^2, ac[2N-2] = h[N-1]^2 (squares, always >= 0)
    - By Cauchy-Schwarz: sum(h)^2 <= N * sum(h^2), so sum(h^2) >= 1/N when sum(h)=1
    - But ac[0] = h[0]^2 which can be small if h[0] is small
    - The key is that NOT all ac[k] need to be positive; we only care about the MAX (most positive)
    
    Strategy: Use alternating signs to make many autoconv entries negative,
    keeping only a few positive ones small.
    
    We'll use iterative optimization with soft-max smoothing.
    """
    
    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-15:
            return np.inf
        ac = np.convolve(h, h)
        return abs(2.0 * N * np.max(ac) / (s * s))
    
    # Start with a pattern that has alternating signs
    # Use a smooth window to avoid sharp transitions
    x = np.arange(N)
    # Create a pattern: positive in first half, negative in second half, with smooth transition
    # This makes the autoconvolution have a dip in the middle
    center = (N - 1) / 2.0
    sigma = N / 8.0
    h = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    # Make it zero-mean by subtracting mean, then add small offset to ensure sum != 0
    h = h - np.mean(h)
    # Add a small constant to ensure sum is positive and not zero
    h = h + 0.1
    
    # Normalize to sum = 1
    h = h / np.sum(h)
    
    # Use soft-max optimization
    # Objective: minimize soft_max(ac, temperature)
    # where ac = convolve(h, h)
    
    temperature = 10.0  # Start with moderate smoothing
    h_current = h.copy()
    
    def soft_max_obj(h):
        ac = np.convolve(h, h)
        # Soft max approximation
        max_ac = np.max(ac)
        # Use log-sum-exp for numerical stability
        shifted = ac - max_ac
        soft = max_ac + (1.0 / temperature) * np.log(np.sum(np.exp(temperature * shifted)))
        # Penalty for sum constraint (we'll project after)
        return soft
    
    def gradient_soft_max(h):
        ac = np.convolve(h, h)
        max_ac = np.max(ac)
        shifted = ac - max_ac
        weights = np.exp(temperature * shifted) / np.sum(np.exp(temperature * shifted))
        # Gradient of soft_max w.r.t. h
        # d(soft_max)/d(h[k]) = sum_j weights[j] * d(ac[j])/d(h[k])
        # ac[j] = sum_i h[i]*h[j-i], so d(ac[j])/d(h[k]) = h[j-k] + h[k-j] (if valid)
        # More precisely: d(ac[j])/d(h[k]) = h[j-k] (if 0<=j-k<N) + h[k-j] (if 0<=k-j<N)
        # But since ac is symmetric in a sense, let's compute directly
        grad = np.zeros(N)
        for k in range(N):
            # For each j, contribution to grad[k] is weights[j] * (h[j-k] if valid + h[k-j] if valid)
            # ac[j] = sum_{i: 0<=i<N, 0<=j-i<N} h[i]*h[j-i]
            # d(ac[j])/d(h[k]) = h[j-k] (if 0<=j-k<N) + h[k-j] (if 0<=k-j<N)
            # Note: these are the same term when j-k = k-j, i.e., j=2k
            for j in range(2 * N - 1):
                term = 0.0
                if 0 <= j - k < N:
                    term += h[j - k]
                if 0 <= k - j < N:
                    term += h[k - j]
                grad[k] += weights[j] * term
        return grad
    
    # Projected gradient descent with sum constraint
    learning_rate = 0.01
    n_iters = 200
    
    best_h = h_current.copy()
    best_val = compute_value(best_h)
    
    for it in range(n_iters):
        # Compute gradient
        ac = np.convolve(h_current, h_current)
        max_ac = np.max(ac)
        shifted = ac - max_ac
        weights = np.exp(temperature * shifted) / np.sum(np.exp(temperature * shifted))
        
        # Compute gradient numerically for reliability
        eps = 1e-7
        grad = np.zeros(N)
        for k in range(N):
            h_plus = h_current.copy()
            h_plus[k] += eps
            ac_plus = np.convolve(h_plus, h_plus)
            max_plus = np.max(ac_plus)
            shifted_plus = ac_plus - max_plus
            soft_plus = max_plus + (1.0 / temperature) * np.log(np.sum(np.exp(temperature * shifted_plus)))
            
            h_minus = h_current.copy()
            h_minus[k] -= eps
            ac_minus = np.convolve(h_minus, h_minus)
            max_minus = np.max(ac_minus)
            shifted_minus = ac_minus - max_minus
            soft_minus = max_minus + (1.0 / temperature) * np.log(np.sum(np.exp(temperature * shifted_minus)))
            
            grad[k] = (soft_plus - soft_minus) / (2 * eps)
        
        # Gradient step
        h_new = h_current - learning_rate * grad
        
        # Project onto sum(h) = 1
        h_new = h_new / np.sum(h_new)
        
        # Check improvement
        val_new = compute_value(h_new)
        if val_new < best_val:
            best_h = h_new.copy()
            best_val = val_new
            h_current = h_new
        else:
            # Reduce learning rate
            learning_rate *= 0.9
        
        # Decrease temperature gradually
        temperature *= 0.995
    
    # Try a few different initial patterns and keep the best
    patterns = [
        # Alternating with smooth window
        lambda x: np.sin(2 * np.pi * x / N) + 0.1,
        # Two-peak pattern
        lambda x: np.exp(-((x - N/4)**2) / (N/20)**2) - np.exp(-((x - 3*N/4)**2) / (N/20)**2) + 0.1,
        # Cosine with offset
        lambda x: np.cos(2 * np.pi * x / N) * 0.5 + 0.1,
    ]
    
    for pattern_func in patterns:
        h_try = pattern_func(np.arange(N))
        h_try = h_try / np.sum(h_try)
        
        # Quick local optimization
        lr = 0.005
        temp = 5.0
        for _ in range(50):
            ac = np.convolve(h_try, h_try)
            max_ac = np.max(ac)
            shifted = ac - max_ac
            weights = np.exp(temp * shifted) / np.sum(np.exp(temp * shifted))
            
            # Numerical gradient
            eps = 1e-7
            grad = np.zeros(N)
            for k in range(N):
                h_p = h_try.copy()
                h_p[k] += eps
                ac_p = np.convolve(h_p, h_p)
                sp = np.max(ac_p) + (1.0/temp) * np.log(np.sum(np.exp(temp * (ac_p - np.max(ac_p)))))
                
                h_m = h_try.copy()
                h_m[k] -= eps
                ac_m = np.convolve(h_m, h_m)
                sm = np.max(ac_m) + (1.0/temp) * np.log(np.sum(np.exp(temp * (ac_m - np.max(ac_m)))))
                
                grad[k] = (sp - sm) / (2 * eps)
            
            h_try = h_try - lr * grad
            h_try = h_try / np.sum(h_try)
            temp *= 0.98
        
        val_try = compute_value(h_try)
        if val_try < best_val:
            best_h = h_try.copy()
            best_val = val_try
    
    # Final check
    final_val = compute_value(best_h)
    seed_val = 2.0
    
    if final_val < seed_val:
        return best_h
    else:
        return np.ones(N)
