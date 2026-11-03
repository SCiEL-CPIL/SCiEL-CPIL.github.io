"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def construct():
    """
    Choose equal-width step heights h on [0,2], bounded between 0 and 1 with integral 1,
    to minimize the maximum overlap of h with shifted 1-h.
    
    The overlap at shift k is:
      O_k = sum_i h[i] * (1 - h[(i+k) % N])
          = sum_i h[i] - sum_i h[i]h[(i+k)%N]
          = 1 - autocorr[k]  (since sum h = 1 in continuous, but here sum h = N/2)
    
    Wait, let's be careful. The task says "integral 1" and "sum is exactly len/2".
    So sum(h) = N/2 = 50.
    
    The overlap of h with shifted (1-h):
      O_k = sum_i h[i] * (1 - h[(i+k) % N])
          = sum_i h[i] - sum_i h[i] h[(i+k)%N]
          = N/2 - A_k
    
    where A_k = sum_i h[i] h[(i+k)%N] is the circular autocorrelation.
    
    We want to minimize max_k O_k = N/2 - min_k A_k.
    So we want to maximize min_k A_k.
    
    Constraints:
      0 <= h[i] <= 1
      sum_i h[i] = N/2
    
    This is a convex problem if we formulate it as:
      maximize t
      subject to: A_k >= t for all k
                  0 <= h[i] <= 1
                  sum h[i] = N/2
    
    But A_k is quadratic in h, so this is not a linear program.
    
    Alternative: The directive suggests frequency domain / PSD approach.
    The autocorrelation A_k is the inverse DFT of |H|^2 where H is the DFT of h.
    
    Let's think differently. Maybe we can use a simpler heuristic first, then refine.
    
    Actually, let's try a direct approach: use scipy.optimize.minimize with a smooth surrogate.
    
    Objective: minimize max_k O_k where O_k = N/2 - A_k
    Equivalent to: maximize min_k A_k
    
    Use log-sum-exp for soft-max or soft-min.
    
    Let's use: f(h) = logsumexp(-A_k / tau) to approximate -min_k A_k, then minimize f(h).
    Or maximize: -logsumexp(-A_k / tau)
    
    Actually, let's just use scipy.optimize.minimize with a penalty or barrier.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s out of 120s, safety margin
    
    # Initial guess: alternating pattern or constant
    h = np.full(N, 0.5)
    
    # Try a few heuristic initializations
    inits = []
    inits.append(np.full(N, 0.5))
    # Alternating
    alt = np.zeros(N)
    alt[::2] = 1.0
    alt[1::2] = 0.0
    # Adjust to sum = N/2
    # sum = N/2 already
    inits.append(alt)
    # Smooth step
    x = np.linspace(0, 1, N)
    inits.append(np.clip(2*x - 0.5, 0, 1))
    # Sine
    inits.append(0.5 + 0.4*np.sin(2*np.pi*x))
    
    best_h = None
    best_obj = np.inf
    
    def compute_A(h):
        """Compute circular autocorrelation A_k = sum_i h[i] h[(i+k)%N]"""
        H = np.fft.fft(h)
        power = np.abs(H)**2
        A = np.fft.ifft(power).real
        return A
    
    def objective(h_flat):
        h = h_flat.reshape(N)
        # Project to [0,1]
        h = np.clip(h, 0, 1)
        # Normalize to sum = N/2
        s = h.sum()
        if s > 0:
            h = h * (N/2.0) / s
        # Re-clip after scaling
        h = np.clip(h, 0, 1)
        # Renormalize again (iterative projection)
        for _ in range(10):
            s = h.sum()
            if abs(s - N/2) < 1e-10:
                break
            h = h * (N/2.0) / s
            h = np.clip(h, 0, 1)
        
        A = compute_A(h)
        # We want to maximize min(A)
        # Use soft-min: -logsumexp(-A/tau)
        tau = 0.01
        soft_min = -np.logsumexp(-A / tau) * tau
        return -soft_min  # minimize negative soft-min = maximize soft-min
    
    def gradient(h_flat):
        h = h_flat.reshape(N)
        h = np.clip(h, 0, 1)
        s = h.sum()
        if s > 0:
            h = h * (N/2.0) / s
        h = np.clip(h, 0, 1)
        for _ in range(10):
            s = h.sum()
            if abs(s - N/2) < 1e-10:
                break
            h = h * (N/2.0) / s
            h = np.clip(h, 0, 1)
        
        A = compute_A(h)
        tau = 0.01
        # Gradient of soft_min = -logsumexp(-A/tau) w.r.t. h
        # d/dh [-logsumexp(-A/tau)] = (1/tau) * sum_k exp(-A_k/tau) / sum_k exp(-A_k/tau) * dA_k/dh
        # dA_k/dh_i = h[(i+k)%N] + h[(i-k)%N] (for k != 0), for k=0: 2*h[i]
        # Actually A_k = sum_i h[i] h[(i+k)%N]
        # dA_k/dh_i = h[(i+k)%N] + (if k != 0) h[(i-k)%N] else 0... 
        # More carefully: A_k = sum_j h[j] h[(j+k)%N]
        # dA_k/dh_i = h[(i-k)%N] + (if k != 0) h[(i+k)%N] else 0
        # For k=0: A_0 = sum h[i]^2, dA_0/dh_i = 2 h[i]
        
        # This is getting complex. Let's just use numerical gradient or a simpler optimizer.
        pass
    
    # Use L-BFGS-B with numerical gradients
    from scipy.optimize import minimize
    
    best_obj = np.inf
    best_h = None
    
    for h0 in inits:
        h0 = h0.copy()
        # Ensure sum is N/2
        s = h0.sum()
        if s > 0:
            h0 = h0 * (N/2.0) / s
        h0 = np.clip(h0, 0, 1)
        
        res = minimize(
            objective,
            h0,
            method='L-BFGS-B',
            bounds=[(0, 1)] * N,
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        h_opt = res.x
        h_opt = np.clip(h_opt, 0, 1)
        s = h_opt.sum()
        if s > 0:
            h_opt = h_opt * (N/2.0) / s
        h_opt = np.clip(h_opt, 0, 1)
        
        A = compute_A(h_opt)
        min_A = A.min()
        max_overlap = N/2 - min_A
        
        if max_overlap < best_obj:
            best_obj = max_overlap
            best_h = h_opt.copy()
        
        if time.time() > deadline:
            break
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    # Final polish: ensure exact sum
    s = best_h.sum()
    if s > 0:
        best_h = best_h * (N/2.0) / s
    best_h = np.clip(best_h, 0, 1)
    # Small adjustment for exact sum
    diff = N/2 - best_h.sum()
    if abs(diff) > 1e-12:
        # Add to the element with most room
        idx = np.argmax(np.where(best_h < 1, 1 - best_h, 0) + np.where(best_h > 0, best_h, 0) * 0)
        # Simpler: just add diff to the first element that can accept it
        for i in range(N):
            if diff > 0 and best_h[i] < 1:
                add = min(diff, 1 - best_h[i])
                best_h[i] += add
                diff -= add
            elif diff < 0 and best_h[i] > 0:
                rem = min(-diff, best_h[i])
                best_h[i] -= rem
                diff += rem
            if abs(diff) < 1e-12:
                break
    
    return best_h
# EVOLVE-BLOCK-END
