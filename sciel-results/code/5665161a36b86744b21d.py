"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Strategy: Min-Max Convex Optimization via Projected Subgradient Descent.
The objective f(h) = max_k C_k(h) is convex (max of linear functions).
We minimize f(h) subject to h in [0,1]^N and sum(h) = N/2.
We use a custom projected subgradient method with Polyak step sizes,
which is globally convergent for convex problems and avoids local minima
associated with smooth approximations or generic NLP solvers.
"""
import time
import numpy as np

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _project(h, target_sum=N / 2.0):
    """
    Project h onto the set: {x | 0 <= x_i <= 1, sum(x) = target_sum}.
    This is the intersection of a box and a hyperplane.
    We use a binary search on a Lagrange multiplier lambda for the sum constraint.
    x_i(lambda) = clip(h_i - lambda, 0, 1)
    We need sum_i x_i(lambda) = target_sum.
    Since sum is monotonic decreasing in lambda, we can binary search.
    """
    h = np.asarray(h, dtype=np.float64)
    
    # If h is already valid, return it
    if np.all(h >= 0) and np.all(h <= 1) and abs(np.sum(h) - target_sum) < 1e-12:
        return h
    
    # Binary search range for lambda
    # If sum(h) > target, we need to subtract (lambda > 0 mostly)
    # If sum(h) < target, we need to add (lambda < 0 mostly)
    lo = -1.0
    hi = 1.0
    
    # Expand range if necessary to ensure bracketing
    # sum(clip(h - lo)) should be >= target
    # sum(clip(h - hi)) should be <= target
    while np.sum(np.clip(h - lo, 0, 1)) < target_sum:
        lo -= 1.0
    while np.sum(np.clip(h - hi, 0, 1)) > target_sum:
        hi += 1.0
        
    for _ in range(60):  # ~1e-18 precision
        mid = (lo + hi) / 2.0
        x = np.clip(h - mid, 0, 1)
        s = np.sum(x)
        if s > target_sum:
            lo = mid
        else:
            hi = mid
            
    x = np.clip(h - (lo + hi) / 2.0, 0, 1)
    
    # Final fine-tuning to ensure exact sum (handle floating point errors)
    diff = target_sum - np.sum(x)
    if abs(diff) > 1e-15:
        # Adjust elements that are not at bounds
        mask = (x > 0) & (x < 1)
        if np.any(mask):
            x[mask] += diff / np.sum(mask)
            x = np.clip(x, 0, 1)
        else:
            # Fallback: adjust one element
            idx = 0
            if diff > 0:
                idx = np.argmin(x)
            else:
                idx = np.argmax(x)
            x[idx] += diff
            x = np.clip(x, 0, 1)
            
    return x


def _subgradient(h):
    """
    Compute a subgradient of f(h) = max_k C_k(h).
    If the max is unique, it's the gradient of that C_k.
    If multiple, we can use the gradient of any active one, or average.
    For stability, we use the gradient of the component achieving the max.
    If ties, we pick the first one or average all active ones.
    Averaging active ones often gives a better direction (centroid of normal cone).
    """
    n = len(h)
    cs = _linear_overlap_values(h)
    max_val = np.max(cs)
    
    # Find active indices (within tolerance)
    tol = 1e-9
    active_mask = (cs >= max_val - tol)
    
    # Gradient of C_k w.r.t h:
    # C_k = (2/n) * (n/2 - sum_{i=k}^{n-1} h_i h_{i-k})
    # dC_k/dh_j = -(2/n) * (h_{j-k} if j>=k + h_{j+k} if j+k<n)
    
    # We can compute the gradient for all active k and average them.
    # To be efficient, we can just pick the index of the max.
    # But averaging active subgradients is standard for min-max.
    
    # Let's compute the subgradient for the single max index first for speed,
    # as computing all active gradients might be slower if many are active.
    # Actually, for a "sharp" max, only 1 is active.
    # Let's use the index of the first max.
    k_max = np.argmax(cs)
    
    grad = np.zeros(n)
    # dC_k/dh_j = -(2/n) * (h_{j-k} if j>=k + h_{j+k} if j+k<n)
    
    # Vectorized computation for a single k
    # Term 1: h_{j-k} for j>=k -> h[0:n-k] shifted by k
    # Term 2: h_{j+k} for j+k<n -> h[k:n] shifted by -k
    
    # Let's construct the vector g_j
    # g_j = -(2/n) * (h_{j-k} + h_{j+k}) with bounds
    
    # Using slicing
    # h_{j-k}: for j in k..n-1, value is h[j-k]. So grad[k:] -= (2/n) * h[:n-k]
    # h_{j+k}: for j in 0..n-1-k, value is h[j+k]. So grad[:n-k] -= (2/n) * h[k:]
    
    factor = 2.0 / n
    grad[k:] -= factor * h[:n-k]
    grad[:n-k] -= factor * h[k:]
    
    return grad, max_val


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    
    # Start with a good initial guess
    # Cosine wave is a good start for this problem
    i = np.arange(N)
    h = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h = _project(h)
    
    best_h = h.copy()
    best_score = _score(h)
    
    # Polyak Step Size parameters
    # f* is the optimal value. We don't know it.
    # We can use a lower bound.
    # Note: C_k >= 0 always? 
    # C_k = (2/N)(N/2 - r_k). r_k is autocorrelation.
    # For h=0.5 constant, r_k = N/4 * 0.25 * N? No.
    # r_0 = sum h^2. For 0.5, r_0 = N * 0.25 = N/4.
    # C_0 = (2/N)(N/2 - N/4) = (2/N)(N/4) = 0.5.
    # So f(0.5) = 0.5.
    # Can we go lower?
    # The theoretical lower bound for this problem is related to the Erdos problem.
    # For N=100, the minimum max overlap is likely around 0.3-0.4?
    # Let's set f_lb = 0.0 as a safe lower bound for the step size calculation,
    # though it will make steps larger.
    # Actually, if f_lb is too low, steps are too big and we oscillate.
    # If f_lb is close to f*, steps are optimal.
    # We can adaptively update f_lb if we find a better solution? No, f* is the min.
    # We can use the current best score as an upper bound f_ub.
    # Step size gamma = (f(h) - f_lb) / ||g||^2.
    # If f_lb is 0, and f(h) is 0.5, gamma is reasonable.
    
    f_lb = 0.0  # Conservative lower bound
    
    # Momentum? Subgradient descent is slow.
    # We can use Nesterov acceleration or just a simple averaging of iterates.
    # Averaging iterates is crucial for subgradient methods to converge to the optimum.
    
    h_avg = h.copy()
    count = 0
    
    # We will run for a fixed number of iterations or until time runs out.
    # Each iteration is fast (FFT + projection).
    
    # Let's try a simple loop with time check.
    # We can also try multiple restarts if time permits, but subgrad is global.
    
    while time.time() - t0 < _DEADLINE - 5:
        # Compute subgradient
        g, f_val = _subgradient(h)
        
        # Check for convergence
        if np.linalg.norm(g) < 1e-8:
            break
            
        # Polyak step size
        # gamma = (f(h) - f_lb) / (||g||^2 + eps)
        norm_sq = np.dot(g, g)
        if norm_sq < 1e-12:
            break
        gamma = (f_val - f_lb) / (norm_sq + 1e-12)
        
        # Cap the step size to avoid huge jumps
        # The diameter of the box is sqrt(N).
        # A safe max step is maybe 0.1 * diameter?
        max_step = 0.5
        if gamma > max_step:
            gamma = max_step
            
        # Update
        h_new = h - gamma * g
        
        # Project
        h_new = _project(h_new)
        
        # Update average
        count += 1
        h_avg = (h_avg * (count - 1) + h_new) / count
        
        # Update best
        s_avg = _score(h_avg)
        if s_avg < best_score:
            best_score = s_avg
            best_h = h_avg.copy()
            
        s_new = _score(h_new)
        if s_new < best_score:
            best_score = s_new
            best_h = h_new.copy()
            
        h = h_new
        
        # Adaptive f_lb update?
        # If we are stuck, maybe f_lb is too low?
        # No, low f_lb means larger steps. If we oscillate, we need smaller steps.
        # Polyak step size naturally handles this if f_lb is correct.
        # If f_lb < f*, steps are too big.
        # We can estimate f* by the minimum value seen so far?
        # No, f* is the global min.
        # Let's stick with f_lb = 0.
        
        # To accelerate convergence, we can reduce the step size over time
        # or use a diminishing step size rule as a fallback if Polyak fails.
        # But Polyak is theoretically sound.
        
        # Let's add a small damping to the step size to prevent oscillation
        # gamma = 0.9 * gamma
        
    # Final polish with a local method?
    # The subgradient method converges slowly near the optimum.
    # We can use the result as a start for a few steps of projected gradient
    # with a smaller fixed step, or just return the best average.
    
    # Given the time limit, the subgradient method might not have fully converged.
    # The "best_h" tracked is the best point seen, which is good.
    
    # One more thing: ensure the sum is exactly N/2 in the final output
    best_h = _project(best_h)
    
    return best_h
