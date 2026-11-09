"""Optimize nonnegative step heights to maximize ||f||_2^2 / (||f||_1 * sup|f*f|).

Key insight: For nonnegative f, the ratio under L1 normalization becomes
||f||_2^2 / sup|f*f|. We use Fourier phase optimization to find signals
where the autoconvolution peak is minimized relative to L2 norm.

Strategy:
1. Represent f as |IFFT(c)| for complex c with fixed L2 norm
2. Optimize phases of c to minimize sup|f*f| / ||f||_2^2
3. Ensure nonnegativity via absolute value
"""
import numpy as np
import time

N = 50
dx = 1.0 / N
START_TIME = time.time()
TIME_LIMIT = 145  # Safety margin before 150s kill


def compute_ratio_normalized(h):
    """Compute ratio = ||h||_2^2 / (||h||_1 * sup|conv|) with proper normalization."""
    h = np.maximum(h, 0)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    # Normalize to L1 = 1
    h_norm = h / l1
    l2_sq = np.sum(h_norm**2) * dx
    # Autoconvolution
    conv = np.convolve(h_norm, h_norm, mode='full') * dx
    sup_conv = np.max(conv)
    if sup_conv == 0:
        return 0.0
    return l2_sq / sup_conv


def compute_ratio_raw(h):
    """Compute raw ratio = ||h||_2^2 / (||h||_1 * sup|conv|) without normalization."""
    h = np.maximum(h, 0)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv == 0:
        return 0.0
    return l2_sq / (l1 * sup_conv)


def fft_phase_optimize(h0, n_iter=200, lr=0.1):
    """Optimize phases of FFT representation to minimize autoconvolution peak."""
    h = h0.copy().astype(float)
    h = np.maximum(h, 0)
    s = np.sum(h)
    if s == 0:
        return h
    h = h / s  # L1 normalize
    
    # Get FFT
    H = np.fft.fft(h)
    magnitudes = np.abs(H)
    phases = np.angle(H)
    
    # Ensure magnitudes match (we only optimize phases)
    # Reconstruct from magnitudes and phases
    def reconstruct(ph):
        H_new = magnitudes * np.exp(1j * ph)
        h_new = np.real(np.fft.ifft(H_new))
        return np.maximum(h_new, 0)
    
    best_h = h.copy()
    best_ratio = compute_ratio_normalized(best_h)
    
    for it in range(n_iter):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        h_curr = reconstruct(phases)
        s_curr = np.sum(h_curr)
        if s_curr == 0:
            break
        h_curr = h_curr / s_curr
        
        # Compute gradient of sup_conv with respect to phases
        # Use numerical gradient
        eps = 1e-6
        grad = np.zeros(N)
        for j in range(N):
            ph_plus = phases.copy()
            ph_plus[j] += eps
            h_plus = reconstruct(ph_plus)
            s_plus = np.sum(h_plus)
            if s_plus > 0:
                h_plus = h_plus / s_plus
                conv_plus = np.convolve(h_plus, h_plus, mode='full')
                sup_plus = np.max(conv_plus)
            else:
                sup_plus = 0.0
            
            ph_minus = phases.copy()
            ph_minus[j] -= eps
            h_minus = reconstruct(ph_minus)
            s_minus = np.sum(h_minus)
            if s_minus > 0:
                h_minus = h_minus / s_minus
                conv_minus = np.convolve(h_minus, h_minus, mode='full')
                sup_minus = np.max(conv_minus)
            else:
                sup_minus = 0.0
            
            grad[j] = (sup_plus - sup_minus) / (2 * eps)
        
        # Update phases (gradient descent on sup_conv)
        phases -= lr * grad
        phases = np.angle(np.exp(1j * phases))  # Wrap to [-pi, pi]
        
        h_new = reconstruct(phases)
        s_new = np.sum(h_new)
        if s_new > 0:
            h_new = h_new / s_new
            r_new = compute_ratio_normalized(h_new)
            if r_new > best_ratio:
                best_ratio = r_new
                best_h = h_new.copy()
    
    return best_h


def random_search(n_starts=30, n_iter_per_start=50):
    """Try multiple random initializations with phase optimization."""
    best_h = np.ones(N) / N
    best_ratio = compute_ratio_normalized(best_h)
    
    for start in range(n_starts):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
        
        # Different initialization strategies
        if start % 5 == 0:
            # Uniform
            h0 = np.ones(N)
        elif start % 5 == 1:
            # Random
            h0 = np.random.rand(N)
        elif start % 5 == 2:
            # Gaussian bump
            x = np.linspace(-1, 1, N)
            h0 = np.exp(-x**2 * 3)
        elif start % 5 == 3:
            # Two bumps
            x = np.linspace(-1, 1, N)
            h0 = np.exp(-(x - 0.4)**2 * 8) + np.exp(-(x + 0.4)**2 * 8)
        else:
            # Sparse random
            h0 = np.zeros(N)
            idx = np.random.choice(N, size=np.random.randint(3, 10), replace=False)
            h0[idx] = 1.0
        
        # Optimize
        h_opt = fft_phase_optimize(h0, n_iter=n_iter_per_start)
        r_opt = compute_ratio_normalized(h_opt)
        
        if r_opt > best_ratio:
            best_ratio = r_opt
            best_h = h_opt
    
    return best_h, best_ratio


def coordinate_ascent(h0, n_iter=100):
    """Coordinate ascent to refine the solution."""
    h = h0.copy()
    s = np.sum(h)
    if s == 0:
        return h
    h = h / s
    
    best_h = h.copy()
    best_ratio = compute_ratio_normalized(best_h)
    
    for it in range(n_iter):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        improved = False
        for i in range(N):
            orig = h[i]
            
            # Try increasing
            h[i] = orig + 0.01
            s = np.sum(h)
            if s > 0:
                h_norm = h / s
                r = compute_ratio_normalized(h_norm)
                if r > best_ratio:
                    best_ratio = r
                    best_h = h_norm.copy()
                    improved = True
                else:
                    h[i] = orig
            
            # Try decreasing
            h[i] = max(0, orig - 0.01)
            s = np.sum(h)
            if s > 0:
                h_norm = h / s
                r = compute_ratio_normalized(h_norm)
                if r > best_ratio:
                    best_ratio = r
                    best_h = h_norm.copy()
                    improved = True
                else:
                    h[i] = orig
            else:
                h[i] = orig
        
        if not improved:
            break
    
    return best_h


def construct():
    elapsed = time.time() - START_TIME
    if elapsed > TIME_LIMIT:
        return [1.0] * N
    
    # Start with uniform (seed)
    best_h = np.ones(N) / N
    best_ratio = compute_ratio_normalized(best_h)
    
    # Try FFT phase optimization from various starts
    h_opt, r_opt = random_search(n_starts=20, n_iter_per_start=30)
    if r_opt > best_ratio:
        best_ratio = r_opt
        best_h = h_opt
    
    # Refine with coordinate ascent
    h_refined = coordinate_ascent(best_h, n_iter=50)
    r_refined = compute_ratio_normalized(h_refined)
    if r_refined > best_ratio:
        best_ratio = r_refined
        best_h = h_refined
    
    # Return normalized heights (L1 = 1)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    
    return [float(x) for x in best_h]
