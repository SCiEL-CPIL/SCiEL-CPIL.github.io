"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def _autoconvolution_peak(h, n_grid):
    """Compute the supremum norm of the autoconvolution of h on [-1/4, 1/4].
    
    h is a step function with N intervals. We compute the continuous autoconvolution
    and find its maximum.
    """
    # The support of h is [-1/4, 1/4]
    # Autoconvolution f * f has support [-1/2, 1/2]
    # f * f (t) = integral f(s) f(t-s) ds
    # For step functions, this is piecewise quadratic
    
    # Discretize the autoconvolution on a fine grid
    # The autoconvolution of a step function with N steps has at most 2N-1 pieces
    # Let's use a fine grid to capture the peak
    
    # Actually, let's compute it more carefully.
    # h(x) = sum_i h_i * 1_{[x_i, x_{i+1}]}(x) where x_i = -1/4 + i/4N for i=0..N-1
    # Wait, the intervals are equal on [-1/4, 1/4], so each interval has width 1/(2N)
    # x_i = -1/4 + i * (1/(2N)) for i = 0, ..., N
    # h(x) = h_i for x in [x_i, x_{i+1})
    
    # Autoconvolution: (h * h)(t) = integral_{-1/4}^{1/4} h(s) h(t-s) ds
    # The integrand is nonzero when s in [-1/4, 1/4] and t-s in [-1/4, 1/4]
    # i.e., s in [max(-1/4, t-1/4), min(1/4, t+1/4)]
    
    # Let's compute on a fine grid
    # The autoconvolution is piecewise quadratic with breakpoints at t = x_i + x_j - 1/4
    # Actually, the breakpoints are at t = (x_i - 1/4) + x_j = x_i + x_j - 1/4
    # where x_i = -1/4 + i/(2N)
    # So breakpoints are at t = -1/2 + (i+j)/(2N) for i,j = 0..N-1
    # i+j ranges from 0 to 2N-2, so breakpoints are at -1/2 + k/(2N) for k = 0..2N-2
    # Plus the endpoints -1/2 and 1/2
    
    # Let's just use a fine numerical integration
    # The peak is likely near t=0
    
    # Use a grid with enough points to resolve the piecewise quadratic structure
    # There are at most 2N-1 pieces in the autoconvolution
    # Let's use 4*(2N-1) points per piece to be safe, but cap it
    
    # Actually, let's think about this differently.
    # The autoconvolution of a step function can be computed exactly.
    # Let's use the FFT-based approach or direct computation.
    
    # Direct computation:
    # (h*h)(t) = sum_i sum_j h_i h_j * measure({s: s in [x_i, x_{i+1}], t-s in [x_j, x_{j+1}]})
    # = sum_i sum_j h_i h_j * measure([max(x_i, t-x_{j+1}), min(x_{i+1}, t-x_j)])
    # (when this interval is non-empty)
    
    # For efficiency, let's sample on a fine grid and interpolate
    # The function is piecewise quadratic, so we need to sample at the breakpoints
    # and possibly at the vertex of each parabola
    
    # Let's compute on a grid that includes all breakpoints
    # Breakpoints at t = -1/2 + k/(2N) for k = 0, 1, ..., 2N
    # That's 2N+1 points, but the function is quadratic between consecutive breakpoints
    # The vertex of each parabola may or may not be at a breakpoint
    
    # For a more accurate estimate, let's use a finer grid
    # Use 8*N points as suggested by the knowledge
    t = np.linspace(-0.5, 0.5, n_grid)
    
    # Compute (h*h)(t) for each t
    # h(s) = h_i where s in [x_i, x_{i+1})
    # x_i = -0.25 + i * (0.5/N)
    dx = 0.5 / N  # width of each interval
    
    conv = np.zeros(len(t))
    for i in range(N):
        x_i = -0.25 + i * dx
        x_ip1 = x_i + dx
        # s in [x_i, x_ip1], need t-s in [-0.25, 0.25]
        # s in [max(x_i, t-0.25), min(x_ip1, t+0.25)]
        s_lo = np.maximum(x_i, t - 0.25)
        s_hi = np.minimum(x_ip1, t + 0.25)
        # For each s in [s_lo, s_hi], h(t-s) = h_j where t-s in [x_j, x_{j+1})
        # This is complex to vectorize directly. Let's use a different approach.
        
        # Alternative: use the fact that h is a step function and compute the convolution
        # via the cumulative sum approach
    
    # Better approach: compute the autoconvolution using the formula for step functions
    # Let's use a numerical approach with fine discretization of the function
    # and use FFT for convolution
    
    # Resample h on a fine grid
    # h is defined on [-0.25, 0.25] with N intervals
    # Let's create a fine grid
    n_fine = 8 * N
    x_fine = np.linspace(-0.25, 0.25, n_fine + 1)
    # h values at the fine grid points (use left endpoint of each interval)
    h_fine = np.zeros(n_fine)
    for i in range(N):
        start_idx = i * (n_fine // N)
        end_idx = (i + 1) * (n_fine // N)
        if i == N - 1:
            end_idx = n_fine
        h_fine[start_idx:end_idx] = h[i]
    
    # Compute autoconvolution using FFT
    # Zero-pad to avoid circular convolution
    n_fft = 2 * n_fine
    h_pad = np.zeros(n_fft)
    h_pad[:n_fine] = h_fine
    
    H = np.fft.rfft(h_pad)
    conv_fft = np.fft.irfft(H * H, n_fft)
    
    # The convolution is on a grid with spacing dx_fine = 0.5/n_fine
    # The support of the convolution is [-0.5, 0.5]
    # The FFT convolution gives us values at t = 0, dx, 2dx, ...
    # We need to find the maximum
    
    # The autoconvolution values
    conv_vals = conv_fft[:n_fine]  # This should give us the convolution on [-0.5, 0.5]
    # Actually, the FFT convolution gives circular convolution, but with zero-padding
    # it should be the linear convolution
    
    # The maximum of the autoconvolution
    peak = np.max(conv_vals)
    
    return peak


def _score(h):
    """Compute the score: squared L2 norm / (L1 norm * supremum norm of autoconvolution)."""
    h = np.array(h)
    l2_sq = np.sum(h ** 2)
    l1 = np.sum(h)
    
    if l1 == 0:
        return 0.0
    
    # Compute the autoconvolution peak
    peak = _autoconvolution_peak(h, 8 * N)
    
    if peak == 0:
        return 0.0
    
    return l2_sq / (l1 * peak)


def _gradient_step(h, lr, n_grid):
    """Compute a subgradient step for maximizing the score."""
    h = np.array(h, dtype=float)
    h = np.maximum(h, 0)
    
    l2_sq = np.sum(h ** 2)
    l1 = np.sum(h)
    
    if l1 == 0:
        return h
    
    peak = _autoconvolution_peak(h, n_grid)
    
    if peak == 0:
        return h
    
    score = l2_sq / (l1 * peak)
    
    # The gradient of the score with respect to h is complex
    # Let's use a numerical gradient approach
    # For each coordinate, compute the score with a small perturbation
    
    grad = np.zeros(N)
    eps = 1e-4
    
    for i in range(N):
        h_plus = h.copy()
        h_plus[i] += eps
        score_plus = _score(h_plus)
        
        h_minus = h.copy()
        h_minus[i] = max(0, h[i] - eps)
        score_minus = _score(h_minus)
        
        grad[i] = (score_plus - score_minus) / (2 * eps)
    
    # Normalize the gradient
    grad_norm = np.linalg.norm(grad)
    if grad_norm > 0:
        grad = grad / grad_norm
    
    # Update
    h_new = h + lr * grad
    h_new = np.maximum(h_new, 0)
    
    return h_new


def construct():
    """Construct a sequence of nonnegative step heights to maximize the score."""
    start_time = time.time()
    deadline = start_time + 140  # Leave 10 seconds margin
    
    # Start with a random initialization
    np.random.seed(42)
    h_best = np.ones(N)
    best_score = _score(h_best)
    
    # Stochastic subgradient method
    n_grid = 8 * N
    lr = 0.1
    n_steps = 500
    
    h_current = h_best.copy()
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Compute gradient step
        h_new = _gradient_step(h_current, lr, n_grid)
        
        # Check if the new point is better
        score_new = _score(h_new)
        
        if score_new > best_score:
            best_score = score_new
            h_best = h_new.copy()
            h_current = h_new
        else:
            # Decay learning rate
            lr *= 0.95
            h_current = h_best.copy()
        
        # Periodically try a local refinement
        if step % 50 == 0:
            # Try a few random perturbations
            for _ in range(5):
                if time.time() > deadline:
                    break
                h_perturbed = h_best + np.random.randn(N) * 0.1
                h_perturbed = np.maximum(h_perturbed, 0)
                score_perturbed = _score(h_perturbed)
                if score_perturbed > best_score:
                    best_score = score_perturbed
                    h_best = h_perturbed.copy()
    
    # Final polish: try to improve with smaller steps
    if time.time() < deadline - 10:
        lr_small = 0.01
        for _ in range(100):
            if time.time() > deadline:
                break
            h_new = _gradient_step(h_best, lr_small, n_grid)
            score_new = _score(h_new)
            if score_new > best_score:
                best_score = score_new
                h_best = h_new
    
    # Ensure nonnegativity
    h_best = np.maximum(h_best, 0)
    
    return list(h_best)
# EVOLVE-BLOCK-END
