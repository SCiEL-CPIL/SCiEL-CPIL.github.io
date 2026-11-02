"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600
# Interval width on [-1/4, 1/4]
h = (1/4 - (-1/4)) / N  # 1/2 / 600 = 1/1200

# EVOLVE-BLOCK-START
def _autoconvolution_max(f, h):
    """Compute the maximum of the autoconvolution (f * f) at the grid points.
    f is an array of step heights. The convolution is computed via FFT for efficiency."""
    n = len(f)
    # The autoconvolution (f * f)(x) = int f(x-y) f(y) dy
    # For step functions on equal intervals, we can approximate via discrete convolution
    # scaled by h.
    # (f * f)(x_k) ≈ h * sum_i f_i f_j where x_k ≈ x_i + x_j
    # More precisely, the continuous autoconvolution at point x is:
    #   (f*f)(x) = ∫ f(x-y) f(y) dy
    # For step functions, this becomes a piecewise linear function. The maximum
    # occurs at some grid point or interval boundary.
    
    # Use FFT-based convolution for efficiency
    # The discrete convolution c_k = sum_i f_i f_{k-i} approximates (1/h) * (f*f) at grid points
    # Actually, let's think carefully.
    
    # f(x) = f_i for x in [x_i, x_{i+1}) where x_i = -1/4 + i*h
    # (f*f)(x) = ∫ f(x-y) f(y) dy
    # This is a convolution. Using FFT:
    # We need to compute the convolution of f with itself.
    # The discrete convolution of two arrays a, b of length n is c_k = sum_{i=0}^{k} a_i b_{k-i} for k=0..2n-2
    # But for the continuous convolution of step functions, we need to be more careful.
    
    # Let's use the fact that for step functions, the autoconvolution is a piecewise quadratic
    # function. However, for a quick approximation, we can use the discrete convolution
    # scaled appropriately.
    
    # The integral (f*f)(x) ≈ h * sum_{i,j: x_i + x_j ≈ x} f_i f_j
    # More precisely, if we define g(x) = f(x), then (f*f)(x) = ∫ g(x-y)g(y)dy
    # Using the trapezoidal rule or rectangle rule on the step functions:
    
    # Let's compute via FFT. The convolution of two sequences a, b of length n:
    # c = np.convolve(a, b) gives c_k = sum_i a_i b_{k-i} for k = 0..2n-2
    # This corresponds to the discrete sum. For the continuous convolution of step
    # functions with width h, we have:
    # (f*f)(x_k) ≈ h * sum_{i: x_i in support} f(x_k - x_i) f(x_i) * h / h
    # Hmm, let me think again.
    
    # f(x) = sum_i f_i * 1_{[x_i, x_{i+1})}(x)
    # (f*f)(x) = ∫ f(x-y) f(y) dy = sum_i sum_j f_i f_j ∫ 1_{[x_i,x_{i+1})}(x-y) 1_{[x_j,x_{j+1})}(y) dy
    # The integral is the length of the intersection of [x-y, x-y+h] and [x_j, x_{j+1})
    # which is a piecewise linear function of x.
    
    # For the purpose of finding the maximum, we can evaluate (f*f) at the midpoints
    # of intervals, or at all grid points, and take the max.
    
    # A simpler approach: use np.convolve with the step heights, then scale by h.
    # The discrete convolution c_k = sum_i f_i f_{k-i} approximates (1/h) * (f*f) at
    # the grid points x_k = x_0 + k*h for k = 0..2n-2.
    # So (f*f)(x_k) ≈ h * c_k
    
    # Let's verify: if f is constant = 1, then (f*f)(x) = ∫ 1 * 1 dy over the overlap
    # The support of f is [-1/4, 1/4], so the support of f*f is [-1/2, 1/2].
    # (f*f)(x) = length of intersection of [x-1/4, x+1/4] and [-1/4, 1/4]
    # At x=0, (f*f)(0) = 1/2.
    # Discrete convolution of [1,1,...,1] (length 600) with itself:
    # c_k = k+1 for k=0..599, c_k = 1200-k for k=600..1198
    # Max is c_599 = 600. h * c_599 = (1/1200) * 600 = 0.5. Correct!
    
    c = np.convolve(f, f, mode='full')
    return np.max(c) * h

def _score(f):
    """Compute the normalized maximum of the autoconvolution.
    The normalization is by the L1 norm of f (or L2? Let's use L1 as it's common
    for probability-like normalizations).
    
    Actually, the "normalized maximum" likely means max(f*f) / (some norm of f)^2
    or similar. Given the context of bounding C1 (first autocorrelation constant),
    the normalization is probably by the square of the L1 norm or L2 norm.
    
    Let's assume the score is max(f*f) / (sum(f)*h)^2, which is the max of the
    autoconvolution of the probability density f/(sum(f)*h).
    """
    s = np.sum(f) * h
    if s == 0:
        return np.inf
    max_conv = _autoconvolution_max(f, h)
    return max_conv / (s * s)

def _random_initial():
    """Generate a random nonnegative step function."""
    return np.random.dirichlet(np.ones(N)) * N  # This doesn't quite work
    
    # Let's just use random positive values
    f = np.random.rand(N)
    return f

def _hill_climb(f, max_iter=5000, deadline=None):
    """Simple hill climbing / coordinate descent."""
    f = f.copy()
    best_score = _score(f)
    step_size = 1.0
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        # Try to improve by adjusting one coordinate
        idx = np.random.randint(N)
        old_val = f[idx]
        
        # Try a few perturbations
        for delta in [step_size, -step_size, step_size*0.5, -step_size*0.5]:
            new_val = old_val + delta
            if new_val < 0:
                new_val = 0
            f[idx] = new_val
            new_score = _score(f)
            if new_score < best_score:
                best_score = new_score
            else:
                f[idx] = old_val
                break
        else:
            # No improvement, reduce step size
            step_size *= 0.9
    
    return f, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with a symmetric bump-like function
    # Try a few initializations
    best_f = None
    best_score = np.inf
    
    np.random.seed(42)
    
    # Initial guess: a smooth bump
    x = np.linspace(-1/4, 1/4, N, endpoint=False)
    f0 = np.exp(-100 * x**2)
    f0 = f0 / np.sum(f0) * N  # Normalize so that sum(f)*h ≈ 1
    
    f, score = _hill_climb(f0, max_iter=2000, deadline=deadline)
    if score < best_score:
        best_score = score
        best_f = f.copy()
    
    if time.time() > deadline:
        return best_f.tolist()
    
    # Try a few more random initializations
    for _ in range(5):
        if time.time() > deadline:
            break
        f_rand = np.random.rand(N)
        f_rand = f_rand / np.sum(f_rand) * N
        f, score = _hill_climb(f_rand, max_iter=1000, deadline=deadline)
        if score < best_score:
            best_score = score
            best_f = f.copy()
    
    return best_f.tolist()
# EVOLVE-BLOCK-END
