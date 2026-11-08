"""Program for minimizing the normalized maximum autoconvolution of a step function."""
import numpy as np
from scipy.optimize import minimize

N = 600

# EVOLVE-BLOCK-START
def construct():
    # We want to find nonnegative step heights h[0..N-1] on equal intervals of [-1/4, 1/4]
    # that minimize the normalized maximum of its autoconvolution.
    # 
    # The autoconvolution of a step function h (constant on each interval) is:
    # (h * h)(t) = sum_i h[i] * h[j] where the intervals [i/N - 1/4, (i+1)/N - 1/4] and
    # [j/N - 1/4, (j+1)/N - 1/4] overlap in a way that their sum covers t.
    #
    # For step functions on equal intervals, the autoconvolution is a piecewise linear function.
    # The maximum of the autoconvolution occurs at specific points. Let's compute it properly.
    #
    # Actually, let's think about this more carefully. If h is constant h_i on interval i,
    # then the convolution (h*h)(t) = ∫ h(s) h(t-s) ds.
    #
    # For discrete step functions, we can compute the autoconvolution by considering the overlap
    # of intervals. The autoconvolution at a point t is the integral over s of h(s)h(t-s)ds.
    #
    # Let's define the intervals: interval i covers [i/N - 1/4, (i+1)/N - 1/4] for i=0,...,N-1.
    # The width of each interval is 1/N.
    #
    # The autoconvolution (h*h)(t) for t in [-1/2, 1/2] (since the support is [-1/4, 1/4],
    # the convolution support is [-1/2, 1/2]).
    #
    # For a step function, the autoconvolution is piecewise quadratic? No, it's piecewise linear
    # if we consider the discrete sum approximation, but actually for true step functions it's
    # piecewise quadratic? Let me think again.
    #
    # Actually, for two step functions f and g, their convolution (f*g)(t) = ∫ f(s)g(t-s)ds.
    # If f is a step function with values f_i on interval i, and g is a step function with values
    # g_j on interval j, then (f*g)(t) is a piecewise linear function (actually piecewise
    # constant? No...).
    #
    # Let's use a simpler approach: the autoconvolution of a step function can be computed by
    # summing over pairs of intervals. For each pair (i,j), the contribution to the convolution
    # is f_i * f_j * (length of overlap of interval i and (t - interval j)).
    #
    # This is getting complex. Let's use a numerical approach: sample the autoconvolution at
    # many points and find its maximum.
    #
    # Actually, for the purpose of this problem, let's use the following approach:
    # 1. Represent h as a vector of N nonnegative values.
    # 2. Compute the autoconvolution using FFT or direct computation.
    # 3. The "normalized" maximum likely means dividing by the integral of h (or h^2?).
    #
    # Let me re-read: "minimize the normalized maximum of its autoconvolution"
    #
    # The normalization is likely by the L1 norm or L2 norm of h. Let's assume it's normalized
    # by the integral of h (i.e., sum(h) * 1/N, since each interval has width 1/N).
    #
    # Actually, let's think about what "normalized" means. If we scale h by a constant c, the
    # autoconvolution scales by c^2. So "normalized" must mean we fix some norm of h.
    #
    # Common normalizations:
    # - L1 norm: ∫|h| = 1, i.e., sum(h)/N = 1
    # - L2 norm: ∫h^2 = 1, i.g., sum(h^2)/N = 1
    #
    # Let's assume L1 normalization: sum(h) = N (so the integral is 1).
    #
    # The autoconvolution (h*h)(t) has integral = (∫h)^2 = 1.
    #
    # We want to minimize max_t (h*h)(t) subject to sum(h) = N and h >= 0.
    #
    # For the uniform function h = 1, the autoconvolution is a triangular function with
    # maximum N/N = 1 at t=0? Let's compute: if h_i = 1 for all i, then (h*h)(t) at t=0
    # is the sum over i,j where intervals overlap at 0... this is getting complicated.
    #
    # Let's use a numerical approach to compute the autoconvolution and its maximum.

    def autoconvolution_max(h, n_samples=4000):
        """Compute the maximum of the autoconvolution of step function h."""
        h = np.asarray(h, dtype=float)
        n = len(h)
        # Intervals: i-th interval is [i/n - 1/4, (i+1)/n - 1/4]
        # Autoconvolution support: [-1/2, 1/2]
        t = np.linspace(-0.5, 0.5, n_samples)
        
        # For each t, compute (h*h)(t) = integral of h(s) h(t-s) ds
        # We can do this by summing over interval pairs
        # For efficiency, use the fact that h is piecewise constant
        
        # Alternative: use FFT-based convolution
        # Pad h to avoid circular convolution effects
        # The support of h is [−1/4, 1/4], length 1/2.
        # The support of h*h is [−1/2, 1/2], length 1.
        # We need to sample on a grid that covers this.
        
        # Let's use direct computation for accuracy
        # For each t_k, (h*h)(t_k) = sum_i h_i * integral over s in interval_i of h(t_k - s) ds
        # = sum_i h_i * (average of h over the shifted interval)
        
        # More precisely: for interval i = [a_i, b_i] with width w = 1/n,
        # integral of h(t-s) ds over s in [a_i, b_i] = integral of h(u) du over u in [t-b_i, t-a_i]
        # This is the integral of h over the interval [t - b_i, t - a_i].
        
        # Since h is piecewise constant, this integral is the sum over j of h_j * (overlap length of [t-b_i, t-a_i] with interval j)
        
        # This is O(n^2 * n_samples) which might be too slow. Let's use FFT.
        
        # FFT approach:
        # Create a sampled version of h on a fine grid
        # The function h is 1 on [-1/4, 1/4] (scaled by h_i on each subinterval)
        # Let's create a grid that covers [-1/2, 1/2] with fine resolution
        
        # Actually, let's just use np.convolve with appropriate padding
        # h is defined on [-1/4, 1/4] with N intervals of width 1/N
        # We want the linear convolution
        
        # Create the signal: h_i on interval i
        # The "sampled" version on a grid of width 1/N:
        # Let's create an array of length M where M is large enough
        # The support of h is length 1/2 = N*(1/N), so N samples
        # The support of h*h is length 1 = 2N samples
        
        # For linear convolution, we need to pad to at least 2N-1
        # But since h is piecewise constant, let's just use the values directly
        
        # Simple approach: treat h as a discrete sequence and compute linear convolution
        # This gives an approximation where each "sample" represents an interval
        conv = np.convolve(h, h, mode='full')
        # The convolution has length 2N-1
        # The maximum of the continuous autoconvolution is approximately the max of this
        # (with appropriate scaling)
        
        # The discrete convolution sum_i h_i h_j approximates the integral times N
        # So the actual autoconvolution value is conv[k] / N (since each term represents
        # an integral over intervals of width 1/N)
        
        return np.max(conv) / n

    # Objective function
    def objective(h):
        h = np.maximum(h, 0)  # Ensure nonnegativity
        if np.sum(h) == 0:
            return 1e10
        # Normalize: L1 norm = 1, i.e., sum(h) = N
        h_norm = h * (N / np.sum(h))
        return autoconvolution_max(h_norm)
    
    # Initial guess: uniform
    h0 = np.ones(N)
    
    # Use gradient descent with projected gradient (for nonnegativity and normalization)
    lr = 0.01
    for iteration in range(200):
        h = np.maximum(h0, 0)
        h = h * (N / (np.sum(h) + 1e-10))
        
        # Compute gradient numerically
        grad = np.zeros(N)
        eps = 1e-4
        for i in range(N):
            h_plus = h.copy()
            h_plus[i] += eps
            h_plus = h_plus * (N / (np.sum(h_plus) + 1e-10))
            f_plus = objective(h_plus)
            
            h_minus = h.copy()
            h_minus[i] = max(h[i] - eps, 0)
            h_minus = h_minus * (N / (np.sum(h_minus) + 1e-10))
            f_minus = objective(h_minus)
            
            grad[i] = (f_plus - f_minus) / (2 * eps)
        
        # Gradient step
        h0 = h0 - lr * grad
        
        # Project: nonnegativity and normalization
        h0 = np.maximum(h0, 0)
        if np.sum(h0) > 0:
            h0 = h0 * (N / np.sum(h0))
    
    h_final = np.maximum(h0, 0)
    h_final = h_final * (N / (np.sum(h_final) + 1e-10))
    
    return list(h_final)
# EVOLVE-BLOCK-END
