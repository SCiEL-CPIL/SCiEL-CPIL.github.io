"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Construct signed step heights minimizing the maximum absolute autoconvolution
    divided by squared integral.
    
    Strategy:
    1. Start with a random binary sequence (+1/-1).
    2. Use iterative gradient descent / projected gradient to minimize the peak autoconvolution.
    3. The step-function constraint is enforced by projecting onto the hypercube {-1, 1}^N.
    
    We minimize max_k |c_k| where c_k = sum_i x_i x_{i+k} (circular autoconvolution).
    Since the integral is sum x_i = S, and for binary sequences S is fixed in magnitude 
    (though can vary), we want to minimize max |c_k| / S^2.
    
    For simplicity and robustness within time limits, we minimize the L_infinity norm 
    of the autoconvolution coefficients, normalized by the square of the sum.
    """
    
    np.random.seed(42)
    
    # Initial random binary sequence
    x = np.random.choice([-1.0, 1.0], size=N)
    
    def compute_autoconv(x):
        # Circular autoconvolution via FFT
        X = np.fft.fft(x)
        power = X * np.conj(X)
        c = np.fft.ifft(power).real
        return c
    
    def objective(x):
        c = compute_autoconv(x)
        S = np.sum(x)
        if abs(S) < 1e-10:
            return 1e10  # Avoid division by zero
        # We want to minimize max |c_k| / S^2
        # But since S^2 is always positive, and we're minimizing, 
        # we can just minimize max |c_k| and keep S large.
        # Actually, the score is max |c_k| / S^2.
        # Let's minimize this directly.
        max_c = np.max(np.abs(c))
        return max_c / (S * S)
    
    def gradient(x):
        # Gradient of max |c_k| / S^2 is tricky.
        # Let's use a subgradient approach or simply use a simple heuristic.
        # Alternative: minimize sum of squares of c_k (L2 norm) as a proxy, 
        # which is easier to differentiate, then project.
        # But the task asks for max absolute value.
        # Let's use a soft-max approximation or just do coordinate descent.
        
        # For now, let's implement a simple iterative improvement:
        # Flip the sign of the element that most reduces the objective.
        # This is O(N^2) per step, which might be slow for N=400.
        # Better: use gradient of the squared autoconvolution.
        
        # Let's try minimizing the L2 norm of the autoconvolution as a proxy,
        # which is equivalent to minimizing the energy in the sidelobes.
        # The gradient of sum(c_k^2) with respect to x_i is 4 * c * x (circular convolution).
        c = compute_autoconv(x)
        # grad = 4 * np.convolve(c, x, mode='same') # linear, but we want circular
        grad = 4 * np.fft.ifft(np.fft.fft(c) * np.fft.fft(x)).real
        return grad
    
    # Use scipy's minimize with projected gradient
    # We minimize the L2 norm of autoconvolution as a proxy for max norm,
    # because it's smooth and differentiable. Then we can do a final step
    # to reduce the peak.
    
    def obj_l2(x):
        c = compute_autoconv(x)
        S = np.sum(x)
        if abs(S) < 1e-10:
            return 1e10
        return np.sum(c**2) / (S * S)
    
    def grad_l2(x):
        c = compute_autoconv(x)
        S = np.sum(x)
        if abs(S) < 1e-10:
            return np.zeros(N)
        # d/dx_i [ sum_k (c_k)^2 / S^2 ]
        # This is complex due to S depending on x.
        # Let's simplify: just minimize sum(c_k^2) and keep S large.
        # grad of sum(c_k^2) = 4 * circular_conv(c, x)
        grad = 4 * np.fft.ifft(np.fft.fft(c) * np.fft.fft(x)).real
        return grad
    
    # Run projected gradient descent
    step_size = 0.1
    for iteration in range(500):
        grad = grad_l2(x)
        x_new = x - step_size * grad
        # Project onto hypercube {-1, 1}
        x_new = np.sign(x_new)
        if np.all(x_new == x):
            break
        x = x_new
    
    # Final refinement: try flipping individual bits to reduce max |c_k| / S^2
    current_obj = objective(x)
    for i in range(N):
        x[i] *= -1
        new_obj = objective(x)
        if new_obj < current_obj:
            current_obj = new_obj
        else:
            x[i] *= -1  # revert
    
    return x

# EVOLVE-BLOCK-END
