import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 120.0

def _autoconv_from_heights(heights):
    """
    Compute the autoconvolution of a step function defined by continuous heights.
    heights: array of shape (N,)
    Returns: array of autoconvolution values on a fine grid.
    """
    # Use a fine grid for accurate convolution calculation
    M = 4 * N
    dx = 0.5 / M  # Domain is [-1/4, 1/4], length 0.5
    
    # Map heights to fine grid
    # Each of the N intervals covers M/N points on the fine grid
    f_fine = np.repeat(heights, M // N)
    
    # Ensure length is correct if there's rounding error in M//N
    if len(f_fine) < M:
        f_fine = np.pad(f_fine, (0, M - len(f_fine)))
    elif len(f_fine) > M:
        f_fine = f_fine[:M]
        
    # Autoconvolution via FFT
    # We want (f * f)(t) = \int f(s) f(t-s) ds
    # Discrete approximation: sum f_i f_j dx where indices match shift
    # FFT method: ifft(fft(f) * conj(fft(f))) gives circular convolution.
    # For linear convolution, we need to zero-pad.
    # Since f is supported on [0, M) in the grid indices (representing [-1/4, 1/4]),
    # the autoconvolution is supported on [-0.5, 0.5].
    # The FFT length needs to be at least 2*M to avoid wrap-around.
    
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    # The result is on a grid of length 2*M, representing [-0.5, 0.5]
    # We care about the max absolute value
    return autoconv

def _c3_and_grad(heights):
    """
    Compute C3 (max |autoconv|) and its gradient.
    """
    autoconv = _autoconv_from_heights(heights)
    c3 = np.max(np.abs(autoconv))
    
    # To get the gradient, we can use the fact that the autoconvolution is quadratic.
    # Let A be the matrix such that autoconv = A @ (heights * heights)? No, it's bilinear.
    # autoconv_k = sum_{i,j} f_i f_j * overlap(i, j, k) * dx
    # This is complex to derive analytically for the max norm.
    # Instead, we can use numerical gradient or a subgradient approximation.
    # However, L-BFGS-B requires gradients.
    
    # Alternative: Use the fact that we are minimizing the max of a set of smooth functions.
    # The gradient of max(|x_i|) is sign(x_i) * grad(x_i) for the maximizing i.
    
    # Let's compute the gradient of the autoconvolution with respect to heights.
    # autoconv = ifft(fft(f) * conj(fft(f))) * dx
    # This is non-linear.
    
    # Simpler approach for gradient:
    # Since N=400 is small, we can compute the autoconvolution matrix explicitly?
    # No, 400x400 is fine, but the autoconvolution has more points.
    
    # Let's use a finite difference approximation for the gradient if analytic is too hard,
    # or better, derive it.
    
    # Actually, let's stick to the continuous relaxation but use a simpler objective:
    # Minimize the L2 norm of the autoconvolution first? No, the task asks for upper bound (likely L-inf).
    
    # Let's try to compute the gradient numerically but efficiently? 
    # Or, realize that for the specific structure of step functions, we can precompute the kernel.
    
    # Let K be a matrix where K[i, j] represents the contribution of f_i f_j to the peak?
    # No, C3 is max over t.
    
    # Let's use a subgradient method or just rely on the fact that the continuous landscape is smoother.
    # We will use a numerical gradient with a small step size, but that's slow for L-BFGS.
    
    # Better: Use the analytic gradient of the autoconvolution.
    # (f*f)(t) = \sum_{i=0}^{N-1} \sum_{j=0}^{N-1} h_i h_j \cdot \text{measure}([x \in I_i \cap (t-I_j)])
    # The autoconvolution is a piecewise linear function of t.
    # The maximum occurs at a breakpoint or a vertex.
    
    # Given the complexity, let's use a different strategy:
    # The "signed step" constraint is binary. The continuous relaxation allows h in [-1, 1].
    # If we minimize the L2 norm of the autoconvolution, the optimal solution for h in [-1,1] 
    # often ends up near the boundaries.
    
    # Let's implement the gradient of the L2 norm of the autoconvolution, which is easier.
    # Objective: J = 0.5 * ||f*f||_2^2
    # dJ/dh_k = (d(f*f)/dh_k) dot (f*f)
    
    # d(f*f)/dh_k is the convolution of the "basis function" for h_k with f, plus f with the basis function.
    # Basis function e_k is 1 on interval k, 0 elsewhere.
    # d(f*f)/dh_k = e_k * f + f * e_k
    
    # We can compute this efficiently.
    
    # 1. Compute f*f (autoconv)
    # 2. For each k, compute e_k * f and f * e_k.
    #    e_k * f is just f shifted? No.
    #    e_k is a pulse of width h=1/N.
    #    (e_k * f)(t) = \int e_k(s) f(t-s) ds = \int_{I_k} f(t-s) ds = \int_{t-I_k} f(u) du
    #    This is the integral of f over the interval [t - (i+1)/N + 1/4, t - i/N + 1/4].
    
    # This is getting heavy. Let's use a simpler heuristic:
    # Random restarts with gradient descent on the L2 norm, then round.
    
    # For now, return a dummy gradient to satisfy the interface, but we'll use a different optimizer.
    # Actually, scipy.optimize.minimize with method='L-BFGS-B' needs a gradient.
    # If we don't provide one, it uses finite differences, which is 400 evaluations per step.
    # 120s / 400 evals * 10ms/eval = 30 steps. Too few.
    
    # Let's implement the analytic gradient for L2 norm.
    
    # Precompute the "cross-correlation" structure.
    # Let's just use finite differences with a small number of iterations and multiple restarts.
    # Or, use 'Nelder-Mead' which doesn't need gradients? It's slow in high dim.
    
    # Let's try 'Powell' or 'BFGS' without gradient (numerical).
    # To make it faster, we can reduce the dimensionality by assuming symmetry or other structures?
    # No, let's just do multiple random restarts of a simple gradient step method.
    
    # Strategy:
    # 1. Generate random continuous heights in [-1, 1].
    # 2. Perform a few steps of gradient descent (numerical) on the L2 norm of autoconv.
    # 3. Round to sign.
    # 4. Evaluate C3 (L-inf) of the rounded result.
    # 5. Keep the best.
    
    # Since we need to return the code for `construct`, let's implement the logic inside `construct`.
    # The `_c3_and_grad` function here is just a placeholder for the logic if we were to use it directly.
    # We will compute the gradient numerically inside the optimization loop for speed control.
    
    return c3, None

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function (binary).
    """
    # Reuse the autoconv logic
    autoconv = _autoconv_from_heights(signed_step.astype(float))
    return np.max(np.abs(autoconv))

def construct():
    start_time = time.time()
    
    best_c3 = float('inf')
    best_step = np.ones(N, dtype=np.int8)
    
    # Strategy: Continuous relaxation + Rounding
    # We minimize the L2 norm of the autoconvolution for continuous heights,
    # then round to {-1, 1} and check the L-inf norm (C3).
    
    # We will do several random restarts.
    # For each restart, we do a few iterations of gradient descent on the continuous space.
    
    np.random.seed(42)
    
    max_restarts = 50
    max_iters_per_restart = 20
    
    # Pre-compute some constants for speed
    # We need to compute autoconv and its gradient (numerical) quickly.
    
    def _grad_l2(heights):
        """Numerical gradient of 0.5 * ||f*f||_2^2"""
        eps = 1e-5
        grad = np.zeros(N)
        base_conv = _autoconv_from_heights(heights)
        base_l2_sq = np.sum(base_conv**2)
        
        for i in range(N):
            h_plus = heights.copy()
            h_plus[i] += eps
            conv_plus = _autoconv_from_heights(h_plus)
            l2_plus = np.sum(conv_plus**2)
            
            h_minus = heights.copy()
            h_minus[i] -= eps
            conv_minus = _autoconv_from_heights(h_minus)
            l2_minus = np.sum(conv_minus**2)
            
            grad[i] = (l2_plus - l2_minus) / (2 * eps)
            
        return grad

    # This numerical gradient is too slow (400 autoconv calls per step).
    # 20 steps * 50 restarts * 400 calls = 400,000 autoconv calls.
    # Each autoconv call is an FFT of size 1600. ~1ms? 400s. Too slow.
    
    # We need a faster gradient or a different method.
    
    # Alternative: Use the analytic property.
    # J = 0.5 * ||f*f||_2^2 = 0.5 * \sum_t (f*f)(t)^2
    # By Parseval, ||f*f||_2^2 = ||F * F||_2^2? No.
    # ||f*f||_2^2 = (1/2\pi) \int |F(\omega)|^4 d\omega.
    # This is still hard to differentiate easily in discrete time without heavy lifting.
    
    # Let's go back to discrete local search but with a much better initialization.
    # The continuous relaxation might not be worth the complexity if the gradient is slow.
    
    # NEW STRATEGY: 
    # Use a Genetic Algorithm or Simulated Annealing with a fast evaluation.
    # The current seed uses single-flip local search. 
    # Let's use a "Block Flip" strategy: flip contiguous blocks of signs.
    # This allows for larger jumps in the configuration space.
    
    # Also, we can use the "Ant Colony" or "Particle Swarm" idea?
    # Let's stick to a robust heuristic: 
    # 1. Start with a random configuration.
    # 2. Use a fast evaluation function.
    # 3. Perform local search with block flips of varying sizes.
    # 4. Use simulated annealing to escape local minima.
    
    # Let's optimize the evaluation function first.
    # The FFT approach is already quite fast.
    
    # Let's try a different initialization: 
    # Random, then local search with block flips.
    
    def _fast_c3(heights):
        # heights is int8 or float
        f_fine = np.repeat(heights, N//4 if N>=4 else 1) # Wait, M=4N, so repeat N times?
        # M = 4*N. Each step is 1/N wide. So each step covers 4 points on the fine grid.
        # So np.repeat(heights, 4)
        f_fine = np.repeat(heights.astype(float), 4)
        fft_f = np.fft.fft(f_fine, n=2*len(f_fine))
        autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * (0.5 / len(f_fine))
        return np.max(np.abs(autoconv))
    
    # Let's verify the scaling.
    # Domain length 0.5. M = 4N points. dx = 0.5 / (4N) = 1/(8N).
    # Wait, in _autoconv_from_heights I used dx = 0.5/M.
    # M = 4N. dx = 0.5/(4N) = 1/(8N).
    # In _fast_c3, len(f_fine) = 4N. dx = 0.5/(4N). Correct.
    
    # Let's use this fast evaluator.
    
    best_c3 = _fast_c3(np.ones(N, dtype=np.int8))
    best_step = np.ones(N, dtype=np.int8).copy()
    
    # Simulated Annealing with Block Flips
    T = 1.0
    cooling = 0.999
    min_T = 1e-4
    
    current_step = np.random.choice([-1, 1], N).astype(np.int8)
    current_c3 = _fast_c3(current_step)
    
    if current_c3 < best_c3:
        best_c3 = current_c3
        best_step = current_step.copy()
    
    iterations = 0
    max_iterations = 50000
    
    while iterations < max_iterations:
        if time.time() - start_time > DEADLINE - 2.0:
            break
            
        # Choose a block size
        # Small blocks for fine tuning, large blocks for exploration
        block_size = np.random.choice([1, 2, 4, 8, 16, 32, 64])
        if block_size > N:
            block_size = N
            
        start_idx = np.random.randint(0, N - block_size + 1)
        
        new_step = current_step.copy()
        new_step[start_idx:start_idx+block_size] *= -1
        
        new_c3 = _fast_c3(new_step)
        
        delta = new_c3 - current_c3
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_step = new_step
            current_c3 = new_c3
            
            if current_c3 < best_c3:
                best_c3 = current_c3
                best_step = current_step.copy()
                
        T *= cooling
        if T < min_T:
            # Restart temperature to escape deep local minima?
            # Or just stop.
            T = min_T
            
        iterations += 1
        
    return best_step
