import numpy as np
import time

N = 600
DELTA_T = (1.0/4 - (-1.0/4)) / N  # = 1/1200

def _compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    The score is max_k (autoconv_k) / (sum_i h_i)^2
    """
    # Ensure nonnegativity
    h = np.maximum(h, 0)
    
    if np.all(h == 0):
        return 0.0  # degenerate case
    
    # Compute autoconvolution via FFT
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    autoconv = np.fft.irfft(H * H, n=n_fft)
    
    # The maximum of the autoconvolution
    max_autoconv = np.max(autoconv)
    
    # Normalization: (sum h_i * delta_t)^2
    integral = np.sum(h) * DELTA_T
    norm = integral ** 2
    
    if norm < 1e-30:
        return 0.0
    
    return max_autoconv / norm

def _compute_score_and_grad(h):
    """
    Compute score and its gradient with respect to h.
    Uses FFT for efficient computation.
    """
    h = np.maximum(h, 0)
    
    if np.all(h == 0):
        return 0.0, np.zeros_like(h)
    
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    autoconv = np.fft.irfft(H * H, n=n_fft)
    
    max_autoconv = np.max(autoconv)
    
    integral = np.sum(h) * DELTA_T
    norm = integral ** 2
    
    if norm < 1e-30:
        return 0.0, np.zeros_like(h)
    
    score = max_autoconv / norm
    
    # Gradient computation
    # score = max_k autoconv[k] / (sum h * delta_t)^2
    # Let m = max_k autoconv[k], I = sum h * delta_t
    # score = m / I^2
    # dm/dh = autoconv' at the argmax (but we need to handle the max)
    # dI/dh = delta_t * ones
    
    # For simplicity, use subgradient: at the argmax index
    k_max = np.argmax(autoconv)
    
    # The autoconvolution is a linear convolution: autoconv[k] = sum_j h[j] * h[k-j] (with appropriate scaling)
    # Actually, autoconv = h * h (linear convolution)
    # d(autoconv[k])/dh[i] = h[k-i] + h[i-k] (if we consider the symmetry)
    
    # More carefully: autoconv = ifft(fft(h) * fft(h))
    # d(autoconv[k])/dh[i] = h[k-i] + h[i-k] for valid indices
    
    # Let's compute the gradient numerically or use the chain rule
    # For now, use a simple finite difference approach for the gradient
    # Or better: use the fact that autoconv[k] = sum_j h[j] * h[k-j]
    
    # Actually, let's just use the subgradient approach
    # The gradient of max_k autoconv[k] w.r.t. h is the vector that has the gradient of autoconv at k_max
    # and zeros elsewhere (subgradient)
    
    # d(autoconv[k_max])/dh[i] = h[k_max - i] + h[i - k_max] (for valid indices, 0 otherwise)
    grad_max = np.zeros(N)
    for i in range(N):
        j1 = k_max - i
        j2 = i - k_max
        val = 0.0
        if 0 <= j1 < N:
            val += h[j1]
        if 0 <= j2 < N:
            val += h[j2]
        grad_max[i] = val
    
    # d(score)/dh = (I^2 * grad_max - m * 2*I * delta_t * ones) / I^4
    #             = grad_max / I^2 - 2*m*delta_t/(I^3) * ones
    #             = grad_max / I^2 - 2*score*delta_t/(I^2) * ones
    #             = (grad_max - 2*score*delta_t*ones) / I^2
    
    ones = np.ones(N)
    grad = (grad_max - 2 * score * DELTA_T * ones) / (norm / DELTA_T**2)
    # Wait, let me redo this
    # score = m / I^2 where I = sum(h) * delta_t
    # d(score)/dh[i] = (dm/dh[i] * I^2 - m * 2*I * dI/dh[i]) / I^4
    #                = dm/dh[i] / I^2 - 2*m*dI/dh[i] / I^3
    # dI/dh[i] = delta_t
    # So: d(score)/dh[i] = grad_max[i] / I^2 - 2*m*delta_t / I^3
    #                = (grad_max[i] * I - 2*m*delta_t) / I^3
    #                = (grad_max[i] * sum(h)*delta_t - 2*m*delta_t) / (sum(h)*delta_t)^3
    
    I = integral
    grad = (grad_max * I - 2 * max_autoconv * DELTA_T) / (I**3)
    
    return score, grad

def _ps_optimize(h_init, max_iter=1000, n_particles=20, learning=0.5, inertia=0.9):
    """
    Particle Swarm Optimization for minimizing the score.
    """
    n_params = len(h_init)
    
    # Initialize particles
    particles = np.zeros((n_particles, n_params))
    velocities = np.zeros((n_particles, n_params))
    p_best = np.zeros((n_particles, n_params))
    p_best_scores = np.zeros(n_particles)
    
    global_best = h_init.copy()
    global_best_score = _compute_score(h_init)
    
    # Initialize particles around the initial solution
    for i in range(n_particles):
        particles[i] = h_init + 0.1 * np.random.randn(n_params)
        particles[i] = np.maximum(particles[i], 0)
        
        score = _compute_score(particles[i])
        p_best[i] = particles[i].copy()
        p_best_scores[i] = score
        
        if score < global_best_score:
            global_best = particles[i].copy()
            global_best_score = score
    
    # Main PSO loop
    for iteration in range(max_iter):
        for i in range(n_particles):
            # Update velocity
            r1 = np.random.rand(n_params)
            r2 = np.random.rand(n_params)
            velocities[i] = (inertia * velocities[i] + 
                           learning * r1 * (p_best[i] - particles[i]) + 
                           learning * r2 * (global_best - particles[i]))
            
            # Update position
            particles[i] = particles[i] + velocities[i]
            particles[i] = np.maximum(particles[i], 0)
            
            # Evaluate
            score = _compute_score(particles[i])
            
            # Update personal best
            if score < p_best_scores[i]:
                p_best[i] = particles[i].copy()
                p_best_scores[i] = score
                
                # Update global best
                if score < global_best_score:
                    global_best = particles[i].copy()
                    global_best_score = score
        
        # Damping to avoid divergence
        velocities *= 0.95
    
    return global_best, global_best_score

def construct():
    """
    Find a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    """
    start_time = time.time()
    deadline = 140.0  # seconds (10s safety margin)
    
    # Start with uniform distribution
    h_best = np.ones(N)
    score_best = _compute_score(h_best)
    
    # Strategy 1: Try some heuristic initializations
    x_grid = np.linspace(-0.25, 0.25, N)
    
    heuristics = [
        np.ones(N),  # uniform
        np.exp(-20 * x_grid**2),  # narrow Gaussian
        np.exp(-10 * x_grid**2),  # medium Gaussian
        np.exp(-5 * x_grid**2),  # wide Gaussian
        1 + 0.5 * np.cos(4 * np.pi * x_grid),  # periodic
        1 + 0.3 * np.cos(8 * np.pi * x_grid),  # higher frequency
    ]
    
    for h_init in heuristics:
        if time.time() > deadline - 20:
            break
        
        h_init = np.maximum(h_init, 0)
        score = _compute_score(h_init)
        
        if score < score_best:
            score_best = score
            h_best = h_init.copy()
    
    # Strategy 2: Use Particle Swarm Optimization
    if time.time() < deadline - 10:
        try:
            h_pso, score_pso = _ps_optimize(h_best, max_iter=500, n_particles=15, learning=0.3, inertia=0.8)
            
            if score_pso < score_best:
                score_best = score_pso
                h_best = h_pso.copy()
        except Exception:
            pass
    
    # Strategy 3: Local refinement with gradient descent
    if time.time() < deadline - 5:
        h_curr = h_best.copy()
        lr = 0.01
        for step in range(200):
            if time.time() > deadline - 5:
                break
            
            score, grad = _compute_score_and_grad(h_curr)
            
            if np.all(np.abs(grad) < 1e-6):
                break
            
            # Gradient step with clipping
            h_new = h_curr - lr * grad
            h_new = np.maximum(h_new, 0)
            
            new_score = _compute_score(h_new)
            
            if new_score < score:
                h_curr = h_new
                score_best = min(score_best, new_score)
                h_best = h_curr.copy()
            else:
                lr *= 0.5
        
        if _compute_score(h_curr) < score_best:
            score_best = _compute_score(h_curr)
            h_best = h_curr.copy()
    
    # Final: ensure we return a valid nonnegative array
    h_best = np.maximum(h_best, 0)
    
    # If all zeros, fall back to uniform
    if np.all(h_best == 0):
        h_best = np.ones(N)
    
    return [float(x) for x in h_best]
