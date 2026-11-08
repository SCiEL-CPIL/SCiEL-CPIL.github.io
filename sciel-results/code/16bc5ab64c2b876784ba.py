"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys


def _autoconvolve(x):
    """Compute the circular autoconvolution of x using FFT."""
    n = len(x)
    fft_x = np.fft.fft(x)
    ac = np.fft.ifft(fft_x * np.conj(fft_x)).real
    return ac


def _compute_c3(x):
    """Compute the C3 upper bound: max of autoconvolution of sign(x)."""
    signs = np.sign(x)
    # Ensure no zeros; map 0 to 1
    signs[signs == 0] = 1.0
    ac = _autoconvolve(signs)
    return np.max(ac)


def _soft_linf(v, tau=0.05):
    """Smooth approximation of L-infinity norm via log-sum-exp."""
    v_shifted = v - np.max(v)
    return tau * np.log(np.sum(np.exp(v_shifted / tau)))


def construct():
    deadline = time.time() + 145  # 150s wall-clock limit, 5s safety margin
    
    best_x = None
    best_c3 = np.inf
    
    # Multiple random restarts with projected gradient descent
    rng = np.random.default_rng(42)
    
    num_restarts = 30
    steps_per_restart = 500
    tau = 0.05  # temperature for soft l-inf
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize: random signs with some structure
        # Start with alternating or random patterns
        if restart == 0:
            x = np.ones(N)
        elif restart == 1:
            x = np.ones(N)
            x[::2] = -1.0
        elif restart == 2:
            x = np.ones(N)
            x[:N//2] = -1.0
        else:
            # Random initialization with bias
            x = rng.choice([-1.0, 1.0], size=N)
        
        # Projected gradient descent on continuous relaxation
        # Minimize soft_linf(autoconvolve(x)) subject to x in [-1, 1]^N
        # Use subgradient of soft l-inf for autoconvolution
        
        step_size = 0.1
        patience = 0
        best_x_iter = x.copy()
        best_c3_iter = _compute_c3(x)
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            # Compute gradient via finite differences or analytical
            # Use FFT-based autoconvolution gradient
            # d/dx soft_linf(autoconv(x))
            
            ac = _autoconvolve(x)
            # Gradient of soft l-inf: softmax weights * gradient of autoconv
            weights = np.exp((ac - np.max(ac)) / tau)
            weights /= np.sum(weights)
            
            # Gradient of autoconvolution w.r.t. x
            # autoconv(x) = ifft(fft(x) * conj(fft(x)))
            # d/dx_k = ifft(fft(2*x) * delta_k) ... use FFT property
            # Simpler: use the fact that autoconv is quadratic
            # grad_k = 2 * sum_j x_j * delta_{k-j} in circular sense
            # Which is 2 * circular_convolve(x, x)[k]... wait, that's the same as autoconv?
            
            # Actually, for circular autoconvolution c = conv(x, x),
            # dc/dx_k = 2 * conv(x, e_k) where e_k is unit vector
            # conv(x, e_k) shifts x by k
            # So grad = 2 * (x rolled by various amounts)
            
            # More efficient: grad_k = 2 * sum_j x_{(j-k) mod N} * x_j / N ... 
            # Let me just use: the gradient of ||autoconv(x)||^2 would be simpler,
            # but we want soft l-inf.
            
            # For now, use a simple approach: gradient of soft_linf composed with autoconv
            # Use chain rule with FFT
            
            fft_x = np.fft.fft(x)
            # d(ac)_k/dx = 2 * ifft(fft(x) * conj(fft(e_k))) where e_k is basis
            # This is complex; let's use a simpler gradient estimate
            
            # Alternative: use the fact that for small perturbations,
            # we can compute directional derivatives
            
            # Let's use a different approach: optimize the signs directly
            # with local search, but use the continuous relaxation for guidance
            
            # Actually, let's just do local search on signs with multi-flip moves
            # since that's what worked before (score 1.001)
            
            # Break out of gradient descent and use local search instead
            break
        
        # Local search phase
        x_signs = np.sign(x)
        x_signs[x_signs == 0] = 1.0
        current_c3 = _compute_c3(x_signs)
        
        if current_c3 < best_c3:
            best_c3 = current_c3
            best_x = x_signs.copy()
        
        # Multi-flip local search
        max_local_steps = 2000
        no_improve = 0
        temperature = 1.0
        
        for ls_step in range(max_local_steps):
            if time.time() > deadline:
                break
            
            # Try multi-flip moves (1-3 simultaneous flips)
            num_flips = rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
            flips = rng.choice(N, size=num_flips, replace=False)
            
            x_trial = x_signs.copy()
            x_trial[flips] *= -1.0
            
            c3_trial = _compute_c3(x_trial)
            
            if c3_trial < current_c3:
                x_signs = x_trial
                current_c3 = c3_trial
                no_improve = 0
                
                if current_c3 < best_c3:
                    best_c3 = current_c3
                    best_x = x_signs.copy()
            else:
                no_improve += 1
                # Simulated annealing: accept worse moves with decreasing probability
                if np.random.random() > np.exp(-(c3_trial - current_c3) / temperature):
                    x_signs = x_trial
                    current_c3 = c3_trial
            
            # Cool down
            temperature *= 0.999
            if temperature < 0.01:
                temperature = 0.01
        
        if no_improve > 200:
            break
    
    # Final high-precision polish: single-flip search until no improvement
    if best_x is not None:
        x_final = best_x.copy()
        c3_final = _compute_c3(x_final)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                x_trial = x_final.copy()
                x_trial[i] *= -1.0
                c3_trial = _compute_c3(x_trial)
                if c3_trial < c3_final:
                    x_final = x_trial
                    c3_final = c3_trial
                    improved = True
                    best_c3 = c3_final
                    best_x = x_final.copy()
    
    if best_x is None:
        return np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
