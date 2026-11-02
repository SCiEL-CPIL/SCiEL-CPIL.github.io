import numpy as np
import time

N = 400
DEADLINE = 140.0  # safety margin before 150s kill

def _compute_c3_bound(signs):
    """
    Compute an upper bound on C3 for a signed step function.
    Uses L1 norm of autoconvolution as the metric.
    """
    dx = 0.5 / N
    n_fft = 2 * N
    F = np.fft.rfft(signs.astype(float), n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft) * dx
    return np.sum(np.abs(conv)) * dx

def _spectral_cost(signs):
    """
    Compute spectral cost: sum of |F(k)|^2 weighted to approximate L1 autoconvolution.
    This is cheaper for gradient-based methods.
    """
    n_fft = 2 * N
    F = np.fft.rfft(signs.astype(float), n=n_fft)
    # L1 norm of autoconvolution relates to integral of |F|^2
    # Use a proxy: sum of |F(k)|^2 with appropriate weighting
    return np.sum(np.abs(F) ** 2)

def _low_discrepancy_init(seed):
    """
    Generate a sign sequence from a low-discrepancy sequence (Halton-like).
    Uses the fractional parts of sqrt(2)*n to create a quasi-random pattern.
    """
    rng = np.random.RandomState(seed)
    # Use a combination of low-frequency and quasi-random patterns
    x = np.linspace(0, 1, N, endpoint=False)
    
    # Halton-like sequence in base 2
    halton = np.zeros(N)
    for i in range(N):
        f = 0.5
        n = i + 1
        while n > 0:
            f = 0.5 * f
            if n % 2 == 1:
                halton[i] += f
            n //= 2
    
    # Combine with a smooth low-frequency component
    pattern = np.sin(2 * np.pi * 3 * x) + 0.5 * np.sin(2 * np.pi * 7 * x) + 0.3 * (2 * halton - 1)
    signs = np.sign(pattern)
    signs[signs == 0] = 1
    return signs

def _gradient_descent_optimize(signs, n_iter=500, lr=0.1, deadline=None):
    """
    Gradient descent in continuous space, then threshold to ±1.
    This allows smoother exploration than discrete flips.
    """
    current = signs.astype(np.float64).copy()
    current_score = _compute_c3_bound(current)
    best = current.copy()
    best_score = current_score
    
    for i in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradient via finite differences on a few coordinates
        # Use FFT-based gradient: d/dx_k of sum|conv| where conv = IFFT(|FFT(x)|^2)
        # Approximate gradient using spectral representation
        n_fft = 2 * N
        F = np.fft.rfft(current, n=n_fft)
        
        # The autoconvolution is IFFT(F^2)
        # Gradient of sum|IFFT(F^2)| w.r.t. x is complex; use subgradient approximation
        conv = np.fft.irfft(F * F, n=n_fft)
        
        # Subgradient of |conv| w.r.t. conv
        subgrad_conv = np.sign(conv)
        
        # Backpropagate through IFFT and the F^2 relation
        # d(conv)/d(F) = 2*F*IFFT basis, then d(F)/d(x) = FFT basis
        # Approximate: gradient in x-space = FFT(2*F*IFFT(subgrad_conv)) / N_fft
        grad_freq = 2 * F * np.fft.rfft(subgrad_conv, n=n_fft)
        grad_x = np.fft.irfft(grad_freq, n=n_fft)[:N]
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad_x)
        if grad_norm > 1e-10:
            grad_x = grad_x / grad_norm
        
        # Update with learning rate decay
        lr_i = lr / (1 + i * 0.01)
        new_current = current - lr_i * grad_x
        
        # Project back to ±1 by thresholding
        new_signs = np.sign(new_current)
        new_signs[new_signs == 0] = 1
        
        new_score = _compute_c3_bound(new_signs)
        
        if new_score < current_score:
            current = new_signs
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best = current.copy()
    
    return best, best_score

def _greedy_flip_optimize(signs, max_rounds=20):
    """
    Greedy coordinate descent: flip individual signs if it reduces the C3 bound.
    Used as a fast local refinement after SA.
    """
    current = signs.astype(np.float64).copy()
    current_score = _compute_c3_bound(current)
    
    for _ in range(max_rounds):
        improved = False
        for i in range(N):
            old_val = current[i]
            current[i] = -old_val
            new_score = _compute_c3_bound(current)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                current[i] = old_val
        if not improved:
            break
    
    return current, current_score

def _block_flip_move(signs, block_size, rng):
    """
    Flip a random contiguous block of signs.
    Returns new signs array.
    """
    new_signs = signs.copy()
    start = rng.randint(0, N - block_size + 1)
    new_signs[start:start + block_size] = -new_signs[start:start + block_size]
    return new_signs

def _simulated_annealing(signs, start_temp, end_temp, n_iter, block_sizes=None, seed=12345, deadline=None):
    """
    Simulated annealing with block-flip moves.
    """
    if block_sizes is None:
        block_sizes = [1, 2, 4, 8, 16, 32, 64]
    
    rng = np.random.RandomState(seed)
    current = signs.astype(np.float64).copy()
    current_score = _compute_c3_bound(current)
    best = current.copy()
    best_score = current_score
    
    for i in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        temp = start_temp * (end_temp / start_temp) ** (i / max(n_iter - 1, 1))
        
        # Choose a random block size with bias toward smaller blocks
        bs = block_sizes[rng.choice(len(block_sizes), p=[0.20, 0.20, 0.15, 0.15, 0.10, 0.10, 0.10])]
        if bs > N:
            bs = N
        
        new_signs = _block_flip_move(current, bs, rng)
        new_score = _compute_c3_bound(new_signs)
        
        delta = new_score - current_score
        if delta < 0 or (temp > 0 and rng.random() < np.exp(-delta / max(temp, 1e-10))):
            current = new_signs
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best = current.copy()
    
    return best, best_score

def construct():
    start_time = time.time()
    
    # Diverse initializations using low-discrepancy and structured patterns
    rng = np.random.RandomState(42)
    
    initializations = []
    
    # Low-discrepancy based initializations (different seeds)
    for seed in [1, 7, 13, 23, 42, 99]:
        initializations.append(_low_discrepancy_init(seed))
    
    # All ones
    initializations.append(np.ones(N))
    
    # Alternating
    initializations.append(np.array([1 if i % 2 == 0 else -1 for i in range(N)]))
    
    # Random
    initializations.append(rng.choice([1, -1], size=N))
    
    # Block structure
    p4 = np.ones(N)
    p4[N//4:N//2] = -1
    p4[3*N//4:] = -1
    initializations.append(p4)
    
    # Multiple frequency patterns
    x = np.linspace(-1/4, 1/4, N, endpoint=False)
    for freq in [2, 3, 4, 5, 6]:
        p = np.sign(np.cos(2 * np.pi * freq * x))
        p[p == 0] = 1
        initializations.append(p)
    
    # Find best initialization
    best_pattern = initializations[0]
    best_score = _compute_c3_bound(best_pattern)
    
    for init in initializations[1:]:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 30:
            break
        s = _compute_c3_bound(init)
        if s < best_score:
            best_score = s
            best_pattern = init
    
    # Primary optimization: gradient descent followed by greedy polish
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 40:
        deadline_for_gd = start_time + DEADLINE - 30
        gd_result, gd_score = _gradient_descent_optimize(
            best_pattern, n_iter=300, lr=0.2, deadline=deadline_for_gd
        )
        if gd_score < best_score:
            best_score = gd_score
            best_pattern = gd_result
        
        # Polish with greedy
        if time.time() < deadline_for_gd - 10:
            polished, pol_score = _greedy_flip_optimize(best_pattern, max_rounds=15)
            if pol_score < best_score:
                best_score = pol_score
                best_pattern = polished
    
    # Secondary: SA from a different random start if time permits
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 35:
        rand_start = rng.choice([1, -1], size=N)
        rand_score = _compute_c3_bound(rand_start)
        if rand_score < best_score:
            best_score = rand_score
            best_pattern = rand_start
        
        deadline_for_sa = start_time + DEADLINE - 20
        optimized2, opt_score2 = _simulated_annealing(
            best_pattern, start_temp=0.3, end_temp=0.01,
            n_iter=1000, seed=99999, deadline=deadline_for_sa
        )
        if opt_score2 < best_score:
            best_score = opt_score2
            best_pattern = optimized2
        
        # Polish again
        if time.time() < deadline_for_sa - 5:
            polished2, pol_score2 = _greedy_flip_optimize(best_pattern, max_rounds=10)
            if pol_score2 < best_score:
                best_score = pol_score2
                best_pattern = polished2
    
    # Final polish if time remains
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10:
        final_polish, final_score = _greedy_flip_optimize(best_pattern, max_rounds=5)
        if final_score < best_score:
            best_score = final_score
            best_pattern = final_polish
    
    # Ensure we return integer signs
    return np.sign(best_pattern).astype(int)
