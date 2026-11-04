import numpy as np
import time
import scipy.optimize

N = 400

def _autoconv_lin(x):
    """Compute linear autoconvolution using FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _metric_offdiag(x):
    """Metric: max |c_k| for k != 0 / S^2"""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    if len(c) > 1:
        return float(np.max(np.abs(c[1:]))) / (s * s)
    return 0.0

def _soft_max_abs(c, beta):
    """Smooth approximation to max|c| using log-sum-exp."""
    abs_c = np.abs(c)
    if len(abs_c) == 0:
        return 0.0
    m = np.max(abs_c)
    if m < 1e-15:
        return 0.0
    return m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta

def _objective_soft(x, beta, deadline):
    """Objective function for L-BFGS-B with time guard."""
    if time.time() > deadline:
        return 1e10
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_lin(x)
    if len(c) <= 1:
        return 0.0
    abs_c = np.abs(c[1:])
    soft_max = _soft_max_abs(abs_c, beta)
    return soft_max / (s * s)

def _generate_random_phase_signal(n, seed):
    """Generate a signal with flat spectrum and random phases."""
    rng = np.random.default_rng(seed)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    phases = rng.uniform(-np.pi, np.pi, size // 2 + 1)
    mag = np.ones(size // 2 + 1)
    spectrum = mag * np.exp(1j * phases)
    spectrum[0] = mag[0]
    if size % 2 == 0:
        spectrum[-1] = mag[-1]
    x_full = np.fft.irfft(spectrum, size)
    x = x_full[:n]
    return x

def _generate_sine_sum(n, K, seed):
    """Generate a signal as sum of K random sines."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    x = np.zeros(n)
    for _ in range(K):
        f = rng.uniform(1, n/2)
        p = rng.uniform(0, 2*np.pi)
        a = rng.uniform(0.5, 1.5)
        x += a * np.sin(2 * np.pi * f * t / n + p)
    return x

def _block_signal(params, n):
    """
    Convert block parameters to a time-domain signal.
    params: array of [start, width, amp] for each block.
    start: continuous position in [0, n)
    width: continuous width in [0, n]
    amp: amplitude
    """
    x = np.zeros(n)
    n_blocks = len(params) // 3
    for i in range(n_blocks):
        start = params[3*i]
        width = params[3*i+1]
        amp = params[3*i+2]
        
        if width <= 0:
            continue
            
        # Create a top-hat with continuous edges using linear interpolation
        # We define the signal as 1.0 in [start, start+width] with smooth transitions
        # For simplicity, let's use a step function but allow sub-pixel positioning via interpolation
        # Actually, the task says "signed step heights", implying discrete samples.
        # But the directive suggests "continuous positions".
        # We will sample the continuous top-hat function at integer points 0..n-1.
        
        # Define the continuous function f(t) = amp if start <= t < start+width else 0
        # But to allow smooth optimization, let's use a smoothed step (sigmoid) or just linear ramp?
        # Let's stick to a simple top-hat sampled at integers, but allow start/width to be real.
        # This creates a piecewise constant signal that changes as start/width cross integer boundaries.
        # This is non-differentiable.
        # Better: Use a smooth approximation of the top-hat, e.g., a raised cosine or just a sigmoid transition.
        # However, the task asks for "step functions". 
        # If we optimize continuous params, the resulting discrete signal is piecewise constant in the params.
        # This might be hard for gradient-based methods.
        # Let's try a different interpretation: The signal is a sum of discrete rectangular blocks, 
        # but we optimize the amplitudes and possibly the positions via discrete search?
        # No, the directive says "optimizing these geometric parameters via differential evolution".
        # Differential evolution is a global optimizer that doesn't need gradients.
        # So we can define the discrete signal from continuous params by rounding or sampling.
        
        # Let's sample the ideal continuous top-hat at the discrete points.
        # f(t) = amp * H(t - start) * H(start + width - t)
        # where H is the Heaviside step.
        # At integer t, this is amp if start <= t < start+width.
        
        t = np.arange(n, dtype=float)
        mask = (t >= start) & (t < start + width)
        x[mask] += amp
        
    return x

def _block_objective(params, deadline):
    """Objective for block parameter optimization."""
    if time.time() > deadline:
        return 1e10
        
    n_blocks = len(params) // 3
    x = _block_signal(params, N)
    
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
        
    c = _autoconv_lin(x)
    if len(c) <= 1:
        return 0.0
    abs_c = np.abs(c[1:])
    # Use a high beta soft-max for a tighter approximation during DE
    beta = 100.0
    soft_max = _soft_max_abs(abs_c, beta)
    return soft_max / (s * s)

def _optimize_blocks(n_blocks, deadline, pop_size=30, max_iter=50):
    """
    Optimize block parameters using Differential Evolution.
    """
    n_params = n_blocks * 3
    
    # Bounds:
    # start: [0, N]
    # width: [1, N] (at least 1 sample wide)
    # amp: [-1, 1]
    
    bounds = []
    for _ in range(n_blocks):
        bounds.append((0.0, float(N)))
        bounds.append((1.0, float(N)))
        bounds.append((-1.0, 1.0))
    
    # Initial population
    rng = np.random.default_rng(int(time.time()))
    pop = np.zeros((pop_size, n_params))
    for i in range(pop_size):
        for j in range(n_blocks):
            pop[i, 3*j] = rng.uniform(0, N)
            pop[i, 3*j+1] = rng.uniform(1, N)
            pop[i, 3*j+2] = rng.uniform(-1, 1)
    
    best_params = pop[0].copy()
    best_score = _block_objective(best_params, deadline)
    
    # Simple DE loop (DE/rand/1/bin)
    F = 0.8
    CR = 0.9
    
    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
            
        for i in range(pop_size):
            # Select distinct random indices
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            if i in [r1, r2, r3]:
                # Re-select if collision (simple retry)
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                if i in [r1, r2, r3]:
                    continue
            
            # Mutation
            v = pop[r1] + F * (pop[r2] - pop[r3])
            
            # Clip to bounds
            for j in range(n_params):
                low, high = bounds[j]
                v[j] = np.clip(v[j], low, high)
            
            # Crossover
            trial = pop[i].copy()
            for j in range(n_params):
                if rng.random() < CR or j == np.random.randint(n_params):
                    trial[j] = v[j]
            
            # Selection
            score_trial = _block_objective(trial, deadline)
            if score_trial < best_score:
                best_score = score_trial
                best_params = trial.copy()
                
            # Compare with current individual
            # We need the score of pop[i]. We didn't store it. Let's compute it.
            score_i = _block_objective(pop[i], deadline)
            if score_trial < score_i:
                pop[i] = trial
            else:
                # Keep old
                pass
                
    return best_params, best_score

def _refine_lbfgs(x_init, deadline, beta_start=10.0, beta_end=200.0, n_anneal_steps=6):
    """
    Refine x_init using L-BFGS-B with annealed soft-max beta.
    """
    x = x_init.copy()
    
    for step in range(n_anneal_steps):
        if time.time() > deadline - 2.0:
            break
        
        t_anneal = step / max(1, n_anneal_steps - 1)
        beta = beta_start + (beta_end - beta_start) * t_anneal
        
        def obj(x):
            return _objective_soft(x, beta, deadline)
        
        try:
            result = scipy.optimize.minimize(
                obj,
                x,
                method='L-BFGS-B',
                options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-10},
                tol=1e-12
            )
            x = result.x
        except Exception:
            pass

    return x

def _polish_subgradient(x, deadline, max_iters=50):
    """
    Polish using subgradient descent on the exact max |c_k| with adaptive step sizing.
    """
    x = x.copy()
    best_x = x.copy()
    best_score = _metric_offdiag(x)
    
    recent_scores = []
    base_step = 0.05
    
    for i in range(max_iters):
        if time.time() > deadline - 1.0:
            break
            
        s = np.sum(x)
        if abs(s) < 1e-10:
            break
            
        c = _autoconv_lin(x)
        abs_c = np.abs(c[1:])
        
        threshold = np.percentile(abs_c, 85)
        active_mask = abs_c > threshold
        if not np.any(active_mask):
            break
            
        grad = np.zeros_like(x)
        active_indices = np.where(active_mask)[0]
        
        for k_idx in active_indices:
            k = k_idx + 1
            sign_c = np.sign(c[k])
            if sign_c == 0:
                sign_c = 1.0
            
            if k < N:
                grad[:N-k] += sign_c * x[k:]
            if k < N:
                grad[k:] += sign_c * x[:N-k]
        
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-15:
            break
            
        if len(recent_scores) >= 3:
            recent = np.array(recent_scores[-3:])
            if np.all(recent < best_score * 1.01):
                base_step = min(base_step * 1.2, 0.1)
            else:
                base_step = max(base_step * 0.7, 0.005)
        
        step_size = base_step * (1.0 - i/max_iters * 0.5)
        x_new = x - step_size * grad / gnorm
        
        sc_new = _metric_offdiag(x_new)
        recent_scores.append(sc_new)
        
        if sc_new < best_score:
            best_score = sc_new
            best_x = x_new.copy()
            x = x_new
        else:
            for shrink in [0.5, 0.25, 0.1]:
                x_try = x - step_size * shrink * grad / gnorm
                sc_try = _metric_offdiag(x_try)
                if sc_try < best_score:
                    best_score = sc_try
                    best_x = x_try.copy()
                    x = x_try
                    break

    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_x = None
    best_score = float('inf')
    
    def update_best(x, score):
        nonlocal best_x, best_score
        if x is not None and score < best_score:
            best_score = score
            best_x = x.copy()
            
    candidates = []
    
    # Phase 1: Block-based generation
    # Try different numbers of blocks
    for n_blocks in [2, 3, 4, 5]:
        if time.time() > deadline - 30.0:
            break
        # Allocate time per block count
        time_per_block = min(15.0, (deadline - time.time() - 10.0) / max(1, (5 - n_blocks)))
        block_deadline = time.time() + time_per_block
        
        params, sc = _optimize_blocks(n_blocks, block_deadline, pop_size=20, max_iter=30)
        x = _block_signal(params, N)
        sc = _metric_offdiag(x)
        update_best(x, sc)
        
        # Refine the block result with L-BFGS
        if time.time() < deadline - 10.0:
            x_ref = _refine_lbfgs(x, deadline)
            sc_ref = _metric_offdiag(x_ref)
            update_best(x_ref, sc_ref)
            
            # Polish
            if time.time() < deadline - 5.0:
                x_pol, sc_pol = _polish_subgradient(x_ref, deadline)
                update_best(x_pol, sc_pol)

    # Phase 2: Traditional random phase candidates as fallback/diversity
    for seed in [1, 2, 3]:
        if time.time() > deadline - 15.0:
            break
        x = _generate_random_phase_signal(N, seed)
        sc = _metric_offdiag(x)
        update_best(x, sc)
        
        if time.time() < deadline - 10.0:
            x_ref = _refine_lbfgs(x, deadline)
            sc_ref = _metric_offdiag(x_ref)
            update_best(x_ref, sc_ref)
            
            if time.time() < deadline - 5.0:
                x_pol, sc_pol = _polish_subgradient(x_ref, deadline)
                update_best(x_pol, sc_pol)

    # Final polish on best
    if best_x is not None and time.time() < deadline - 2.0:
        x_final, sc_final = _polish_subgradient(best_x, deadline, max_iters=80)
        update_best(x_final, sc_final)
        
        if time.time() < deadline - 3.0:
            x_precise = _refine_lbfgs(best_x, deadline, beta_start=200.0, beta_end=500.0, n_anneal_steps=3)
            sc_precise = _metric_offdiag(x_precise)
            update_best(x_precise, sc_precise)
            
            if time.time() < deadline - 2.0:
                x_final2, sc_final2 = _polish_subgradient(best_x, deadline, max_iters=30)
                update_best(x_final2, sc_final2)
        
    if best_x is None:
        return np.ones(N)
        
    return best_x
