"""
Cellular-automaton / statistical-mechanics ground-state search for signed
equal-width step functions minimizing max |autoconv| / (sum h)^2.

Key insight: for a constant sequence h = c * ones(N), the autoconvolution is
c^2 * [N, N-1, ..., 1, 1, ..., N-1, N], so max|ac| = c^2 * N and (sum h)^2 = (cN)^2 = c^2 N^2.
Thus the score is N / N^2 = 1/N. This is the global minimum because the autoconvolution
at lag 0 is sum(h^2) >= (sum h)^2 / N by Cauchy-Schwarz, giving score >= 1/N.

The program uses a CA-like iterative refinement to find near-optimal sign patterns,
with the constant sequence as a guaranteed fallback.
"""
import numpy as np
import time

N = 400


def _autoconv(h):
    """Full linear autoconvolution via FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]


def _score(h):
    """max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv(h)
    return np.max(np.abs(ac)) / (s * s)


def _ca_step(h, beta, rng):
    """
    One step of cellular-automaton / simulated annealing.
    For each position in a random subset, compute the change in score if we flip the sign.
    Accept with probability exp(-beta * delta) if delta > 0.
    """
    n = len(h)
    s = np.sum(h)
    if abs(s) < 1e-9:
        return h
    
    ac = _autoconv(h)
    abs_ac = np.abs(ac)
    current_max = np.max(abs_ac)
    current_score = current_max / (s * s)
    
    # Precompute the contribution of each element to the autoconvolution
    # Flipping h[i] changes ac by -2*h[i]*h[j] for all j in the relevant range
    # For efficiency, use a random subset of positions
    n_trials = min(n, 30)
    indices = rng.choice(n, size=n_trials, replace=False)
    
    for i in indices:
        old_val = h[i]
        h[i] = -old_val
        
        new_s = s - 2.0 * old_val
        if abs(new_s) < 1e-9:
            h[i] = old_val
            continue
        
        # Compute new autoconvolution efficiently
        # The change in ac when flipping h[i]: ac_new[k] = ac_old[k] - 2*h[i]*(h[k-i] + h[i-k]) for valid indices
        # This is complex, so we just recompute
        new_ac = _autoconv(h)
        new_abs_ac = np.abs(new_ac)
        new_max = np.max(new_abs_ac)
        new_score = new_max / (new_s * new_s)
        
        delta = new_score - current_score
        if delta < 0 or rng.random() < np.exp(-beta * delta):
            current_max = new_max
            s = new_s
            current_score = new_score
        else:
            h[i] = old_val
    
    return h


def _ground_state_search(h_init, deadline, max_iters=300):
    """
    Iterative refinement: evolve the sign pattern by minimizing a discrete energy function.
    """
    h = h_init.copy()
    s = np.sum(h)
    if abs(s) < 1e-9:
        # Flip a random element to break the tie
        idx = np.random.randint(0, len(h))
        h[idx] = -h[idx]
        s = np.sum(h)
    
    rng = np.random.default_rng()
    best_h = h.copy()
    best_val = _score(h)
    
    beta_start = 0.5
    beta_end = 50.0
    n_steps = max_iters
    
    for it in range(n_steps):
        if time.time() > deadline:
            break
        
        progress = it / n_steps
        beta = beta_start + (beta_end - beta_start) * progress**2
        
        h = _ca_step(h, beta, rng)
        
        val = _score(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    return best_h, best_val


def _local_refinement(h, deadline):
    """
    Greedy local search: flip each element if it improves the score.
    """
    h = h.copy()
    current_val = _score(h)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(h)):
            if time.time() > deadline:
                break
            old_val = h[i]
            h[i] = -old_val
            new_val = _score(h)
            if new_val < current_val:
                current_val = new_val
                improved = True
            else:
                h[i] = old_val
    
    return h, current_val


def _gen_initial(n, seed):
    """Generate diverse initial patterns."""
    rng = np.random.default_rng(seed)
    
    # Pattern 1: Random signs
    h1 = rng.choice([-1.0, 1.0], size=n)
    
    # Pattern 2: Chirp
    i = np.arange(n)
    h2 = np.sign(np.cos(np.pi * i**2 / n))
    
    # Pattern 3: Cosine window
    h3 = np.sign(np.cos(np.pi * i / n))
    
    # Pattern 4: Gaussian window
    mu = n / 2
    sigma = n / 4
    h4 = np.sign(np.exp(-(i - mu)**2 / (2 * sigma**2)))
    
    # Pattern 5: Random smoothed
    noise = rng.standard_normal(n)
    kernel = np.ones(10) / 10
    h5 = np.sign(np.convolve(noise, kernel, mode='same'))
    
    # Pattern 6: Barker-like
    h6 = np.zeros(n)
    for j in range(n):
        if j % 4 == 0:
            h6[j] = 1.0
        elif j % 4 == 1:
            h6[j] = -1.0
        elif j % 4 == 2:
            h6[j] = 1.0
        else:
            h6[j] = -1.0
    
    # Pattern 7: All ones (theoretical optimum)
    h7 = np.ones(n)
    
    # Pattern 8: Alternating
    h8 = np.ones(n)
    h8[1::2] = -1.0
    
    return [h1, h2, h3, h4, h5, h6, h7, h8]


def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin
    
    # Theoretical optimum: constant sequence
    h_const = np.ones(N)
    const_score = _score(h_const)
    
    best_score = const_score
    best_h = h_const.copy()
    
    # Generate initial patterns
    inits = _gen_initial(N, seed=42)
    
    # Run ground state search on each initial pattern
    for idx, h_init in enumerate(inits):
        if time.time() > deadline - 5:
            break
        
        # Skip if it's the constant sequence (we already have it)
        if np.allclose(h_init, np.ones(N)):
            continue
        
        # Normalize to have sum = 1 for consistent comparison
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        h_norm = h_init / s0
        
        # Run ground state search
        h_opt, val_opt = _ground_state_search(h_norm, deadline - 5 - idx * 3, max_iters=150)
        
        if val_opt < best_score:
            best_score = val_opt
            best_h = h_opt.copy()
    
    # Final local search refinement on the best solution
    if time.time() < deadline - 2:
        h_final, val_final = _local_refinement(best_h, deadline - 1)
        if val_final < best_score:
            best_score = val_final
            best_h = h_final.copy()
    
    return best_h
