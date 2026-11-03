import time
import numpy as np

N = 400
L = 1024  # FFT length >= 2*N-1 = 799
_SEED = 12345

def _autocorr_fast(h):
    """Linear one-sided autocorrelation ac[k] = sum_i h[i] h[i+k], k = 0..N-1."""
    F = np.fft.rfft(h, n=L)
    ac = np.fft.irfft(F * np.conj(F), n=L)[:N]
    return ac

def _hard_score(h):
    """The actual metric: peak autoconvolution concentration (lower is better)."""
    ac = _autocorr_fast(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    return float(np.max(np.abs(ac[1:])) / dc)

def _smooth_score(h, p=4.0):
    """
    Compute the L_p norm of the sidelobes, normalized by DC.
    This serves as a smooth surrogate for the peak sidelobe when p is large.
    """
    ac = _autocorr_fast(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    sidelobes = ac[1:]
    # Normalize to prevent overflow/underflow issues in power calculation
    y = np.abs(sidelobes) / dc
    if np.max(y) < 1e-30:
        return 0.0
    # L_p norm
    return float(np.sum(y**p) ** (1.0/p))

def _gradient_l_p(x, p=4.0):
    """
    Compute the gradient of the L_p norm of the sidelobes with respect to x.
    x is a continuous signal of length N (padded to L for FFT).
    """
    # Pad x to length L
    x_padded = np.zeros(L)
    x_padded[:N] = x
    
    # 1. Compute Autocorrelation R
    F = np.fft.rfft(x_padded)
    P = F * np.conj(F)
    R = np.fft.irfft(P, n=L)
    
    # R[0] is DC. We ignore it in the objective.
    # We want to minimize sum_{k=1}^{L-1} |R[k]|^p (approximating one-sided)
    # Actually, for real signals, R is symmetric. The one-sided sidelobes are R[1:N].
    # The full autocorrelation sum includes negative lags too.
    # Minimizing the full L_p norm of the full autocorrelation (excluding DC)
    # is equivalent to minimizing the one-sided norm because R[k] = R[-k].
    
    # Gradient with respect to R
    # d/dR[k] (|R[k]|^p) = p * |R[k]|^(p-2) * R[k]
    dR = np.zeros(L)
    # Exclude DC (index 0)
    # Note: R is real
    dR[1:] = p * (np.abs(R[1:]) ** (p - 2)) * R[1:]
    # Since R is symmetric, dR should be symmetric. The formula above handles it if R is symmetric.
    # However, irfft returns a symmetric array.
    
    # 2. Compute Gradient with respect to x
    # J = sum_k |R[k]|^p
    # R = C x (where C is convolution operator)
    # dJ/dx = C^T (dJ/dR)
    # C is circulant defined by x_padded.
    # C^T is circulant defined by x_padded reversed.
    
    x_rev = x_padded[::-1].copy()
    # Wait, for real convolution y = x * h, the adjoint is x * h_rev.
    # Here R[k] = sum_i x[i] x[i+k].
    # This is (x * x_rev)[k].
    # So R = conv(x, x_rev).
    # dJ/dx = conv(x_rev, dR).
    
    F_dR = np.fft.rfft(dR)
    F_x_rev = np.fft.rfft(x_rev)
    F_grad = F_x_rev * F_dR
    grad = np.fft.irfft(F_grad, n=L)
    
    return grad[:N]

def _gradient_descent_polish(init_h, time_budget, rng, p=8.0):
    """
    Perform gradient descent on a continuous relaxation, then polish with local search.
    """
    t_start = time.time()
    x = init_h.astype(np.float64)
    
    # Phase 1: Continuous Gradient Descent
    # We use a decreasing step size or adaptive step size.
    # Let's use a fixed small step size and many iterations, or a decaying one.
    lr = 0.01
    n_iters = 1000
    
    # We monitor the hard score occasionally to track progress
    best_hard = _hard_score(init_h)
    best_x = x.copy()
    
    for i in range(n_iters):
        if time.time() - t_start > time_budget * 0.6:
            break
            
        grad = _gradient_l_p(x, p)
        
        # Normalize gradient to control step size
        g_norm = np.linalg.norm(grad)
        if g_norm < 1e-10:
            break
            
        x -= lr * (grad / g_norm) * 0.1 # Scale down
        
        # Every 100 steps, check hard score of the sign-quantized version
        if i % 100 == 0:
            h_curr = np.sign(x)
            h_curr[h_curr == 0] = 1.0
            s_curr = _hard_score(h_curr)
            if s_curr < best_hard:
                best_hard = s_curr
                best_x = x.copy()
                best_h = h_curr.copy()
                
    # Ensure we have a valid best_h
    h_curr = np.sign(x)
    h_curr[h_curr == 0] = 1.0
    s_curr = _hard_score(h_curr)
    if s_curr < best_hard:
        best_hard = s_curr
        best_h = h_curr.copy()
    else:
        best_h = np.sign(best_x)
        best_h[best_h == 0] = 1.0

    # Phase 2: Local Polish
    h = best_h.copy()
    cur_hard = best_hard
    
    # Single bit polish
    improved = True
    while improved and time.time() - t_start < time_budget - 2.0:
        improved = False
        for i in range(N):
            if time.time() - t_start > time_budget - 2.0:
                break
            old_val = h[i]
            h[i] *= -1.0
            new_hard = _hard_score(h)
            if new_hard < cur_hard - 1e-12:
                cur_hard = new_hard
                improved = True
            else:
                h[i] = old_val
                
    # Block polish (size 2, 4)
    for size in [2, 4]:
        improved = True
        while improved and time.time() - t_start < time_budget - 1.0:
            improved = False
            for start in range(0, N - size + 1):
                if time.time() - t_start > time_budget - 1.0:
                    break
                sl = slice(start, start + size)
                old_block = h[sl].copy()
                h[sl] *= -1.0
                new_hard = _hard_score(h)
                if new_hard < cur_hard - 1e-12:
                    cur_hard = new_hard
                    improved = True
                else:
                    h[sl] = old_block

    return h, cur_hard

def construct():
    try:
        t0 = time.time()
        deadline = t0 + 145.0  # Safety margin
        rng = np.random.default_rng(_SEED)

        best_h_global = None
        best_score_global = 1e9
        
        # We will run multiple iterations of the Gradient+Polish strategy
        # with different initializations.
        
        num_restarts = 5
        time_per_restart = (deadline - t0 - 5.0) / num_restarts
        
        for r in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
                
            # 1. Generate diverse initialization
            if r == 0:
                # Chirp
                idx = np.arange(N)
                h_init = np.sign(np.cos(2 * np.pi * idx ** 2 / (2 * N)))
            elif r == 1:
                # Random
                h_init = rng.choice(np.array([-1.0, 1.0]), size=N)
            elif r == 2:
                # Golden ratio
                phi = (1 + np.sqrt(5)) / 2
                h_init = np.sign(np.cos(2 * np.pi * phi * np.arange(N) / N))
            elif r == 3:
                # Low frequency sine
                h_init = np.sign(np.cos(2 * np.pi * 3 * np.arange(N) / N))
            else:
                # M-sequence like
                h_init = rng.choice(np.array([-1.0, 1.0]), size=N)
                # Apply a simple filter to create structure
                # Just use random for now, gradient will shape it
            
            h_init = np.sign(h_init)
            h_init[h_init == 0] = 1.0
            
            # 2. Run Gradient Descent + Polish
            h_candidate, score_candidate = _gradient_descent_polish(
                h_init, 
                time_per_restart, 
                rng, 
                p=8.0  # High power to target peak
            )
            
            if score_candidate < best_score_global:
                best_score_global = score_candidate
                best_h_global = h_candidate.copy()
                
            # Optional: Use the best found so far as a perturbed start for next iteration
            # to stay in a good basin? No, diverse starts are better for global escape.
            
        if best_h_global is None:
            return np.ones(N)
            
        return best_h_global

    except Exception:
        return np.ones(N)
