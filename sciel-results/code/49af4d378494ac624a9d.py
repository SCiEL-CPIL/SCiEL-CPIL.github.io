"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 145

def _check_deadline(margin=5):
    return (time.time() - START_TIME) > (TIME_LIMIT - margin)

def _linear_autoconv(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _score(x):
    """Compute the C3' score: max|autoconv| / (sum(x))^2"""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    ac = _linear_autoconv(x)
    max_abs = np.max(np.abs(ac))
    return max_abs / (s * s)

def _soft_max_grad(x, beta=100.0):
    """
    Compute soft-max surrogate for max|ac| and its gradient w.r.t. x.
    Uses the identity: d(ac[k])/dx[n] = 2*x[k-n] (with boundary handling).
    """
    ac = _linear_autoconv(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    
    # Soft max with temperature beta
    exps = np.exp(beta * (abs_ac - m))
    sum_exps = np.sum(exps)
    if sum_exps < 1e-300:
        weights = np.zeros_like(abs_ac)
        weights[np.argmax(abs_ac)] = 1.0
    else:
        weights = exps / sum_exps
    
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs  # length 2N-1
    
    # grad_x[n] = sum_k grad_ac[k] * 2 * x[k-n] for valid k-n in [0, N-1]
    # This is a cross-correlation
    L_corr = 1
    while L_corr < 4 * N:
        L_corr *= 2
    
    g_padded = np.zeros(L_corr)
    g_padded[:2*N-1] = grad_ac
    
    X_f = np.fft.rfft(x, n=L_corr)
    G_f = np.fft.rfft(g_padded, n=L_corr)
    
    corr = np.fft.irfft(G_f * np.conj(X_f), n=L_corr)
    grad_x = 2.0 * corr[:N]
    
    soft_val = m + np.log(sum_exps) / beta if sum_exps > 0 else m
    return soft_val, grad_x

def _psd_optimize(mag_init, n_iters=300, time_budget=40):
    """
    Optimize the magnitude spectrum (PSD) to minimize L-infinity norm
    of the corresponding autoconvolution, subject to unit DC constraint.
    
    Uses projected gradient descent in the frequency domain.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    mag = mag_init.copy()
    mag = np.maximum(mag, 0.01)  # Keep positive
    
    # Ensure DC component gives unit sum
    # sum(x) = X[0] = mag[0] * exp(i*0) = mag[0]
    # For binary ±1 with sum = S, we want |S| to be large
    # But we'll normalize later
    
    best_mag = mag.copy()
    best_val = float('inf')
    
    start_time = time.time()
    lr = 0.01
    
    for it in range(n_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Reconstruct a candidate x from current mag with random phase
        # Use the phase that minimizes peak (via simple iteration)
        phase = np.zeros(n_freq)
        for _ in range(10):
            spectrum = mag * np.exp(1j * phase)
            spectrum[0] = mag[0]
            x_trial = np.fft.irfft(spectrum, n=L_fft)[:N]
            s = np.sum(x_trial)
            if s < 0:
                x_trial = -x_trial
                phase = -phase
                phase[0] = 0
            
            # Binary projection
            x_bin = np.sign(x_trial)
            
            # Compute gradient w.r.t. x
            soft_val, grad_x = _soft_max_grad(x_bin, beta=50.0)
            
            # Transform gradient to frequency domain
            G = np.fft.rfft(grad_x, n=L_fft)
            
            # Gradient of peak w.r.t. magnitude
            # d(peak)/dmag[k] = 2 * Re(X[k] * conj(G[k])) / |X[k]| * mag[k]
            X_curr = mag * np.exp(1j * phase)
            mag_grad = np.zeros(n_freq)
            for k in range(n_freq):
                if mag[k] > 1e-10:
                    mag_grad[k] = 2.0 * np.real(X_curr[k] * np.conj(G[k])) / mag[k]
            
            # Update magnitude
            mag_new = mag - lr * mag_grad
            mag_new = np.maximum(mag_new, 0.01)
            
            # Project: ensure DC is preserved (sum constraint)
            # The DC component mag[0] determines the sum
            # We want to keep mag[0] fixed to maintain the integral
            mag_new[0] = mag[0]
            
            # Check if this is better
            spectrum = mag_new * np.exp(1j * phase)
            spectrum[0] = mag_new[0]
            x_trial = np.fft.irfft(spectrum, n=L_fft)[:N]
            s = np.sum(x_trial)
            if s < 0:
                x_trial = -x_trial
            
            x_bin = np.sign(x_trial)
            val = _score(x_bin)
            
            if val < best_val:
                best_val = val
                best_mag = mag_new.copy()
            
            # Adaptive learning rate
            if it % 20 == 0 and it > 0:
                lr = max(0.001, lr * 0.9)
            
            mag = mag_new
    
    return best_mag, best_val

def _phase_retrieval(mag, n_iters=500, time_budget=30):
    """
    Given a fixed magnitude spectrum, find the phase that minimizes
    the peak autoconvolution, with binary projection.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    best_x = None
    best_val = float('inf')
    
    start_time = time.time()
    
    for init in range(5):
        if time.time() - start_time > time_budget:
            break
        
        if init == 0:
            phase = np.zeros(n_freq)
        else:
            phase = np.random.uniform(0, 2 * np.pi, size=n_freq)
            phase[0] = 0
        
        lr = 0.05
        
        for it in range(n_iters):
            if time.time() - start_time > time_budget:
                break
            
            spectrum = mag * np.exp(1j * phase)
            spectrum[0] = mag[0]
            x_trial = np.fft.irfft(spectrum, n=L_fft)[:N]
            s = np.sum(x_trial)
            if s < 0:
                x_trial = -x_trial
                phase = -phase
                phase[0] = 0
            
            x_bin = np.sign(x_trial)
            val = _score(x_bin)
            
            if val < best_val:
                best_val = val
                best_x = x_bin.copy()
            
            # Compute gradient
            soft_val, grad_x = _soft_max_grad(x_bin, beta=50.0)
            G = np.fft.rfft(grad_x, n=L_fft)
            
            # Phase gradient
            X_curr = mag * np.exp(1j * phase)
            phase_grad = np.imag(X_curr * np.conj(G)) / (mag + 1e-10)
            phase_grad[0] = 0
            
            # Update phase
            phase -= lr * phase_grad
            phase = np.mod(phase, 2 * np.pi)
            
            # Adaptive learning rate
            if it % 50 == 0 and it > 0:
                lr = max(0.005, lr * 0.95)
    
    return best_x, best_val

def _local_search(x0, max_iters=300, time_budget=20):
    """Local search by flipping individual bits to reduce score."""
    x = x0.copy()
    best_score = _score(x)
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Find the peak location in autoconv
        ac = _linear_autoconv(x)
        abs_ac = np.abs(ac)
        peak_idx = np.argmax(abs_ac)
        
        # Try flipping each bit and see if it reduces the peak
        best_flip = -1
        best_improvement = 0
        
        # Sample a subset of positions to try
        n_tries = min(150, len(x))
        positions = np.random.choice(len(x), n_tries, replace=False)
        
        for pos in positions:
            x_trial = x.copy()
            x_trial[pos] = -x_trial[pos]
            
            # Quick score check
            ac_t = _linear_autoconv(x_trial)
            max_abs_t = np.max(np.abs(ac_t))
            integral_t = np.sum(x_trial)
            if abs(integral_t) < 1e-12:
                continue
            score_t = max_abs_t / (integral_t * integral_t)
            
            if score_t < best_score - 1e-10:
                best_improvement = best_score - score_t
                best_flip = pos
        
        if best_flip >= 0:
            x[best_flip] = -x[best_flip]
            best_score = _score(x)
        else:
            # No improvement found, try random perturbation
            if np.random.rand() < 0.15:
                n_perturb = np.random.randint(1, 8)
                perturb_positions = np.random.choice(len(x), n_perturb, replace=False)
                x[perturb_positions] = -x[perturb_positions]
                new_score = _score(x)
                if new_score < best_score:
                    best_score = new_score
                else:
                    x[perturb_positions] = -x[perturb_positions]  # revert
    
    return x, best_score

def _gradient_descent_polish(x0, n_iters=200, time_budget=25):
    """
    Gradient descent polish using soft-max surrogate.
    Optimizes in continuous space, then projects to binary.
    """
    x = x0.astype(float).copy()
    best_x_bin = x0.copy()
    best_val = _score(x0)
    
    start_time = time.time()
    lr = 0.01
    beta = 50.0
    
    for it in range(n_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Compute objective and gradient
        s = np.sum(x)
        if abs(s) < 1e-10:
            break
        
        soft_val, grad_soft = _soft_max_grad(x, beta)
        val = soft_val / (s * s)
        grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
        
        # Gradient step
        x_new = x - lr * grad
        
        # Project to binary
        x_bin = np.sign(x_new)
        
        # Check if better
        val_bin = _score(x_bin)
        if val_bin < best_val:
            best_val = val_bin
            best_x_bin = x_bin.copy()
        
        # Keep x in continuous space for next iteration
        x = x_new
        
        # Adaptive learning rate
        if it % 30 == 0 and it > 0:
            lr = max(0.001, lr * 0.9)
    
    return best_x_bin, best_val

def construct():
    start_time = time.time()
    
    # Initialize best solution
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    # Strategy 1: PSD optimization with multiple initial spectra
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    freqs = np.fft.rfftfreq(L_fft)
    
    # Different initial magnitude spectra
    mag_inits = []
    
    # Flat spectrum
    mag_inits.append(np.ones(n_freq))
    
    # Tapered spectrum
    mag_taper = np.ones(n_freq)
    taper = np.minimum(1.0, np.abs(freqs) * L_fft / 5)
    mag_inits.append(mag_taper * taper)
    
    # Cosine window
    mag_cos = 0.5 + 0.5 * np.cos(2 * np.pi * freqs * L_fft * 0.5)
    mag_inits.append(mag_cos)
    
    # Gaussian window
    mag_gauss = np.exp(-20 * (freqs * L_fft / 2) ** 2)
    mag_inits.append(mag_gauss)
    
    # Low-pass with smooth transition
    cutoff = 0.3
    transition = 0.05
    mag_lp = np.where(np.abs(freqs) < cutoff - transition, 1.0,
                      np.where(np.abs(freqs) < cutoff + transition,
                               0.5 * (1 + np.cos(np.pi * (np.abs(freqs) - cutoff) / transition)),
                               0.1))
    mag_inits.append(mag_lp)
    
    # Optimize each initial PSD
    for i, mag_init in enumerate(mag_inits):
        if _check_deadline(margin=30):
            break
        
        # PSD optimization
        mag_opt, psd_val = _psd_optimize(mag_init, n_iters=200, time_budget=15)
        
        # Phase retrieval with optimized PSD
        x_candidate, val_candidate = _phase_retrieval(mag_opt, n_iters=300, time_budget=20)
        
        if x_candidate is not None and val_candidate < best_score:
            best_score = val_candidate
            best_x = x_candidate.copy()
    
    # Strategy 2: Direct spectral phase retrieval with various initial phases
    for attempt in range(10):
        if _check_deadline(margin=20):
            break
        
        # Different initial phases
        if attempt == 0:
            phase_init = np.zeros(n_freq)
        elif attempt == 1:
            phase_init = np.pi * np.random.randn(n_freq)
        else:
            phase_init = np.random.uniform(0, 2 * np.pi, n_freq)
        
        # Use a reasonable magnitude spectrum
        if attempt % 2 == 0:
            mag = np.ones(n_freq)
        else:
            mag = 0.5 + 0.5 * np.cos(2 * np.pi * freqs * L_fft * 0.3)
        
        spectrum = mag * np.exp(1j * phase_init)
        spectrum[0] = mag[0]
        x = np.fft.irfft(spectrum, n=L_fft)[:N]
        s = np.sum(x)
        if s < 0:
            x = -x
        x_bin = np.sign(x)
        
        score = _score(x_bin)
        if score < best_score:
            best_score = score
            best_x = x_bin.copy()
        
        # Refine with local search
        if _check_deadline(margin=10):
            break
        x_refined, score_refined = _local_search(x_bin, max_iters=100, time_budget=5)
        if score_refined < best_score:
            best_score = score_refined
            best_x = x_refined.copy()
    
    # Strategy 3: Gradient descent polish on the best solution
    if not _check_deadline(margin=15):
        x_polished, score_polished = _gradient_descent_polish(best_x, n_iters=200, time_budget=20)
        if score_polished < best_score:
            best_score = score_polished
            best_x = x_polished.copy()
    
    # Strategy 4: Final local search
    if not _check_deadline(margin=5):
        x_final, score_final = _local_search(best_x, max_iters=300, time_budget=10)
        if score_final < best_score:
            best_score = score_final
            best_x = x_final.copy()
    
    return best_x
# EVOLVE-BLOCK-END
