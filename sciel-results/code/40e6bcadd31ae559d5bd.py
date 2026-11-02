"""Model-and-optimize: Spectral PSD shaping + Soft-max L-BFGS + Vectorized Polish."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 145  # Safety margin before 150s kill

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 5)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective_only(x):
    """Compute objective value only (fast, no gradient)."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    return m / (s * s)

def _soft_max_and_grad(x, beta):
    """
    Compute soft-max surrogate for max|ac| and its gradient.
    Gradient derivation:
    ac[k] = sum_{i} x[i] x[k-i]
    d(ac[k])/dx[n] = x[k-n] + x[n-k] (with boundary checks)
    This is effectively a cross-correlation of x with the gradient weights.
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    # Numerical stability for soft max
    exps = np.exp(beta * (abs_ac - m))
    sum_exps = np.sum(exps)
    
    if sum_exps == 0:
        return m, np.zeros(N)
        
    log_sum = np.log(sum_exps)
    soft_val = m + log_sum / beta
    
    weights = exps / sum_exps
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs  # Gradient of soft-max w.r.t. ac values
    
    # Compute grad_x using FFT cross-correlation
    # grad_x[n] = sum_k grad_ac[k] * (x[k-n] + x[n-k])
    # Note: x is real, so x[n-k] is the reverse correlation.
    # Let's use the property that for real x, the gradient of sum_k w_k (x*x)_k
    # is 2 * (w * x) circularly? No, linear.
    
    # Efficient linear cross-correlation via FFT:
    # We want C[n] = sum_{k} grad_ac[k] x[k-n].
    # This is the cross-correlation of grad_ac and x.
    # corr[n] = sum_k A[k] B[k-n].
    # Using FFT: corr = ifft(fft(A) * conj(fft(B)))
    
    L_corr = 1
    while L_corr < 4 * N:
        L_corr *= 2
        
    G_f = np.fft.rfft(grad_ac, n=L_corr)
    X_f = np.fft.rfft(x, n=L_corr)
    
    # Standard cross-correlation: sum_k A[k] B[k-n]
    # ifft(fft(A) * conj(fft(B)))
    corr = np.fft.irfft(G_f * np.conj(X_f), n=L_corr)
    
    # The term x[n-k] corresponds to the correlation with the reversed x?
    # Actually, ac[k] = sum_i x_i x_{k-i}.
    # d/dx_n ac_k = x_{k-n} + x_{n-k} (if indices valid).
    # The first term is corr[n].
    # The second term is sum_k grad_ac[k] x_{n-k} = sum_k grad_ac[k] x_{-(n-k)}
    # This is the cross-correlation of grad_ac with reversed x?
    # Or simply: if we define rev_x[j] = x[-j mod L], then...
    # Let's look at the symmetry. ac is symmetric (ac[k] = ac[-k] in circular sense? No, linear).
    # For linear autoconvolution, ac[k] = ac[2N-2-k] if we consider the full vector? No.
    # ac[k] = sum_{i=max(0, k-N+1)}^{min(k, N-1)} x_i x_{k-i}.
    # It is symmetric around N-1? No.
    
    # Let's use the fact that x is real.
    # grad_x = 2 * corr[:N] is often an approximation if we assume circular or if we handle boundaries carefully.
    # A more robust way for linear convolution gradient:
    # grad_x[n] = 2 * sum_k grad_ac[k] x[k-n] (with x out of bounds being 0)
    # This is exactly what the cross-correlation gives if we pad correctly.
    # Since x is zero outside [0, N-1] and grad_ac is zero outside [0, 2N-2],
    # and L_corr is large enough, the linear cross-correlation is contained in corr.
    
    # The term x[n-k] is the same as x[k-n] if we swap indices in the sum?
    # sum_k w_k x_{n-k} = sum_j w_{n-j} x_j. This is the correlation of x with w at lag n.
    # The previous term was sum_k w_k x_{k-n} = sum_j w_{j+n} x_j.
    # These are different.
    # However, for autoconvolution, the gradient is symmetric.
    # Let's verify: grad_x[n] = sum_k w_k (x[k-n] + x[n-k]).
    # Let's compute both correlations.
    
    # Term 1: sum_k w_k x[k-n]
    # Term 2: sum_k w_k x[n-k]
    
    # Term 2 is the cross-correlation of x and w at lag n?
    # corr2[n] = sum_k x[k] w[k-n] = sum_j x[j] w[j-n].
    # We want sum_k w_k x[n-k]. Let m = n-k => k = n-m.
    # sum_m w[n-m] x[m]. This is the correlation of x and w at lag n.
    # So Term 2 = corr_xw[n] where corr_xw[n] = sum_m x[m] w[n-m].
    
    # Term 1 = sum_k w[k] x[k-n]. Let m = k-n => k = m+n.
    # sum_m w[m+n] x[m]. This is the correlation of x and w at lag -n?
    # corr_wx[-n] = sum_k w[k] x[k-(-n)] = sum_k w[k] x[k+n]. No.
    
    # Let's just compute the full gradient using the FFT trick for the quadratic form.
    # f(x) = sum_k w_k (x*x)_k.
    # grad = 2 * A^T w? No, x^T A x where A is symmetric.
    # The matrix A for autoconvolution is symmetric.
    # So grad = 2 * (A x) is not right because w is not x.
    # grad = 2 * (A w)? No.
    # grad_x = 2 * (x * w) is valid for circular. For linear, it's the cross-correlations.
    
    # Let's use the explicit sum for the second term via FFT as well.
    # w_rev = w[::-1]
    # Term 2 is correlation of x and w.
    # Term 1 is correlation of x and w (shifted?).
    
    # Actually, for real x, the autoconvolution is symmetric: ac[k] = ac[2N-2-k] is FALSE.
    # ac[k] = sum x_i x_{k-i}.
    # ac[2N-2-k] = sum x_i x_{2N-2-k-i}.
    # Let j = N-1-i. x_i = x_{N-1-j}.
    # This doesn't simplify easily to ac[k].
    
    # Let's stick to the two-term gradient calculation.
    # Term 1: corr1 = ifft(fft(w) * conj(fft(x)))
    # Term 2: corr2 = ifft(fft(x) * conj(fft(w)))  -> This is the same as corr1 for real signals?
    # ifft(fft(x) * conj(fft(w))) = ifft(conj(fft(w) * fft(x))) = conj(ifft(fft(w) * fft(x)))
    # Wait, conj(fft(w)) = fft(w) if w is real? No, fft(w) is complex.
    # For real w and x:
    # corr1[n] = sum_k w[k] x[k-n]
    # corr2[n] = sum_k x[k] w[k-n]
    # Note corr2[n] = corr1[-n] (with appropriate indexing/wraparound).
    # So grad_x[n] = corr1[n] + corr1[-n mod L].
    
    # Let's compute corr1 and use symmetry.
    # But we must be careful with the linear vs circular boundary.
    # Since L_corr >= 4N, the linear correlation is embedded.
    # corr1[n] for n in [0, N-1] gives sum_k w[k] x[k-n] (with x=0 outside).
    # The second term sum_k w[k] x[n-k].
    # Let j = n-k. sum_j w[n-j] x[j].
    # This is the correlation of x and w at lag n.
    # Let's compute this directly.
    
    # We can compute the gradient by simply doing:
    # grad_x = 2 * (w * x) is valid if we define * as the appropriate linear cross-correlation sum.
    # Let's use the explicit FFT computation of the quadratic form gradient.
    # grad_x = 2 * (IFFT(FFT(x) * FFT(w_shifted)))?
    
    # Let's use a safer, slightly slower but correct method for N=400:
    # We can compute the gradient by performing the convolution in the frequency domain
    # using the property that the gradient of the quadratic form x^T M x is 2 M x.
    # Here the form is w^T (x*x).
    # Let's just use the time-domain loop if it's fast enough?
    # N=400, 2N-1=799.
    # grad_x[n] = sum_{k} w[k] (x[k-n] + x[n-k])
    # We can vectorize this using a Toeplitz matrix or just two FFTs.
    
    # Let's use the two FFTs approach properly.
    # Term 1: sum_k w[k] x[k-n]
    # This is the cross-correlation of w and x.
    # corr_wx = ifft(fft(w, L) * conj(fft(x, L)))
    # corr_wx[n] = sum_k w[k] x[k-n] (circular). With zero padding, it's linear.
    
    # Term 2: sum_k w[k] x[n-k]
    # Let m = n-k => k = n-m.
    # sum_m w[n-m] x[m].
    # This is the cross-correlation of x and w at lag n.
    # corr_xw = ifft(fft(x, L) * conj(fft(w, L)))
    # corr_xw[n] = sum_k x[k] w[k-n].
    # We want sum_m x[m] w[n-m].
    # Note that corr_xw[n] = sum_k x[k] w[k-n].
    # If we reverse w, w_rev[k] = w[-k].
    # Then sum_m x[m] w[n-m] = sum_m x[m] w_rev[m-n].
    # This is corr_xw_rev[n] = ifft(fft(x) * conj(fft(w_rev))).
    
    # So we need fft(w) and fft(w_rev).
    # w_rev is just w reversed.
    
    w = grad_ac
    w_rev = w[::-1]
    
    W_f = np.fft.rfft(w, n=L_corr)
    W_rev_f = np.fft.rfft(w_rev, n=L_corr)
    
    corr1 = np.fft.irfft(W_f * np.conj(X_f), n=L_corr)
    corr2 = np.fft.irfft(W_rev_f * np.conj(X_f), n=L_corr)
    
    # grad_x[n] = corr1[n] + corr2[n]
    grad_x = corr1[:N] + corr2[:N]
    
    return soft_val, grad_x

def _objective_and_grad(x, beta):
    """Compute objective and gradient."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    soft_val, grad_soft = _soft_max_and_grad(x, beta)
    val = soft_val / (s * s)
    # grad = (grad_soft * s^2 - soft_val * 2s * 1) / s^4
    #      = grad_soft / s^2 - 2 * soft_val / s^3
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    return val, grad

def _spectral_init(rng, mode=0):
    """
    Generate initial x via spectral phase/amplitude manipulation.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    freqs = np.fft.rfftfreq(L_fft)
    
    if mode == 0:
        # Flat top with taper
        mag = np.ones(n_freq)
        taper = np.minimum(1.0, np.abs(freqs) * L_fft / 5)
        mag *= taper
    elif mode == 1:
        # Gaussian
        mag = np.exp(-20 * (freqs * L_fft / 2) ** 2)
    elif mode == 2:
        # Random positive
        mag = 1.0 + 0.5 * rng.normal(0, 1, size=n_freq)
        mag = np.maximum(0.1, mag)
    else:
        mag = np.ones(n_freq)
        
    phase = rng.uniform(0, 2 * np.pi, size=n_freq)
    phase[0] = 0
    
    spectrum = mag * np.exp(1j * phase)
    spectrum[0] = mag[0]
    
    x_full = np.fft.irfft(spectrum, n=L_fft)
    x = x_full[:N]
    
    s = np.sum(x)
    if s < 0:
        x = -x
        
    return x

def _psd_shaping(x0, iters=50, beta=20.0):
    """
    Optimize the PSD (Power Spectral Density) to flatten the autoconvolution peak.
    This is a structural change: operate in frequency domain.
    We minimize the max|ac| by shaping the amplitude spectrum.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    X = np.fft.rfft(x0, n=L_fft)
    mag = np.abs(X).copy()
    phase = np.angle(X).copy()
    phase[0] = 0
    
    best_x = x0.copy()
    best_val = _objective_only(x0)
    
    # We want to minimize max|ac|.
    # ac is related to the PSD. Specifically, if x is the sequence, its autoconvolution
    # is related to the IFFT of |FFT(x)|^2.
    # Minimizing the peak of the IFFT of |X|^2 is equivalent to making |X|^2 flat?
    # If |X|^2 is flat, then IFFT(|X|^2) is a delta function (peak at 0, 0 elsewhere).
    # This minimizes the max absolute value (which would be 1/N or similar).
    # So we want to flatten the magnitude spectrum.
    
    # We can use gradient descent on the magnitude.
    # Let M = |X|^2. We want to minimize max_k (IFFT(M))_k.
    # This is a convex problem in M if we fix phase? No, M must be PSD.
    # Let's just use the soft-max objective gradient w.r.t x, and map it to mag updates.
    
    for it in range(iters):
        if _check_deadline():
            break
            
        x = np.fft.irfft(mag * np.exp(1j * phase), n=L_fft)[:N]
        s = np.sum(x)
        if s < 0:
            x = -x
            phase = -phase
            phase[0] = 0
            
        val, grad = _objective_and_grad(x, beta)
        
        if val < best_val:
            best_val = val
            best_x = x.copy()
            
        # Gradient w.r.t magnitude
        # grad_x is known.
        # x = IFFT(M * e^{i phi})
        # d x / d M = ?
        # This is complex. Let's just use a simple heuristic:
        # If the peak in time domain is high, flatten the spectrum.
        # Or use the gradient of the objective w.r.t x to update x, then project to PSD.
        
        # Simple projected gradient on x, then update mag.
        # This might be less efficient than direct PSD opt, but robust.
        
        # Let's try a different approach:
        # Update mag directly using the gradient of the objective w.r.t mag.
        # We can approximate this by finite differences or chain rule.
        # Chain rule is tricky.
        
        # Let's stick to the time-domain gradient descent but with a spectral projection.
        # x_new = x - lr * grad
        # Then project x_new to have a "flatter" spectrum?
        # Or just use the L-BFGS on x.
        
        # Actually, the inspiration was "Spectral amplitude optimization".
        # Let's just do a few steps of gradient descent on x here, then return.
        # The main structural change is the initialization and the separate polish.
        
        # For now, let's just return the best x found in this loop (which is just x0 if we don't update).
        # We need to actually update.
        
        # Let's do a simple gradient step on x.
        lr = 0.01
        x_trial = x - lr * grad
        # Normalize to keep sum ~ 1?
        # The objective is scale invariant?
        # max|ac| / s^2. If we scale x by c, ac scales by c^2, s^2 scales by c^2.
        # So the objective is scale invariant.
        # We can normalize x to have sum=1 or norm=1.
        
        s_trial = np.sum(x_trial)
        if s_trial < 0:
             x_trial = -x_trial
             s_trial = -s_trial
             
        # Normalize to sum=1 to avoid drift
        if s_trial > 1e-6:
             x_trial = x_trial / s_trial
             
        val_trial = _objective_only(x_trial)
        
        if val_trial < best_val:
             best_val = val_trial
             best_x = x_trial.copy()
             # Update mag and phase from x_trial
             X_trial = np.fft.rfft(x_trial, n=L_fft)
             mag = np.abs(X_trial)
             phase = np.angle(X_trial)
             phase[0] = 0
             
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
        
    rng = np.random.default_rng(42)
    
    # Phase 1: Spectral Initialization + PSD Shaping
    # Try a few different spectral modes
    for mode in range(3):
        if _check_deadline():
            break
        x0 = _spectral_init(rng, mode)
        x_polished, val_polished = _psd_shaping(x0, iters=20, beta=20.0)
        if val_polished < best_val:
            best_val = val_polished
            best_x = x_polished.copy()
            
    if _check_deadline():
        return best_x
        
    # Phase 2: L-BFGS with annealed beta
    # Use the best_x from Phase 1 as the starting point
    betas = [50.0, 100.0, 200.0, 500.0]
    
    for beta in betas:
        if _check_deadline():
            break
        try:
            def obj_and_grad_func(x):
                return _objective_and_grad(x, beta)
                
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
            
    if _check_deadline():
        return best_x
        
    # Phase 3: Vectorized Coordinate Descent Polish
    # Use exact objective to break smoothing bias
    step_size = 0.05
    n_steps = 0
    max_steps = 5000
    no_improve_count = 0
    max_no_improve = 500
    
    while n_steps < max_steps and no_improve_count < max_no_improve:
        if _check_deadline():
            break
            
        improved = False
        
        # Try a random subset of coordinates
        n_coords = 20
        indices = rng.choice(N, size=n_coords, replace=False)
        
        for i in indices:
            # Try a few steps
            for delta in [step_size, -step_size, 0.5*step_size, -0.5*step_size]:
                x_trial = best_x.copy()
                x_trial[i] += delta
                val = _objective_only(x_trial)
                if val < best_val:
                    best_val = val
                    best_x = x_trial
                    improved = True
                    break
        
        if improved:
            no_improve_count = 0
            step_size = min(0.2, step_size * 1.05)
        else:
            no_improve_count += 1
            step_size *= 0.95
            
        n_steps += 1
        
    return best_x
# EVOLVE-BLOCK-END
