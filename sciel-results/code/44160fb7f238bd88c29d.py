"""Model-and-optimize approach for signed step functions minimizing max|autoconv|/(sum x)^2."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 115
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _psd_objective_and_grad(params, beta=100.0):
    """
    Objective and gradient for optimizing the Power Spectral Density (PSD).
    params: log-PSD values for k=0..N/2.
    P[k] = exp(params[k]).
    P is extended to full N with symmetry P[N-k] = P[k].
    c[n] = IFFT(P_full)[n].
    Objective: SoftMax(|c|) / P[0].
    """
    half = N // 2
    P_half = np.exp(params)
    
    # Construct full P (length N)
    P_full = np.zeros(N)
    P_full[:half+1] = P_half
    P_full[half+1:] = P_half[:half-1][::-1]
    
    # Calculate c = IFFT(P_full)
    c = np.fft.ifft(P_full).real
    
    s_sq = P_full[0]
    if s_sq < 1e-10:
        return 1e10, np.zeros_like(params)
    
    abs_c = np.abs(c)
    max_c = np.max(abs_c)
    
    if max_c < 1e-15:
        return 1e10, np.zeros_like(params)
    
    # Soft-max
    shifted = beta * (abs_c - max_c)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_c + (1.0 / beta) * np.log(sum_exp)
    
    val = soft_max / s_sq
    
    # Gradient calculation
    W = weights * np.sign(c)
    d_soft_dP_full = np.fft.fft(W) / N
    
    d_val_dP_full = d_soft_dP_full / P_full[0]
    d_val_dP_full[0] -= soft_max / (P_full[0] ** 2)
    
    grad = np.zeros(half+1)
    for k in range(half+1):
        if k == 0:
            contrib = d_val_dP_full[0] * P_full[0]
        elif k == N // 2:
            contrib = d_val_dP_full[k] * P_full[k]
        else:
            contrib = (d_val_dP_full[k] + d_val_dP_full[N-k]) * P_full[k]
        grad[k] = contrib
        
    return val, grad

def _recover_x_from_psd(params):
    """Recover x from log-PSD params using zero phase."""
    half = N // 2
    P_half = np.exp(params)
    P_full = np.zeros(N)
    P_full[:half+1] = P_half
    P_full[half+1:] = P_half[:half-1][::-1]
    
    X = np.sqrt(P_full)
    x = np.fft.ifft(X).real
    return x

def _peak_targeted_polish(x, max_iters=500, lr=0.01):
    """
    Directly reduce the dominant autoconvolution peaks by gradient descent.
    For each iteration, find the peak lag, compute gradient w.r.t. x, and step.
    """
    best_x = x.copy()
    best_val = _objective(best_x)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        ac = _autoconv_fft(best_x)
        s = np.sum(best_x)
        s_sq = s * s
        
        if abs(s_sq) < 1e-10:
            break
        
        # Find peak absolute value location
        abs_ac = np.abs(ac)
        peak_idx = np.argmax(abs_ac)
        peak_val = abs_ac[peak_idx]
        
        if peak_val < 1e-12:
            break
        
        # Gradient of |ac[peak_idx]| w.r.t. x
        # ac[k] = sum_j x[j] * x[k-j]
        # dac[k]/dx[i] = x[k-i] + x[i-k] (for valid indices)
        # For linear autoconv: ac[k] = sum_{j} x[j]*x[k-j] where 0<=j<N, 0<=k-j<N
        # dac[k]/dx[i] = sum_j [delta(j=i)*x[k-j] + delta(k-j=i)*x[j]]
        #               = x[k-i] (if 0<=k-i<N) + x[k-i] (if 0<=k-i<N, same term)
        # Wait, let me be more careful.
        # ac[k] = sum_{j=max(0,k-N+1)}^{min(k,N-1)} x[j]*x[k-j]
        # dac[k]/dx[i] = x[k-i] (if 0<=k-i<N and i in valid range) + x[k-i] (same)
        # Actually: dac[k]/dx[i] = x[k-i] if 0<=k-i<N, else 0
        # But we need to handle both terms in the product rule.
        # d/dx[i] (x[j]*x[k-j]) summed over j:
        # = sum_j [delta(j=i)*x[k-j] + x[j]*delta(k-j=i)]
        # = x[k-i] (from first term, j=i, if 0<=k-i<N) + x[k-i] (from second term, k-j=i => j=k-i, if 0<=k-i<N)
        # = 2*x[k-i] if 0<=k-i<N, else 0
        # Wait no. The second term: x[j]*delta(k-j=i) means j=k-i, so value is x[k-i] if 0<=k-i<N.
        # First term: delta(j=i)*x[k-j] means j=i, so value is x[k-i] if 0<=k-i<N.
        # So total: 2*x[k-i] if 0<=k-i<N, else 0.
        
        # But we want to minimize |ac[peak_idx]|, so we need the sign.
        # d|ac[k]|/dx[i] = sign(ac[k]) * dac[k]/dx[i] = sign(ac[k]) * 2*x[k-i] (if 0<=k-i<N)
        
        sign_ac = np.sign(ac[peak_idx])
        k = peak_idx
        
        # Build gradient vector
        grad = np.zeros(N)
        # For each i, grad[i] = sign_ac * 2 * x[k-i] if 0<=k-i<N
        valid = (k - np.arange(N) >= 0) & (k - np.arange(N) < N)
        if np.any(valid):
            grad[valid] = sign_ac * 2.0 * x[k - np.arange(N)][valid]
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        
        # Step
        trial = best_x - lr * grad / grad_norm
        
        # Also normalize to keep sum roughly constant (optional)
        # trial = trial - np.mean(trial - best_x)  # keep mean close
        
        val = _objective(trial)
        if val < best_val:
            best_val = val
            best_x = trial.copy()
        else:
            # Reduce learning rate
            lr *= 0.5
            if lr < 1e-8:
                break
    
    return best_x, best_val

def _fast_coordinate_descent(x, max_rounds=50, step_sizes=None):
    """
    Faster coordinate descent using vectorized trial evaluations.
    """
    if step_sizes is None:
        step_sizes = [0.1, 0.05, 0.02, 0.01]
    
    best_x = x.copy()
    best_val = _objective(best_x)
    
    for _ in range(max_rounds):
        if _check_deadline():
            break
        
        improved = False
        for j in range(N):
            if _check_deadline():
                return best_x, best_val
            
            for delta in step_sizes:
                for sign in [-1, 1]:
                    trial = best_x.copy()
                    trial[j] += sign * delta
                    val = _objective(trial)
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            break
    
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    half = N // 2
    
    # Generate diverse initial PSD candidates
    candidates = []
    
    # Flat
    candidates.append(np.zeros(half+1))
    
    # Low-pass
    for alpha in [0.05, 0.2, 0.5, 1.0, 2.0, 5.0]:
        k = np.arange(half+1)
        candidates.append(-alpha * k)
        
    # High-pass
    for alpha in [0.05, 0.2, 0.5, 1.0, 2.0]:
        k = np.arange(half+1)
        candidates.append(-alpha * (N - k))
        
    # Band-pass
    for center in [N//8, N//4, N//2, 3*N//4]:
        for width in [N//8, N//4, N//2]:
            k = np.arange(half+1)
            candidates.append(-(k - center)**2 / width)
    
    # Random smooth PSDs
    rng = np.random.RandomState(42)
    for _ in range(5):
        # Smooth random spectrum
        freqs = np.linspace(0, 1, half+1)
        noise = rng.randn(half+1)
        # Low-pass filter the noise to make it smooth
        from scipy.signal import lfilter
        b = np.ones(5) / 5.0
        a = [1.0]
        smooth = lfilter(b, a, noise)
        candidates.append(smooth)

    # Optimize each candidate with PSD optimization
    for p0 in candidates:
        if _check_deadline():
            break
        
        try:
            result = minimize(_psd_objective_and_grad, p0, method='L-BFGS-B', jac=True,
                             options={'maxiter': 150, 'gtol': 1e-6, 'ftol': 1e-12})
            
            if result.fun < best_val:
                x_rec = _recover_x_from_psd(result.x)
                val_rec = _objective(x_rec)
                
                if val_rec < best_val:
                    best_val = val_rec
                    best_x = x_rec.copy()
                    
        except Exception:
            pass

    if _check_deadline():
        return best_x

    # Peak-targeted polish on best_x
    best_x, best_val = _peak_targeted_polish(best_x, max_iters=300, lr=0.05)
    
    if _check_deadline():
        return best_x

    # Re-optimize PSD from the polished x
    X_best = np.fft.rfft(best_x)
    P_best = np.abs(X_best)**2
    P_best[P_best < 1e-10] = 1e-10
    params_best = np.log(P_best)
    
    # Try with different betas
    for beta in [50, 100, 200, 500]:
        if _check_deadline():
            break
        
        try:
            result = minimize(lambda p: _psd_objective_and_grad(p, beta=beta), params_best,
                             method='L-BFGS-B', jac=True,
                             options={'maxiter': 200, 'gtol': 1e-7, 'ftol': 1e-13})
            if result.fun < best_val:
                x_rec = _recover_x_from_psd(result.x)
                val_rec = _objective(x_rec)
                if val_rec < best_val:
                    best_val = val_rec
                    best_x = x_rec.copy()
                    # Update params_best for next beta
                    X_tmp = np.fft.rfft(x_rec)
                    P_tmp = np.abs(X_tmp)**2
                    P_tmp[P_tmp < 1e-10] = 1e-10
                    params_best = np.log(P_tmp)
        except Exception:
            pass

    if _check_deadline():
        return best_x

    # Another round of peak-targeted polish
    best_x, best_val = _peak_targeted_polish(best_x, max_iters=200, lr=0.02)
    
    if _check_deadline():
        return best_x

    # Fast coordinate descent for final refinement
    best_x, best_val = _fast_coordinate_descent(best_x, max_rounds=30, 
                                                 step_sizes=[0.05, 0.02, 0.01, 0.005])
    
    return best_x
# EVOLVE-BLOCK-END
