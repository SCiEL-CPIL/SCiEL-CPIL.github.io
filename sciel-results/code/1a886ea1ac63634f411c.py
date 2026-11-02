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
    # Ensure symmetry for even N: P[N-k] = P[k].
    # Indices: 0..N-1.
    # P[0] is DC. P[N/2] is Nyquist.
    # If N is even:
    # P[1..N/2-1] mirror to P[N-1..N/2+1].
    # P[N/2] is self-symmetric.
    
    # Calculate c = IFFT(P_full)
    c = np.fft.ifft(P_full).real
    # c is real and even.
    
    s_sq = P_full[0] # (sum x)^2 = |X[0]|^2 = P[0]
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
    # d(soft_max)/dP[k] = sum_n weights[n] * sign(c[n]) * (1/N) * exp(i * 2*pi * n * k / N)
    # Let W[n] = weights[n] * sign(c[n])
    # Then d(soft_max)/dP[k] = (1/N) * FFT(W)[k]  (using standard FFT convention where IFFT has 1/N)
    # Actually, if c = IFFT(P), then P = FFT(c).
    # c[n] = 1/N sum_k P[k] exp(i 2pi nk/N).
    # dc[n]/dP[k] = 1/N exp(i 2pi nk/N).
    # d(soft_max)/dP[k] = sum_n w[n] sgn(c[n]) (1/N) exp(i 2pi nk/N)
    # This is (1/N) * FFT(w * sgn(c))[k].
    
    W = weights * np.sign(c)
    d_soft_dP_full = np.fft.fft(W) / N
    
    # We need d(val)/dparams[j]
    # val = soft_max / P[0]
    # d(val)/dP[k] = (1/P[0]) * d(soft_max)/dP[k] - (soft_max / P[0]^2) * d(P[0])/dP[k]
    # d(P[0])/dP[k] is 1 if k==0 else 0.
    
    d_val_dP_full = d_soft_dP_full / P_full[0]
    d_val_dP_full[0] -= soft_max / (P_full[0] ** 2)
    
    # Map d_val_dP_full to d_val_dparams
    # params[j] corresponds to P[j] for j=0..half.
    # P[k] = exp(params[k]).
    # dP[k]/dparams[k] = P[k].
    
    grad = np.zeros(half+1)
    for k in range(half+1):
        # Sum contributions from P[k] and P[N-k] if they are distinct variables in P_full
        # But P_full is constructed from params.
        # dP_full[m]/dparams[j] = P[j] if m==j or (m==N-j and N-j != j).
        # If m==j and m==N-j (i.e., j=0 or j=N/2), then dP_full[j]/dparams[j] = P[j].
        # Otherwise, for j in 1..N/2-1, P_full[j] and P_full[N-j] both depend on params[j].
        
        contrib = 0.0
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
    
    # X = sqrt(P) * exp(i*phase). Phase = 0.
    X = np.sqrt(P_full)
    x = np.fft.ifft(X).real
    return x

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Initial PSD: Flat spectrum (White noise)
    # P[k] = 1 for all k.
    # params = log(1) = 0.
    half = N // 2
    params0 = np.zeros(half+1)
    
    # Try a few different initial PSD shapes
    # 1. Flat
    # 2. Low-pass (DC heavy)
    # 3. High-pass (Nyquist heavy)
    
    candidates = []
    
    # Flat
    candidates.append(np.zeros(half+1))
    
    # Low-pass: P[k] ~ exp(-alpha * k)
    for alpha in [0.1, 0.5, 1.0, 2.0]:
        k = np.arange(half+1)
        candidates.append(-alpha * k)
        
    # High-pass: P[k] ~ exp(-alpha * (N-k))
    for alpha in [0.1, 0.5, 1.0, 2.0]:
        k = np.arange(half+1)
        candidates.append(-alpha * (N - k))
        
    # Band-pass
    for center in [N//4, N//2, 3*N//4]:
        k = np.arange(half+1)
        candidates.append(-(k - center)**2 / (N/4))

    # Optimize each candidate
    for p0 in candidates:
        if _check_deadline():
            break
        
        try:
            result = minimize(_psd_objective_and_grad, p0, method='L-BFGS-B', jac=True,
                             options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-12})
            
            if result.success or result.fun < best_val:
                # Recover x
                x_rec = _recover_x_from_psd(result.x)
                val_rec = _objective(x_rec)
                
                if val_rec < best_val:
                    best_val = val_rec
                    best_x = x_rec.copy()
                    
        except Exception as e:
            pass

    if _check_deadline():
        return best_x

    # Refine the best PSD found
    # Find the best params from the last successful optimization or re-optimize from best_x's PSD
    # Let's just do a final polish on the best_x using time-domain coordinate descent or L-BFGS on x
    # Since we changed representation, let's stick to PSD optimization for the final step too.
    
    # Get PSD of current best_x
    X_best = np.fft.rfft(best_x)
    P_best = np.abs(X_best)**2
    P_best[P_best < 1e-10] = 1e-10
    params_best = np.log(P_best)
    
    # Optimize params_best
    try:
        result = minimize(_psd_objective_and_grad, params_best, method='L-BFGS-B', jac=True,
                         options={'maxiter': 300, 'gtol': 1e-8, 'ftol': 1e-14})
        if result.fun < best_val:
            x_rec = _recover_x_from_psd(result.x)
            val_rec = _objective(x_rec)
            if val_rec < best_val:
                best_val = val_rec
                best_x = x_rec.copy()
    except Exception:
        pass

    # Final time-domain polish
    if not _check_deadline():
        # Simple coordinate descent
        best_val = _objective(best_x)
        for _ in range(1000):
            if _check_deadline():
                break
            improved = False
            for j in range(N):
                for delta in [-0.05, 0.05]:
                    trial = best_x.copy()
                    trial[j] += delta
                    val = _objective(trial)
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        improved = True
                        break
            if not improved:
                break

    return best_x
# EVOLVE-BLOCK-END
