"""All-pass phase-optimized autoconvolution peak minimization."""
import numpy as np
import time

N = 400
# Number of all-pass stages (each contributes 2 phase params effectively via real coeff)
M_STAGES = 8

def _autoconv(x):
    """Full autoconvolution via FFT."""
    n = len(x)
    xp = np.zeros(2 * n, dtype=np.float64)
    xp[:n] = x
    X = np.fft.rfft(xp)
    C = np.fft.irfft(X * X.conj(), n=2 * n)
    return C

def _max_abs_ratio(x):
    """max|autoconv| / (sum x)^2"""
    S = np.sum(x)
    if abs(S) < 1e-12:
        return np.inf, 0.0, np.zeros(len(x))
    C = _autoconv(x)
    peak = np.max(np.abs(C))
    idx = int(np.argmax(np.abs(C)))
    ratio = peak / (S * S)
    return ratio, peak, C

def _allpass_impulse_response(coeffs, N):
    """
    Compute first N samples of impulse response of cascade of first-order
    real all-pass filters.
    coeffs: array of shape (M,) with |a_m| < 1.
    Each stage: y[n] = a_m * y[n-1] + x[n] + a_m * x[n-1]  (wait, standard form)
    
    Standard first-order all-pass (real): 
        H(z) = (a + z^{-1}) / (1 + a z^{-1})
    Difference equation: y[n] = -a*y[n-1] + x[n] + a*x[n-1]
    Actually let's use: H(z) = (z^{-1} + a)/(1 + a z^{-1})
    => y[n] + a*y[n-1] = x[n-1] + a*x[n]
    => y[n] = -a*y[n-1] + x[n-1] + a*x[n]
    """
    n = N
    x = np.zeros(n)
    x[0] = 1.0  # impulse
    y = np.zeros(n)
    for m in range(len(coeffs)):
        a = coeffs[m]
        # y_new[n] = -a * y[n-1] + x[n-1] + a * x[n]
        y_new = np.zeros(n)
        for k in range(n):
            val = 0.0
            if k - 1 >= 0:
                val += -a * y[k-1] + x[k-1]
            if k < n:
                val += a * x[k]
            y_new[k] = val
        x = y_new
        y = y_new
    return y

def _allpass_impulse_response_fast(coeffs, N):
    """
    Faster: use the fact that cascade of all-pass can be computed via
    polynomial division or direct convolution of numerator/denominator.
    
    For a single all-pass (a + z^-1)/(1 + a z^-1), impulse response is:
    h[0] = a
    h[n] = (-a)^n * (1 - a)  for n >= 1?  Let me verify.
    
    H(z) = (a + z^{-1})/(1 + a z^{-1}) = (a z + 1)/(z + a)
    h[0]: coefficient of z^0 in H(z) = a (from a*z/(z+a) at z=inf gives a, plus 1/(z+a) gives 0 at z=inf... )
    Actually let's just do the difference equation but vectorized per stage.
    """
    n = N
    x = np.zeros(n)
    x[0] = 1.0
    for m in range(len(coeffs)):
        a = coeffs[m]
        y = np.zeros(n)
        # y[k] = -a*y[k-1] + x[k-1] + a*x[k]
        y[0] = a * x[0]  # x[-1]=0, y[-1]=0
        for k in range(1, n):
            y[k] = -a * y[k-1] + x[k-1] + a * x[k]
        x = y
    return x

def _allpass_via_fft(coeffs, N):
    """
    Compute impulse response via FFT: H(z) = prod (a_m + z^{-1})/(1 + a_m z^{-1}).
    Numerator: prod (a_m z + 1) = sum of z^k terms
    Denominator: prod (z + a_m)
    h[n] = coeff of z^{-(n)} in H(z) = coeff of z^{L-n} in (num * 1/den) where L is degree.
    
    Simpler: compute H at DFT points and ifft.
    """
    # Use a large FFT size to avoid wrap-around
    fft_size = 4 * N
    freqs = np.fft.fftfreq(fft_size) * 2 * np.pi
    # H(e^{jw}) = prod (a_m + e^{-jw}) / (1 + a_m e^{-jw})
    H = np.ones(fft_size, dtype=np.complex128)
    ew = np.exp(-1j * freqs)
    for m in range(len(coeffs)):
        a = coeffs[m]
        H *= (a + ew) / (1 + a * ew)
    h = np.fft.ifft(H).real
    return h[:N]

def _objective_and_grad(coeffs, N):
    """
    Compute max|autoconv|/(sum)^2 and its gradient w.r.t. coeffs.
    We use the continuous relaxation: x = allpass_impulse_response(coeffs), 
    then we want to minimize f(x) = max_k |C[k]| / S^2.
    
    For gradient, we use the subgradient: if peak is at lag k*, then
    grad_f = (grad_{x} |C[k*]| * S^2 - 2*S*|C[k*]|*grad_x S) / S^4
    But max is non-smooth. We use a smooth surrogate: 
    g(x) = (sum_k C[k]^4)^{1/4} / S^2  (approximates max)
    or simply minimize sum_{k != 0} C[k]^2 / S^2 as a smooth proxy,
    plus a penalty on the peak.
    
    Let's use: obj = max(|C|) approximated by (sum |C|^p)^{1/p} for large p,
    but that's expensive. Instead, use the smooth energy + peak tracking.
    
    Actually, let's use the approach: minimize E = sum_{k=1}^{N-1} C[k]^2 / S^2
    which is smooth and differentiable, as a proxy. The all-pass structure
    already fixes the total off-peak energy (since |H|^2 = 1, the infinite
    autocorr is delta, so sum C[k]^2 for k != 0 is bounded by truncation effects).
    We can then do local search on the rounded sequence.
    """
    x = _allpass_via_fft(coeffs, N)
    S = np.sum(x)
    if abs(S) < 1e-12:
        return np.inf, np.zeros(len(coeffs))
    
    C = _autoconv(x)
    # Smooth objective: sum of squared off-peak autocorr / S^2
    # C is length 2N, centered at N
    off_peak_sq = np.sum(C[N+1:]**2) + np.sum(C[1:N]**2)
    obj = off_peak_sq / (S * S)
    
    # Gradient via finite differences (since analytic is complex)
    # For speed, use central differences with small step
    grad = np.zeros(len(coeffs))
    h = 1e-5
    for i in range(len(coeffs)):
        cp = coeffs.copy()
        cm = coeffs.copy()
        cp[i] += h
        cm[i] -= h
        xp = _allpass_via_fft(cp, N)
        xm = _allpass_via_fft(cm, N)
        Sp = np.sum(xp)
        Sm = np.sum(xm)
        if abs(Sp) < 1e-12 or abs(Sm) < 1e-12:
            grad[i] = 0.0
            continue
        Cp = _autoconv(xp)
        Cm = _autoconv(xm)
        opp = (np.sum(Cp[N+1:]**2) + np.sum(Cp[1:N]**2)) / (Sp * Sp)
        om = (np.sum(Cm[N+1:]**2) + np.sum(Cm[1:N]**2)) / (Sm * Sm)
        grad[i] = (opp - om) / (2 * h)
    
    return obj, grad

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    np.random.seed(12345)
    
    # Initial random all-pass coefficients
    # |a_m| < 1, keep away from boundaries
    coeffs = np.random.uniform(-0.7, 0.7, size=M_STAGES)
    
    # Evaluate initial
    x = _allpass_via_fft(coeffs, N)
    best_x = np.sign(x)
    best_x[best_x == 0] = 1.0
    best_score, _, _ = _max_abs_ratio(best_x)
    
    # Phase 1: Gradient descent on continuous all-pass coefficients
    lr = 0.05
    for step in range(3000):
        if time.time() > deadline:
            break
        
        obj, grad = _objective_and_grad(coeffs, N)
        if not np.isfinite(obj):
            coeffs = np.random.uniform(-0.7, 0.7, size=M_STAGES)
            continue
        
        # Gradient step
        coeffs -= lr * grad
        # Keep |a_m| < 0.95
        coeffs = np.clip(coeffs, -0.95, 0.95)
        
        # Evaluate rounded
        if step % 20 == 0:
            x = _allpass_via_fft(coeffs, N)
            x_rounded = np.sign(x)
            x_rounded[x_rounded == 0] = 1.0
            s, _, _ = _max_abs_ratio(x_rounded)
            if s < best_score:
                best_score = s
                best_x = x_rounded.copy()
        
        # Adaptive learning rate
        if step % 500 == 0 and step > 0:
            lr *= 0.8
    
    # Final evaluation
    x = _allpass_via_fft(coeffs, N)
    x_rounded = np.sign(x)
    x_rounded[x_rounded == 0] = 1.0
    s, _, _ = _max_abs_ratio(x_rounded)
    if s < best_score:
        best_score = s
        best_x = x_rounded.copy()
    
    # Phase 2: Local search on the rounded sequence
    x = best_x.astype(np.float64)
    current_score = best_score
    
    # Hill climbing: flip bits that improve
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            x[i] = -x[i]
            new_score, _, _ = _max_abs_ratio(x)
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
                improved = True
            else:
                x[i] = -x[i]
    
    # Phase 3: Simulated annealing for escape from local minima
    if time.time() < deadline:
        T = 0.005
        T_min = 1e-7
        cooling = 0.998
        
        for step in range(20000):
            if time.time() > deadline:
                break
            if T < T_min:
                break
            
            idx = np.random.randint(N)
            old_val = x[idx]
            x[idx] = -old_val
            
            new_score, _, _ = _max_abs_ratio(x)
            delta = new_score - current_score
            
            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-12)):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
            else:
                x[idx] = old_val
            
            T *= cooling
    
    return best_x.astype(np.int8)
