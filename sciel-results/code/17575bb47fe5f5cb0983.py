"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill


def _time_remaining():
    return _DEADLINE - (time.time() - _start_time)


def _autoconv_linear(x):
    """Compute linear autoconvolution using FFT."""
    n = len(x)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    X = np.fft.rfft(x, n=fft_size)
    C = np.fft.irfft(X * X.conj(), n=fft_size)
    return C[:size]


def _score(x):
    """Compute max|autoconv| / (sum x)^2. Return inf if sum is zero."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    c = _autoconv_linear(x)
    return np.max(np.abs(c)) / (s * s)


def _hann_window(n):
    """Hann window of length n."""
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / (n - 1)))


def _raised_cosine_window(n, alpha=0.5):
    """Raised cosine (Kaiser-like) window with parameter alpha in [0,1]."""
    t = np.arange(n)
    if alpha <= 0:
        return np.ones(n)
    # Smooth transition at edges
    x = np.ones(n)
    edge = int(alpha * n)
    if edge > 0:
        t_edge = np.arange(edge)
        x[:edge] = 0.5 * (1 + np.cos(np.pi * t_edge / edge - np.pi))
        x[-edge:] = 0.5 * (1 + np.cos(np.pi * t_edge / edge - np.pi))
    return x


def _signed_from_continuous(x):
    """Convert continuous vector to signed ±1 with small perturbation for tie-breaking."""
    return np.sign(x)


def _gradient_step(x, lr):
    """
    Subgradient step for minimizing max|autoconv| / (sum x)^2.
    Uses FFT-based gradient of the objective.
    """
    n = len(x)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return x
    
    # Compute autoconvolution
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    X = np.fft.rfft(x, n=fft_size)
    C = np.fft.irfft(X * X.conj(), n=fft_size)
    C = C[:size]
    
    # The objective is max_k |C[k]| / s^2
    # Find the index of maximum absolute value
    idx = np.argmax(np.abs(C))
    c_max = C[idx]
    
    # Gradient of C[k] w.r.t. x[i] for linear autoconvolution:
    # C[k] = sum_j x[j] * x[j + k]  (for k >= 0)
    # dC[k]/dx[i] = x[i+k] + x[i-k] (with appropriate bounds)
    # For the sign, we need gradient of |C[idx]|
    sign_c = np.sign(c_max)
    
    # Build gradient vector
    grad = np.zeros(n)
    for i in range(n):
        g = 0.0
        # Contribution from x[i] as first factor: x[i+k]
        j = i + idx
        if j < n:
            g += sign_c * x[j]
        # Contribution from x[i] as second factor: x[i-k]
        j = i - idx
        if j >= 0:
            g += sign_c * x[j]
        grad[i] = g
    
    # Objective: f = |C[idx]| / s^2
    # df/dx_i = (grad[i] * s^2 - |C[idx]| * 2 * s) / s^4
    #         = grad[i] / s^2 - 2*|C[idx]| / s^3
    #         = (grad[i] * s - 2*|C[idx]|) / s^3
    
    obj_grad = (grad * s - 2 * abs(c_max)) / (s ** 3)
    
    # Gradient descent step
    x_new = x - lr * obj_grad
    
    # Keep values bounded (project to [-2, 2] to maintain sign structure)
    x_new = np.clip(x_new, -2.0, 2.0)
    
    return x_new


def construct():
    """Construct signed step function minimizing max|autoconv|/integral^2."""
    # Phase 1: Try deterministic tapered windows
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    # Try various window shapes
    windows = []
    windows.append(('ones', np.ones(N)))
    windows.append(('hann', _hann_window(N)))
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        windows.append((f'rc_{alpha}', _raised_cosine_window(N, alpha)))
    
    # Also try sign-flipped and shifted versions
    for name, w in windows[:]:
        for shift in range(0, N, max(1, N // 10)):
            w_shifted = np.roll(w, shift)
            # Ensure nonzero sum
            s = np.sum(w_shifted)
            if abs(s) > 1e-12:
                sc = _score(w_shifted)
                if sc < best_score:
                    best_score = sc
                    best_x = w_shifted.copy()
        
        # Sign-flipped version
        w_neg = -w
        s = np.sum(w_neg)
        if abs(s) > 1e-12:
            sc = _score(w_neg)
            if sc < best_score:
                best_score = sc
                best_x = w_neg.copy()
    
    # Phase 2: Continuous gradient-based refinement
    x_cont = best_x.astype(float)
    
    # Adaptive learning rate gradient descent
    lr = 0.1
    patience = 0
    best_cont_score = best_score
    
    for step in range(2000):
        if _time_remaining() < 10.0:
            break
        
        x_new = _gradient_step(x_cont, lr)
        
        # Evaluate
        sc = _score(x_new)
        if sc < best_cont_score:
            best_cont_score = sc
            x_cont = x_new.copy()
            patience = 0
            # Slowly decay learning rate
            lr *= 0.99
        else:
            patience += 1
            lr *= 0.95
            if patience > 50:
                # Restart with smaller lr
                lr = max(0.001, lr * 0.5)
                patience = 0
    
    # Phase 3: Round to signs and do local sign-flip polish
    x_sign = np.sign(x_cont)
    # Handle zeros by using the continuous value's sign with small bias
    zeros = np.where(x_sign == 0)[0]
    if len(zeros) > 0:
        x_sign[zeros] = np.sign(x_cont[zeros] + 1e-6)
    
    sc = _score(x_sign)
    if sc < best_score:
        best_score = sc
        best_x = x_sign.copy()
    
    # Phase 4: Local sign-flip search on the best candidate
    x = best_x.copy()
    improved = True
    max_rounds = 20
    for _ in range(max_rounds):
        if _time_remaining() < 5.0:
            break
        improved = False
        for i in range(N):
            if _time_remaining() < 3.0:
                break
            old_val = x[i]
            x[i] = -old_val
            sc = _score(x)
            if sc < best_score:
                best_score = sc
                best_x = x.copy()
                improved = True
            else:
                x[i] = old_val
        if not improved:
            break
    
    # Ensure output is ±1
    return np.sign(best_x).astype(float)
# EVOLVE-BLOCK-END
