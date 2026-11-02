"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv(f):
    """Compute full linear autoconvolution via FFT."""
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    F = np.fft.rfft(f, size)
    ac = np.fft.irfft(F * F, size)[:2 * n - 1]
    return ac


def _score(f):
    """Compute max|autoconv| / (sum f)^2."""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autoconv(f)
    return np.max(np.abs(ac)) / (s * s)


def _spectral_objective(f):
    """
    Compute the peak of |F(ω)|^2 / (sum f)^2, which is proportional to 
    max|autoconv| / (sum f)^2.
    
    The autoconvolution is the inverse FFT of |F|^2, so its peak is bounded
    by the peak of |F|^2 in frequency domain.
    """
    n = len(f)
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    
    # Compute DFT
    size = 1
    while size < 2 * n - 1:
        size *= 2
    F = np.fft.rfft(f, size)
    power = np.abs(F) ** 2
    
    # The peak of the autoconvolution is the max of the inverse FFT of power
    # But we can use the spectral peak as a surrogate
    peak_power = np.max(power)
    
    return peak_power / (s * s)


def _spectral_gradient(f):
    """
    Compute gradient of the spectral objective.
    We use a smooth approximation: max_k |F_k|^2 ≈ exp(β * max_k |F_k|^2)
    """
    n = len(f)
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.zeros(n)
    
    size = 1
    while size < 2 * n - 1:
        size *= 2
    F = np.fft.rfft(f, size)
    power = np.abs(F) ** 2
    
    # Find peak frequency
    peak_idx = np.argmax(power)
    
    # Gradient of |F_k|^2 with respect to f
    # F_k = sum_j f_j * exp(-2πi*k*j/size)
    # |F_k|^2 = F_k * conj(F_k)
    # d|F_k|^2/df_j = conj(F_k) * exp(-2πi*k*j/size) + F_k * exp(2πi*k*j/size)
    #               = 2 * Re[conj(F_k) * exp(-2πi*k*j/size)]
    
    k = peak_idx
    phase = np.exp(-2j * np.pi * k * np.arange(n) / size)
    grad_power = 2 * np.real(np.conj(F[k]) * phase)
    
    # Gradient of ratio: d(peak_power / s^2) / df
    # = (d(peak_power)/df * s^2 - peak_power * 2 * s * 1) / s^4
    # = (grad_power * s - 2 * peak_power) / s^3
    
    grad = (grad_power * s - 2 * power[peak_idx]) / (s ** 3)
    
    return grad


def _spectral_descent(f, deadline, max_iters=2000, lr=0.01):
    """
    Spectral gradient descent to minimize the peak of |F|^2 / (sum f)^2.
    """
    current = f.copy()
    best = f.copy()
    best_obj = _spectral_objective(f)
    
    for it in range(max_iters):
        if time.time() > deadline - 10.0:
            break
        
        # Compute gradient
        grad = _spectral_gradient(current)
        
        # Gradient step
        current = current - lr * grad
        
        # Renormalize to keep sum = N
        s = np.sum(current)
        if abs(s) < 1e-10:
            current = f.copy()
            continue
        current = current * (N / s)
        
        # Evaluate
        obj = _spectral_objective(current)
        if obj < best_obj:
            best_obj = obj
            best = current.copy()
        
        # Adaptive learning rate
        if it % 100 == 0:
            lr = max(0.001, lr * 0.9)
    
    return best, best_obj


def construct():
    deadline = time.time() + 120.0  # safety margin before 150s kill
    
    # Start with all ones (the seed)
    best = np.ones(N)
    best_score = _score(best)
    
    # Phase 1: Spectral optimization
    f = best.copy().astype(float)
    f_opt, obj_opt = _spectral_descent(f, deadline, max_iters=1500, lr=0.005)
    
    # Evaluate the optimized continuous solution
    sc_opt = _score(f_opt)
    if sc_opt < best_score:
        best_score = sc_opt
        best = f_opt.copy()
    
    # Phase 2: Round to signed values
    result = np.sign(best)
    result[result == 0] = 1.0
    
    rounded_score = _score(result)
    if rounded_score < best_score:
        best_score = rounded_score
        best = result.copy()
    
    # Phase 3: Local search on the rounded result
    f = best.copy().astype(float)
    current_score = best_score
    np.random.seed(42)
    
    for iteration in range(30000):
        if time.time() > deadline - 10.0:
            break
        
        # Pick a random index
        idx = np.random.randint(N)
        old_val = f[idx]
        f[idx] += np.random.choice([-1, 1]) * 0.5
        
        # Renormalize
        s = np.sum(f)
        if abs(s) < 1e-10:
            f[idx] = old_val
            continue
        f = f * (N / s)
        
        sc = _score(f)
        if sc < current_score:
            current_score = sc
            if current_score < best_score:
                best_score = current_score
                best = f.copy()
        else:
            f[idx] = old_val
            s = np.sum(f)
            if abs(s) >= 1e-10:
                f = f * (N / s)
    
    # Final rounding
    result = np.sign(best)
    result[result == 0] = 1.0
    
    final_score = _score(result)
    if final_score < best_score:
        best_score = final_score
        best = result.copy()
    
    # Phase 4: Try a few random ±1 sequences
    for _ in range(50):
        if time.time() > deadline - 5.0:
            break
        rand_seq = np.random.choice([-1.0, 1.0], N)
        rs = _score(rand_seq)
        if rs < best_score:
            best_score = rs
            best = rand_seq
    
    return np.sign(best)
# EVOLVE-BLOCK-END
