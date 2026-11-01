"""Rewritten program: smooth surrogate + projected gradient descent for C3 minimization."""
import numpy as np
import time
import random

N = 400
DEADLINE = 145.0  # safety margin before 150s kill

def _autocorr_from_continuous(z):
    """
    Compute the autoconvolution sidelobe magnitudes for a continuous vector z
    (relaxed signed step function). Uses FFT for efficiency.
    Returns (sidelobe_mags, R0) where sidelobe_mags has shape (2*(N-1),) 
    and R0 is the zero-lag energy.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(z, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    # R[0] = sum z[i]^2
    # R[1:N] positive lags, R[N:2N-1] negative lags
    R0 = R[0]
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    return sidelobes, R0

def _c3_from_sidelobes(sidelobes, R0):
    """Compute C3 = max|sidelobe| / R0."""
    if R0 < 1e-15:
        return 1.0
    return np.max(np.abs(sidelobes)) / R0

def _c3_exact(z):
    """Exact C3 for a ±1 vector."""
    z = np.sign(z)
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(z, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    R0 = R[0]
    if R0 < 1e-15:
        return 1.0
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    return np.max(np.abs(sidelobes)) / R0

def _softmax_c3(z, tau=1.0):
    """
    Smooth surrogate for C3 using log-sum-exp.
    C3_soft = (1/tau) * log(sum exp(tau * |R[k]|)) / R0
    This is a smooth upper bound on max|sidelobe|/R0.
    """
    sidelobes, R0 = _autocorr_from_continuous(z)
    if R0 < 1e-15:
        return 1.0, np.zeros(N)
    abs_sidelobes = np.abs(sidelobes)
    # log-sum-exp for stability
    m = np.max(abs_sidelobes)
    log_sum_exp = m + np.log(np.sum(np.exp(tau * (abs_sidelobes - m))))
    c3_soft = (1.0 / tau) * log_sum_exp / R0
    return c3_soft, abs_sidelobes, R0

def _softmax_c3_grad(z, tau=1.0):
    """
    Compute the smooth C3 value and its gradient w.r.t. z.
    Uses automatic differentiation via finite differences on the FFT path
    for correctness, but we derive analytically for speed.
    
    For z in R^N, the autocorrelation R[k] = sum_i z[i] z[i+k].
    dR[k]/dz[j] = z[j+k] (if j+k < N) + z[j-k] (if j-k >= 0)
    
    The log-sum-exp gradient: 
    d/dz_j [ (1/tau) log(sum_l exp(tau |R[l]|)) / R0 ]
    
    This is complex. Instead, we use a simpler approach:
    Minimize sum of squared sidelobes (L2) as a smooth surrogate,
    which is related but different. Or use the max-sidelobe with
    subgradient.
    
    Actually, let's use a different smooth objective:
    Objective = (1/tau) * log(sum_k exp(tau * |R[k]|^2 / R0))
    This is smooth and its gradient is well-defined.
    
    Alternatively, minimize the L2 norm of sidelobes: sum_k R[k]^2 / R0^2
    This is smooth and easy to differentiate.
    """
    # Use L2-based smooth objective: sum_k (R[k]/R0)^2 for k != 0
    # This is equivalent to minimizing the total sidelobe energy
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(z, n=n_fft)
    # Power spectrum
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    R0 = R[0]
    if R0 < 1e-15:
        return 1.0, np.zeros(N)
    
    # Normalize sidelobes
    sidelobes_norm = np.concatenate([R[1:N], R[N:2*N-1]]) / R0
    obj = np.sum(sidelobes_norm ** 2)
    
    # Gradient of sum_k (R[k]/R0)^2 w.r.t. z
    # d/dz_j [sum_k R[k]^2 / R0^2]
    # = (2/R0^2) * sum_k R[k] * dR[k]/dz_j - 2*sum_k R[k]^2 * dR0/dz_j / R0^3
    # dR0/dz_j = 2*z[j]
    # dR[k]/dz_j = z[j+k] (if valid) + z[j-k] (if valid)
    
    # Efficient computation using FFT
    # sum_k R[k] * dR[k]/dz_j = sum_k R[k] * (z[j+k] + z[j-k])
    # This is a cross-correlation of R with z
    # Let's compute it directly for each j using the structure
    
    # Actually, for efficiency, note that:
    # sum_k R[k] * dR[k]/dz_j = (R * z)[j] in some sense
    # Let's compute the gradient via the frequency domain
    
    # d/dz_j sum_k R[k]^2 = 2 * sum_k R[k] * dR[k]/dz_j
    # dR[k]/dz_j = z[j+k] (for j+k<N) + z[j-k] (for j-k>=0)
    # So sum_k R[k] * dR[k]/dz_j = sum_k R[k]*z[j+k] + sum_k R[k]*z[j-k]
    # = (R * z)[j] + (R * z)[N-1-j] approximately (circular)
    
    # For linear autocorrelation, let's just compute it directly
    # But that's O(N^2). For N=400, that's 160K ops per gradient eval, which is fine.
    
    # Let's compute the gradient more carefully.
    # R[k] = sum_i z[i]*z[i+k] for k=0..N-1
    # The full autocorrelation (both signs) has 2N-1 values.
    
    # Gradient of sum_{k=1}^{N-1} R[k]^2 w.r.t. z[j]:
    # = 2 * sum_{k=1}^{N-1} R[k] * dR[k]/dz[j]
    # dR[k]/dz[j] = z[j+k] (if j+k<N) + z[j-k] (if j-k>=0)
    
    # Let's compute this efficiently.
    # sum_{k=1}^{N-1} R[k] * z[j+k] (for j+k<N)
    # + sum_{k=1}^{N-1} R[k] * z[j-k] (for j-k>=0)
    
    # This is like a correlation of R[1:N] with z.
    # Let's use FFT for this too.
    
    # Actually, for N=400, direct computation is fast enough.
    grad = np.zeros(N)
    R_pos = R[1:N]  # positive lags, length N-1
    
    for j in range(N):
        # sum_{k=1}^{N-1} R[k] * z[j+k] for j+k < N
        # k from 1 to N-1-j
        if j < N - 1:
            s1 = np.dot(R_pos[:N-1-j], z[j+1:N])
        else:
            s1 = 0.0
        # sum_{k=1}^{N-1} R[k] * z[j-k] for j-k >= 0
        # k from 1 to j
        if j > 0:
            s2 = np.dot(R_pos[:j], z[j-1::-1])
        else:
            s2 = 0.0
        grad[j] = 2.0 * (s1 + s2)
    
    # Now account for the 1/R0^2 factor and the derivative of R0
    # d/dz_j [sum R[k]^2 / R0^2] = (2/R0^2) * (s1+s2) - 2*sum R[k]^2 * 2*z[j] / R0^3
    sum_R2 = np.sum(R_pos ** 2)
    grad = (2.0 / R0 ** 2) * grad - (4.0 * sum_R2 / R0 ** 3) * z
    
    return obj, grad

def _projected_gradient_descent(z_init, max_iter=5000, lr=0.01, tau=1.0, time_limit=None):
    """
    Projected gradient descent on the continuous relaxation.
    At each step, we take a gradient step and then project back to the
    ±1 constraint (by taking sign).
    """
    start_time = time.time()
    z = z_init.copy()
    z = np.sign(z) if np.all(np.abs(z) > 0.5) else z  # start from continuous
    
    best_z = z.copy()
    best_obj = _c3_exact(z)
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - 5.0:
            break
        
        # Compute gradient of smooth objective
        obj, grad = _softmax_c3_grad(z, tau=tau)
        
        # Gradient step
        z_new = z - lr * grad
        
        # Project to ±1 (hard projection)
        z_new = np.sign(z_new)
        
        # Evaluate exact C3
        c3_new = _c3_exact(z_new)
        
        if c3_new < best_obj - 1e-12:
            best_z = z_new.copy()
            best_obj = c3_new
        z = z_new
    
    return best_z, best_obj

def _local_search(x, max_iter=2000, time_limit=None):
    """
    Local search with single-bit flips, using FFT-based C3 computation.
    """
    start_time = time.time()
    current_c3 = _c3_exact(x)
    best_x = x.copy()
    best_c3 = current_c3
    no_improve = 0
    max_no_improve = 50
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - 5.0:
            break
        
        idx = random.randint(0, N - 1)
        x_new = x.copy()
        x_new[idx] = -x_new[idx]
        new_c3 = _c3_exact(x_new)
        
        if new_c3 < current_c3 - 1e-12:
            x = x_new
            current_c3 = new_c3
            no_improve = 0
            if current_c3 < best_c3:
                best_x = x.copy()
                best_c3 = current_c3
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
    
    return best_x, best_c3

def _construct():
    """
    Main construction: combine smooth gradient search with local search.
    """
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    # Initialize with random ±1 vector
    z = np.random.choice([-1.0, 1.0], size=N)
    best_x = z.copy()
    best_c3 = _c3_exact(z)
    
    # Phase 1: Projected gradient descent from multiple random starts
    for trial in range(3):
        if time.time() > DEADLINE - 10.0:
            break
        z_init = np.random.choice([-1.0, 1.0], size=N)
        z_trial, c3_trial = _projected_gradient_descent(
            z_init, max_iter=2000, lr=0.05, tau=1.0,
            time_limit=min(30.0, DEADLINE - 10.0 - time.time())
        )
        if c3_trial < best_c3:
            best_x = z_trial
            best_c3 = c3_trial
    
    # Phase 2: Local search polish
    if time.time() < DEADLINE - 5.0:
        x_polished, c3_polished = _local_search(
            best_x, max_iter=3000,
            time_limit=min(40.0, DEADLINE - 5.0 - time.time())
        )
        if c3_polished < best_c3:
            best_x = x_polished
            best_c3 = c3_polished
    
    # Phase 3: Another round of gradient search from best
    if time.time() < DEADLINE - 5.0:
        z_trial, c3_trial = _projected_gradient_descent(
            best_x.astype(float), max_iter=2000, lr=0.02, tau=1.0,
            time_limit=min(30.0, DEADLINE - 5.0 - time.time())
        )
        if c3_trial < best_c3:
            best_x = z_trial
            best_c3 = c3_trial
    
    # Final local search
    if time.time() < DEADLINE - 3.0:
        x_final, c3_final = _local_search(
            best_x, max_iter=2000,
            time_limit=max(1.0, DEADLINE - 3.0 - time.time())
        )
        if c3_final < best_c3:
            best_x = x_final
            best_c3 = c3_final
    
    return best_x

def construct():
    """
    Entry point: return a sequence of signed step heights (±1) of length N=400.
    """
    x = _construct()
    return np.sign(x).astype(int)
