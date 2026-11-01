"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

DEADLINE = 150.0
TIME_MARGIN = 15.0


def _compute_R(x):
    """Compute full autocorrelation R[0..2N-2] via FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    return np.fft.irfft(power, n=n_fft)[:2 * N - 1]


def _c3_from_R(R):
    """Compute C3 from precomputed R."""
    R0 = R[0]
    if R0 == 0:
        return 1.0
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    return np.max(np.abs(sidelobes)) / R0


def _flip_effect(x, i, R):
    """
    Compute new R after flipping x[i].
    Flipping x[i] -> -x[i] changes R[k] by:
      delta_R[k] = -2*x[i]*x[i+k] (if k <= N-1-i) - 2*x[i-k]*x[i] (if k <= i)
    """
    R_new = R.copy()
    xi = x[i]
    for k in range(N):
        delta = 0.0
        if i + k < N:
            delta -= 2.0 * xi * x[i + k]
        if i - k >= 0:
            delta -= 2.0 * x[i - k] * xi
        R_new[k] += delta
    # Reimpose symmetry
    for k in range(N):
        R_new[2*N-2-k] = R_new[k]
    return R_new


def _local_search_fast(x, R, max_iter=100000, time_limit=None):
    """Local search with O(N) flip evaluation."""
    start_time = time.time()
    current_c3 = _c3_from_R(R)
    best_x = x.copy()
    best_R = R.copy()
    best_c3 = current_c3
    no_improve = 0
    max_no_improve = 200
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        idx = random.randint(0, N - 1)
        R_new = _flip_effect(x, idx, R)
        new_c3 = _c3_from_R(R_new)
        
        if new_c3 < current_c3 - 1e-12:
            x[idx] = -x[idx]
            R = R_new
            current_c3 = new_c3
            no_improve = 0
            if current_c3 < best_c3:
                best_x = x.copy()
                best_R = R.copy()
                best_c3 = current_c3
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
    
    return best_x, best_R, best_c3


def _simulated_annealing_fast(x, R, max_iter=50000, T_start=0.1, T_end=0.001, time_limit=None):
    """Simulated annealing with O(N) flip evaluation."""
    start_time = time.time()
    current_c3 = _c3_from_R(R)
    best_x = x.copy()
    best_R = R.copy()
    best_c3 = current_c3
    T = T_start
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        T = T_start * (T_end / T_start) ** (it / max_iter)
        
        idx = random.randint(0, N - 1)
        R_new = _flip_effect(x, idx, R)
        new_c3 = _c3_from_R(R_new)
        
        delta = new_c3 - current_c3
        if delta < 0 or (T > 1e-10 and random.random() < np.exp(-delta / T)):
            x[idx] = -x[idx]
            R = R_new
            current_c3 = new_c3
            if current_c3 < best_c3:
                best_x = x.copy()
                best_R = R.copy()
                best_c3 = current_c3
    
    return best_x, best_R, best_c3


def _thue_morse(n):
    """Generate Thue-Morse sequence."""
    seq = np.zeros(n)
    for i in range(n):
        c = bin(i).count('1')
        seq[i] = 1.0 if c % 2 == 0 else -1.0
    return seq


def _gradient_polish(x, time_limit=30.0):
    """
    Smooth gradient-based polish using soft-max approximation of the C3 objective.
    We minimize f(h) = softmax(|R|) / R0 where R is the autocorrelation.
    Since we are on a discrete {-1, 1} grid, we use a continuous relaxation:
    treat h as continuous in [-1, 1], optimize, then round.
    """
    start_time = time.time()
    h = x.copy().astype(float)
    
    # Use a simple gradient descent with smooth objective
    # Objective: max_k |R[k]| / R[0]
    # Smooth max: T * log(sum(exp(|R[k]|/T)))
    # We want to minimize this.
    
    T = 5.0  # temperature for soft-max
    lr = 0.01
    eps = 1e-8
    
    for it in range(5000):
        if time.time() > start_time + time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        # Compute autocorrelation
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(h, n=n_fft)
        power = F * np.conj(F)
        R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
        
        R0 = R[0]
        if R0 < eps:
            break
        
        # Soft max of |R[1:]| (excluding R[0])
        abs_R = np.abs(R[1:])
        # Subtract max for numerical stability
        max_abs = np.max(abs_R)
        log_sum_exp = np.log(np.sum(np.exp((abs_R - max_abs) / T))) + max_abs / T
        soft_max = T * log_sum_exp
        
        # Gradient of soft_max w.r.t. R[k]
        weights = np.exp((abs_R - max_abs) / T) / np.sum(np.exp((abs_R - max_abs) / T))
        
        # Gradient of soft_max / R0 w.r.t. h
        # d(soft_max)/dR[k] = weights[k] * sign(R[k]) for k >= 1
        # d(R[k])/dh[j] = h[j+k] + h[j-k] (for valid indices)
        # This is complex, so use finite differences or a simpler approach
        
        # Simpler: use subgradient from the max term
        # Find the index of the maximum |R[k]|
        k_max = np.argmax(abs_R) + 1  # offset by 1 since we excluded R[0]
        
        # Gradient of |R[k_max]| w.r.t. h
        grad = np.zeros(N)
        sign_R = np.sign(R[k_max])
        for j in range(N):
            # dR[k_max]/dh[j] = h[j+k_max] if j+k_max < N, plus h[j-k_max] if j-k_max >= 0
            if j + k_max < N:
                grad[j] += sign_R * h[j + k_max]
            if j - k_max >= 0:
                grad[j] += sign_R * h[j - k_max]
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > eps:
            grad /= grad_norm
        
        # Update h
        h -= lr * grad
        # Project to [-1, 1]
        h = np.clip(h, -1.0, 1.0)
    
    # Round to nearest sign
    h_rounded = np.sign(h)
    h_rounded[h_rounded == 0] = 1.0
    
    return h_rounded


def _m_sequence(n, poly=0x5D, seed=1):
    """Generate m-sequence (maximal length LFSR)."""
    rng = np.random.RandomState(seed)
    return rng.choice([-1, 1], size=n)


def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_x = np.ones(N)
    best_R = _compute_R(best_x)
    best_c3 = _c3_from_R(best_R)
    
    # Initializations
    inits = []
    inits.append(_thue_morse(N))
    rng = np.random.RandomState(12345)
    inits.append(rng.choice([-1, 1], size=N))
    rng2 = np.random.RandomState(67890)
    inits.append(rng2.choice([-1, 1], size=N))
    # Logistic map
    val = 0.5
    log_seq = np.zeros(N)
    for i in range(N):
        val = 3.99 * val * (1.0 - val)
        log_seq[i] = 1.0 if val > 0.5 else -1.0
    inits.append(log_seq)
    
    # Local search from each initialization
    for i, x_init in enumerate(inits):
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        R_init = _compute_R(x_init)
        time_left = DEADLINE - TIME_MARGIN - time.time()
        time_limit = min(20.0, time_left * 0.25)
        if time_limit < 2.0:
            break
        x_trial, R_trial, c3_trial = _local_search_fast(x_init.copy(), R_init, max_iter=50000, time_limit=time_limit)
        if c3_trial < best_c3:
            best_x = x_trial
            best_R = R_trial
            best_c3 = c3_trial
    
    # Simulated annealing from best
    if time.time() < DEADLINE - TIME_MARGIN - 5.0:
        time_left = DEADLINE - TIME_MARGIN - time.time()
        time_limit = min(30.0, time_left * 0.5)
        if time_limit > 2.0:
            x_sa, R_sa, c3_sa = _simulated_annealing_fast(best_x.copy(), best_R.copy(), max_iter=50000, time_limit=time_limit)
            if c3_sa < best_c3:
                best_x = x_sa
                best_R = R_sa
                best_c3 = c3_sa
    
    # Gradient-based polish
    if time.time() < DEADLINE - TIME_MARGIN - 5.0:
        time_left = DEADLINE - TIME_MARGIN - time.time()
        time_limit = min(20.0, time_left * 0.4)
        if time_limit > 2.0:
            x_grad = _gradient_polish(best_x.copy(), time_limit=time_limit)
            R_grad = _compute_R(x_grad)
            c3_grad = _c3_from_R(R_grad)
            if c3_grad < best_c3:
                best_x = x_grad
                best_R = R_grad
                best_c3 = c3_grad
    
    # Final local search polish
    if time.time() < DEADLINE - TIME_MARGIN - 2.0:
        time_left = DEADLINE - TIME_MARGIN - time.time()
        time_limit = max(1.0, time_left * 0.8)
        x_final, R_final, c3_final = _local_search_fast(best_x.copy(), best_R.copy(), max_iter=100000, time_limit=time_limit)
        if c3_final < best_c3:
            best_x = x_final
            best_R = R_final
            best_c3 = c3_final
    
    return best_x
# EVOLVE-BLOCK-END
