"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

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


def _flip_effect_vectorized(x, i, R):
    """Compute new R after flipping x[i] using vectorized O(N) update."""
    xi = x[i]
    # For each lag k, the contribution from x[i]*x[i+k] and x[i-k]*x[i]
    # R_new[k] = R[k] - 2*xi*(x[i+k] if i+k<N else 0) - 2*xi*(x[i-k] if i-k>=0 else 0)
    # But we need to be careful: when k=0, we have x[i]*x[i] = xi^2 = 1, so delta = -2*xi*xi = -2
    # Actually, flipping x[i] to -x[i]:
    # R_new[k] = sum_j x'_j * x'_{j+k}
    # where x'_j = x_j for j != i, x'_i = -x_i
    # R_new[k] = R[k] - x_i * (x_{i+k} + x_{i-k}) + x_i * (x_{i+k} + x_{i-k}) ... 
    # Wait, let me redo this.
    # R[k] = sum_j x_j * x_{j+k}
    # After flipping x_i: R_new[k] = sum_j x'_j * x'_{j+k}
    # The terms that change are those where j=i or j+k=i, i.e., j=i or j=i-k.
    # Term j=i: x'_i * x'_{i+k} = (-x_i) * x_{i+k} (if i+k != i, i.e., k != 0)
    #           if k=0: (-x_i)*(-x_i) = x_i^2 = 1, original was x_i^2 = 1, so no change for k=0? 
    #           Wait, original R[0] includes x_i^2. New R[0] includes (-x_i)^2 = x_i^2. Same.
    #           But for k != 0, term j=i contributes -x_i * x_{i+k} instead of x_i * x_{i+k}, delta = -2*x_i*x_{i+k}
    # Term j=i-k: x'_{i-k} * x'_i = x_{i-k} * (-x_i) = -x_{i-k}*x_i, original was x_{i-k}*x_i, delta = -2*x_{i-k}*x_i
    # So for k != 0: R_new[k] = R[k] - 2*x_i*(x_{i+k} + x_{i-k}) (with out-of-bounds = 0)
    # For k = 0: R_new[0] = R[0] (since x_i^2 unchanged)
    
    R_new = R.copy()
    if N > 1:
        # k from 1 to N-1
        ks = np.arange(1, N)
        # x_{i+k} valid when i+k < N
        idx_pos = i + ks
        valid_pos = idx_pos < N
        x_pos = np.zeros(N-1)
        x_pos[valid_pos] = x[idx_pos[valid_pos]]
        # x_{i-k} valid when i-k >= 0
        idx_neg = i - ks
        valid_neg = idx_neg >= 0
        x_neg = np.zeros(N-1)
        x_neg[valid_neg] = x[idx_neg[valid_neg]]
        
        delta = -2.0 * xi * (x_pos + x_neg)
        R_new[1:N] += delta
        # Symmetry: R_new[2N-2-k] = R_new[k]
        R_new[N:2*N-1] = R_new[1:N][::-1]
    
    return R_new


def _local_search_fast(x, R, max_iter=100000, time_limit=None):
    """Local search with vectorized O(N) flip evaluation."""
    start_time = time.time()
    current_c3 = _c3_from_R(R)
    best_x = x.copy()
    best_R = R.copy()
    best_c3 = current_c3
    no_improve = 0
    max_no_improve = 300
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        idx = random.randint(0, N - 1)
        R_new = _flip_effect_vectorized(x, idx, R)
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
    """Simulated annealing with vectorized O(N) flip evaluation."""
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
        R_new = _flip_effect_vectorized(x, idx, R)
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


def _spectral_optimize(x0, time_limit=30.0, start_time=None):
    """
    Optimize the continuous power spectrum to minimize peak autocorrelation sidelobe.
    """
    if start_time is None:
        start_time = time.time()
    
    n_fft = 1 << (2 * N - 1).bit_length()
    
    F0 = np.fft.rfft(x0, n=n_fft)
    P0 = np.abs(F0)**2
    logP0 = np.log(P0 + 1e-10)
    
    def objective(logP):
        """Minimize soft-max of |IFFT(P)[k]| for k != 0, normalized by IFFT(P)[0]."""
        if time.time() - start_time > DEADLINE - TIME_MARGIN - 2.0:
            return 1e10
        
        P = np.exp(logP)
        R = np.fft.irfft(P, n=n_fft)[:2*N-1]
        R0 = R[0]
        if R0 < 1e-10:
            return 1e10
        
        sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
        
        abs_s = np.abs(sidelobes) / R0
        eps = 1e-10
        abs_s_smooth = np.sqrt(abs_s**2 + eps)
        
        T = 0.3
        max_val = np.max(abs_s_smooth)
        if max_val < 1e-15:
            return 0.0
        shifted = (abs_s_smooth - max_val) / T
        shifted = np.clip(shifted, -100, 100)
        lse = np.log(np.sum(np.exp(shifted))) + max_val / T
        soft_max = T * lse
        
        return soft_max
    
    def gradient(logP):
        """Compute gradient of objective w.r.t. logP via FFT chain rule."""
        if time.time() - start_time > DEADLINE - TIME_MARGIN - 2.0:
            return np.zeros_like(logP)
        
        P = np.exp(logP)
        R = np.fft.irfft(P, n=n_fft)[:2*N-1]
        R0 = R[0]
        if R0 < 1e-10:
            return np.zeros_like(logP)
        
        sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
        abs_s = np.abs(sidelobes) / R0
        eps = 1e-10
        abs_s_smooth = np.sqrt(abs_s**2 + eps)
        
        T = 0.3
        max_val = np.max(abs_s_smooth)
        if max_val < 1e-15:
            return np.zeros_like(logP)
        shifted = (abs_s_smooth - max_val) / T
        shifted = np.clip(shifted, -100, 100)
        weights = np.exp(shifted) / np.sum(np.exp(shifted))
        
        dR = np.zeros(2*N-1)
        dR[1:N] = weights[:N-1] * np.sign(sidelobes[:N-1]) / R0
        dR[N:2*N-1] = weights[N-1:] * np.sign(sidelobes[N-1:]) / R0
        
        dR_full = np.zeros(n_fft)
        dR_full[:2*N-1] = dR
        grad_logP = np.real(np.fft.ifft(dR_full) * P) / n_fft
        
        return grad_logP
    
    result = minimize(
        objective, logP0, method='L-BFGS-B', jac=gradient,
        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 1000}
    )
    
    if time.time() - start_time > DEADLINE - TIME_MARGIN - 2.0:
        return None
    
    P_opt = np.exp(result.x)
    F_mag = np.sqrt(P_opt + 1e-15)
    F_phase = F0 / (np.abs(F0) + 1e-15)
    F_recon = F_mag * F_phase
    h_cont = np.real(np.fft.irfft(F_recon, n=n_fft))[:N]
    
    h_disc = np.sign(h_cont)
    h_disc[h_disc == 0] = 1.0
    return h_disc


def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_x = np.ones(N)
    best_R = _compute_R(best_x)
    best_c3 = _c3_from_R(best_R)
    
    candidates = []
    candidates.append(np.ones(N))
    
    alt = np.ones(N)
    alt[1::2] = -1.0
    candidates.append(alt)
    
    candidates.append(_thue_morse(N))
    
    rng = np.random.RandomState(123)
    candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    for block_size in [5, 10, 20, 40]:
        n_blocks = N // block_size
        block_vals = np.random.RandomState(100 + block_size).choice([-1, 1], size=n_blocks)
        candidates.append(np.repeat(block_vals, block_size))
    
    # Also try some structured patterns
    # Low-density pattern
    sparse = np.ones(N)
    sparse[::7] = -1.0
    candidates.append(sparse)
    
    # Quadratic phase
    quad = np.sign(np.cos(2 * np.pi * np.arange(N) / N))
    candidates.append(quad)
    
    spectral_results = []
    for i, x_init in enumerate(candidates):
        if time.time() > DEADLINE - TIME_MARGIN - 5.0:
            break
        time_left = DEADLINE - TIME_MARGIN - time.time()
        if time_left < 5.0:
            break
        time_limit = min(15.0, time_left * 0.3)
        if time_limit < 3.0:
            break
        
        h_disc = _spectral_optimize(x_init.copy(), time_limit=time_limit, start_time=start_time)
        if h_disc is not None:
            spectral_results.append(h_disc)
    
    all_candidates = candidates + spectral_results
    
    # Local search from each candidate
    for i, x_init in enumerate(all_candidates):
        if time.time() > DEADLINE - TIME_MARGIN - 3.0:
            break
        R_init = _compute_R(x_init)
        c3_init = _c3_from_R(R_init)
        
        time_left = DEADLINE - TIME_MARGIN - time.time()
        if time_left < 3.0:
            break
        time_limit = min(12.0, time_left * 0.2)
        if time_limit < 2.0:
            break
        
        x_trial, R_trial, c3_trial = _local_search_fast(
            x_init.copy(), R_init, max_iter=80000, time_limit=time_limit
        )
        if c3_trial < best_c3:
            best_x = x_trial
            best_R = R_trial
            best_c3 = c3_trial
    
    # Simulated annealing from best
    if time.time() < DEADLINE - TIME_MARGIN - 5.0:
        time_left = DEADLINE - TIME_MARGIN - time.time()
        time_limit = min(25.0, time_left * 0.4)
        if time_limit > 2.0:
            x_sa, R_sa, c3_sa = _simulated_annealing_fast(
                best_x.copy(), best_R.copy(), max_iter=80000, time_limit=time_limit
            )
            if c3_sa < best_c3:
                best_x = x_sa
                best_R = R_sa
                best_c3 = c3_sa
    
    # Final local search polish
    if time.time() < DEADLINE - TIME_MARGIN - 2.0:
        time_left = DEADLINE - TIME_MARGIN - time.time()
        time_limit = max(1.0, time_left * 0.8)
        x_final, R_final, c3_final = _local_search_fast(
            best_x.copy(), best_R.copy(), max_iter=150000, time_limit=time_limit
        )
        if c3_final < best_c3:
            best_x = x_final
            best_R = R_final
            best_c3 = c3_final
    
    return best_x
# EVOLVE-BLOCK-END
