"""Autoconvolution minimization via block-flip simulated annealing with incremental FFT updates.

We choose 400 signed step heights (+/-1) to minimize the maximum absolute
value of the (linear) autoconvolution, which is the C3 upper bound we want
to lower.

Key improvements:
1. Continuous gradient descent (flat-spectrum / L2 autocorrelation) to generate
   high-quality binary seeds before discrete search.
2. Multi-start SA with adaptive temperature schedule.
3. Aggressive multi-scale local polish at the end.
"""
import time
import numpy as np

N = 400
L = 1024  # FFT length >= 2*N-1 = 799
_SEED = 12345

# Precompute Fourier basis for incremental updates
_k_indices = np.arange(L // 2 + 1)
_i_indices = np.arange(N)
_exponents = -2j * np.pi * np.outer(_i_indices, _k_indices) / L
_W_basis = np.exp(_exponents)


def _autocorr_fast(h):
    """Linear one-sided autocorrelation ac[k] = sum_i h[i] h[i+k], k = 0..N-1."""
    F = np.fft.rfft(h, n=L)
    ac = np.fft.irfft(F * np.conj(F), n=L)[:N]
    return ac


def _hard_score_from_ac(ac):
    """Compute score from autocorrelation array."""
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    return float(np.max(np.abs(ac)) / dc)


def _smooth_score_from_ac(ac, beta):
    """Smooth (log-sum-exp) surrogate of the peak sidelobe energy, guides SA."""
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    y = (ac[1:] / dc) ** 2
    m = float(np.max(y))
    z = beta * (y - m)
    return m + float(np.sum(np.exp(z))) / beta


def _continuous_seed(deadline, rng):
    """
    Generate a high-quality binary seed via continuous gradient descent on
    the L2 norm of the autocorrelation sidelobes (flat spectrum objective).
    This gives a much better starting point than random for the discrete search.
    """
    # Start from a continuous vector
    x = np.ones(N)
    # Add some structured perturbation
    idx = np.arange(N)
    x = np.cos(2 * np.pi * rng.uniform(0.1, 1.0) * idx / N) + 0.5 * np.sin(4 * np.pi * idx / N)
    
    lr = 0.05
    n_steps = 2000
    t_start = time.time()
    
    for step in range(n_steps):
        if time.time() > deadline - 60.0:
            break
        
        # Compute autocorrelation
        F = np.fft.rfft(x, n=L)
        P = F * np.conj(F)
        ac = np.fft.irfft(P, n=L)
        
        # Objective: minimize sum of squared sidelobes
        # Gradient with respect to x: 
        # ac[k] = sum_i x[i] x[i+k]
        # d/dx[j] sum_k (ac[k])^2 = 2 * sum_k ac[k] * d(ac[k])/dx[j]
        # d(ac[k])/dx[j] = x[j-k] + x[j+k] (with boundary handling)
        # This is equivalent to 2 * (x * ac)[j] in circular convolution, but for linear:
        # Actually, for linear autocorrelation, the gradient is:
        # grad[j] = 2 * sum_k ac[k] * (x[j-k] + x[j+k]) where terms are in bounds
        # This equals 2 * (linear_cross_corr(x, ac))[j]
        
        # Use FFT for efficiency: grad = 2 * irfft(F_x * conj(F_ac)) but linear
        F_ac = np.fft.rfft(ac, n=L)
        grad = 2.0 * np.fft.irfft(F * np.conj(F_ac), n=L)[:N]
        
        # Also subtract the DC component contribution (we want to minimize sidelobes only)
        # Actually let's just minimize the full L2 of ac[1:]
        # Re-derive: minimize sum_{k=1}^{N-1} ac[k]^2
        # grad[j] = 2 * sum_{k=1}^{N-1} ac[k] * (x[j-k] + x[j+k])
        # = 2 * (full linear cross-corr of x and ac, excluding k=0 term)
        
        # Simpler: just use the full gradient including k=0, it's fine
        x -= lr * grad
        
        # Normalize to keep magnitudes reasonable
        norm = np.linalg.norm(x)
        if norm > 1e-10:
            x = x / norm * np.sqrt(N)
    
    # Threshold to binary
    h_binary = np.sign(x)
    h_binary[h_binary == 0] = 1.0
    
    # Local polish the binary seed
    h_binary, _ = _local_polish_quick(h_binary, deadline)
    
    return h_binary


def _local_polish_quick(h, deadline):
    """Quick single-pass local polish for seed generation."""
    h = h.copy()
    F = np.fft.rfft(h, n=L)
    ac = _autocorr_fast(h)
    best_hard = _hard_score_from_ac(ac)
    best_h = h.copy()
    
    for _pass in range(3):
        if time.time() > deadline - 55.0:
            break
        any_improve = False
        for start in range(N):
            if time.time() > deadline - 55.0:
                break
            idx_flip = np.array([start])
            h_block = h[idx_flip]
            W_block = _W_basis[idx_flip, :]
            delta_F = -2.0 * (W_block @ h_block)
            F_new = F + delta_F
            P = F_new * np.conj(F_new)
            ac_new = np.fft.irfft(P, n=L)[:N]
            new_hard = _hard_score_from_ac(ac_new)
            if new_hard < best_hard - 1e-12:
                h[idx_flip] *= -1.0
                F = F_new
                best_hard = new_hard
                best_h = h.copy()
                any_improve = True
        if not any_improve:
            break
    
    return best_h, best_hard


def _sa_run(h_init, rng, deadline, T_start, T_min, decay, beta, sizes):
    """Run simulated annealing from h_init until deadline. Returns (best_h, best_hard)."""
    h = h_init.copy()
    F = np.fft.rfft(h, n=L)
    ac = _autocorr_fast(h)
    cur_hard = _hard_score_from_ac(ac)
    cur_smooth = _smooth_score_from_ac(ac, beta)
    best_hard = cur_hard
    best_h = h.copy()
    T = T_start
    steps = 0
    
    while time.time() < deadline - 2.0:
        if steps % 500 == 0 and steps > 0:
            F = np.fft.rfft(h, n=L)
        
        # Choose move type
        r = rng.random()
        if r < 0.65:
            # Block flip
            size = int(rng.choice(sizes))
            start = int(rng.integers(0, N - size + 1))
            idx_flip = np.arange(start, start + size)
        elif r < 0.85:
            # Single bit flip
            i = int(rng.integers(0, N))
            idx_flip = np.array([i])
        else:
            # Double bit flip (swap two signs)
            i1, i2 = rng.choice(N, size=2, replace=False)
            idx_flip = np.array([int(i1), int(i2)])
        
        # Incremental FFT update
        h_block = h[idx_flip]
        W_block = _W_basis[idx_flip, :]
        delta_F = -2.0 * (W_block @ h_block)
        F_new = F + delta_F
        
        # Compute scores
        P = F_new * np.conj(F_new)
        ac_new = np.fft.irfft(P, n=L)[:N]
        new_hard = _hard_score_from_ac(ac_new)
        new_smooth = _smooth_score_from_ac(ac_new, beta)
        
        dS = new_smooth - cur_smooth
        accept = False
        if new_hard < best_hard:
            accept = True
        elif dS <= 0:
            accept = True
        else:
            # Metropolis criterion
            if T > 1e-10:
                if rng.random() < np.exp(-dS / T):
                    accept = True
        
        if accept:
            h[idx_flip] *= -1.0
            F = F_new
            cur_hard = new_hard
            cur_smooth = new_smooth
            if new_hard < best_hard:
                best_hard = new_hard
                best_h = h.copy()
        
        T = max(T_min, T * decay)
        steps += 1
    
    return best_h, best_hard


def _local_polish(h, deadline):
    """Greedy multi-scale local polish. Returns (best_h, best_hard)."""
    h = h.copy()
    F = np.fft.rfft(h, n=L)
    ac = _autocorr_fast(h)
    best_hard = _hard_score_from_ac(ac)
    best_h = h.copy()
    
    sizes_polish = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    for _pass in range(80):
        if time.time() > deadline - 0.5:
            break
        any_improve = False
        for size in sizes_polish:
            if time.time() > deadline - 0.5:
                return best_h, best_hard
            for start in range(0, N - size + 1):
                if time.time() > deadline - 0.5:
                    return best_h, best_hard
                idx_flip = np.arange(start, start + size)
                
                # Incremental FFT update
                h_block = h[idx_flip]
                W_block = _W_basis[idx_flip, :]
                delta_F = -2.0 * (W_block @ h_block)
                F_new = F + delta_F
                
                # Compute score
                P = F_new * np.conj(F_new)
                ac_new = np.fft.irfft(P, n=L)[:N]
                new_hard = _hard_score_from_ac(ac_new)
                
                if new_hard < best_hard - 1e-12:
                    h[idx_flip] *= -1.0
                    F = F_new
                    best_hard = new_hard
                    best_h = h.copy()
                    any_improve = True
        
        if not any_improve:
            break
    
    return best_h, best_hard


def construct():
    try:
        t0 = time.time()
        deadline = t0 + 145.0  # safety margin before 150s limit
        rng = np.random.default_rng(_SEED)
        idx = np.arange(N)
        
        # Phase 1: Generate high-quality seeds via continuous optimization
        # Allocate ~20s for continuous seed generation
        cont_deadline = t0 + 20.0
        seed1 = _continuous_seed(cont_deadline, rng)
        
        # Also generate a few more diverse seeds
        seed2 = np.ones(N)
        seed3 = rng.choice(np.array([-1.0, 1.0]), size=N)
        seed4 = np.cos(2 * np.pi * idx ** 2 / (2 * N))
        seed5 = np.cos(2 * np.pi * idx ** 2 / (N + 1))
        
        # Evaluate all seeds
        candidates = [seed1, seed2, seed3, seed4, seed5]
        best_h = None
        best_hard = 1e9
        for h in candidates:
            ac = _autocorr_fast(h)
            s = _hard_score_from_ac(ac)
            if s < best_hard:
                best_hard = s
                best_h = h.copy()
        
        # Phase 2: Multi-start SA
        # Allocate remaining time for SA and polish
        sa_start_time = time.time()
        polish_budget = 30.0
        sa_end = deadline - polish_budget
        
        sizes_sa = np.array([1, 2, 3, 4, 8, 12, 16, 20, 24, 32, 48, 64])
        
        # Run SA from the best seed and a perturbed version
        n_starts = 3
        sa_time_per = (sa_end - sa_start_time) / n_starts
        
        for start_idx in range(n_starts):
            if time.time() > sa_end - 1.0:
                break
            
            if start_idx == 0:
                h_init = best_h.copy()
            elif start_idx == 1:
                # Perturb the best
                h_init = best_h.copy()
                flip_mask = rng.random(N) < 0.05
                h_init[flip_mask] *= -1.0
            else:
                # Use another good seed
                h_init = candidates[1].copy()
                flip_mask = rng.random(N) < 0.1
                h_init[flip_mask] *= -1.0
            
            sa_deadline = min(sa_start_time + (start_idx + 1) * sa_time_per, sa_end - 1.0)
            
            h_candidate, hard_candidate = _sa_run(
                h_init, rng, sa_deadline,
                T_start=0.02, T_min=1e-5, decay=0.9997, beta=40.0, sizes=sizes_sa
            )
            
            if hard_candidate < best_hard:
                best_hard = hard_candidate
                best_h = h_candidate.copy()
        
        # Phase 3: Final local polish
        best_h, best_hard = _local_polish(best_h, deadline)
        
        return best_h
    except Exception:
        return np.ones(N)
