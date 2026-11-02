"""Autoconvolution minimization via block-flip simulated annealing with incremental FFT updates.

We choose 400 signed step heights (+/-1) to minimize the maximum absolute
value of the (linear) autoconvolution, which is the C3 upper bound we want
to lower. 

Optimization:
- Uses incremental FFT updates for small block flips.
- Precomputes Fourier basis vectors to update the spectrum in O(kL) instead of O(L log L) for the forward transform.
- Multi-start restarts to escape local minima.
- Aggressive multi-scale local polish at the end.
"""
import time
import numpy as np

N = 400
L = 1024  # FFT length >= 2*N-1 = 799
_SEED = 12345

# Precompute Fourier basis for incremental updates
# W[i, k] = exp(-2j * pi * i * k / L) for i in 0..N-1, k in 0..L//2
_k_indices = np.arange(L // 2 + 1)
_i_indices = np.arange(N)
# Create meshgrid for exponent calculation
# Shape: (N, L//2 + 1)
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
    for _pass in range(60):
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
        
        # Generate diverse initializations
        inits = []
        inits.append(np.ones(N))
        inits.append(rng.choice(np.array([-1.0, 1.0]), size=N))
        inits.append(np.cos(2 * np.pi * idx ** 2 / (2 * N)))
        inits.append(np.cos(2 * np.pi * idx ** 2 / (N + 1)))
        inits.append(np.cos(2 * np.pi * idx ** 2 / (3 * N)))
        inits.append(rng.choice(np.array([-1.0, 1.0]), size=N))
        inits.append(np.cos(2 * np.pi * idx / N))
        inits.append(np.sin(2 * np.pi * idx / N))
        inits.append(np.cos(2 * np.pi * 2 * idx / N))
        inits.append(rng.choice(np.array([-1.0, 1.0]), size=N))
        inits.append(np.cos(2 * np.pi * 3 * idx / N))
        inits.append(rng.choice(np.array([-1.0, 1.0]), size=N))
        
        # Evaluate all inits
        best_h = None
        best_hard = 1e9
        for h in inits:
            ac = _autocorr_fast(h)
            s = _hard_score_from_ac(ac)
            if s < best_hard:
                best_hard = s
                best_h = h.copy()
        
        # Multi-start SA: run SA from several good initializations
        # Allocate time budget across multiple starts
        n_starts = 3
        sa_budget = (deadline - (t0 + 10.0) - 30.0) / n_starts  # leave 10s for init, 30s for polish
        
        sizes_sa = np.array([1, 2, 3, 4, 8, 12, 16, 20, 24, 32, 48, 64])
        
        for start_idx in range(n_starts):
            if time.time() > deadline - 35.0:
                break
            
            # Choose initialization for this start
            if start_idx == 0:
                h_init = best_h.copy()
            else:
                # Perturb the current best or use another good init
                if rng.random() < 0.5:
                    h_init = best_h.copy()
                    # Randomly flip a fraction of bits
                    flip_mask = rng.random(N) < 0.1
                    h_init[flip_mask] *= -1.0
                else:
                    # Pick a random init from the list
                    h_init = inits[rng.integers(0, len(inits))].copy()
            
            sa_deadline = t0 + 10.0 + (start_idx + 1) * sa_budget
            sa_deadline = min(sa_deadline, deadline - 35.0)
            
            h_candidate, hard_candidate = _sa_run(
                h_init, rng, sa_deadline,
                T_start=0.03, T_min=1e-4, decay=0.9995, beta=30.0, sizes=sizes_sa
            )
            
            if hard_candidate < best_hard:
                best_hard = hard_candidate
                best_h = h_candidate.copy()
        
        # Final local polish
        best_h, best_hard = _local_polish(best_h, deadline)
        
        return best_h
    except Exception:
        return np.ones(N)
