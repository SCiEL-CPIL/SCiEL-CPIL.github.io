"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize

N = 400

# EVOLVE-BLOCK-START

def _autoconv_full(x):
    """Full linear autoconvolution of x (length 2N-1)."""
    n = len(x)
    size = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(x, n=size)
    conv = np.fft.irfft(F * F, n=size)
    return conv[:2 * n - 1]


def _l2_offdiag_energy(x):
    """L2 energy of off-diagonal autoconvolution."""
    n = len(x)
    size = 1 << (2 * n - 1).bit_length()
    H = np.fft.rfft(x, size)
    ac = np.fft.irfft(H * H, size)[:2 * n - 1]
    if len(ac) > 1:
        return np.sum(ac[1:] ** 2)
    return 0.0


def _l2_offdiag_grad(x):
    """
    Analytical gradient of L2 off-diagonal autoconvolution energy.
    d/dx_i [sum_{k!=0} (x*x)[k]^2] = 2 * sum_{k!=0} 2*(x*x)[k] * d(x*x)[k]/dx_i
    where d(x*x)[k]/dx_i = x[i+k] + x[i-k] (with appropriate bounds)
    
    More efficiently: grad_i = 4 * sum_{k=1}^{n-1} ac[k] * (x[i+k] + x[i-k])
    This can be computed as a convolution: grad = 4 * (ac_offdiag * x) + 4 * (ac_offdiag * x_flipped)
    Actually: grad_i = 4 * sum_{k=1}^{n-1} ac[k] * (x[(i+k)%n] + x[(i-k)%n])
    But we need linear (not circular) convolution. Let's use the FFT approach.
    
    ac[k] = sum_j x[j]*x[j+k] for k>=0
    d(ac[k])/dx[i] = x[i+k] + x[i-k] (if in bounds)
    
    grad[i] = 4 * sum_{k=1}^{n-1} ac[k] * (x[i+k] if i+k<n else 0 + x[i-k] if i-k>=0 else 0)
    
    This is: 4 * (ac * x) where * is cross-correlation-like.
    Specifically: grad[i] = 4 * sum_k ac[k]*x[i+k] + 4 * sum_k ac[k]*x[i-k]
    = 4 * (ac_rev * x)[i] + 4 * (ac * x)[i]  where ac_rev is reversed
    """
    n = len(x)
    size = 1 << (2 * n - 1).bit_length()
    H = np.fft.rfft(x, size)
    ac = np.fft.irfft(H * H, size)[:2 * n - 1]
    ac_off = ac[1:]  # length n-1
    
    # grad[i] = 4 * sum_{k=1}^{n-1} ac_off[k-1] * (x[i+k] + x[i-k])
    # First term: sum_k ac_off[k-1] * x[i+k] = (ac_off_rev * x)[i] where ac_off_rev[k]=ac_off[n-1-k]
    # This is a cross-correlation. Use FFT.
    
    # Method: convolve ac_off with x (linear convolution)
    # sum_{k} ac_off[k] * x[i+k] = (ac_off_flipped * x)[i + n-1] ... let's just use direct for clarity
    # Actually for n=400, O(n^2) = 160K ops is fine per gradient eval.
    
    grad = np.zeros(n)
    for k in range(1, n):
        val = 4.0 * ac_off[k-1]
        # x[i+k] for i=0..n-1-k
        grad[:n-k] += val * x[k:]
        # x[i-k] for i=k..n-1
        grad[k:] += val * x[:n-k]
    
    return grad


def _score_offdiag(x):
    """
    Compute the metric: max(|autoconv|) for k != 0 / (sum(x))^2
    We want to minimize this.
    """
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return 1e12
    conv = _autoconv_full(x)
    if len(conv) <= 1:
        return 0.0
    max_abs_offdiag = np.max(np.abs(conv[1:]))
    return max_abs_offdiag / (integral * integral)


def _score_std(x):
    """
    Compute the metric: max(|autoconv|) for all k / (sum(x))^2
    """
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return 1e12
    conv = _autoconv_full(x)
    max_abs = np.max(np.abs(conv))
    return max_abs / (integral * integral)


def _param_to_signal(params):
    """
    Convert spectral coefficients to a time-domain signal.
    params: array of length 2*M+1, representing [c0, c1, d1, c2, d2, ..., cM, dM]
    x[n] = c0 + sum_{k=1}^{M} c_k * cos(2*pi*k*n/N) + d_k * sin(2*pi*k*n/N)
    """
    M = len(params) // 2
    c0 = params[0]
    x = np.full(N, c0)
    n = np.arange(N)
    for k in range(1, M + 1):
        c_k = params[2 * k - 1]
        d_k = params[2 * k]
        x += c_k * np.cos(2 * np.pi * k * n / N) + d_k * np.sin(2 * np.pi * k * n / N)
    return x


def _soft_max_offdiag(x, beta=50.0):
    """
    Smoothed objective for off-diagonal metric using log-sum-exp for soft max.
    """
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    conv = _autoconv_full(x)
    if len(conv) <= 1:
        return 0.0
    abs_c = np.abs(conv[1:])
    m = np.max(abs_c)
    if m < 1e-15:
        return 0.0
    soft_max = m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta
    return soft_max / (s * s)


def _m_sequence(n, k):
    """Generate an m-sequence of length n using LFSR with k stages."""
    polys = {
        4: [3, 0],
        5: [4, 3, 0],
        6: [5, 3, 0],
        7: [6, 5, 3, 0],
        8: [7, 5, 3, 2, 0],
    }
    if k not in polys:
        return np.sign(np.sin(np.arange(n) * 2 * np.pi / n * 3))[:n]
    
    taps = polys[k]
    state = [0] * k
    state[0] = 1
    seq = []
    for _ in range(n):
        fb = 0
        for t in taps:
            fb ^= state[t]
        for i in range(k - 1, 0, -1):
            state[i] = state[i - 1]
        state[0] = fb
        seq.append(1.0 if state[0] else -1.0)
    return np.array(seq, dtype=np.float64)


def _block_pattern(n, block_size, neg_frac):
    """Generate block pattern with alternating +1/-1 blocks."""
    h = np.zeros(n, dtype=np.float64)
    for i in range(n):
        block_idx = i // block_size
        if block_idx % 2 == 0:
            pos_in_block = i % block_size
            n_neg = int(neg_frac * block_size)
            h[i] = -1.0 if pos_in_block < n_neg else 1.0
        else:
            pos_in_block = i % block_size
            n_neg = int(neg_frac * block_size)
            h[i] = 1.0 if pos_in_block < n_neg else -1.0
    return h


def _chunk_flip_search(current, current_score, best_score, best_h, rng, deadline, n, use_offdiag=True):
    """
    Structural change: chunk-based local search with peak-targeting.
    Instead of single-bit flips, flip contiguous chunks of bits.
    Proposals are biased toward positions near autoconvolution peaks.
    """
    size = 1 << (2 * n - 1).bit_length()
    
    # Track peak location for targeting (off-diagonal)
    ac = _autoconv_full(current)
    if use_offdiag and len(ac) > 1:
        peak_idx = np.argmax(np.abs(ac[1:])) + 1
    else:
        peak_idx = np.argmax(np.abs(ac))
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            break
        
        # Choose chunk size
        r = rng.random()
        if r < 0.4:
            chunk_sz = 1
        elif r < 0.6:
            chunk_sz = rng.choice([2, 3, 4])
        elif r < 0.8:
            chunk_sz = rng.choice([5, 8, 12])
        else:
            chunk_sz = rng.choice([16, 24, 32])
        
        # Peak-targeted proposal
        if rng.random() < 0.6:
            target = peak_idx % n
            spread = max(chunk_sz, n // 8)
            start = int(rng.integers(max(0, target - spread), min(n - chunk_sz, target + spread) + 1))
        else:
            start = int(rng.integers(0, max(1, n - chunk_sz + 1)))
        
        start = max(0, min(start, n - chunk_sz))
        end = start + chunk_sz
        
        # Perform the chunk flip
        old_vals = current[start:end].copy()
        current[start:end] = -old_vals
        
        # Compute new score
        s = np.sum(current)
        if abs(s) < 1e-12:
            new_score = np.inf
        else:
            H = np.fft.rfft(current, size)
            ac_new = np.fft.irfft(H * H, size)[:2 * n - 1]
            if use_offdiag and len(ac_new) > 1:
                max_abs = np.max(np.abs(ac_new[1:]))
            else:
                max_abs = np.max(np.abs(ac_new))
            new_score = max_abs / (s * s)
        
        # Accept or reject
        if new_score <= current_score:
            current_score = new_score
            no_improve_count = 0
            if new_score < best_score:
                best_score = new_score
                best_h = current.copy()
                if use_offdiag and len(ac_new) > 1:
                    peak_idx = np.argmax(np.abs(ac_new[1:])) + 1
                else:
                    peak_idx = np.argmax(np.abs(ac_new))
        else:
            current[start:end] = old_vals
            no_improve_count += 1
    
    return current, current_score, best_score, best_h


def _single_flip_refine(current, current_score, best_score, best_h, rng, deadline, n, use_offdiag=True):
    """
    Fast single-bit flip refinement pass. Tries flipping each bit once
    in a greedy manner, accepting improvements.
    """
    size = 1 << (2 * n - 1).bit_length()
    improved = True
    passes = 0
    
    while improved and time.time() < deadline:
        improved = False
        passes += 1
        
        for i in range(n):
            if time.time() > deadline:
                break
            
            old_val = current[i]
            current[i] = -old_val
            
            s = np.sum(current)
            if abs(s) < 1e-12:
                new_score = np.inf
            else:
                H = np.fft.rfft(current, size)
                ac_new = np.fft.irfft(H * H, size)[:2 * n - 1]
                if use_offdiag and len(ac_new) > 1:
                    max_abs = np.max(np.abs(ac_new[1:]))
                else:
                    max_abs = np.max(np.abs(ac_new))
                new_score = max_abs / (s * s)
            
            if new_score <= current_score:
                current_score = new_score
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_h = current.copy()
            else:
                current[i] = old_val
    
    return current, current_score, best_score, best_h


def _continuous_relaxation(x_init, deadline, rng):
    """
    Continuous relaxation via gradient descent on L2 autoconvolution energy
    using analytical gradient. Then project to binary and polish.
    """
    n = len(x_init)
    
    # Start from the initial signal
    x = x_init.astype(np.float64).copy()
    
    # Gradient descent with adaptive step size using analytical gradient
    lr = 0.05
    best_x = x.copy()
    best_energy = _l2_offdiag_energy(x)
    
    for step in range(500):
        if time.time() > deadline:
            break
        
        grad = _l2_offdiag_grad(x)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        # Update with line search
        x_new = x - lr * grad
        x_new = np.clip(x_new, -2.0, 2.0)
        
        energy_new = _l2_offdiag_energy(x_new)
        if energy_new < best_energy:
            best_energy = energy_new
            best_x = x_new.copy()
            x = x_new
            lr = min(lr * 1.05, 0.5)
        else:
            lr *= 0.8
            if lr < 1e-6:
                break
    
    # Project to binary
    x_binary = np.sign(best_x)
    x_binary[x_binary == 0] = 1.0
    
    # Ensure non-zero sum
    s = np.sum(x_binary)
    if abs(s) < 1e-12:
        x_binary[0] = 1.0
    
    score = _score_offdiag(x_binary)
    return x_binary, score


def _soft_max_lbfgs(x_init, deadline, rng):
    """
    L-BFGS-B optimization on soft-max off-diagonal autoconvolution metric.
    Uses the continuous heights directly.
    """
    n = len(x_init)
    
    def objective(x):
        s = np.sum(x)
        if abs(s) < 1e-12:
            return 1e10
        conv = _autoconv_full(x)
        if len(conv) <= 1:
            return 0.0
        abs_c = np.abs(conv[1:])
        m = np.max(abs_c)
        if m < 1e-15:
            return 0.0
        # Soft max with moderate beta for smoothness
        beta = 20.0
        soft_max = m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta
        return soft_max / (s * s)
    
    x0 = x_init.astype(np.float64).copy()
    bounds = [(-2.0, 2.0)] * n
    
    try:
        result = scipy.optimize.minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        if result.fun < 1e10:
            return result.x, _score_offdiag(result.x)
    except Exception:
        pass
    
    return x_init, _score_offdiag(x_init)


def _spectral_optimize(x_init, deadline, rng):
    """
    Optimize using spectral parameterization with differential evolution
    and a soft-max surrogate for the off-diagonal metric.
    """
    # Initialize parameters from the given signal
    n = len(x_init)
    # Use M frequency bins
    M = 20
    n_params = 2 * M + 1
    
    # Initialize params: try to fit the initial signal
    # Simple approach: start with random params and let DE optimize
    params_init = rng.uniform(-5, 5, size=n_params)
    
    bounds = [(-20, 20)] * n_params
    
    def objective(params):
        x = _param_to_signal(params)
        return _soft_max_offdiag(x, beta=30.0)
    
    try:
        result = scipy.optimize.differential_evolution(
            objective,
            bounds=bounds,
            maxiter=50,
            popsize=12,
            tol=1e-6,
            seed=42,
            polish=True,
            updating='immediate'
        )
        
        if result.fun < 1e10:
            x_opt = _param_to_signal(result.x)
            score_opt = _score_offdiag(x_opt)
            return x_opt, score_opt
    except Exception:
        pass
    
    return x_init, _score_offdiag(x_init)


def _multi_restart_search(best_h, best_score, rng, deadline, n):
    """
    Multiple restarts from perturbed versions of the best solution.
    """
    current_best = best_score
    current_best_h = best_h.copy()
    
    num_restarts = 15
    per_restart_time = (deadline - time.time()) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        h = current_best_h.copy()
        flip_frac = rng.choice([0.05, 0.1, 0.15, 0.2])
        n_flips = int(flip_frac * n)
        flip_indices = rng.choice(n, size=n_flips, replace=False)
        h[flip_indices] = -h[flip_indices]
        
        s = np.sum(h)
        if abs(s) < 1e-12:
            continue
        size = 1 << (2 * n - 1).bit_length()
        H = np.fft.rfft(h, size)
        ac = np.fft.irfft(H * H, size)[:2 * n - 1]
        if len(ac) > 1:
            max_abs = np.max(np.abs(ac[1:]))
        else:
            max_abs = np.max(np.abs(ac))
        score = max_abs / (s * s)
        
        if score < current_best:
            h = current_best_h.copy()
            score = current_best
        
        restart_deadline = min(time.time() + per_restart_time, deadline)
        h, score, restart_best, restart_best_h = _chunk_flip_search(
            h, score, current_best, current_best_h.copy(), rng, restart_deadline, n, use_offdiag=True
        )
        
        if restart_best < current_best:
            current_best = restart_best
            current_best_h = restart_best_h
    
    return current_best_h, current_best


def _zadoff_chu_real(n, q, phase=0.0):
    """
    Generate the real part of a Zadoff-Chu sequence with quadratic chirp phase.
    z[k] = exp(-j * pi * q * k * (k + delta) / n)
    where delta = 0 if q is even, 1 if q is odd
    phase: additional linear phase shift
    """
    k = np.arange(n)
    delta = 0 if q % 2 == 0 else 1
    # Quadratic chirp phase
    phase_quad = -np.pi * q * k * (k + delta) / n
    # Add linear phase
    phase_lin = phase * k / n
    total_phase = phase_quad + phase_lin
    z = np.exp(1j * total_phase)
    return np.real(z)


def _zadoff_chu_search(deadline, rng):
    """
    Efficient search over Zadoff-Chu parameters (q, phase) to find
    the configuration with minimum off-diagonal autoconvolution peak.
    Vectorized evaluation.
    """
    n = N
    best_score = np.inf
    best_h = np.ones(n)
    best_q = None
    best_phase = 0.0
    
    # Try various q values (must be coprime with n for proper ZC)
    valid_q = [q for q in range(1, n, 2) if np.gcd(q, n) == 1 and q % 5 != 0]
    extra_q = list(range(1, min(50, n), 2))
    all_q = list(set(valid_q + extra_q))
    
    size = 1 << (2 * n - 1).bit_length()
    
    # Phase 1: Coarse grid search over q and phase
    phase_grid = np.linspace(-np.pi, np.pi, 13)
    
    for q in all_q:
        if time.time() > deadline:
            break
        k = np.arange(n)
        delta = 0 if q % 2 == 0 else 1
        phase_quad = -np.pi * q * k * (k + delta) / n
        for phase in phase_grid:
            phase_lin = phase * k / n
            total_phase = phase_quad + phase_lin
            h = np.real(np.exp(1j * total_phase))
            s = np.sum(h)
            if abs(s) < 1e-12:
                continue
            H = np.fft.rfft(h, size)
            ac = np.fft.irfft(H * H, size)[:2 * n - 1]
            if len(ac) > 1:
                max_abs = np.max(np.abs(ac[1:]))
            else:
                max_abs = np.max(np.abs(ac))
            score = max_abs / (s * s)
            if score < best_score:
                best_score = score
                best_h = h.copy()
                best_q = q
                best_phase = phase
    
    # Phase 2: Fine-tune around best (q, phase)
    if time.time() < deadline - 2 and best_q is not None:
        fine_phases = np.linspace(best_phase - 0.3, best_phase + 0.3, 41)
        k = np.arange(n)
        delta = 0 if best_q % 2 == 0 else 1
        phase_quad = -np.pi * best_q * k * (k + delta) / n
        for phase in fine_phases:
            if time.time() > deadline:
                break
            phase_lin = phase * k / n
            total_phase = phase_quad + phase_lin
            h = np.real(np.exp(1j * total_phase))
            s = np.sum(h)
            if abs(s) < 1e-12:
                continue
            H = np.fft.rfft(h, size)
            ac = np.fft.irfft(H * H, size)[:2 * n - 1]
            if len(ac) > 1:
                max_abs = np.max(np.abs(ac[1:]))
            else:
                max_abs = np.max(np.abs(ac))
            score = max_abs / (s * s)
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    return best_h, best_score


def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    rng = np.random.default_rng(42)
    
    # Diverse initializations
    candidates = []
    
    # 1. Hanning window (good off-diagonal behavior)
    candidates.append(np.hanning(N))
    
    # 2. Hamming window
    candidates.append(np.hamming(N))
    
    # 3. M-sequences
    for k in [6, 7, 8]:
        candidates.append(_m_sequence(N, k))
    
    # 4. Block patterns
    for bs in [8, 16, 32, 64]:
        for nf in [0.0, 0.2, 0.3, 0.5]:
            candidates.append(_block_pattern(N, bs, nf))
    
    # 5. Random with varying bias
    for bias in [0.0, 0.1, 0.2, 0.3]:
        p_pos = 0.5 + bias / 2
        h = rng.choice([-1.0, 1.0], size=N, p=[1 - p_pos, p_pos])
        candidates.append(h)
    
    # 6. Sinusoidal modulation
    for freq in [1, 2, 3, 5]:
        h = np.sign(np.sin(2 * np.pi * freq * np.arange(N) / N))
        candidates.append(h)
    
    # 7. Walsh-Hadamard-like patterns
    for level in range(4):
        h = np.ones(N)
        for i in range(N):
            val = 1.0
            for bit in range(level + 1):
                if (i >> bit) & 1:
                    val *= -1.0
            h[i] = val
        candidates.append(h)
    
    # 8. Gaussian window
    x = np.arange(N)
    sigma = N / 4
    candidates.append(np.exp(-0.5 * ((x - N/2) / sigma) ** 2))
    
    # 9. Zadoff-Chu sequences
    zc_deadline = time.time() + 12
    zc_h, zc_score = _zadoff_chu_search(zc_deadline, rng)
    candidates.append(zc_h)
    
    # Evaluate all candidates using off-diagonal metric
    best_score = np.inf
    best_h = np.ones(N)
    
    for h in candidates:
        if time.time() > deadline - 60:
            break
        s = _score_offdiag(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    n = N
    
    # Phase 1: Continuous relaxation with analytical gradient (fast)
    if time.time() < deadline - 55:
        relax_deadline = time.time() + 20
        best_h, best_score = _continuous_relaxation(best_h, relax_deadline, rng)
    
    # Phase 2: Soft-max L-BFGS-B optimization (key improvement)
    if time.time() < deadline - 45:
        lbfgs_deadline = time.time() + 25
        best_h, best_score = _soft_max_lbfgs(best_h, lbfgs_deadline, rng)
    
    # Phase 3: Chunk-flip local search
    phase3_deadline = min(t0 + 105, deadline - 25)
    best_h, best_score, _, best_h = _chunk_flip_search(
        best_h.copy(), best_score, best_score, best_h.copy(), rng, phase3_deadline, n, use_offdiag=True
    )
    
    # Phase 4: Single-bit refinement
    if time.time() < deadline - 15:
        refine_deadline = time.time() + 12
        best_h, best_score, _, best_h = _single_flip_refine(
            best_h.copy(), best_score, best_score, best_h.copy(), rng, refine_deadline, n, use_offdiag=True
        )
    
    # Phase 5: Multi-restart search
    if time.time() < deadline - 10:
        best_h, best_score = _multi_restart_search(
            best_h, best_score, rng, deadline - 10, n
        )
    
    # Phase 6: Final refinement
    if time.time() < deadline:
        best_h, best_score, _, best_h = _chunk_flip_search(
            best_h.copy(), best_score, best_score, best_h.copy(), rng, deadline, n, use_offdiag=True
        )
    
    return best_h

# EVOLVE-BLOCK-END
