"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_fft(h):
    """Compute linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac


def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    ac = _autoconv_fft(h)
    return np.max(np.abs(ac))


def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(h) / (s * s)


def _soft_max_abs(ac, beta=20.0):
    """
    Smooth approximation of max(|ac|) using log-sum-exp.
    This makes the objective differentiable for continuous optimization.
    """
    # Use log-sum-exp trick for numerical stability
    x = np.abs(ac)
    m = np.max(x)
    if m < 1e-300:
        return 0.0
    return m + np.log(np.sum(np.exp(beta * (x - m)))) / beta


def _objective_smooth(x, beta=20.0):
    """
    Smooth objective: soft_max(|autoconv|) / (sum x)^2.
    This is differentiable and suitable for gradient-based methods.
    """
    s = np.sum(x)
    if abs(s) < 1e-6:
        return 1e10
    ac = _autoconv_fft(x)
    smax = _soft_max_abs(ac, beta)
    return smax / (s * s)


def _local_sign_flip(h, max_passes=5, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves.
    """
    h = h.copy()
    best_val = _score(h)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val - 1e-12:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]
        if not improved:
            break
    return h, best_val


def _simulated_annealing_continuous(x0, deadline, rng):
    """
    Simulated annealing on the continuous relaxation.
    x is in [-1, 1]^N, and we minimize _objective_smooth.
    Periodically project to signs and evaluate the true objective.
    """
    x = x0.copy()
    best_continuous = _objective_smooth(x)
    best_sign_h = np.sign(x)
    best_sign_h[best_sign_h == 0] = 1.0
    best_sign_val = _score(best_sign_h)
    
    T_start = 0.5
    T_end = 0.001
    n_steps = 5000
    cooling = (T_end / T_start) ** (1.0 / n_steps)
    
    T = T_start
    step = 0
    
    while step < n_steps and time.time() < deadline:
        # Propose a random perturbation
        n_perturb = rng.integers(1, 6)  # perturb 1-5 coordinates
        idx = rng.integers(0, N, size=n_perturb)
        deltas = rng.normal(0, 0.3, size=n_perturb)
        
        x_new = x.copy()
        x_new[idx] = np.clip(x_new[idx] + deltas, -1.0, 1.0)
        
        f_old = _objective_smooth(x)
        f_new = _objective_smooth(x_new)
        
        delta_f = f_new - f_old
        
        if delta_f < 0 or rng.random() < np.exp(-delta_f / T):
            x = x_new
            if f_new < best_continuous:
                best_continuous = f_new
                # Project to signs and check true objective
                h_proj = np.sign(x)
                h_proj[h_proj == 0] = 1.0
                val_proj = _score(h_proj)
                if val_proj < best_sign_val:
                    best_sign_val = val_proj
                    best_sign_h = h_proj
        
        T *= cooling
        step += 1
        
        # Every 500 steps, do a local sign flip polish
        if step % 500 == 0 and time.time() < deadline - 2:
            h_polish, val_polish = _local_sign_flip(best_sign_h, max_passes=2, deadline=deadline)
            if val_polish < best_sign_val:
                best_sign_val = val_polish
                best_sign_h = h_polish
                # Update continuous x to match
                x = best_sign_h.copy()
                best_continuous = _objective_smooth(x)
    
    return best_sign_h, best_sign_val


def _gen_chirp(n):
    """Generate a chirp-like sequence."""
    t = np.arange(n)
    return np.sign(np.cos(2 * np.pi * t * (t + 1) / (2 * n)))


def _gen_barker_like(n, rng):
    """Generate a random sequence with slight bias toward +1."""
    x = rng.choice([-1.0, 1.0], size=n)
    # Slight bias
    n_flip = rng.integers(0, n // 10)
    if n_flip > 0:
        idx = rng.choice(n, size=n_flip, replace=False)
        x[idx] = 1.0
    return x


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    # Initialize best
    best_h = np.ones(N)
    best_val = _score(best_h)

    rng = np.random.default_rng(42)

    # Multiple restarts with different initializations, each followed by SA
    n_restarts = 12
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break

        # Different initialization strategies
        if r == 0:
            x0 = np.ones(N)
        elif r == 1:
            x0 = _gen_chirp(N)
        elif r == 2:
            x0 = _gen_barker_like(N, rng)
        elif r == 3:
            # Van der Corput
            t = np.arange(N)
            vdc = np.zeros(N)
            for i in range(N):
                f = 0.5
                num = i
                val = 0.0
                while num > 0:
                    val += f * (num % 2)
                    num //= 2
                    f /= 2
                vdc[i] = val
            x0 = np.sign(vdc - 0.5)
            x0[x0 == 0] = 1.0
        elif r == 4:
            # Kronecker with golden ratio
            alpha = (np.sqrt(5) - 1) / 2
            kr = np.mod(np.arange(N) * alpha, 1.0)
            x0 = np.sign(kr - 0.5)
            x0[x0 == 0] = 1.0
        elif r == 5:
            # Random
            x0 = rng.choice([-1.0, 1.0], size=N)
        elif r == 6:
            # Sine wave
            t = np.arange(N)
            x0 = np.sign(np.sin(2 * np.pi * t / N * 3))
            x0[x0 == 0] = 1.0
        elif r == 7:
            # Triangle wave
            t = np.arange(N)
            phase = (2 * t / N) % 2.0
            x0 = np.where(phase < 1.0, 1.0, -1.0)
        elif r == 8:
            # Two-tone chirp
            t = np.arange(N)
            x0 = np.sign(np.cos(2 * np.pi * (3 * t / N + t**2 / N**2)))
            x0[x0 == 0] = 1.0
        elif r == 9:
            # Random with +1 bias
            x0 = rng.choice([-1.0, 1.0], size=N, p=[0.35, 0.65])
        elif r == 10:
            # Block pattern
            block_size = N // 10
            pattern = [1, 1, -1, 1, -1, -1, 1, -1, 1, 1]
            x0 = np.repeat(pattern, block_size)
        else:
            # Another random
            x0 = rng.choice([-1.0, 1.0], size=N)

        # Ensure sum is not too small
        s = np.sum(x0)
        if abs(s) < N * 0.1:
            if s < 0:
                x0 = -x0
            x0 = np.sign(x0 + 0.3)
            x0[x0 == 0] = 1.0

        val = _score(x0)
        if val < best_val:
            best_val = val
            best_h = x0.copy()

        # Run simulated annealing on continuous relaxation
        h_sa, val_sa = _simulated_annealing_continuous(x0, deadline - 5, rng)
        if val_sa < best_val:
            best_val = val_sa
            best_h = h_sa

    # Final local polish
    if time.time() < deadline - 2:
        best_h, best_val = _local_sign_flip(best_h, max_passes=3, deadline=deadline)

    return best_h
# EVOLVE-BLOCK-END
