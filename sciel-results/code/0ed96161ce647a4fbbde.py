"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, minimize_scalar

def construct():
    """
    Minimize value = abs(2*N*max(convolve(h,h)) / (sum(h))^2)
    using real part of Zadoff-Chu * complex chirp * raised-cosine window,
    with optimized DC offset. Complex chirp rate enables amplitude shaping.
    """
    best_h = np.ones(N)
    best_val = 2.0

    def evaluate(h):
        if not np.all(np.isfinite(h)):
            return float('inf')
        s = np.sum(h)
        if abs(s) < 1e-12:
            return float('inf')
        conv = np.convolve(h, h)
        m = np.max(conv)
        val = abs(2 * N * m / (s * s))
        return val

    t = np.arange(N, dtype=np.float64)
    n_norm = (t - (N - 1) / 2.0) / ((N - 1) / 2.0)
    t_norm = t / N

    def get_h_ac(params):
        """
        params: [mu, a_real, a_imag, beta, win_param]
        mu: Zadoff-Chu parameter
        a_real: real part of complex chirp rate
        a_imag: imaginary part (controls Gaussian amplitude envelope)
        beta: global phase
        win_param: additional raised-cosine window [0,1]
        """
        mu, a_real, a_imag, beta, win_param = params

        # Zadoff-Chu like sequence
        zc = np.exp(-1j * np.pi * mu * t * (t + 1) / N)

        # Complex chirp: exp(1j * pi * (a_real + 1j*a_imag) * t_norm^2)
        # = exp(1j * pi * a_real * t_norm^2) * exp(-pi * a_imag * t_norm^2)
        chirp_phase = np.exp(1j * np.pi * a_real * t_norm * t_norm)
        chirp_amp = np.exp(-np.pi * a_imag * t_norm * t_norm)
        chirp = chirp_phase * chirp_amp

        # Global phase
        phase = np.exp(1j * beta)

        signal = zc * chirp * phase

        # Raised cosine window
        window = 1.0 - 0.5 * win_param + 0.5 * win_param * np.cos(np.pi * n_norm)

        h_ac = np.real(signal) * window
        return h_ac

    def optimize_dc(x):
        """Finds the optimal DC offset c to add to x to minimize the value."""
        S_x = np.sum(x)
        C_x = np.convolve(x, x)
        U = np.convolve(x, np.ones(N))
        T = np.convolve(np.ones(N), np.ones(N))

        def obj(c):
            conv_c = C_x + 2 * c * U + c**2 * T
            m = np.max(conv_c)
            s = S_x + N * c
            if abs(s) < 1e-12:
                return float('inf')
            val = abs(2 * N * m / (s * s))
            return val

        best_c = 0.0
        best_obj = obj(0.0)

        for c_start in np.linspace(-5, 5, 21):
            res = minimize_scalar(obj, bounds=(c_start - 0.5, c_start + 0.5),
                                  method='bounded', options={'xatol': 1e-9})
            if res.success and res.fun < best_obj:
                best_obj = res.fun
                best_c = res.x

        return x + best_c, best_obj

    def objective(params):
        try:
            h_ac = get_h_ac(params)
            h, val = optimize_dc(h_ac)
            return val
        except Exception:
            return float('inf')

    np.random.seed(42)

    n_starts = 60
    best_val_found = float('inf')
    best_params_found = None
    best_h_found = None

    for i in range(n_starts):
        if i < 15:
            # Structured: vary mu, moderate chirp
            mu = i * 1.5
            a_real = np.random.uniform(-1, 1)
            a_imag = np.random.uniform(0, 3)
            beta = np.random.uniform(0, 2 * np.pi)
            win_param = np.random.uniform(0.3, 1.0)
        elif i < 30:
            # Emphasize amplitude shaping via a_imag
            mu = np.random.uniform(0, 20)
            a_real = np.random.uniform(-2, 2)
            a_imag = np.random.uniform(0.5, 5.0)
            beta = np.random.uniform(0, 2 * np.pi)
            win_param = np.random.uniform(0, 0.8)
        elif i < 45:
            # Strong amplitude shaping
            mu = np.random.uniform(0, 30)
            a_real = np.random.uniform(-3, 3)
            a_imag = np.random.uniform(1.0, 8.0)
            beta = np.random.uniform(0, 2 * np.pi)
            win_param = np.random.uniform(0, 1.0)
        else:
            # Fully random
            mu = np.random.uniform(0, 40)
            a_real = np.random.uniform(-4, 4)
            a_imag = np.random.uniform(0, 10)
            beta = np.random.uniform(0, 2 * np.pi)
            win_param = np.random.uniform(0, 1.0)

        x0 = np.array([mu, a_real, a_imag, beta, win_param])

        res = minimize(
            objective,
            x0,
            method='Powell',
            options={'maxiter': 200, 'maxfev': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
        )

        if res.fun < best_val_found:
            best_val_found = res.fun
            best_params_found = res.x.copy()
            h_ac = get_h_ac(best_params_found)
            best_h_found, _ = optimize_dc(h_ac)

            if best_val_found < 1.05:
                break

    if best_params_found is not None and best_val_found < best_val:
        return best_h_found
    return best_h
# EVOLVE-BLOCK-END
