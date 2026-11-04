import numpy as np
import time
import scipy.optimize

N = 400

def _fft_size(n):
    size = 1
    while size < 2 * n - 1:
        size *= 2
    return size

def _autoconv_lin(x):
    n = len(x)
    size = _fft_size(n)
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _metric_std(x):
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    return float(np.max(np.abs(c))) / (s * s)

# ---------------- Frequency-domain phase optimization ----------------

def _phase_objective(phi, beta, weight_peak):
    """Minimize spectral flatness ratio + soft-max peak penalty on autocorrelation."""
    n = len(phi)
    mag = np.ones(n)
    X = mag * np.exp(1j * phi)
    x = np.real(np.fft.ifft(X))
    s = np.sum(x)
    if abs(s) < 1e-8:
        return 1e10

    # autocorrelation from |X|^2 (periodic); take linear part
    ac = np.fft.ifft(np.abs(X) ** 2).real
    ac = np.concatenate([ac[:n], ac[2 * n - 1:0:-1]])  # length 2n-1, lags -(n-1)..(n-1)

    # spectral flatness ratio (geometric/mean) of |X|^2
    p = np.abs(X) ** 2
    p = p + 1e-12
    flat = np.exp(np.mean(np.log(p))) / (np.mean(p) + 1e-12)

    # soft-max peak of |ac| over off-diagonal lags
    off = np.concatenate([ac[:n - 1], ac[n + 1:]])
    off_abs = np.abs(off)
    off_abs = off_abs / (np.mean(p) + 1e-12)  # normalize
    soft = np.logaddexp.reduce(np.log(off_abs + 1e-12) * beta) / beta

    return flat + weight_peak * soft

def _phase_to_binary(phi):
    X = np.exp(1j * phi)
    x = np.real(np.fft.ifft(X))
    b = np.sign(x)
    b[b == 0] = 1.0
    return b

def _optimize_phases(phi0, deadline, beta=40.0, weight_peak=1.5):
    def obj(phi):
        return _phase_objective(phi, beta, weight_peak)

    res = scipy.optimize.minimize(
        obj, phi0, method='L-BFGS-B',
        options={'maxiter': 300, 'maxfun': 400, 'gtol': 1e-3, 'ftol': 1e-6},
    )
    return res.x

def _generate_phase_candidates(rng):
    cands = []
    # random phases
    for _ in range(4):
        cands.append(rng.uniform(0.0, 2.0 * np.pi, size=N))
    # sine-sum phases: phi_k = sum_j a_j * sin(2 pi f_j k + p_j)
    for _ in range(4):
        k = np.arange(N)
        phi = np.zeros(N)
        for _ in range(3):
            f = rng.uniform(0.01, 0.5)
            a = rng.uniform(0.5, 2.0)
            p = rng.uniform(0.0, 2.0 * np.pi)
            phi += a * np.sin(2.0 * np.pi * f * k + p)
        cands.append(phi)
    # low-frequency ramp
    k = np.arange(N)
    cands.append(2.0 * np.pi * (k / N))
    cands.append(2.0 * np.pi * (2 * k / N))
    return cands

# ---------------- Fast discrete polish (O(N) per flip) ----------------

def _c_from_ac(ac, n):
    return np.concatenate([ac[:n], ac[2 * n - 1:0:-1]])

def _ac_update(ac, x, i, delta, n):
    """Return updated autocorrelation after x[i] -> x[i]+delta."""
    ac = ac.copy()
    xi = x[i]
    new = xi + delta
    # lag 0
    ac[0] += 2.0 * delta * xi + delta * delta
    # positive lags k = i+1 .. n-1 : ac[k] += 2*delta*x[i+k]
    if i + 1 < n:
        ac[1:n] += 2.0 * delta * x[i + 1:]
    # negative lags k = -(i-1) .. -1  (stored at index 2n-1-k)
    if i >= 1:
        neg_idx = 2 * n - 1 - np.arange(1, i)
        ac[neg_idx] += 2.0 * delta * x[i - 1::-1]
    return ac

def _polish(x, ac, deadline, max_iters=40000):
    n = len(x)
    x = x.copy()
    ac = ac.copy()
    rng = np.random.default_rng(777)
    s = np.sum(x)
    ss = s * s
    c = _c_from_ac(ac, n)
    cur = float(np.max(np.abs(c))) / ss
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        i = rng.integers(0, n)
        delta = -2.0 * x[i]
        ac2 = _ac_update(ac, x, i, delta, n)
        c2 = _c_from_ac(ac2, n)
        v = float(np.max(np.abs(c2))) / ss
        if v < cur - 1e-12:
            x[i] += delta
            ac = ac2
            c = c2
            cur = v
    return x, cur

def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(42)

    best_x = None
    best_val = float('inf')

    def consider(x):
        nonlocal best_x, best_val
        if time.time() > deadline - 3.0:
            return
        v = _metric_std(x)
        if v < best_val:
            best_val = v
            best_x = x.copy()

    # Phase candidates -> binary -> polish
    for phi0 in _generate_phase_candidates(rng):
        if time.time() > deadline - 15.0:
            break
        phi_opt = _optimize_phases(phi0, deadline - 5.0)
        x_bin = _phase_to_binary(phi_opt)
        consider(x_bin)
        if best_x is not None:
            ac = _autoconv_lin(best_x)
            x_pol, v_pol = _polish(best_x, ac, deadline - 2.0)
            consider(x_pol)

    # Also seed from plain random signs for safety
    if time.time() < deadline - 15.0:
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            x_r = rng.choice([-1.0, 1.0], size=N).astype(float)
            consider(x_r)
            if best_x is not None:
                ac = _autoconv_lin(best_x)
                x_pol, v_pol = _polish(best_x, ac, deadline - 2.0)
                consider(x_pol)

    if best_x is None:
        return np.ones(N)
    return best_x
