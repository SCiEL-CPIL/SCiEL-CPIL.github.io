"""Optimize step heights to minimize max overlap of h with shifted 1-h."""
import time
import numpy as np

N = 100


def _overlap_matrix(h):
    # C_k = sum_i h_i * (1 - h_{i+k mod N}) for k in 0..N-1
    # Use FFT for speed: 1-h shifted by k, dot with h
    # C_k = sum_i h_i - sum_i h_i * h_{i+k} = sum(h) - autocorr_k
    # autocorr_k = sum_i h_i * h_{i+k} (cyclic)
    return None


def _max_overlap(h):
    h = np.asarray(h)
    s = np.sum(h)
    # cyclic autocorrelation: A_k = sum_i h_i * h_{(i+k) mod N}
    # use FFT
    F = np.fft.fft(h)
    A = np.fft.ifft(F * np.conj(F)).real
    # A_k = sum_i h_i * h_{(i+k) mod N}? Check: ifft(|F|^2) gives circular correlation
    # np: ifft(fft(h)*conj(fft(h)))_k = sum_i h_i * h_{(i-k) mod N}
    # So A_k = sum_i h_i h_{i-k}. We want sum_i h_i h_{i+k} = A_{-k} = A_{N-k}.
    # max over k of C_k = s - min over k of A_k. So maximize min A_k.
    C = s - A  # C_k = s - A_k, but A here is sum h_i h_{i-k}; max C = s - min A
    return np.max(C)


def _soft_max_overlap(h, lam=50.0):
    h = np.asarray(h)
    s = np.sum(h)
    F = np.fft.fft(h)
    A = np.fft.ifft(F * np.conj(F)).real
    C = s - A
    # soft max via log-sum-exp
    m = np.max(C)
    return m + np.log(np.sum(np.exp(lam * (C - m)))) / lam


def _objective(x):
    # x are free variables; we optimize over h directly with bounds and equality via penalty or use SLSQP
    return _soft_max_overlap(x)


def construct():
    t0 = time.time()
    DEADLINE = 110.0  # safety margin

    # Initial guess: uniform
    h0 = np.full(N, 0.5)

    from scipy.optimize import minimize

    best_h = h0.copy()
    best_val = _max_overlap(best_h)

    # Try a few initializations and optimize
    inits = [h0]

    # Block initialization: half 0, half 1
    hb = np.zeros(N)
    hb[:N // 2] = 1.0
    inits.append(hb)

    # Random
    rng = np.random.default_rng(0)
    for _ in range(4):
        hr = rng.random(N)
        hr = (hr - hr.mean()) / (np.std(hr) + 1e-12) * 0.3 + 0.5
        hr = np.clip(hr, 0, 1)
        inits.append(hr)

    for h_init in inits:
        if time.time() - t0 > DEADLINE:
            break
        try:
            res = minimize(
                _objective,
                h_init,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2}],
                options={'maxiter': 200, 'ftol': 1e-9},
            )
            h_try = res.x
            # project to satisfy constraints exactly
            h_try = np.clip(h_try, 0, 1)
            # fix sum exactly
            diff = N / 2 - np.sum(h_try)
            if abs(diff) > 1e-12:
                # distribute
                h_try = h_try + diff / N
                h_try = np.clip(h_try, 0, 1)
            val = _max_overlap(h_try)
            if val < best_val - 1e-12:
                best_val = val
                best_h = h_try.copy()
        except Exception:
            pass

    # Final exact projection to make sum exactly N/2
    h = best_h.copy()
    h = np.clip(h, 0, 1)
    diff = N / 2 - np.sum(h)
    if abs(diff) > 1e-12:
        # add to elements not at bound
        mask = (h > 1e-9) & (h < 1 - 1e-9)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0, 1)
        else:
            h = h + diff / N
            h = np.clip(h, 0, 1)

    return h
