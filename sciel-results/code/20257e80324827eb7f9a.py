import numpy as np
import time

def _sup_norm(coeffs, n_points=2048):
    """Compute sup norm of P(z)=sum c_k z^k on |z|=1."""
    k = np.arange(len(coeffs))
    c = np.array(coeffs, dtype=complex)
    thetas = 2 * np.pi * np.arange(n_points) / n_points
    z = np.exp(1j * thetas)
    powers = z[:, None] ** k[None, :]
    vals = powers @ c
    mag = np.abs(vals)
    peak_idx = int(np.argmax(mag))
    # Refine
    if peak_idx == 0:
        lo, hi = 0.0, thetas[1]
    elif peak_idx == n_points - 1:
        lo, hi = thetas[-2], 2 * np.pi
    else:
        lo, hi = thetas[peak_idx - 1], thetas[peak_idx + 1]
    phi = (np.sqrt(5) - 1) / 2
    for _ in range(50):
        x1 = lo + (1 - phi) * (hi - lo)
        x2 = lo + phi * (hi - lo)
        f1 = abs(np.sum(c * np.exp(1j * x1 * k)))
        f2 = abs(np.sum(c * np.exp(1j * x2 * k)))
        if f1 < f2:
            hi = x2
        else:
            lo = x1
    x = (lo + hi) / 2
    return abs(np.sum(c * np.exp(1j * x * k)))

def construct():
    start_time = time.time()
    deadline = 140.0

    def gr(n):
        a = [1] * n
        if n > 0: a[0] = 1
        if n > 1: a[1] = 1
        for i in range(2, n):
            a[i] = a[i // 2] if i % 2 == 0 else -a[i // 2]
        return a

    best = gr(24)
    best_score = _sup_norm(best)

    n = 24
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for i in range(n):
            if (time.time() - start_time) > deadline:
                break
            cand = best.copy()
            cand[i] = -cand[i]
            s = _sup_norm(cand)
            if s < best_score:
                best, best_score, improved = cand, s, True
        for i in range(n):
            for j in range(i + 1, n):
                if (time.time() - start_time) > deadline:
                    break
                cand = best.copy()
                cand[i] = -cand[i]
                cand[j] = -cand[j]
                s = _sup_norm(cand)
                if s < best_score:
                    best, best_score, improved = cand, s, True
            if (time.time() - start_time) > deadline:
                break

    return best
