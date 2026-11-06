"""Search for the flattest plus-minus-one polynomial with 24 coefficients."""
import time
import random
import numpy as np

N = 24
TARGET = 1.2583758
WALL_LIMIT = 150.0
SAFETY_MARGIN = 5.0


def _eval_mags(coeffs, angles):
    """Evaluate |P(e^{i*angle})| for given angles. coeffs: np.array of shape (24,)."""
    k = np.arange(N)
    phases = np.exp(1j * np.outer(angles, k))
    return np.abs(phases @ coeffs)


def _fast_score(coeffs, angles):
    """Coarse sup-norm score for inner-loop use."""
    mags = _eval_mags(coeffs, angles)
    return float(np.max(mags)) / np.sqrt(N)


def _refined_score(coeffs, n_coarse=4096, n_peaks=6, n_gs=80):
    """Refined sup-norm score with golden-section peak refinement."""
    angles = np.linspace(0, 2 * np.pi, n_coarse, endpoint=False)
    mags = _eval_mags(coeffs, angles)
    top_idx = np.argsort(mags)[::-1][:n_peaks]
    best = float(mags[top_idx[0]])
    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    for idx in top_idx:
        lo = angles[idx - 1] if idx > 0 else angles[idx - 1] + 2 * np.pi
        hi = angles[idx + 1] if idx < n_coarse - 1 else angles[idx + 1] - 2 * np.pi
        if lo > hi:
            lo -= 2 * np.pi
        c = hi - invphi * (hi - lo)
        d = lo + invphi * (hi - lo)
        fc = _eval_mags(coeffs, np.array([c]))[0]
        fd = _eval_mags(coeffs, np.array([d]))[0]
        for _ in range(n_gs):
            if fc < fd:
                lo, c, fc = c, d, fd
                d = lo + invphi * (hi - lo)
                fd = _eval_mags(coeffs, np.array([d]))[0]
            else:
                hi, d, fd = d, c, fc
                c = hi - invphi * (hi - lo)
                fc = _eval_mags(coeffs, np.array([c]))[0]
        v = max(fc, fd)
        if v > best:
            best = v
    return best / np.sqrt(N)


def _greedy_polish(coeffs, angles, max_rounds=500):
    """Greedy 1-flip then 2-flip local search."""
    coeffs = list(coeffs)
    cur = _fast_score(np.array(coeffs), angles)
    for _ in range(max_rounds):
        improved = False
        # 1-flip
        for i in range(N):
            coeffs[i] = -coeffs[i]
            ns = _fast_score(np.array(coeffs), angles)
            if ns < cur - 1e-13:
                cur = ns
                improved = True
                break
            coeffs[i] = -coeffs[i]
        # 2-flip
        if not improved:
            for i in range(N):
                for j in range(i + 1, N):
                    coeffs[i] = -coeffs[i]
                    coeffs[j] = -coeffs[j]
                    ns = _fast_score(np.array(coeffs), angles)
                    if ns < cur - 1e-13:
                        cur = ns
                        improved = True
                        break
                    coeffs[i] = -coeffs[i]
                    coeffs[j] = -coeffs[j]
                if improved:
                    break
        if not improved:
            break
    return coeffs, cur


def construct():
    t0 = time.time()
    deadline = t0 + WALL_LIMIT - SAFETY_MARGIN

    # Initial Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    best = list(p[:N])
    best_score = _refined_score(np.array(best))

    angles = np.linspace(0, 2 * np.pi, 2048, endpoint=False)
    rng = random.Random(12345)

    # Multi-restart SA + greedy
    while time.time() < deadline:
        # Generate starting point
        r = rng.random()
        if r < 0.3:
            cur = [rng.choice([-1, 1]) for _ in range(N)]
        elif r < 0.7:
            cur = list(best)
            for _ in range(rng.randint(2, 4)):
                cur[rng.randint(0, N - 1)] *= -1
        else:
            cur = list(p[:N])
            for _ in range(rng.randint(1, 3)):
                cur[rng.randint(0, N - 1)] *= -1

        # Simulated annealing
        sa = list(cur)
        sa_score = _fast_score(np.array(sa), angles)
        sa_best = list(sa)
        sa_best_score = sa_score
        temp = 0.06
        cool = 0.9965
        max_it = 4000

        for it in range(max_it):
            if time.time() > deadline:
                break
            # Move
            if rng.random() < 0.65:
                i = rng.randint(0, N - 1)
                sa[i] = -sa[i]
                new_score = _fast_score(np.array(sa), angles)
                flip_positions = (i,)
            else:
                i, j = rng.sample(range(N), 2)
                sa[i] = -sa[i]
                sa[j] = -sa[j]
                new_score = _fast_score(np.array(sa), angles)
                flip_positions = (i, j)

            delta = new_score - sa_score
            if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-6)):
                sa_score = new_score
                if new_score < sa_best_score:
                    sa_best_score = new_score
                    sa_best = list(sa)
            else:
                # Undo
                for pos in flip_positions:
                    sa[pos] = -sa[pos]
            temp *= cool

        # Polish
        polished, pol_score = _greedy_polish(sa_best, angles, max_rounds=400)
        full = _refined_score(np.array(polished))
        if full < best_score:
            best_score = full
            best = list(polished)
            if best_score <= TARGET + 1e-7:
                break

    return best
