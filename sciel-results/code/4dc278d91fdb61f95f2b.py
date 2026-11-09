"""Spectral shaping: construct signed steps by flattening |F|^2 then polish."""
import numpy as np
import time

N = 400
FULL = 2 * N - 1


def _autoconv(f):
    size = 1
    while size < FULL:
        size <<= 1
    F = np.fft.rfft(f, size)
    power = F * np.conj(F)
    return np.fft.irfft(power, size)[:FULL]


def _score(auto, s):
    if abs(s) < 1e-12:
        return float('inf')
    return np.max(np.abs(auto)) / (s * s)


def _spectral_candidate(rng, flat_frac=0.5, target_energy=None):
    """Build a candidate by shaping the spectrum toward a flat magnitude."""
    size = 1
    while size < FULL:
        size <<= 1
    # Target magnitude: mix a flat floor with a random shape to add diversity
    k = np.arange(size // 2 + 1)
    if target_energy is None:
        target_energy = N  # sum of f^2 = N for ±1
    # flat magnitude would give |F|^2 = target_energy uniformly -> |F| = sqrt(E)
    mag_target = np.full(size // 2 + 1, np.sqrt(target_energy))
    # blend in a random magnitude profile for diversity
    rand_mag = np.sqrt(target_energy) * (0.5 + rng.random(size // 2 + 1))
    mag = (1 - flat_frac) * mag_target + flat_frac * rand_mag
    phase = 2 * np.pi * rng.random(size // 2 + 1)
    F = mag * np.exp(1j * phase)
    # enforce conjugate symmetry (real signal)
    F_full = np.empty(size, dtype=complex)
    F_full[:size // 2 + 1] = F
    F_full[size // 2 + 1:] = np.conj(F[1:size // 2][::-1])
    g = np.fft.ifft(F_full).real
    # round to ±1
    f = np.where(g >= 0, 1.0, -1.0)
    return f[:N]


def _local_search(f, auto, s, current, deadline, rng, max_sweeps=200):
    """Greedy sign-flip local search with incremental autoconv update."""
    f = f.copy()
    auto = auto.copy()
    n = N
    for _ in range(max_sweeps):
        if time.time() > deadline - 3:
            break
        improved = False
        order = rng.permutation(n)
        for i in order:
            if time.time() > deadline - 3:
                break
            old = f[i]
            delta = -2.0 * old
            new_s = s + delta
            if abs(new_s) < 1e-12:
                continue
            # incremental update
            left = np.zeros(FULL)
            left[i:i + n] = f
            right = np.zeros(FULL)
            right[:n - i] = f[i:]
            new_auto = auto + delta * (left + right)
            if 0 <= 2 * i < FULL:
                new_auto[2 * i] += delta * delta
            # fast max-abs check via tracking
            new_max = max(np.max(np.abs(new_auto)), current * s * s)
            new_score = new_max / (new_s * new_s)
            if new_score < current - 1e-12:
                f[i] = old + delta
                auto = new_auto
                s = new_s
                current = new_score
                improved = True
        if not improved:
            break
    return f, auto, s, current


def _anneal(f, auto, s, current, deadline, rng):
    """Simulated annealing with incremental update."""
    f = f.copy()
    auto = auto.copy()
    n = N
    t0 = time.time()
    t_start = 0.05
    t_end = 0.0005
    while time.time() < deadline - 3:
        frac = (time.time() - t0) / max(deadline - 3 - t0, 1e-9)
        temp = t_start * (t_end / t_start) ** min(frac, 1.0)
        i = rng.integers(n)
        old = f[i]
        delta = -2.0 * old
        new_s = s + delta
        if abs(new_s) < 1e-12:
            continue
        left = np.zeros(FULL)
        left[i:i + n] = f
        right = np.zeros(FULL)
        right[:n - i] = f[i:]
        new_auto = auto + delta * (left + right)
        if 0 <= 2 * i < FULL:
            new_auto[2 * i] += delta * delta
        new_score = np.max(np.abs(new_auto)) / (new_s * new_s)
        if new_score < current:
            f[i] = old + delta
            auto = new_auto
            s = new_s
            current = new_score
        else:
            d = new_score - current
            if d > 0 and temp > 0 and rng.random() < np.exp(-d / temp):
                f[i] = old + delta
                auto = new_auto
                s = new_s
                current = new_score
    return f, auto, s, current


def construct():
    deadline = time.time() + 145
    rng = np.random.default_rng(2024)

    best_f = None
    best_score = float('inf')

    # 1. Spectral shaping: generate diverse candidates, keep the best, polish each.
    n_spec = 12
    for c in range(n_spec):
        if time.time() > deadline - 40:
            break
        flat_frac = 0.3 + 0.6 * (c / max(n_spec - 1, 1))
        f = _spectral_candidate(rng, flat_frac=flat_frac)
        auto = _autoconv(f)
        s = np.sum(f)
        sc = _score(auto, s)
        if sc < best_score:
            best_score = sc
            best_f = f.copy()
            best_auto = auto.copy()
            best_s = s
        # quick polish on this candidate
        if time.time() > deadline - 40:
            break
        f2, auto2, s2, sc2 = _local_search(f, auto, s, sc, deadline, rng, max_sweeps=30)
        if sc2 < best_score:
            best_score = sc2
            best_f = f2.copy()
            best_auto = auto2.copy()
            best_s = s2

    # 2. Also seed from all +1 and random for robustness.
    for seed_f in [np.ones(N), np.where(rng.random(N) < 0.5, 1.0, -1.0)]:
        if time.time() > deadline - 40:
            break
        auto = _autoconv(seed_f)
        s = np.sum(seed_f)
        sc = _score(auto, s)
        if sc < best_score:
            best_score = sc
            best_f = seed_f.copy()
            best_auto = auto.copy()
            best_s = s

    # 3. Deep polish on the best.
    if time.time() < deadline - 20:
        rng2 = np.random.default_rng(11)
        best_f, best_auto, best_s, best_score = _local_search(
            best_f, best_auto, best_s, best_score, deadline, rng2, max_sweeps=400)

    # 4. Anneal to escape local minima.
    if time.time() < deadline - 8:
        rng3 = np.random.default_rng(22)
        f, auto, s, sc = _anneal(best_f, best_auto, best_s, best_score, deadline, rng3)
        if sc < best_score:
            best_score = sc
            best_f = f.copy()
            best_auto = auto.copy()
            best_s = s

    # 5. Final polish.
    if time.time() < deadline - 2:
        rng4 = np.random.default_rng(33)
        best_f, best_auto, best_s, best_score = _local_search(
            best_f, best_auto, best_s, best_score, deadline, rng4, max_sweeps=300)

    return best_f
