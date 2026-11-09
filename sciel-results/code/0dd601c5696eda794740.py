import time
import numpy as np

N = 51975  # 3^3 * 5^2 * 7 * 11, odd


def _divisors(n):
    ds = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            ds.append(i)
            if i != n // i:
                ds.append(n // i)
        i += 1
    return ds


# All odd divisors > 1 of N (N is odd, so every divisor is odd).
MODS = sorted(d for d in _divisors(N) if d > 1)
K = {m: N // m for m in MODS}  # fiber size for modulus m
MODS_LIST = list(MODS)
N_MODS = len(MODS_LIST)


def _cov(assign):
    """Boolean array: covered[x] = exists (m,r) with x == r (mod m)."""
    cov = np.zeros(N, dtype=bool)
    for m in MODS_LIST:
        cov[assign[m]::m] = True
    return cov


def _uncovered(assign):
    return int((~_cov(assign)).sum())


def _build_count(assign):
    """Build integer coverage count array."""
    cnt = np.zeros(N, dtype=np.int32)
    for m in MODS_LIST:
        cnt[assign[m]::m] += 1
    return cnt


def _uncovered_from_count(cnt):
    return int((cnt == 0).sum())


def _cd(assign, max_passes=200):
    """Coordinate descent with incremental coverage counting."""
    cnt = _build_count(assign)
    for _ in range(max_passes):
        improved = False
        for m in MODS_LIST:
            old_r = assign[m]
            # Remove this modulus's contribution
            cnt[old_r::m] -= 1
            # Uncovered positions (only this modulus was covering some)
            unc = (cnt == 0)
            # For each residue r mod m, count uncovered positions in that fiber
            cnt_reshaped = unc.reshape(K[m], m)
            fiber_counts = cnt_reshaped.sum(axis=0)
            br = int(np.argmax(fiber_counts))
            # Add back with best residue
            cnt[br::m] += 1
            if br != old_r:
                assign[m] = br
                improved = True
        if not improved:
            break
    return assign


def _cd_limited(assign, max_passes=3):
    """Limited coordinate descent for faster SA moves."""
    cnt = _build_count(assign)
    for _ in range(max_passes):
        improved = False
        for m in MODS_LIST:
            old_r = assign[m]
            cnt[old_r::m] -= 1
            unc = (cnt == 0)
            cnt_reshaped = unc.reshape(K[m], m)
            fiber_counts = cnt_reshaped.sum(axis=0)
            br = int(np.argmax(fiber_counts))
            cnt[br::m] += 1
            if br != old_r:
                assign[m] = br
                improved = True
        if not improved:
            break
    return assign


def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin well under the 150s limit
    rng = np.random.default_rng(20240517)

    best_assign = None
    best_unc = N

    def consider(a):
        nonlocal best_assign, best_unc
        u = _uncovered(a)
        if u < best_unc:
            best_unc = u
            best_assign = dict(a)
        return u

    # Deterministic baseline: all residues 0, then coordinate descent.
    a0 = {m: 0 for m in MODS_LIST}
    _cd(a0)
    consider(a0)

    # Try greedy with different orderings
    for order_seed in [1, 2, 3, 4, 5]:
        rng_g = np.random.default_rng(order_seed)
        mods_shuffled = list(MODS_LIST)
        rng_g.shuffle(mods_shuffled)
        a_g = {m: 0 for m in MODS_LIST}
        # Greedy: process moduli in shuffled order, pick best residue given current coverage
        for m in mods_shuffled:
            cov = np.zeros(N, dtype=bool)
            for m2 in MODS_LIST:
                if m2 != m:
                    cov[a_g[m2]::m2] = True
            unc = ~cov
            cnt = unc.reshape(K[m], m).sum(axis=0)
            a_g[m] = int(np.argmax(cnt))
        _cd(a_g)
        consider(a_g)

    # Random restarts + coordinate descent.
    while time.time() < deadline - 40.0:
        a = {m: int(rng.integers(0, m)) for m in MODS_LIST}
        _cd(a)
        u = consider(a)
        if u == 0:
            break

    # Simulated annealing on the best solution to escape local minima.
    if best_assign is not None and time.time() < deadline - 5.0:
        a = dict(best_assign)
        cur_u = best_unc
        cnt = _build_count(a)
        T = 5.0
        T_min = 0.01
        while time.time() < deadline - 3.0 and T > T_min:
            # Random single-residue perturbation
            m = MODS_LIST[int(rng.integers(0, N_MODS))]
            old_r = a[m]
            new_r = int(rng.integers(0, m))
            if new_r == old_r:
                continue
            # Remove old, add new
            cnt[old_r::m] -= 1
            cnt[new_r::m] += 1
            u = _uncovered_from_count(cnt)
            if u < cur_u:
                cur_u = u
                a[m] = new_r
                if u < best_unc:
                    best_unc = u
                    best_assign = dict(a)
                    if u == 0:
                        break
            else:
                delta = u - cur_u
                if rng.random() < np.exp(-delta / max(T, 1e-9)):
                    cur_u = u
                    a[m] = new_r
                else:
                    # Revert
                    cnt[new_r::m] -= 1
                    cnt[old_r::m] += 1
            T *= 0.998

        # Final polish: full CD on best
        if best_assign is not None and time.time() < deadline - 1.0:
            a_final = dict(best_assign)
            _cd(a_final, max_passes=300)
            consider(a_final)

    return [[m, int(best_assign[m])] for m in MODS_LIST]
