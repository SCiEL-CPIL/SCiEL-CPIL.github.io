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


MODS = sorted(d for d in _divisors(N) if d > 1)
K = {m: N // m for m in MODS}
MODS_LIST = list(MODS)
N_MODS = len(MODS_LIST)

# Precompute position arrays: pos[m][r] = all x in [0,N) with x % m == r
# Stored as flat arrays for fast fancy indexing
_pos_cache = {}
def _get_pos(m):
    if m not in _pos_cache:
        km = K[m]
        # positions for residue r: r, r+m, r+2m, ..., r+(km-1)*m
        # shape (m, km)
        base = np.arange(km) * m
        _pos_cache[m] = (base[None, :] + np.arange(m)[:, None]).reshape(-1)
        # Actually we want pos[m] as a 2D array (m, km) so pos[m][r] gives the km positions
        _pos_cache[m] = base[None, :] + np.arange(m)[:, None]
    return _pos_cache[m]

# Precompute all
for m in MODS_LIST:
    _get_pos(m)


def _build_count(assign):
    cnt = np.zeros(N, dtype=np.int32)
    for m in MODS_LIST:
        r = assign[m]
        cnt[_pos_cache[m][r]] += 1
    return cnt


def _uncovered_from_count(cnt):
    return int((cnt == 0).sum())


def _cd(assign, max_passes=300, deadline=None):
    cnt = _build_count(assign)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for m in MODS_LIST:
            old_r = assign[m]
            cnt[_pos_cache[m][old_r]] -= 1
            unc = (cnt == 0)
            # Fiber counts: reshape unc to (K[m], m), sum over axis 0
            km = K[m]
            fiber_counts = unc.reshape(km, m).sum(axis=0)
            br = int(np.argmax(fiber_counts))
            cnt[_pos_cache[m][br]] += 1
            if br != old_r:
                assign[m] = br
                improved = True
        if not improved:
            break
    return assign


def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    rng = np.random.default_rng(20240517)

    best_assign = None
    best_unc = N

    def consider(a):
        nonlocal best_assign, best_unc
        u = _uncovered_from_count(_build_count(a))
        if u < best_unc:
            best_unc = u
            best_assign = dict(a)
        return u

    # Deterministic baseline: all residues 0, then CD
    a0 = {m: 0 for m in MODS_LIST}
    _cd(a0, max_passes=500, deadline=deadline)
    consider(a0)

    # Greedy with different orderings (largest to smallest, smallest to largest, random)
    orderings = [
        MODS_LIST[::-1],  # largest first
        MODS_LIST,         # smallest first
    ]
    for _ in range(5):
        orderings.append(list(rng.choice(MODS_LIST, size=N_MODS, replace=False)))

    for order in orderings:
        if time.time() > deadline - 30.0:
            break
        a_g = {m: 0 for m in MODS_LIST}
        # Greedy: process in order, pick best residue given current coverage
        cov = np.zeros(N, dtype=bool)
        for m in order:
            unc = ~cov
            km = K[m]
            fiber_counts = unc.reshape(km, m).sum(axis=0)
            a_g[m] = int(np.argmax(fiber_counts))
            cov[_pos_cache[m][a_g[m]]] = True
        _cd(a_g, max_passes=300, deadline=deadline)
        consider(a_g)

    # Random restarts + CD
    while time.time() < deadline - 40.0:
        a = {m: int(rng.integers(0, m)) for m in MODS_LIST}
        _cd(a, max_passes=200, deadline=deadline)
        u = consider(a)
        if u == 0:
            break

    # Simulated annealing on the best solution
    if best_assign is not None and time.time() < deadline - 5.0:
        a = dict(best_assign)
        cur_u = best_unc
        cnt = _build_count(a)
        T = 10.0
        T_min = 0.01
        # Precompute for faster SA
        while time.time() < deadline - 3.0 and T > T_min:
            m = MODS_LIST[int(rng.integers(0, N_MODS))]
            old_r = a[m]
            new_r = int(rng.integers(0, m))
            if new_r == old_r:
                continue
            cnt[_pos_cache[m][old_r]] -= 1
            cnt[_pos_cache[m][new_r]] += 1
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
                    cnt[_pos_cache[m][new_r]] -= 1
                    cnt[_pos_cache[m][old_r]] += 1
            T *= 0.999

        # Final polish
        if best_assign is not None and time.time() < deadline - 1.0:
            a_final = dict(best_assign)
            _cd(a_final, max_passes=500, deadline=deadline)
            consider(a_final)

    return [[m, int(best_assign[m])] for m in MODS_LIST]
