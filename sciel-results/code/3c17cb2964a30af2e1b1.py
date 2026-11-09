import time, math, copy
import numpy as np

# ---------------- fast local TM band solver ----------------
def _fourier_eps(cell, n):
    # cell: (N,N) 0/1 ; eps = 1 + 11*cell
    N = cell.shape[0]
    eps = 1.0 + 11.0 * cell.astype(float)
    c = np.fft.fft2(eps) / (N * N)  # c[m] = (1/N^2) sum eps(x) exp(-i 2pi m.x/N)
    idx = list(range(-n, n + 1))
    m = np.array(idx, dtype=int)
    G = np.zeros((2 * n + 1, 2 * n + 1, 2), dtype=float)
    for ii in range(2 * n + 1):
        for jj in range(2 * n + 1):
            G[ii, jj, 0] = m[ii]
            G[ii, jj, 1] = m[jj]
    C = np.zeros((2 * n + 1, 2 * n + 1), dtype=complex)
    for ii in range(2 * n + 1):
        for jj in range(2 * n + 1):
            C[ii, jj] = c[m[ii], m[jj]]
    return G, C

def _tm_eigs(G, C, k, band_lo, band_hi, n):
    N = 2 * n + 1
    G2 = (G[:, :, 0] ** 2 + G[:, :, 1] ** 2)
    A = np.zeros((N, N), dtype=complex)
    for i in range(N):
        gi = G[i, 0]; gj = G[i, 1]
        for j in range(N):
            # (k+G_i)·(k+G_j) * C[G_i - G_j]
            kx = k[0] + gi; ky = k[1] + gj
            dx = G[i, 0] - G[j, 0]; dy = G[i, 1] - G[j, 1]
            A[i, j] = (kx * (k[0] + G[j, 0]) + ky * (k[1] + G[j, 1])) * C[dx, dy]
    # TM: eigenvalue = (k+G_i)^2
    # build Hermitian H = (k+G_i)^2 delta - A ; solve (H) v = lambda v
    # Actually standard: (k+G_i)^2 v_i - sum_j C[G_i-G_j] (k+G_i).(k+G_j) v_j = lambda v_i
    diag = (G[:, :, 0] + k[0]) ** 2 + (G[:, :, 1] + k[1]) ** 2
    H = np.diag(diag) - A
    H = (H + H.conj().T) / 2.0
    vals = np.linalg.eigvalsh(H)
    # omega = sqrt(lambda)
    lo = max(0, band_lo - 1)
    hi = min(len(vals), band_hi)
    return np.sqrt(np.maximum(vals[lo:hi], 0.0))

def _gap_ratio(cell, n_pw, kpts, b_lo=4, b_hi=5):
    G, C = _fourier_eps(cell, n_pw)
    max4 = 0.0
    min5 = float('inf')
    for k in kpts:
        ev = _tm_eigs(G, C, k, b_lo, b_hi, n_pw)
        if len(ev) >= 2:
            max4 = max(max4, float(ev[0]))
            min5 = min(min5, float(ev[-1]))
    if not (min5 > 0 and max4 > 0):
        return -1.0
    return 2.0 * (min5 - max4) / (min5 + max4)

# ---------------- k-points (irreducible BZ) ----------------
def _make_kpts(count=14):
    pts = []
    def add(x, y):
        # map into irreducible wedge: x>=y, x>=0, x+y<=pi (use units pi)
        # keep as-is; we just need a good spread
        pts.append((x, y))
    # high symmetry
    add(0.0, 0.0)          # Gamma
    add(0.5, 0.0)          # X
    add(0.5, 0.5)          # M
    # lines
    for t in [0.25, 0.35, 0.42]:
        add(t, 0.0)        # Gamma-X
    for t in [0.25, 0.35, 0.42]:
        add(t, t)          # Gamma-M
    for t in [0.15, 0.3, 0.42]:
        add(0.5, t)        # X-M
    for t in [0.2, 0.3]:
        add(0.5, 0.5 - t)  # interior
    # dedupe
    uniq = []
    for p in pts:
        if not any(abs(p[0]-q[0]) < 1e-6 and abs(p[1]-q[1]) < 1e-6 for q in uniq):
            uniq.append(p)
    return uniq[:count]

# ---------------- D4 symmetric parameterization (Fourier synthesis) ----------------
def _build_cell(params, size=32):
    # params: [dc, a0, a1, a2, a3, a4, a5]  (continuous)
    dc, a0, a1, a2, a3, a4, a5 = params
    # continuous field over pixel centers
    X, Y = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    u = (X + 0.5) / size
    v = (Y + 0.5) / size
    # fundamental domain coordinate (D4): use min/max to fold
    # value = dc + sum a_k * mode_k ; modes are D4-symmetric (use |u-0.5|,|v-0.5| combos)
    du = np.abs(u - 0.5)
    dv = np.abs(v - 0.5)
    # radial-ish and square-symmetric basis functions (all invariant under D4 about center)
    r2 = du * du + dv * dv
    # square radius
    sq = np.maximum(du, dv)
    f0 = np.ones_like(u)                       # DC
    f1 = np.cos(2 * math.pi * (u + v))          # X/M harmonic (symmetric under D4? cos(2pi(u+v)) is)
    f2 = np.cos(2 * math.pi * (u - v))
    f3 = np.cos(4 * math.pi * u) * np.cos(4 * math.pi * v)
    f4 = np.cos(2 * math.pi * np.maximum(du, dv))
    f5 = np.cos(2 * math.pi * np.sqrt(r2))
    field = (dc
             + a0 * f0
             + a1 * f1
             + a2 * f2
             + a3 * f3
             + a4 * f4
             + a5 * f5)
    # threshold at 0
    cell = (field > 0.0).astype(int)
    return cell

def _enforce_d4(cell):
    N = cell.shape[0]
    # average over the 8 D4 operations then threshold (keeps exact symmetry)
    ops = []
    def rot90(c): return c.T[::-1, :]
    def flipx(c): return c[:, ::-1]
    c = cell.astype(float)
    ops.append(c)
    ops.append(flipx(c))
    ops.append(rot90(c))
    ops.append(rot90(rot90(c)))
    ops.append(rot90(rot90(rot90(c))))
    ops.append(flipx(rot90(c)))
    ops.append(flipx(rot90(rot90(c))))
    ops.append(flipx(rot90(rot90(rot90(c)))))
    avg = sum(ops) / 8.0
    return (avg > 0.5).astype(int)

def _score_cell(cell, kpts, n_pw=6):
    return _gap_ratio(cell, n_pw, kpts)

# ---------------- local pixel-flip polish (D4 respecting) ----------------
def _polish(cell, kpts, n_pw, max_rounds=6, time_budget=60.0):
    N = cell.shape[0]
    cell = cell.astype(int)
    best = _score_cell(cell, kpts, n_pw)
    t0 = time.time()
    # fundamental domain pixels: i<=j and i+j < N (about center)
    # We'll operate on the full cell but apply flips symmetrically.
    for rnd in range(max_rounds):
        if time.time() - t0 > time_budget:
            break
        improved = False
        # iterate over fundamental domain
        for i in range(N // 2 + 1):
            for j in range(i, N - i):
                if i + j >= N:
                    continue
                # gather orbit under D4
                orbit = set()
                def add_orbit(a, b):
                    orbit.add((a, b))
                    orbit.add((b, a))
                    orbit.add((a, N - 1 - b))
                    orbit.add((N - 1 - a, b))
                    orbit.add((N - 1 - a, N - 1 - b))
                    orbit.add((b, N - 1 - a))
                    orbit.add((N - 1 - b, a))
                    orbit.add((N - 1 - b, N - 1 - a))
                add_orbit(i, j)
                # flip the orbit
                for (a, b) in orbit:
                    cell[a, b] = 1 - cell[a, b]
                sc = _score_cell(cell, kpts, n_pw)
                if sc > best + 1e-9:
                    best = sc
                    improved = True
                else:
                    for (a, b) in orbit:
                        cell[a, b] = 1 - cell[a, b]
        if not improved:
            break
    return cell, best

# ---------------- main ----------------
def design(problem):
    size = problem.get("pixels", 32)
    t_start = time.time()
    WALL_LIMIT = 200.0  # safety margin before 230s kill

    kpts = _make_kpts(14)
    n_pw = 6  # fast solver for search

    # initial guess: centered rod (seed) as a starting cell
    def rod_cell(r=0.2):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            for j in range(size):
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < r * r:
                    cell[i, j] = 1
        return cell

    best_cell = rod_cell(0.2)
    best_score = _score_cell(best_cell, kpts, n_pw)

    # ---- Fourier synthesis + Nelder-Mead over continuous amplitudes ----
    def objective(params):
        cell = _build_cell(params, size)
        cell = _enforce_d4(cell)
        return -_score_cell(cell, kpts, n_pw)

    # initial params around a rod-like profile
    p0 = [0.0, 0.3, 0.0, 0.0, -0.2, 0.15, 0.0]

    try:
        from scipy.optimize import minimize
        t_nm = time.time()
        res = minimize(objective, p0, method='Nelder-Mead',
                       options={'maxiter': 400, 'xatol': 1e-3, 'fatol': 1e-4,
                                'adaptive': True})
        if res.x is not None:
            cell = _build_cell(res.x, size)
            cell = _enforce_d4(cell)
            sc = _score_cell(cell, kpts, n_pw)
            if sc > best_score:
                best_score = sc
                best_cell = cell
    except Exception:
        pass

    if time.time() - t_start > WALL_LIMIT - 40:
        return best_cell.tolist()

    # ---- local pixel-flip polish ----
    try:
        polished, sc = _polish(best_cell.copy(), kpts, n_pw,
                               max_rounds=6, time_budget=min(120.0, WALL_LIMIT - (time.time() - t_start) - 30))
        if sc > best_score:
            best_score = sc
            best_cell = polished
    except Exception:
        pass

    if time.time() - t_start > WALL_LIMIT - 20:
        return best_cell.tolist()

    # ---- final high-precision check with more plane waves / k-points ----
    try:
        kpts_hi = _make_kpts(20)
        sc_hi = _score_cell(best_cell, kpts_hi, n_pw=8)
        # try a couple of radius/param tweaks if time
        if time.time() - t_start < WALL_LIMIT - 30:
            for r in [0.18, 0.22, 0.25]:
                c2 = rod_cell(r)
                c2 = _enforce_d4(c2)
                s2 = _score_cell(c2, kpts_hi, n_pw=8)
                if s2 > best_score:
                    best_score = s2
                    best_cell = c2
    except Exception:
        pass

    return best_cell.tolist()
