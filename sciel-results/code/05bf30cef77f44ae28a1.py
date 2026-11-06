import time
import numpy as np
from numpy.linalg import eigh


def _cheb(n, x):
    """Chebyshev polynomial T_n(x), x in [-1,1]."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x
    t0 = np.ones_like(x)
    t1 = x
    for k in range(1, n):
        t2 = 2.0 * x * t1 - t0
        t0, t1 = t1, t2
    return t1


def _build_cell(params, size=32):
    """Build a C4v-symmetric cell from 4 Chebyshev contour superpositions.
    params: list of 4 tuples (r, order, thickness, phase)
    """
    x = (np.arange(size) + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X * X + Y * Y)
    # theta in [0, 2*pi)
    TH = np.arctan2(Y, X)
    TH = np.where(TH < 0, TH + 2 * np.pi, TH)

    # 4-fold symmetric angular function: cos(4*theta) -> C4v symmetric
    # Map theta to a variable in [-1,1] using the 4-fold symmetry
    # u = cos(4*theta) is already 4-fold symmetric and in [-1,1]
    U = np.cos(4.0 * TH)

    # For each contour, compute field = thickness * T_order(U) * (1 - smoothstep(R/r))
    # Superpose and threshold
    field = np.zeros((size, size))
    for (r, order, thick, phase) in params:
        if r <= 0:
            continue
        # Chebyshev on U (4-fold symmetric angular variable)
        T = _cheb(int(order), U)
        # Radial falloff: 1 inside r, 0 outside
        rad = np.clip(1.0 - R / max(r, 1e-6), 0.0, 1.0)
        # Smooth it a bit
        rad = rad * rad * (3 - 2 * rad)
        field += thick * T * rad

    # Threshold at a level that gives reasonable fill fraction
    # Normalize field to [-1,1] roughly
    fmax = np.max(np.abs(field))
    if fmax > 1e-6:
        field = field / fmax
    # Threshold: pixels where field > -0.3 become dielectric
    cell = (field > -0.3).astype(int)
    return cell


def _enforce_c4v(cell):
    """Enforce exact C4v symmetry by mirroring top-left quadrant."""
    N = len(cell)
    c = np.array(cell, dtype=int)
    # Use top-left quadrant and mirror
    q = c[:N//2, :N//2].copy()
    c2 = np.zeros((N, N), dtype=int)
    c2[:N//2, :N//2] = q
    c2[:N//2, N//2:] = q[:, ::-1]
    c2[N//2:, :N//2] = q[::-1, :]
    c2[N//2:, N//2:] = q[::-1, ::-1]
    return c2.tolist()


def _compute_gap_fast(cell, n_k=8, pw_radius=10):
    N = len(cell)
    inv_eps = 1.0 / (1.0 + 11.0 * np.array(cell, dtype=float))
    fft_inv = np.fft.fft2(inv_eps) / (N * N)

    G_list = []
    for Gx in range(-pw_radius, pw_radius + 1):
        for Gy in range(-pw_radius, pw_radius + 1):
            if Gx*Gx + Gy*Gy <= pw_radius * pw_radius:
                G_list.append((Gx, Gy))
    n_pw = len(G_list)
    G_arr = np.array(G_list, dtype=float)

    Gx_diff = G_arr[:, 0][:, None] - G_arr[:, 0][None, :]
    Gy_diff = G_arr[:, 1][:, None] - G_arr[:, 1][None, :]
    dqx = Gx_diff.astype(int)
    dqy = Gy_diff.astype(int)
    inv_eps_mat = fft_inv[dqx % N, dqy % N]

    k_points = [
        (0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.5, 0.25),
        (0.5, 0.5), (0.35, 0.35), (0.4, 0.2), (0.2, 0.1)
    ][:n_k]

    max_w4 = 0.0
    min_w5 = float('inf')

    for k in k_points:
        kvec = np.array(k, dtype=float)
        kG = kvec + G_arr
        dots = kG @ kG.T
        M = dots * inv_eps_mat
        M = (M + M.conj().T) / 2
        try:
            eigvals = eigh(M, eigvals_only=True)
            eigvals = np.sqrt(np.maximum(eigvals, 0))
            if len(eigvals) >= 5:
                max_w4 = max(max_w4, eigvals[3])
                min_w5 = min(min_w5, eigvals[4])
        except:
            pass

    if min_w5 == float('inf') or max_w4 <= 0:
        return -1.0
    return 2 * (min_w5 - max_w4) / (min_w5 + max_w4)


def _evaluate_params(params, size=32, n_k=6, pw_radius=8):
    cell = _enforce_c4v(_build_cell(params, size))
    return _compute_gap_fast(cell, n_k=n_k, pw_radius=pw_radius)


def _gen_seed_params():
    """Generate diverse initial parameter sets."""
    rng = np.random.default_rng(42)
    seeds = []
    # Various combinations of radius, order, thickness
    for r in [0.2, 0.3, 0.4, 0.45, 0.5]:
        for order in [0, 1, 2, 3, 4]:
            for thick in [0.5, 1.0, 1.5]:
                params = []
                for i in range(4):
                    rr = r * (0.5 + 0.25 * i)
                    oo = order + (i % 2)
                    tt = thick * (1.0 - 0.15 * i)
                    ph = 0.0
                    params.append((rr, oo, tt, ph))
                seeds.append(params)
    # Random seeds
    for _ in range(30):
        params = []
        for i in range(4):
            rr = rng.uniform(0.15, 0.55)
            oo = int(rng.integers(0, 5))
            tt = rng.uniform(0.3, 2.0)
            ph = 0.0
            params.append((rr, oo, tt, ph))
        seeds.append(params)
    return seeds


def _mutate_params(params, rate=0.3):
    """Perturb parameters."""
    rng = np.random.default_rng()
    new_params = []
    for (r, o, t, p) in params:
        nr = r + rng.normal(0, 0.05 * rate * 5)
        nr = max(0.05, min(0.6, nr))
        no = o
        if rng.random() < 0.3 * rate:
            no = int(np.clip(o + int(rng.integers(-1, 2)), 0, 6))
        nt = t + rng.normal(0, 0.2 * rate * 5)
        nt = max(0.1, min(2.5, nt))
        new_params.append((nr, no, nt, p))
    return new_params


def design(problem):
    start_time = time.time()
    deadline = start_time + 200.0

    size = problem["pixels"]
    assert size == 32

    # Phase 1: Evaluate seed parameter sets
    seeds = _gen_seed_params()
    best_score = -1.0
    best_params = None
    best_cell = None

    for params in seeds:
        if time.time() > deadline - 60:
            break
        try:
            score = _evaluate_params(params, size=size, n_k=5, pw_radius=7)
            if score > best_score:
                best_score = score
                best_params = params
                best_cell = _enforce_c4v(_build_cell(params, size))
        except:
            pass

    # Phase 2: Local search in parameter space
    if best_params is not None and time.time() < deadline - 40:
        current_params = best_params
        current_score = best_score
        no_improve = 0
        max_no_improve = 40

        while time.time() < deadline - 25 and no_improve < max_no_improve:
            mutated = _mutate_params(current_params, rate=0.2)
            try:
                s = _evaluate_params(mutated, size=size, n_k=5, pw_radius=7)
                if s > current_score:
                    current_score = s
                    current_params = mutated
                    no_improve = 0
                    if s > best_score:
                        best_score = s
                        best_params = mutated
                        best_cell = _enforce_c4v(_build_cell(mutated, size))
                else:
                    no_improve += 1
            except:
                no_improve += 1

    # Phase 3: Fine-tune best parameters with smaller steps
    if best_params is not None and time.time() < deadline - 20:
        current_params = best_params
        current_score = best_score
        no_improve = 0
        max_no_improve = 30

        while time.time() < deadline - 15 and no_improve < max_no_improve:
            mutated = _mutate_params(current_params, rate=0.05)
            try:
                s = _evaluate_params(mutated, size=size, n_k=6, pw_radius=8)
                if s > current_score:
                    current_score = s
                    current_params = mutated
                    no_improve = 0
                    if s > best_score:
                        best_score = s
                        best_params = mutated
                        best_cell = _enforce_c4v(_build_cell(mutated, size))
                else:
                    no_improve += 1
            except:
                no_improve += 1

    # Phase 4: If time allows, try a few more random restarts
    if time.time() < deadline - 20:
        rng = np.random.default_rng(123)
        for _ in range(20):
            if time.time() > deadline - 15:
                break
            params = []
            for i in range(4):
                rr = rng.uniform(0.15, 0.55)
                oo = int(rng.integers(0, 5))
                tt = rng.uniform(0.3, 2.0)
                ph = 0.0
                params.append((rr, oo, tt, ph))
            try:
                s = _evaluate_params(params, size=size, n_k=5, pw_radius=7)
                if s > best_score:
                    best_score = s
                    best_params = params
                    best_cell = _enforce_c4v(_build_cell(params, size))
                    # Quick local search
                    for _inner in range(15):
                        if time.time() > deadline - 15:
                            break
                        mutated = _mutate_params(params, rate=0.15)
                        s2 = _evaluate_params(mutated, size=size, n_k=5, pw_radius=7)
                        if s2 > best_score:
                            best_score = s2
                            best_params = mutated
                            best_cell = _enforce_c4v(_build_cell(mutated, size))
                            params = mutated
            except:
                pass

    # Fallback
    if best_cell is None:
        best_cell = _enforce_c4v(_build_cell([(0.3, 1, 1.0, 0), (0.4, 2, 0.8, 0),
                                              (0.45, 3, 0.6, 0), (0.5, 4, 0.5, 0)], size))

    # Ensure exact symmetry one final time
    best_cell = _enforce_c4v(best_cell)
    return best_cell
