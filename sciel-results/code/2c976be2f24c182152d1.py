import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell to maximize the TM band gap between bands 4 and 5.

    STRUCTURAL CHANGE: Represent the unit cell as the superposition of four
    C4v-symmetric Chebyshev-contour dielectric regions (multi-scale fractal-like
    distribution), and optimize the continuous contour parameters (radii and
    exponents) with a Nelder-Mead simplex over a fast PWE surrogate. This is a
    different design family (smooth multi-scale contours) and a different
    optimizer (derivative-free continuous search) than the previous pixel-flip
    local search.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)

    # Explicit deadline guard: 230 s wall-clock kill, keep 15 s safety margin.
    start_time = time.time()
    deadline = start_time + 215.0

    def time_left():
        return deadline - time.time()

    # Precompute coordinate grids (fractional, centered) once.
    coords = (np.arange(size) + 0.5) / size - 0.5
    X, Y = np.meshgrid(coords, coords, indexing='ij')
    # Four C4v-symmetric centers.
    CENTERS = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]

    # Fast internal PWE solver for TM modes (vectorized).
    def compute_tm_bands(cell, n_pw=15, n_kpts=9):
        """
        Compute TM band structure using plane-wave expansion.
        Returns omega (n_kpts, n_bands).
        """
        cell = np.asarray(cell, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell
        inv_eps = 1.0 / eps
        fft_coeffs = np.fft.fft2(inv_eps) / (size * size)

        M = n_pw // 2
        g_indices = []
        for m in range(-M, M + 1):
            for n in range(-M, M + 1):
                if m*m + n*n <= (M + 1)*(M + 1):
                    g_indices.append((m, n))
        n_total = len(g_indices)
        g_idx = np.array(g_indices)  # (n_total, 2)

        # S[i,j] = coeff(g_i - g_j)
        diff = g_idx[:, None, :] - g_idx[None, :, :]  # (n_total, n_total, 2)
        gm = (diff[:, :, 0] % size)
        gn = (diff[:, :, 1] % size)
        S = fft_coeffs[gm, gn]  # (n_total, n_total)

        # Irreducible BZ k-points: x in [0,0.5], y in [0,x].
        k_points = []
        n_seg = n_kpts
        for t in np.linspace(0, 0.5, n_seg):
            k_points.append((t, 0))
        for t in np.linspace(0, 0.5, n_seg):
            k_points.append((0.5, t))
        for t in np.linspace(0.5, 0, n_seg):
            k_points.append((t, t))
        k_points_unique = []
        for kp in k_points:
            if not any(np.allclose(kp, kp2, atol=1e-10) for kp2 in k_points_unique):
                k_points_unique.append(kp)
        kpts = np.array(k_points_unique)
        n_kpts_actual = len(kpts)

        G = 2.0 * np.pi * g_idx  # (n_total, 2)
        omega = np.zeros((n_kpts_actual, n_total))
        for ki in range(n_kpts_actual):
            kx, ky = kpts[ki]
            Gk = G + 2.0 * np.pi * np.array([kx, ky])
            k_mag_sq = Gk[:, 0]**2 + Gk[:, 1]**2
            Mmat = S * k_mag_sq[np.newaxis, :]
            try:
                eigvals = np.linalg.eigvalsh(Mmat)
                eigvals = np.sort(np.real(eigvals))
                eigvals = np.maximum(eigvals, 0)
                omega[ki, :] = np.sqrt(eigvals)
            except Exception:
                omega[ki, :] = np.inf
        return omega

    def score_cell(cell, n_pw=15, n_kpts=9):
        try:
            bands = compute_tm_bands(cell, n_pw=n_pw, n_kpts=n_kpts)
            omega_4 = bands[:, 3]
            omega_5 = bands[:, 4]
            max_omega_4 = np.max(omega_4)
            min_omega_5 = np.min(omega_5)
            if min_omega_5 > max_omega_4:
                gap = min_omega_5 - max_omega_4
                ratio = 2.0 * gap / (min_omega_5 + max_omega_4)
            else:
                overlap = max_omega_4 - min_omega_5
                midgap = 0.5 * (min_omega_5 + max_omega_4)
                ratio = -2.0 * overlap / midgap if midgap > 0 else -1.0
            return ratio
        except Exception:
            return -1.0

    def enforce_d4_symmetry(cell):
        cell = np.array(cell, dtype=int)
        half = size // 2
        tl = cell[:half, :half].copy()
        cell_new = np.zeros((size, size), dtype=int)
        cell_new[:half, :half] = tl
        cell_new[:half, half:] = tl[:, ::-1]
        cell_new[half:, :half] = tl[::-1, :]
        cell_new[half:, half:] = tl[::-1, ::-1]
        return cell_new

    def make_chebyshev_cell(radii, exponents):
        """
        Build a D4-symmetric binary cell as the superposition of four
        concentric (per-center) Chebyshev contours. A pixel is dielectric if it
        lies inside ANY of the four per-center contours (multi-scale fractal
        distribution).
        radii: (4,) center radii
        exponents: (4,) Chebyshev exponents (Lp-norm shape control)
        """
        cell = np.zeros((size, size), dtype=int)
        for c, (cx, cy) in enumerate(CENTERS):
            r = max(float(radii[c]), 1e-3)
            p = max(float(exponents[c]), 0.5)
            DX = (X - cx) / r
            DY = (Y - cy) / r
            val = np.abs(DX)**p + np.abs(DY)**p
            cell[val <= 1.0 + 1e-9] = 1
        return enforce_d4_symmetry(cell)

    # Nelder-Mead objective (minimize negative score).
    def objective(params):
        if time_left() < 8.0:
            return 1e6
        radii = np.clip(params[:4], 0.05, 0.28)
        exponents = np.clip(params[4:8], 1.0, 5.0)
        cell = make_chebyshev_cell(radii, exponents)
        s = score_cell(cell)
        return -s

    # ---- Phase 1: coarse multi-start grid over the 8-dim parameter space ----
    best_score = -np.inf
    best_cell = None
    best_params = None

    # Coarse grid: 2-3 levels per dimension to seed the simplex.
    r_levels = [0.09, 0.13, 0.17]
    p_levels = [1.5, 2.5, 3.5]

    for r0 in r_levels:
        if time_left() < 30:
            break
        for p0 in p_levels:
            if time_left() < 15:
                break
            # Base: all centers same radius/exponent, plus a few asymmetric variants.
            for variant in range(4):
                if time_left() < 10:
                    break
                radii = np.full(4, r0)
                exponents = np.full(4, p0)
                if variant == 1:
                    radii = np.array([r0, r0*0.7, r0, r0*0.7])
                elif variant == 2:
                    radii = np.array([r0*0.8, r0, r0, r0*0.8])
                    exponents = np.array([p0, p0*1.3, p0, p0*1.3])
                elif variant == 3:
                    radii = np.array([r0*1.1, r0*0.9, r0*0.9, r0*1.1])
                    exponents = np.array([p0*0.9, p0*1.2, p0*1.2, p0*0.9])
                params = np.concatenate([radii, exponents])
                cell = make_chebyshev_cell(radii, exponents)
                s = score_cell(cell)
                if s > best_score:
                    best_score = s
                    best_cell = cell
                    best_params = params.copy()

    if best_cell is None:
        radii = np.full(4, 0.13)
        exponents = np.full(4, 2.0)
        best_cell = make_chebyshev_cell(radii, exponents)
        best_params = np.concatenate([radii, exponents])
        best_score = score_cell(best_cell)

    # ---- Phase 2: Nelder-Mead continuous refinement over the 8 params ----
    if time_left() > 40:
        p0 = best_params.copy()
        x0 = p0
        # Simplex initialization: perturb each dimension.
        simplex = [x0.copy()]
        for d in range(8):
            x = x0.copy()
            step = 0.02 if d < 4 else 0.3
            x[d] = x0[d] + step
            simplex.append(x)

        # Manual Nelder-Mead (avoids scipy dependency, full time control).
        fvals = [objective(x) for x in simplex]
        alpha, gamma, rho, sigma = 1.0, 2.0, -0.5, 0.5
        max_iter = 300
        for it in range(max_iter):
            if time_left() < 15:
                break
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            if fvals[0] < -best_score:
                best_score = -fvals[0]
                best_params = simplex[0].copy()
            # Convergence check
            if abs(fvals[0] - fvals[-1]) < 1e-6:
                break
            centroid = np.mean(simplex[:-1], axis=0)
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = objective(xr)
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (xr - centroid)
                fe = objective(xe)
                if fe < fr:
                    simplex[-1] = xe; fvals[-1] = fe
                else:
                    simplex[-1] = xr; fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr; fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    fc = objective(xc)
                    if fc <= fr:
                        simplex[-1] = xc; fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = objective(simplex[i])
                else:
                    xc = centroid - rho * (simplex[-1] - centroid)
                    fc = objective(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc; fvals[-1] = fc
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = objective(simplex[i])
            if time_left() < 15:
                break

        # Adopt the best found by Nelder-Mead if it beats the grid best.
        if best_params is not None:
            radii = np.clip(best_params[:4], 0.05, 0.28)
            exponents = np.clip(best_params[4:8], 1.0, 5.0)
            cell = make_chebyshev_cell(radii, exponents)
            s = score_cell(cell)
            if s > best_score:
                best_score = s
                best_cell = cell

    # ---- Phase 3: short pixel-level local refinement (keep D4) ----
    if best_cell is not None and time_left() > 15:
        best_arr = np.array(best_cell, dtype=int)
        half = size // 2
        tl_best = best_arr[:half, :half].copy()
        pixels = [(i, j) for i in range(half) for j in range(half)]
        np.random.seed(7)
        np.random.shuffle(pixels)
        improved = True
        passes = 0
        while improved and passes < 3 and time_left() > 15:
            improved = False
            passes += 1
            for (i, j) in pixels:
                if time_left() < 8:
                    break
                tl_new = tl_best.copy()
                tl_new[i, j] = 1 - tl_new[i, j]
                cell_new = np.zeros((size, size), dtype=int)
                cell_new[:half, :half] = tl_new
                cell_new[:half, half:] = tl_new[:, ::-1]
                cell_new[half:, :half] = tl_new[::-1, :]
                cell_new[half:, half:] = tl_new[::-1, ::-1]
                s = score_cell(cell_new)
                if s > best_score:
                    best_score = s
                    best_cell = cell_new
                    best_arr = cell_new
                    tl_best = tl_new
                    improved = True

    if best_cell is None:
        best_cell = make_chebyshev_cell(np.full(4, 0.13), np.full(4, 2.0))

    best_arr = np.array(best_cell, dtype=int)
    best_arr = enforce_d4_symmetry(best_arr)

    return best_arr.tolist()

# EVOLVE-BLOCK-END
