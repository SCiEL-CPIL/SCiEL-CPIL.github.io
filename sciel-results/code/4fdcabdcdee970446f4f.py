"""Photonic crystal unit cell design for large TM 4-5 band gap.

Fast cyclic plane-wave solver + dense geometric seeding + orbit local search.
"""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def design(problem):
    """Design a symmetric 32x32 unit cell maximizing the TM 4-5 gap-midgap ratio."""
    t_start = time.time()
    DEADLINE = 222.0  # safety margin before 230s kill

    size = problem["pixels"]
    eps_high = problem["eps_high"]
    eps_low = problem["eps_low"]

    # ---------- Fast TM plane-wave solver (17x17 = 289 plane waves) ----------
    n_pw = 17
    half = n_pw // 2
    G_vals = np.arange(-half, half + 1)
    Gx_grid, Gy_grid = np.meshgrid(G_vals, G_vals, indexing='ij')
    Gx = Gx_grid.ravel()
    Gy = Gy_grid.ravel()
    n_pw_total = n_pw * n_pw  # 289

    # k-points: 30 points on the boundary/interior of the irreducible BZ
    k_points = []
    for i in range(10):
        t = i / 9.0
        k_points.append((t * np.pi, 0.0))
    for i in range(10):
        t = i / 9.0
        k_points.append((np.pi, t * np.pi))
    for i in range(10):
        t = i / 9.0
        k_points.append((np.pi * (1 - t), np.pi * (1 - t)))
    k_points = np.array(k_points)
    n_k = len(k_points)

    # (k+G)^2 for every (k-point, plane-wave) pair
    k2_all = np.zeros((n_k, n_pw_total))
    for i, (kx, ky) in enumerate(k_points):
        k2_all[i] = (kx + Gx) ** 2 + (ky + Gy) ** 2

    # Cyclic coupling: H[i,j] = k2[i] * coeffs[(Gx[i]-Gx[j], Gy[i]-Gy[j])]
    # -> H is a sum over d of (k2 * coeffs[d]) * Circ(d).  Build via FFT.
    diff_coeffs = np.zeros((n_pw_total, n_pw_total), dtype=complex)
    for i in range(n_pw_total):
        for j in range(n_pw_total):
            d_idx = (int(Gx[i]) - int(Gx[j]) + half) * n_pw + (int(Gy[i]) - int(Gy[j]) + half)
            diff_coeffs[i, j] = coeffs = None  # placeholder (overwritten below)
    # (properly fill)
    diff_coeffs = np.zeros((n_pw_total, n_pw_total), dtype=complex)
    for i in range(n_pw_total):
        for j in range(n_pw_total):
            d_idx = (int(Gx[i]) - int(Gx[j]) + half) * n_pw + (int(Gy[i]) - int(Gy[j]) + half)
            diff_coeffs[i, j] = 1.0
    # diff_coeffs[i,j] will be multiplied by coeffs[d_idx(i,j)] at solve time via
    # a precomputed mapping: for each i, the vector of coeffs indices over j.
    diff_idx = np.zeros((n_pw_total, n_pw_total), dtype=int)
    for i in range(n_pw_total):
        for j in range(n_pw_total):
            diff_idx[i, j] = (int(Gx[i]) - int(Gx[j]) + half) * n_pw + (int(Gy[i]) - int(Gy[j]) + half)

    def get_fourier_coeffs(cell):
        """Fourier coefficients of 1/eps from the pixel map (via FFT)."""
        eps_map = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_map
        fft_2d = np.fft.fft2(inv_eps)
        coeffs = np.empty(n_pw_total, dtype=complex)
        for idx in range(n_pw_total):
            coeffs[idx] = fft_2d[int(Gx[idx]) % size, int(Gy[idx]) % size] / (size * size)
        return coeffs

    def build_H(coeffs, k_idx):
        """Assemble the 289x289 Hermitian matrix for one k-point (cyclic structure)."""
        k2 = k2_all[k_idx]
        H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
        for i in range(n_pw_total):
            H[i, :] = k2[i] * coeffs[diff_idx[i, :]]
        H = (H + H.conj().T) * 0.5
        return H

    def _bands(H):
        ev = np.linalg.eigvalsh(H)
        return ev[3], ev[4]

    def compute_band_ratio_full(coeffs):
        """Exact gap-midgap ratio over all 30 k-points."""
        max_b4 = -np.inf
        min_b5 = np.inf
        for k_idx in range(n_k):
            b4, b5 = _bands(build_H(coeffs, k_idx))
            if b4 > max_b4:
                max_b4 = b4
            if b5 < min_b5:
                min_b5 = b5
        gap = min_b5 - max_b4
        midgap = (min_b5 + max_b4) * 0.5
        return gap / midgap if midgap > 1e-10 else 0.0

    def compute_band_ratio_fast(coeffs):
        """Fast ratio over a representative subset of k-points."""
        k_subset = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 29)
        max_b4 = -np.inf
        min_b5 = np.inf
        for k_idx in k_subset:
            b4, b5 = _bands(build_H(coeffs, k_idx))
            if b4 > max_b4:
                max_b4 = b4
            if b5 < min_b5:
                min_b5 = b5
        gap = min_b5 - max_b4
        midgap = (min_b5 + max_b4) * 0.5
        return gap / midgap if midgap > 1e-10 else 0.0

    def evaluate_cell(cell):
        return compute_band_ratio_full(get_fourier_coeffs(cell))

    def evaluate_cell_fast(cell):
        return compute_band_ratio_fast(get_fourier_coeffs(cell))

    # ---------- Symmetry enforcement (D4 about cell centre) ----------
    def enforce_symmetry(cell):
        cell = np.asarray(cell, dtype=int)
        result = (cell
                  + cell[::-1, ::-1]
                  + cell[:, ::-1]
                  + cell[::-1, :]
                  + cell.T
                  + cell[::-1, ::-1].T
                  + cell[:, ::-1].T
                  + cell[::-1, :].T)
        return (result >= 4).astype(int)

    # ---------- Vectorized geometric generators ----------
    ii, jj = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    x = (ii + 0.5) / size - 0.5
    y = (jj + 0.5) / size - 0.5
    r2 = x * x + y * y

    def gen_rod(radius):
        return enforce_symmetry((r2 < radius * radius).astype(int))

    def gen_diamond(radius):
        return enforce_symmetry(((np.abs(x) + np.abs(y)) < radius).astype(int))

    def gen_ring(ri, ro):
        return enforce_symmetry(((r2 > ri * ri) & (r2 < ro * ro)).astype(int))

    def gen_cross(aw, al):
        m = ((np.abs(y) < aw / 2) & (np.abs(x) < al / 2)) | \
            ((np.abs(x) < aw / 2) & (np.abs(y) < al / 2))
        return enforce_symmetry(m.astype(int))

    def gen_four_rods(radius, off):
        m = np.zeros((size, size), dtype=bool)
        for sx in (1, -1):
            for sy in (1, -1):
                dx = x - sx * off
                dy = y - sy * off
                m |= (dx * dx + dy * dy) < radius * radius
        return enforce_symmetry(m.astype(int))

    def gen_four_rings(ri, ro, off):
        m = np.zeros((size, size), dtype=bool)
        for sx in (1, -1):
            for sy in (1, -1):
                dx = x - sx * off
                dy = y - sy * off
                rr = dx * dx + dy * dy
                m |= (rr > ri * ri) & (rr < ro * ro)
        return enforce_symmetry(m.astype(int))

    def gen_moire(f1, f2, p1, p2, thr):
        v1 = np.sin(2 * np.pi * f1 * (x + y) + p1)
        v2 = np.sin(2 * np.pi * f2 * (x - y) + p2)
        m = ((v1 > 0).astype(int) * (v2 > 0).astype(int)) >= thr
        return enforce_symmetry(m.astype(int))

    def gen_moire_sin(f1, f2, p1, p2, thr):
        v1 = np.sin(2 * np.pi * f1 * (x + y) + p1)
        v2 = np.sin(2 * np.pi * f2 * (x - y) + p2)
        m = (v1 * v2) >= thr
        return enforce_symmetry(m.astype(int))

    def gen_checkerboard(bs, ph):
        bi = (ii // bs + ph) % 2
        bj = (jj // bs) % 2
        return enforce_symmetry((bi != bj).astype(int))

    def gen_chevron(fr, th, ph):
        v = np.sin(2 * np.pi * fr * (np.abs(x) + y) + ph)
        m = (v > 0) & (np.abs(v) < th)
        return enforce_symmetry(m.astype(int))

    def gen_concentric(n_r, ph):
        r = np.sqrt(r2)
        ring_idx = (r / (0.5 / n_r) + ph) % n_r
        return enforce_symmetry((ring_idx < n_r // 2).astype(int))

    # ---------- Symmetry orbits (independent D4 orbits) ----------
    orbits = []
    seen = set()
    for i in range(size // 2):
        for j in range(size // 2):
            orbit = [(i, j), (i, size - 1 - j), (size - 1 - i, j),
                     (size - 1 - i, size - 1 - j),
                     (j, i), (j, size - 1 - i), (size - 1 - i, j),
                     (size - 1 - i, size - 1 - i)]
            orbit = list(dict.fromkeys(orbit))
            key = tuple(sorted(orbit))
            if key not in seen:
                seen.add(key)
                orbits.append(orbit)
    n_orbits = len(orbits)

    def apply_orbit(cell, orbit, val):
        for (oi, oj) in orbit:
            cell[oi, oj] = val

    # ---------- Hill climbing over orbits (fast eval) ----------
    def hill_climb(start_cell, fast_score, budget):
        current = start_cell.copy()
        cur = fast_score
        for _ in range(2000):
            if time.time() > budget:
                break
            improved = False
            for _ in range(2):  # two passes per iteration
                random.shuffle(orbits)
                for orbit in orbits:
                    if time.time() > budget:
                        break
                    test = current.copy()
                    apply_orbit(test, orbit, 1 - test[orbit[0][0], orbit[0][1]])
                    s = evaluate_cell_fast(test)
                    if s > cur + 1e-12:
                        current = test
                        cur = s
                        improved = True
                        break
            if not improved:
                break
        return current, cur

    # ---------- Main search ----------
    best_score = -np.inf
    best_cell = None

    def update_best(cell, score):
        nonlocal best_score, best_cell
        if score > best_score:
            best_score = score
            best_cell = cell.copy()

    print("Phase 1: dense geometric seeding...")
    candidates = []
    for r in np.arange(0.12, 0.34, 0.02):
        candidates.append(gen_rod(r))
    for r in np.arange(0.18, 0.40, 0.02):
        candidates.append(gen_diamond(r))
    for ri in np.arange(0.04, 0.14, 0.02):
        for ro in np.arange(0.14, 0.28, 0.02):
            if ro > ri + 0.02:
                candidates.append(gen_ring(ri, ro))
    for aw in (0.04, 0.06, 0.08, 0.10):
        for al in (0.28, 0.34, 0.40):
            candidates.append(gen_cross(aw, al))
    for r in np.arange(0.05, 0.13, 0.02):
        for off in np.arange(0.18, 0.32, 0.02):
            candidates.append(gen_four_rods(r, off))
    for ri in (0.02, 0.03, 0.04, 0.05):
        for ro in (0.08, 0.10, 0.12, 0.14):
            for off in np.arange(0.18, 0.32, 0.02):
                candidates.append(gen_four_rings(ri, ro, off))
    for f1 in (2, 3, 4, 5):
        for f2 in (2, 3, 4, 5):
            if f1 == f2:
                continue
            for p1 in (0.0, np.pi / 2):
                for p2 in (0.0, np.pi / 2):
                    for thr in (0.5, 0.75):
                        candidates.append(gen_moire(f1, f2, p1, p2, thr))
                    for thr in (0.2, 0.4, 0.6):
                        candidates.append(gen_moire_sin(f1, f2, p1, p2, thr))
    for bs in (3, 4, 5, 6, 8):
        for ph in (0, 1):
            candidates.append(gen_checkerboard(bs, ph))
    for fr in (2, 3, 4):
        for th in (0.3, 0.5, 0.7):
            for ph in (0.0, np.pi / 2):
                candidates.append(gen_chevron(fr, th, ph))
    for n_r in (2, 3, 4):
        for ph in (0.0, 0.5):
            candidates.append(gen_concentric(n_r, ph))

    print(f"Evaluating {len(candidates)} candidates (fast)...")
    fast_scores = []
    for cell in candidates:
        if time.time() > DEADLINE - 30:
            break
        fast_scores.append((evaluate_cell_fast(cell), cell))
    fast_scores.sort(key=lambda z: -z[0])

    # Exact-evaluate the top seeds
    top_n = min(40, len(fast_scores))
    for score, cell in fast_scores[:top_n]:
        if time.time() > DEADLINE - 30:
            break
        update_best(cell, evaluate_cell(cell))
    print(f"Best after Phase 1: {best_score:.4f}")

    # Phase 2: multi-start hill climbing from the best fast seeds
    if best_cell is not None and time.time() < DEADLINE - 25:
        print("Phase 2: multi-start orbit hill climbing...")
        starts = [c for s, c in fast_scores[:6]]
        if best_cell not in starts:
            starts.append(best_cell)
        for sc in starts:
            if time.time() > DEADLINE - 25:
                break
            polished, pfast = hill_climb(sc, evaluate_cell_fast(sc), DEADLINE - 25)
            fs = evaluate_cell(polished)
            if fs > best_score:
                update_best(polished, fs)
        print(f"Best after Phase 2: {best_score:.4f}")

    # Phase 3: final exhaustive single-orbit polish (exact 30-point eval)
    if best_cell is not None and time.time() < DEADLINE - 3:
        print("Phase 3: exact single-orbit polish...")
        current = best_cell.copy()
        cur = best_score
        for _ in range(3):  # a few sweeps for local convergence
            if time.time() > DEADLINE - 3:
                break
            for orbit in orbits:
                if time.time() > DEADLINE - 3:
                    break
                test = current.copy()
                apply_orbit(test, orbit, 1 - test[orbit[0][0], orbit[0][1]])
                s = evaluate_cell(test)
                if s > cur + 1e-12:
                    current = test
                    cur = s
                    update_best(current, cur)
        print(f"Final best: {best_score:.4f}")

    if best_cell is None:
        best_cell = gen_four_rods(0.10, 0.25)

    best_cell = enforce_symmetry(best_cell)
    return [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]
# EVOLVE-BLOCK-END
