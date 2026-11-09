"""Photonic crystal design: vectorized PWE + multi-resolution SA for max TM 4-5 gap."""
import numpy as np
import time

def design(problem):
    start_time = time.time()
    size = problem["pixels"]      # 32
    eps_high = problem["eps_high"]  # 12
    N_pw = 8
    num_pw = (2 * N_pw + 1) ** 2  # 289

    G_range = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_range, G_range, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel())).astype(int)  # (289, 2)

    # Precompute difference-index matrix: C[a, b] = index of G_a - G_b
    # G_a - G_b has components in [-16, 16]; only in [-8, 8] is valid
    G_diff_matrix = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]  # (289, 289, 2)
    diff_gx = G_diff_matrix[:, :, 0]
    diff_gy = G_diff_matrix[:, :, 1]
    valid = (np.abs(diff_gx) <= N_pw) & (np.abs(diff_gy) <= N_pw)
    # Map (gx, gy) in [-8, 8] to flat index: (gx + 8) * 17 + (gy + 8)
    diff_idx = (diff_gx + N_pw) * (2 * N_pw + 1) + (diff_gy + N_pw)
    diff_idx = np.where(valid, diff_idx, -1).astype(np.int32)

    # k-points: irreducible BZ wedge 0 <= ky <= kx <= 0.5
    # 30 points covering boundary and interior
    k_points = []
    # High-symmetry
    k_points.append((0.0, 0.0))    # Gamma
    k_points.append((0.5, 0.0))    # X
    k_points.append((0.5, 0.5))    # M
    # Along Gamma-X (kx from 0 to 0.5, ky=0)
    for t in [0.1, 0.2, 0.3, 0.4]:
        k_points.append((t, 0.0))
    # Along X-M (kx=0.5, ky from 0 to 0.5)
    for t in [0.1, 0.2, 0.3, 0.4]:
        k_points.append((0.5, t))
    # Along Gamma-M (kx=ky, from 0 to 0.5)
    for t in [0.1, 0.2, 0.3, 0.4]:
        k_points.append((t, t))
    # Interior of the wedge
    interior = [
        (0.15, 0.05), (0.25, 0.1), (0.35, 0.15), (0.45, 0.2),
        (0.2, 0.15), (0.3, 0.2), (0.4, 0.25), (0.45, 0.35),
        (0.15, 0.1), (0.25, 0.2), (0.35, 0.3), (0.45, 0.4),
    ]
    k_points.extend(interior)
    k_points = np.array(k_points)  # (30, 2)

    def compute_fourier_inv_eps(cell):
        """Compute Fourier coefficients of 1/eps via 2D FFT, extract (2N+1)^2 subset."""
        inv_eps = np.where(cell == 0, 1.0, 1.0 / eps_high)
        # 2D FFT: F[u, v] = sum_{i,j} f[i,j] exp(-2pi i (u*i + v*j)/size)
        # We want F at frequencies Gx, Gy in [-8, 8]
        # FFT gives frequencies 0, 1, ..., size-1; negative freqs are at size-|g|
        F = np.fft.fft2(inv_eps) / (size * size)
        # Extract: for g in [-8, 8], index is g % size
        idx = np.arange(-N_pw, N_pw + 1) % size
        subset = F[np.ix_(idx, idx)]  # (17, 17)
        return subset.ravel()  # (289,) in row-major order matching G_vectors

    def compute_bands_fast(inv_eps_G, kx, ky):
        """Vectorized band computation. Returns sorted omega^2."""
        k_plus_G = G_vectors.astype(float) + np.array([kx, ky])
        kG2 = np.sum(k_plus_G ** 2, axis=1)  # (289,)

        # C[a, b] = inv_eps_G[diff_idx[a, b]] if valid, else 0
        # Use np.take with where
        C = np.zeros((num_pw, num_pw), dtype=complex)
        mask = diff_idx >= 0
        C[mask] = inv_eps_G[diff_idx[mask]]

        # H = C * kG2[None, :]  (each column b scaled by kG2[b])
        H = C * kG2[np.newaxis, :]
        # H should be Hermitian; symmetrize to remove numerical noise
        H = 0.5 * (H + H.conj().T)

        eigvals = np.linalg.eigvalsh(H)
        return np.real(eigvals)

    def evaluate_cell(cell):
        """Compute gap-midgap ratio for a full cell."""
        inv_eps_G = compute_fourier_inv_eps(cell)
        min_w5_sq = np.inf
        max_w4_sq = 0.0
        for kx, ky in k_points:
            bands = compute_bands_fast(inv_eps_G, kx, ky)
            w4_sq = bands[3]
            w5_sq = bands[4]
            if w5_sq < min_w5_sq:
                min_w5_sq = w5_sq
            if w4_sq > max_w4_sq:
                max_w4_sq = w4_sq
        if max_w4_sq <= 0:
            return 0.0
        min_w5 = np.sqrt(min_w5_sq)
        max_w4 = np.sqrt(max_w4_sq)
        midgap = 0.5 * (min_w5 + max_w4)
        if midgap == 0:
            return 0.0
        return 2.0 * (min_w5 - max_w4) / midgap

    # --- Multi-resolution search ---
    def get_orbits(n):
        """Symmetry orbits for n x n cell under square group about center."""
        visited = set()
        orbits = []
        for i in range(n):
            for j in range(n):
                if (i, j) not in visited:
                    orbit = set()
                    orbit.add((i, j))
                    orbit.add((n - 1 - i, n - 1 - j))
                    orbit.add((i, n - 1 - j))
                    orbit.add((n - 1 - i, j))
                    orbit.add((j, i))
                    orbit.add((n - 1 - j, n - 1 - i))
                    visited.update(orbit)
                    orbits.append(list(orbit))
        return orbits

    def orbit_to_cell_orbit(orbit_vals, orbits):
        cell = np.zeros((len(orbits[0]) and orbits[0][0][0] * 0 + 1, 1), dtype=np.uint8)  # dummy
        n = None
        for o in orbits:
            for (i, j) in o:
                n = max(n or 0, i + 1)
                n = max(n, j + 1)
        cell = np.zeros((n, n), dtype=np.uint8)
        for o_idx, orbit in enumerate(orbits):
            val = orbit_vals[o_idx]
            for (i, j) in orbit:
                cell[i, j] = val
        return cell

    def upscale_cell(cell_small, factor):
        """Upscale a small cell to larger by nearest-neighbor (block replication)."""
        n_small = cell_small.shape[0]
        n_large = n_small * factor
        cell_large = np.zeros((n_large, n_large), dtype=np.uint8)
        for i in range(n_small):
            for j in range(n_small):
                cell_large[i * factor:(i + 1) * factor, j * factor:(j + 1) * factor] = cell_small[i, j]
        return cell_large

    def sa_search(cell, n_orbits, orbits, time_limit, T0=1.0, T1=0.01, alpha=0.95):
        """Simulated annealing on orbit values."""
        n = cell.shape[0]
        current = np.zeros(n_orbits, dtype=int)
        for o_idx, orbit in enumerate(orbits):
            i, j = orbit[0]
            current[o_idx] = cell[i, j]

        current_score = evaluate_cell(orbit_to_cell_orbit(current, orbits))
        best_score = current_score
        best_vals = current.copy()

        t = time.time()
        elapsed = 0.0
        while time.time() - t < time_limit:
            # Pick 1-3 random orbits to flip
            n_flips = np.random.randint(1, 4)
            idxs = np.random.choice(n_orbits, n_flips, replace=False)
            new_vals = current.copy()
            new_vals[idxs] = 1 - new_vals[idxs]

            new_score = evaluate_cell(orbit_to_cell_orbit(new_vals, orbits))
            delta = new_score - current_score

            if delta > 0 or np.random.random() < np.exp(delta / max(T0 * (T1 / T0) ** (elapsed / time_limit), 1e-10)):
                current = new_vals
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_vals = current.copy()

            elapsed = time.time() - t

        return orbit_to_cell_orbit(best_vals, orbits)

    # Phase 1: Coarse search on 8x8
    coarse_size = 8
    coarse_orbits = get_orbits(coarse_size)
    n_coarse = len(coarse_orbits)

    # Random initial coarse cell
    coarse_cell = np.random.randint(0, 2, (coarse_size, coarse_size), dtype=np.uint8)
    # Enforce symmetry
    coarse_cell = (coarse_cell + coarse_cell[::-1, ::-1] + coarse_cell[:, ::-1] +
                   coarse_cell[::-1, :] + coarse_cell.T + coarse_cell[::-1, ::-1].T) // 2
    coarse_cell = (coarse_cell > 0).astype(np.uint8)

    coarse_best = sa_search(coarse_cell, n_coarse, coarse_orbits, time_limit=15.0)

    # Phase 2: Upscale to 16x16 and refine
    mid_size = 16
    mid_cell = upscale_cell(coarse_best, 2)
    mid_orbits = get_orbits(mid_size)
    n_mid = len(mid_orbits)
    mid_best = sa_search(mid_cell, n_mid, mid_orbits, time_limit=25.0)

    # Phase 3: Upscale to 32x32 and refine
    full_cell = upscale_cell(mid_best, 2)
    full_orbits = get_orbits(size)
    n_full = len(full_orbits)
    full_best = sa_search(full_cell, n_full, full_orbits, time_limit=60.0)

    # Phase 4: Final polish with more iterations
    remaining = 120.0 - (time.time() - start_time)
    if remaining > 10:
        full_best = sa_search(full_best, n_full, full_orbits, time_limit=min(remaining - 5, 40.0),
                              T0=0.5, T1=0.01, alpha=0.95)

    # Safety: ensure symmetry
    full_best = (full_best + full_best[::-1, ::-1] + full_best[:, ::-1] +
                 full_best[::-1, :] + full_best.T + full_best[::-1, ::-1].T) // 2
    full_best = (full_best > 0).astype(np.uint8)

    return full_best.tolist()
