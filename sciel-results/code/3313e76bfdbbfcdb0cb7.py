"""Rewritten program: square-pillar photonic crystal with D4 symmetry,
accurate plane-wave solver, and efficient local optimization."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np


def design(problem):
    size = 32
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)

    t_start = time.time()
    DEADLINE = 215  # safety margin before 230s kill

    # ---------- D4 symmetry ----------
    def enforce_d4(cell):
        """Enforce D4 symmetry by majority vote over the 8 orbit elements."""
        s = size
        # Build orbit for each pixel
        result = [[0] * s for _ in range(s)]
        # Precompute orbit membership
        for i in range(s):
            for j in range(s):
                orbit = set()
                ii = i
                jj = j
                for _ in range(4):
                    orbit.add((ii, jj))
                    orbit.add((jj, ii))  # transpose
                    ii, jj = s - 1 - ii, s - 1 - jj  # reverse both
                # Actually need all 8 elements of D4
                orbit = set()
                a, b = i, j
                for _ in range(4):
                    orbit.add((a, b))
                    orbit.add((b, a))
                    a, b = s - 1 - a, s - 1 - b
                vals = [cell[r][c] for (r, c) in orbit]
                result[i][j] = 1 if sum(vals) >= len(vals) / 2.0 else 0
        return result

    # ---------- Fourier coefficients via FFT ----------
    def compute_coeffs(cell):
        """Compute 1/eps Fourier coefficients using 2D FFT."""
        eps = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                eps[i, j] = eps_high if cell[i][j] == 1 else eps_low
        inv_eps = 1.0 / eps
        fft2 = np.fft.fft2(inv_eps) / (size * size)
        return fft2

    # ---------- Band solver ----------
    # G-vectors: use a set that covers the needed range
    # For TM mode: H[k][k'] = |k+G|^2 * eps^{-1}(G-G')
    # We need enough G-vectors for convergence with 289 PWs (17x17)
    # Use G-vectors with |m|,|n| <= 8 (17x17 = 289)

    def build_g_vectors(g_max):
        """Build list of G-vector indices (m,n) for the basis."""
        gs = []
        for m in range(-g_max, g_max + 1):
            for n in range(-g_max, g_max + 1):
                gs.append((m, n))
        return gs

    # Precompute G-vectors for the solver
    G_MAX = 8  # 17x17 = 289 G-vectors, matching the evaluator
    G_VECTORS = build_g_vectors(G_MAX)
    N_G = len(G_VECTORS)
    G_IDX = {(m, n): idx for idx, (m, n) in enumerate(G_VECTORS)}

    # K-points in the irreducible BZ (M/2, 0) -> (M/2, M/2) -> (0, 0)
    # In reduced coordinates (fraction of reciprocal lattice vector):
    # Irreducible BZ of square lattice: triangle with vertices (0,0), (0.5,0), (0.5,0.5)
    # We use a set of k-points covering this triangle.
    K_POINTS = []
    # Along Gamma-X: (0,0) to (0.5, 0)
    for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        K_POINTS.append((t, 0.0))
    # Along X-M: (0.5, 0) to (0.5, 0.5)
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        K_POINTS.append((0.5, t))
    # Along M-Gamma: (0.5, 0.5) to (0, 0)
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        K_POINTS.append((0.5 - t, 0.5 - t))
    # Interior points
    K_POINTS.append((0.25, 0.125))
    K_POINTS.append((0.375, 0.125))
    K_POINTS.append((0.25, 0.25))
    K_POINTS.append((0.375, 0.25))
    K_POINTS.append((0.5, 0.25))
    K_POINTS.append((0.5, 0.375))
    K_POINTS.append((0.125, 0.125))
    K_POINTS.append((0.375, 0.375))
    K_POINTS.append((0.25, 0.375))
    K_POINTS.append((0.375, 0.125))

    def compute_bands(coeffs, kx, ky):
        """
        Compute TM band eigenvalues at wave vector (kx, ky) in reduced coords.
        H[i,j] = |k+G_i|^2 * eps^{-1}(G_i - G_j)
        Eigenvalues give omega^2.
        """
        H = np.zeros((N_G, N_G), dtype=complex)
        for i, (mi, ni) in enumerate(G_VECTORS):
            kx_i = kx + mi
            ky_i = ky + ni
            k_sq = kx_i * kx_i + ky_i * ky_i
            if k_sq < 1e-12:
                # At k=0, G=0: H[0,0] = 0 (trivial), but actually for TM
                # the zero-frequency mode is spurious; we still include it
                pass
            for j, (mj, nj) in enumerate(G_VECTORS):
                dm = mi - mj
                dn = ni - nj
                # Get Fourier coefficient
                dm_idx = dm % size
                dn_idx = dn % size
                eps_inv = coeffs[dm_idx, dn_idx]
                H[i, j] = k_sq * eps_inv
        # Solve eigenvalue problem
        eigvals = np.linalg.eigvalsh(H)
        eigvals = np.sort(eigvals.real)
        # Filter out negative or near-zero values (spurious)
        eigvals = eigvals[eigvals > 1e-10]
        return eigvals

    def gap_ratio(cell, use_fewer_kpoints=False):
        """Compute the gap-midgap ratio between bands 4 and 5."""
        coeffs = compute_coeffs(cell)
        min_omega_5 = float('inf')
        max_omega_4 = 0.0

        kpts = K_POINTS[:8] if use_fewer_kpoints else K_POINTS

        for kx, ky in kpts:
            eigvals = compute_bands(coeffs, kx, ky)
            if len(eigvals) >= 5:
                # omega^2 values; bands 4 and 5 are indices 3 and 4
                w4_sq = eigvals[3]
                w5_sq = eigvals[4]
                w4 = math.sqrt(w4_sq)
                w5 = math.sqrt(w5_sq)
                if w5 < min_omega_5:
                    min_omega_5 = w5
                if w4 > max_omega_4:
                    max_omega_4 = w4

        if min_omega_5 == float('inf') or max_omega_4 <= 0:
            return -1.0
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap <= 0:
            return -1.0
        return gap / midgap

    # ---------- Design generators (square pillars) ----------
    def generate_square_rod(half_width):
        """Central square pillar with half-width in pixels."""
        cell = [[0] * size for _ in range(size)]
        cx = size / 2.0
        cy = size / 2.0
        hw = half_width
        for i in range(size):
            for j in range(size):
                x = (j + 0.5) - cx
                y = (i + 0.5) - cy
                if abs(x) < hw and abs(y) < hw:
                    cell[i][j] = 1
        return cell

    def generate_2x2_square_rods(half_width, offset_frac):
        """Four square pillars in a 2x2 arrangement."""
        cell = [[0] * size for _ in range(size)]
        cx = size / 2.0
        cy = size / 2.0
        off = offset_frac * size
        hw = half_width
        centers = [
            (cx - off, cy - off),
            (cx + off, cy - off),
            (cx - off, cy + off),
            (cx + off, cy + off),
        ]
        for i in range(size):
            for j in range(size):
                x = j + 0.5
                y = i + 0.5
                for ccx, ccy in centers:
                    if abs(x - ccx) < hw and abs(y - ccy) < hw:
                        cell[i][j] = 1
                        break
        return cell

    def generate_square_ring(half_inner, half_outer):
        """Square ring (annulus)."""
        cell = [[0] * size for _ in range(size)]
        cx = size / 2.0
        cy = size / 2.0
        hi = half_inner
        ho = half_outer
        for i in range(size):
            for j in range(size):
                x = abs((j + 0.5) - cx)
                y = abs((i + 0.5) - cy)
                r = max(x, y)  # Chebyshev distance for square
                if hi <= r < ho:
                    cell[i][j] = 1
        return cell

    def generate_cross(arm_half_width, arm_length):
        """Cross/plus shape with square arms."""
        cell = [[0] * size for _ in range(size)]
        cx = size / 2.0
        cy = size / 2.0
        hw = arm_half_width
        al = arm_length
        for i in range(size):
            for j in range(size):
                x = abs((j + 0.5) - cx)
                y = abs((i + 0.5) - cy)
                if (x < hw and y < al) or (y < hw and x < al):
                    cell[i][j] = 1
        return cell

    def generate_diamond(half_width, size_frac):
        """Diamond (rotated square) shape."""
        cell = [[0] * size for _ in range(size)]
        cx = size / 2.0
        cy = size / 2.0
        s = size_frac * size
        hw = half_width
        for i in range(size):
            for j in range(size):
                x = (j + 0.5) - cx
                y = (i + 0.5) - cy
                if abs(x) / s + abs(y) / s <= 1.0:
                    # Check thickness
                    dist = 1.0 - (abs(x) / s + abs(y) / s)
                    if dist * s >= hw * 0.5:
                        cell[i][j] = 1
        return cell

    def generate_concentric_squares(radii_fracs):
        """Multiple concentric square rings."""
        cell = [[0] * size for _ in range(size)]
        cx = size / 2.0
        cy = size / 2.0
        for i in range(size):
            for j in range(size):
                x = abs((j + 0.5) - cx)
                y = abs((i + 0.5) - cy)
                r = max(x, y)
                for k in range(len(radii_fracs) - 1):
                    r_in = radii_fracs[k] * size
                    r_out = radii_fracs[k + 1] * size
                    if r_in <= r < r_out:
                        cell[i][j] = 1
                        break
        return cell

    # ---------- Generate candidates ----------
    candidates = []

    # Central square rods
    for hw in [3, 4, 5, 6, 7, 8]:
        candidates.append(generate_square_rod(hw))

    # 2x2 square rods
    for hw in [2, 3, 4, 5]:
        for off in [4, 5, 6, 7, 8]:
            candidates.append(generate_2x2_square_rods(hw, off / size))

    # Square rings
    for hi, ho in [(2, 5), (3, 6), (2, 6), (3, 7), (4, 7), (2, 4), (3, 5)]:
        candidates.append(generate_square_ring(hi, ho))

    # Crosses
    for hw, al in [(2, 8), (3, 7), (2, 6), (3, 8), (4, 6), (2, 10)]:
        candidates.append(generate_cross(hw, al))

    # Diamonds
    for hw, sf in [(1, 0.4), (2, 0.4), (1, 0.5), (2, 0.5), (1, 0.35), (2, 0.35)]:
        candidates.append(generate_diamond(hw, sf))

    # Concentric squares
    for radii in [(0, 0.25), (0.1, 0.35), (0, 0.2, 0.4), (0.1, 0.3, 0.45)]:
        candidates.append(generate_concentric_squares(radii))

    # ---------- Evaluate candidates ----------
    best_score = -1.0
    best_cell = candidates[0]

    for cell in candidates:
        if time.time() > DEADLINE:
            break
        cell_sym = enforce_d4(cell)
        score = gap_ratio(cell_sym, use_fewer_kpoints=True)
        if score > best_score:
            best_score = score
            best_cell = cell_sym

    # ---------- Local search: flip pixels in D4 orbits ----------
    if time.time() < DEADLINE - 20:
        base = best_cell
        base_score = best_score
        improved = True
        max_rounds = 5
        for rnd in range(max_rounds):
            if time.time() > DEADLINE - 15:
                break
            improved = False
            # Build unique D4 orbits
            orbits = []
            visited = set()
            for i in range(size):
                for j in range(size):
                    if (i, j) in visited:
                        continue
                    orbit = []
                    a, b = i, j
                    for _ in range(4):
                        orbit.append((a, b))
                        orbit.append((b, a))
                        a, b = size - 1 - a, size - 1 - b
                    for pos in orbit:
                        visited.add(pos)
                    orbits.append(orbit)

            for orbit in orbits:
                if time.time() > DEADLINE - 10:
                    break
                # Try flipping this orbit
                cell_trial = [row[:] for row in base]
                for (r, c) in orbit:
                    cell_trial[r][c] = 1 - base[r][c]
                cell_sym = enforce_d4(cell_trial)
                if cell_sym == base:
                    continue
                score = gap_ratio(cell_sym, use_fewer_kpoints=True)
                if score > base_score + 1e-8:
                    base = cell_sym
                    base_score = score
                    improved = True
                    # Restart orbit iteration
                    break

            if not improved:
                break

        best_score = base_score
        best_cell = base

    # ---------- Final high-precision evaluation and polish ----------
    if time.time() < DEADLINE - 5:
        # Do a more precise evaluation of the best cell
        final_score = gap_ratio(best_cell, use_fewer_kpoints=False)
        if final_score > best_score:
            best_score = final_score

        # Try a few more local flips with full k-point set
        for _ in range(3):
            if time.time() > DEADLINE - 3:
                break
            best_local = best_score
            best_local_cell = best_cell
            visited = set()
            orbits = []
            for i in range(size):
                for j in range(size):
                    if (i, j) in visited:
                        continue
                    orbit = []
                    a, b = i, j
                    for _ in range(4):
                        orbit.append((a, b))
                        orbit.append((b, a))
                        a, b = size - 1 - a, size - 1 - b
                    for pos in orbit:
                        visited.add(pos)
                    orbits.append(orbit)

            for orbit in orbits:
                if time.time() > DEADLINE - 2:
                    break
                cell_trial = [row[:] for row in best_cell]
                for (r, c) in orbit:
                    cell_trial[r][c] = 1 - best_cell[r][c]
                cell_sym = enforce_d4(cell_trial)
                if cell_sym == best_cell:
                    continue
                score = gap_ratio(cell_sym, use_fewer_kpoints=False)
                if score > best_local + 1e-8:
                    best_local = score
                    best_local_cell = cell_sym

            if best_local > best_score:
                best_score = best_local
                best_cell = best_local_cell
            else:
                break

    final_cell = enforce_d4(best_cell)
    return final_cell

# EVOLVE-BLOCK-END
