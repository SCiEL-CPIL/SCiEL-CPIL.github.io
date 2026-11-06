"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def design(problem):
    """Optimize D4-symmetric dielectric pattern for 4th-5th TM band gap."""
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    start_time = time.time()
    deadline = start_time + 200  # 30s safety margin before 230s kill

    def enforce_d4_symmetry(cell):
        """Average cell over all 8 D4 symmetry operations, then threshold."""
        sym = np.zeros_like(cell, dtype=float)
        sym += cell
        sym += np.rot90(cell, 1)
        sym += np.rot90(cell, 2)
        sym += np.rot90(cell, 3)
        sym += np.fliplr(cell)
        sym += np.flipud(cell)
        sym += np.fliplr(np.flipud(cell))
        sym += np.rot90(np.fliplr(cell), 1)
        return (sym / 8.0 >= 0.5).astype(np.uint8)

    def compute_epsilon_fourier(cell):
        """Compute the 2D FFT of the dielectric function (centered)."""
        eps = 1.0 + (eps_high - 1.0) * cell.astype(float)
        eps_centered = eps - eps.mean()
        fft = np.fft.fft2(eps_centered)
        return fft

    def compute_gap_approx(cell, n_pw=17):
        """
        Approximate band gap between bands 4 and 5 using plane-wave method.
        Uses a reduced number of plane waves for speed.
        """
        eps_fft = compute_epsilon_fourier(cell)

        # Build reciprocal lattice vectors
        # For a 32x32 cell, we use indices from -n_pw//2 to n_pw//2
        half = n_pw // 2
        idx = np.arange(-half, half + 1)
        Gx, Gy = np.meshgrid(idx, idx, indexing='ij')
        G = np.stack([Gx.ravel(), Gy.ravel()], axis=1).astype(float)
        n_g = len(G)

        # Dielectric function in reciprocal space
        # epsilon(G) = FFT of epsilon(r)
        # The FFT gives us epsilon at G indices 0..31, but we need -16..16
        # For a 32x32 cell, fftfreq gives us the right indices
        eps_g = np.zeros(n_g, dtype=complex)
        for i, (gx, gy) in enumerate(G):
            igx = int(gx) % size
            igy = int(gy) % size
            eps_g[i] = eps_fft[igy, igx]

        # epsilon_0 (DC component)
        eps_0 = eps_g[0].real  # Should be the average epsilon

        # For TM polarization: (1/epsilon) matrix
        # H(k) = (1/epsilon(k-G)) * |k-G|^2
        # We need 1/epsilon for all G

        inv_eps_g = np.zeros(n_g, dtype=complex)
        inv_eps_g[0] = 1.0 / eps_0
        for i in range(1, n_g):
            # For non-zero G, 1/epsilon(G) is the inverse of the Fourier coefficient
            # But actually we need 1/epsilon(r) expanded in Fourier
            # 1/epsilon(r) = sum_G (1/epsilon)_G e^{iG.r}
            # This is NOT simply 1/epsilon_G
            # We need to compute the Fourier transform of 1/epsilon(r)
            pass

        # Actually, let's compute 1/epsilon(r) directly and FFT it
        eps_real = 1.0 + (eps_high - 1.0) * cell.astype(float)
        inv_eps_real = 1.0 / eps_real
        inv_eps_fft = np.fft.fft2(inv_eps_real - inv_eps_real.mean())

        inv_eps_g = np.zeros(n_g, dtype=complex)
        for i, (gx, gy) in enumerate(G):
            igx = int(gx) % size
            igy = int(gy) % size
            inv_eps_g[i] = inv_eps_fft[igy, igx]

        # inv_eps_0
        inv_eps_0 = inv_eps_real.mean()

        # For TM: eigenvalue problem is
        # sum_G' [ (1/epsilon)_{G-G'} * |k-G'|^2 ] a_{G'} = omega^2 * a_G
        # where (1/epsilon)_{G-G'} is the (G-G')-th Fourier coefficient of 1/epsilon

        # Build the matrix for a given k
        def solve_bands(kx, ky, n_bands=5):
            # k is in units of 2*pi/a
            K = np.column_stack([G[:, 0] + kx, G[:, 1] + ky])
            k_mag_sq = np.sum(K**2, axis=1)

            # Build matrix M[G, G'] = (1/epsilon)_{G-G'} * |k-G'|^2
            M = np.zeros((n_g, n_g), dtype=complex)
            for i in range(n_g):
                for j in range(n_g):
                    dgx = int(G[i, 0] - G[j, 0])
                    dgy = int(G[i, 1] - G[j, 1])
                    # Get (1/epsilon)_{(dgx, dgy)}
                    idgx = dgx % size
                    idgy = dgy % size
                    if idgx == 0 and idgy == 0:
                        val = inv_eps_0
                    else:
                        val = inv_eps_fft[idgy, idgx]
                    M[i, j] = val * k_mag_sq[j]

            # Solve eigenvalue problem
            try:
                eigvals = np.linalg.eigvalsh(M)
                eigvals = np.real(eigvals)
                # Sort and take first n_bands
                eigvals_sorted = np.sort(eigvals)
                return eigvals_sorted[:n_bands]
            except:
                return None

        # Evaluate at several high-symmetry points and some interior points
        # Irreducible BZ: 0 <= kx <= ky (for square lattice, actually kx,ky >= 0 and kx <= ky for the triangle)
        # Actually for square lattice, the irreducible BZ is the triangle with vertices (0,0), (pi/a, 0), (pi/a, pi/a)
        # In units of 2*pi/a: (0,0), (0.5, 0), (0.5, 0.5)

        # Key points: Gamma(0,0), X(0.5,0), M(0.5,0.5)
        # And some interior points
        k_points = [
            (0.0, 0.0),      # Gamma
            (0.5, 0.0),      # X
            (0.5, 0.5),      # M
            (0.25, 0.0),     # Midpoint Gamma-X
            (0.25, 0.25),    # Midpoint Gamma-M
            (0.5, 0.25),     # Midpoint X-M
            (0.1, 0.0),
            (0.4, 0.0),
            (0.1, 0.1),
            (0.4, 0.4),
            (0.3, 0.1),
            (0.3, 0.3),
        ]

        omega4_max = -np.inf
        omega5_min = np.inf

        for kx, ky in k_points:
            bands = solve_bands(kx, ky, n_bands=5)
            if bands is None or len(bands) < 5:
                continue
            omega4_max = max(omega4_max, bands[3])  # 4th band (0-indexed: 3)
            omega5_min = min(omega5_min, bands[4])  # 5th band (0-indexed: 4)

        if omega4_max == -np.inf or omega5_min == np.inf:
            return 0.0

        gap = omega5_min - omega4_max
        midgap = 0.5 * (omega5_min + omega4_max)
        if midgap <= 0:
            return 0.0
        return 2.0 * gap / midgap

    def score_cell_fast(cell):
        """
        Fast scoring using Fourier analysis as a proxy.
        Used for initial screening.
        """
        eps_fft = compute_epsilon_fourier(cell)
        eps_mag = np.abs(eps_fft)

        kx = np.fft.fftfreq(size, d=1.0/size)
        ky = np.fft.fftfreq(size, d=1.0/size)
        KX, KY = np.meshgrid(kx, ky)
        Kmag = np.sqrt(KX**2 + KY**2)

        # For 4th-5th TM bands, relevant G-vectors are in a specific range
        mask_low = (Kmag > 2.0) & (Kmag < 3.5)
        mask_high = (Kmag >= 3.5) & (Kmag < 6.0)

        power_low = np.sum(eps_mag[mask_low]**2)
        power_high = np.sum(eps_mag[mask_high]**2)

        if power_low < 1e-10:
            return 0.0

        frac = cell.mean()
        frac_penalty = max(0.0, 1.0 - abs(frac - 0.4) * 2.5)

        return (power_high / (power_low + 1e-10)) * frac_penalty

    def make_wavelet_pattern(scale1, scale2, phase1, phase2, threshold):
        """
        Create a pattern from superposition of two sinusoidal waves at different scales.
        This captures multi-scale features.
        """
        x = np.linspace(-0.5, 0.5, size, endpoint=False)
        y = np.linspace(-0.5, 0.5, size, endpoint=False)
        X, Y = np.meshgrid(x, y)

        # Two wave components at different frequencies
        wave1 = np.sin(2 * np.pi * scale1 * X + phase1) * np.sin(2 * np.pi * scale1 * Y + phase1)
        wave2 = np.sin(2 * np.pi * scale2 * X + phase2) * np.sin(2 * np.pi * scale2 * Y + phase2)

        # Superposition
        field = wave1 + 0.7 * wave2

        # Threshold to binary
        cell = (field > threshold).astype(np.uint8)
        return enforce_d4_symmetry(cell)

    def make_ring_pattern(inner_r, outer_r, angular_freq=4):
        """
        Create a ring-like pattern with angular modulation.
        """
        x = np.linspace(-0.5, 0.5, size, endpoint=False)
        y = np.linspace(-0.5, 0.5, size, endpoint=False)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)

        # Ring with angular modulation
        mod = 1.0 + 0.3 * np.cos(angular_freq * Theta)
        inner = inner_r * mod
        outer = outer_r * mod

        cell = ((R > inner) & (R < outer)).astype(np.uint8)
        return enforce_d4_symmetry(cell)

    def make_square_ring_pattern(inner_w, outer_w):
        """
        Create a square ring pattern.
        """
        x = np.linspace(-0.5, 0.5, size, endpoint=False)
        y = np.linspace(-0.5, 0.5, size, endpoint=False)
        X, Y = np.meshgrid(x, y)

        cell = ((np.abs(X) > inner_w) | (np.abs(Y) > inner_w)) & \
               ((np.abs(X) < outer_w) & (np.abs(Y) < outer_w))
        cell = cell.astype(np.uint8)
        return enforce_d4_symmetry(cell)

    def make_cross_pattern(arm_width, gap_width):
        """
        Create a cross pattern with a gap in the center.
        """
        x = np.linspace(-0.5, 0.5, size, endpoint=False)
        y = np.linspace(-0.5, 0.5, size, endpoint=False)
        X, Y = np.meshgrid(x, y)

        # Cross arms
        cross = (np.abs(X) < arm_width) | (np.abs(Y) < arm_width)
        # Central gap (air hole in the middle of the cross)
        center_hole = (np.abs(X) < gap_width) & (np.abs(Y) < gap_width)

        cell = (cross & ~center_hole).astype(np.uint8)
        return enforce_d4_symmetry(cell)

    def make_diamond_pattern(half_width):
        """
        Create a diamond (rotated square) pattern.
        """
        x = np.linspace(-0.5, 0.5, size, endpoint=False)
        y = np.linspace(-0.5, 0.5, size, endpoint=False)
        X, Y = np.meshgrid(x, y)

        # Diamond: |x| + |y| < half_width
        cell = (np.abs(X) + np.abs(Y) < half_width).astype(np.uint8)
        return enforce_d4_symmetry(cell)

    def local_search(cell, max_steps=50, timeout=None):
        """
        Local search: flip individual pixels and keep if it improves the score.
        Only flips pixels that maintain D4 symmetry (flip the orbit).
        """
        current_cell = cell.copy()
        current_score = compute_gap_approx(current_cell, n_pw=13)

        # Get all unique orbits under D4
        orbits = []
        visited = set()
        for i in range(size):
            for j in range(size):
                if (i, j) in visited:
                    continue
                # Get the orbit of (i, j)
                orbit = [(i, j)]
                visited.add((i, j))
                for rot in range(1, 4):
                    # Rotate 90*rot degrees about center
                    ci, cj = size // 2 - 0.5, size // 2 - 0.5
                    di, dj = i - ci, j - cj
                    if rot == 1:
                        ni, nj = int(round(ci - dj)), int(round(cj + di))
                    elif rot == 2:
                        ni, nj = int(round(ci - di)), int(round(cj - dj))
                    else:
                        ni, nj = int(round(ci + dj)), int(round(cj - di))
                    if 0 <= ni < size and 0 <= nj < size and (ni, nj) not in visited:
                        orbit.append((ni, nj))
                        visited.add((ni, nj))
                # Also add reflections
                for ri, rj in list(orbit):
                    # Reflect about vertical axis
                    ni, nj = ri, size - 1 - rj
                    if 0 <= ni < size and 0 <= nj < size and (ni, nj) not in visited:
                        orbit.append((ni, nj))
                        visited.add((ni, nj))
                    # Reflect about horizontal axis
                    ni, nj = size - 1 - ri, rj
                    if 0 <= ni < size and 0 <= nj < size and (ni, nj) not in visited:
                        orbit.append((ni, nj))
                        visited.add((ni, nj))

                orbits.append(orbit)

        # Shuffle orbits for diversity
        np.random.seed(42)
        np.random.shuffle(orbits)

        improved = True
        while improved:
            improved = False
            for orbit in orbits:
                if timeout is not None and time.time() > timeout:
                    break
                # Try flipping this orbit
                test_cell = current_cell.copy()
                current_val = test_cell[orbit[0][0], orbit[0][1]]
                for (oi, oj) in orbit:
                    test_cell[oi, oj] = 1 - current_val

                new_score = compute_gap_approx(test_cell, n_pw=13)
                if new_score > current_score + 1e-6:
                    current_cell = test_cell
                    current_score = new_score
                    improved = True

            if timeout is not None and time.time() > timeout:
                break

        return current_cell, current_score

    # Generate diverse candidate patterns
    candidates = []

    # Wavelet-like patterns (multi-scale sinusoidal)
    for s1 in [2, 3, 4, 5, 6]:
        for s2 in [1, 2, 3, 4]:
            if s1 == s2:
                continue
            for p1 in [0, np.pi/2, np.pi]:
                for p2 in [0, np.pi/2]:
                    for thresh in [-0.3, 0.0, 0.3]:
                        cell = make_wavelet_pattern(s1, s2, p1, p2, thresh)
                        candidates.append(cell)

    # Ring patterns
    for inner in [0.1, 0.15, 0.2, 0.25]:
        for outer in [0.3, 0.35, 0.4, 0.45]:
            if outer <= inner:
                continue
            for angf in [4, 6, 8]:
                cell = make_ring_pattern(inner, outer, angf)
                candidates.append(cell)

    # Square ring patterns
    for inner in [0.05, 0.1, 0.15]:
        for outer in [0.25, 0.3, 0.35, 0.4]:
            if outer <= inner:
                continue
            cell = make_square_ring_pattern(inner, outer)
            candidates.append(cell)

    # Cross patterns
    for aw in [0.08, 0.1, 0.12, 0.15]:
        for gw in [0.03, 0.05, 0.08]:
            if gw >= aw:
                continue
            cell = make_cross_pattern(aw, gw)
            candidates.append(cell)

    # Diamond patterns
    for hw in [0.15, 0.2, 0.25, 0.3]:
        cell = make_diamond_pattern(hw)
        candidates.append(cell)

    # Square pillars (from previous knowledge)
    for hw in [0.1, 0.12, 0.15, 0.18, 0.2]:
        x = np.linspace(-0.5, 0.5, size, endpoint=False)
        y = np.linspace(-0.5, 0.5, size, endpoint=False)
        X, Y = np.meshgrid(x, y)
        cell = (np.abs(X) < hw) & (np.abs(Y) < hw)
        cell = cell.astype(np.uint8)
        candidates.append(enforce_d4_symmetry(cell))

    # Screen candidates with fast score
    scored = []
    for i, cell in enumerate(candidates):
        if time.time() > deadline - 60:
            break
        s = score_cell_fast(cell)
        frac = cell.mean()
        if 0.15 < frac < 0.65:
            scored.append((s, i, cell))

    scored.sort(key=lambda x: -x[0])

    # Take top candidates and do local search
    best_cell = None
    best_score = -np.inf

    top_n = min(8, len(scored))
    for rank in range(top_n):
        if time.time() > deadline - 30:
            break
        _, idx, cell = scored[rank]
        # Do local search with a timeout
        search_timeout = deadline - 25
        optimized_cell, opt_score = local_search(cell, max_steps=30, timeout=search_timeout)

        if opt_score > best_score:
            best_score = opt_score
            best_cell = optimized_cell

    # If no good result, use the best screened candidate
    if best_cell is None:
        if scored:
            best_cell = scored[0][2]
        else:
            # Default: square pillar
            x = np.linspace(-0.5, 0.5, size, endpoint=False)
            y = np.linspace(-0.5, 0.5, size, endpoint=False)
            X, Y = np.meshgrid(x, y)
            best_cell = ((np.abs(X) < 0.15) & (np.abs(Y) < 0.15)).astype(np.uint8)
            best_cell = enforce_d4_symmetry(best_cell)

    return best_cell.tolist()
# EVOLVE-BLOCK-END
