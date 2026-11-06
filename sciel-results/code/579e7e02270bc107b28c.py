"""Design a 32x32 C4v-symmetric photonic crystal unit cell maximizing the TM band-4/5 gap."""
import time
import numpy as np

_DEADLINE = 215.0  # safety margin before 230s kill


def _build_fourier(pixel_map, Npw):
    """Compute inverse-permittivity Fourier coefficients for a pixel map.

    pixel_map: 32x32 array of 0/1 (1 = dielectric eps_high, 0 = eps_low=1)
    Returns dict G -> 1/eps(G) for |Gx|,|Gy| <= Npw//2 (circular or square cutoff).
    """
    S = pixel_map.shape[0]
    eps_low, eps_high = 1.0, 12.0
    # Build inverse-epsilon on the pixel grid
    inv_eps = np.where(pixel_map == 1, 1.0 / eps_high, 1.0 / eps_low)
    # Compute Fourier coefficients via DFT (exact for the pixel sampling)
    F = np.fft.fft2(inv_eps) / (S * S)
    # Reorder so that G=0 is at center; indices: Gx = (col - S//2) * 2pi, etc.
    # We want coefficients for reciprocal vectors G = (2pi * m, 2pi * n)
    # with m, n in range. fft2 gives frequency index k corresponding to k/S cycles,
    # i.e. G = 2pi * (k - S//2 if k>S//2 else k) / S ... standard fftfreq.
    freqs = np.fft.fftfreq(S) * S  # gives integer m values: 0,1,...,S/2-1,-S/2,...,-1
    # Build the coefficient dictionary for |m|,|n| <= Npw//2
    half = Npw // 2
    coeffs = {}
    for im, m in enumerate(freqs):
        if abs(m) > half:
            continue
        for in_, n in enumerate(freqs):
            if abs(n) > half:
                continue
            coeffs[(int(m), int(n))] = F[im, in_]
    return coeffs


def _pwe_eigenvalues(coeffs, kx, ky, Npw, n_bands=5):
    """Solve the PWE eigenvalue problem for TM bands at wavevector (kx,ky).

    Returns the lowest n_bands eigenvalues (omega^2 in units of (2pi/a)^2 ... actually
    we work in dimensionless units where a=1, so eigenvalue = (ka)^2).
    """
    # Build list of G vectors
    Glist = [g for g in coeffs.keys()]
    N = len(Glist)
    Gidx = {g: i for i, g in enumerate(Glist)}
    # Build A matrix: A[i,j] = (k - G_i) x (k - G_j) dot ... for TM:
    # (k - G_i)_x * (k - G_j)_x + (k - G_i)_y * (k - G_j)_y
    # times 1/eps(G_i - G_j)
    # Actually the standard TM PWE: for each pair, the operator is
    # [(k-G_i)x (k-G_j)x + (k-G_i)y (k-G_j)y] * eps_inv(G_i - G_j)
    # Build dense matrix
    A = np.zeros((N, N), dtype=float)
    for i, gi in enumerate(Glist):
        kxi, ky_i = kx - gi[0] * 2.0 * np.pi, ky - gi[1] * 2.0 * np.pi
        for j, gj in enumerate(Glist):
            diff = (gi[0] - gj[0], gi[1] - gj[1])
            if diff in coeffs:
                kxj, kyj = kx - gj[0] * 2.0 * np.pi, ky - gj[1] * 2.0 * np.pi
                A[i, j] = (kxi * kxj + ky_i * kyj) * coeffs[diff]
    # Solve eigenvalue problem
    vals, vecs = np.linalg.eigh(A)
    vals = np.sort(vals)
    return vals[:n_bands], vecs[:, :n_bands]


def _irreducible_kpoints(n_per_edge=6):
    """Generate k-points in the irreducible BZ of a square lattice (GAMMA->X->M->GAMMA).
    Returns list of (kx, ky) in units of 1/a (i.e. k = 2pi * (kx/a, ky/a) with a=1).
    We use kx, ky in [0, 1] in units of 2pi/a, so actual k = 2pi*(kx, ky).
    """
    kpts = []
    # Along GAMMA(0,0) -> X(0.5, 0)
    for t in np.linspace(0, 0.5, n_per_edge):
        kpts.append((t, 0.0))
    # Along X(0.5,0) -> M(0.5, 0.5)
    for t in np.linspace(0, 0.5, n_per_edge - 1):
        kpts.append((0.5, t))
    # Along M(0.5,0.5) -> GAMMA(0,0)  (line kx=ky)
    for t in np.linspace(0.5, 0.0, n_per_edge - 1):
        kpts.append((t, t))
    return kpts


def _compute_gap_and_fields(pixel_map, Npw, kpoints, n_bands=5):
    """Compute max of band 4 and min of band 5 over k-points, plus field intensities.
    
    Returns (gap_ratio, max_b4, min_b5, field4, field5) where field4/field5 are
    32x32 arrays of integrated |E|^2 for band 4 (at k of max_b4) and band 5 (at k of min_b5).
    """
    coeffs = _build_fourier(pixel_map, Npw)
    max_b4 = -np.inf
    min_b5 = np.inf
    best_k4 = None
    best_k5 = None
    best_vec4 = None
    best_vec5 = None

    for kx, ky in kpoints:
        kx_actual = kx * 2.0 * np.pi
        ky_actual = ky * 2.0 * np.pi
        vals, vecs = _pwe_eigenvalues(coeffs, kx_actual, ky_actual, Npw, n_bands)
        b4 = vals[3]
        b5 = vals[4]
        if b4 > max_b4:
            max_b4 = b4
            best_k4 = (kx_actual, ky_actual)
            best_vec4 = vecs[:, 3]
        if b5 < min_b5:
            min_b5 = b5
            best_k5 = (kx_actual, ky_actual)
            best_vec5 = vecs[:, 4]

    if max_b4 < 0:
        return -1.0, max_b4, min_b5, None, None

    gap = 2.0 * (min_b5 - max_b4) / (min_b5 + max_b4) if (min_b5 + max_b4) != 0 else -1.0

    # Compute field intensity on pixel grid for band 4 and band 5
    S = pixel_map.shape[0]
    field4 = _field_intensity(pixel_map, coeffs, best_k4, best_vec4, S)
    field5 = _field_intensity(pixel_map, coeffs, best_k5, best_vec5, S)
    return gap, max_b4, min_b5, field4, field5


def _field_intensity(pixel_map, coeffs, kvec, vec, S):
    """Compute |E(z)|^2 integrated over each pixel for a given eigenmode.
    
    kvec: (kx, ky) actual wavevector
    vec: eigenvector (coefficients for each G)
    """
    Glist = list(coeffs.keys())
    Gidx = {g: i for i, g in enumerate(Glist)}
    kx, ky = kvec
    # E(r) = sum_G c_G exp(i (k - G) . r)
    # |E|^2 = sum_{G1,G2} c_{G1} c*_{G2} exp(i (G2 - G1) . r)
    # We sample on a fine grid within the unit cell and bin into pixels.
    n_sample = 4  # samples per pixel
    grid = np.linspace(0, 1, S * n_sample, endpoint=False)
    E2 = np.zeros((S * n_sample, S * n_sample))
    for i, gi in enumerate(Glist):
        if abs(vec[i]) < 1e-12:
            continue
        kx_i = kx - gi[0] * 2.0 * np.pi
        ky_i = ky - gi[1] * 2.0 * np.pi
        phase_x = np.exp(1j * kx_i * grid)
        phase_y = np.exp(1j * ky_i * grid)
        E2 += vec[i] * np.outer(phase_x, phase_y)
    E2 = np.abs(E2) ** 2
    # Bin into S x S pixels
    E2 = E2.reshape(S, n_sample, S, n_sample).mean(axis=(1, 3))
    return E2


def _apply_c4v_flip(cell, i, j, new_val):
    """Set pixel (i,j) and all its C4v-symmetric partners to new_val.
    Returns list of positions flipped.
    """
    S = cell.shape[0]
    # C4v orbit of (i,j): (i,j), (j,i), (S-1-i, j), (S-1-j, i), (S-1-i, S-1-j), (S-1-j, S-1-i), (i, S-1-j), (S-1-j, S-1-i)
    positions = set()
    r, c = i, j
    for _ in range(4):
        positions.add((r, c))
        positions.add((c, r))
        positions.add((S - 1 - r, c))
        positions.add((S - 1 - c, r))
        r, c = c, S - 1 - r
    positions = list(positions)
    for (pi, pj) in positions:
        cell[pi, pj] = new_val
    return positions


def design(problem):
    global _DEADLINE
    _DEADLINE = time.time() + 215.0

    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    eps_low = problem.get("eps_low", 1)

    # Start from the seed: a centered dielectric rod
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < 0.2 ** 2:
                cell[i, j] = 1

    # Coarse PWE settings for optimization
    Npw_coarse = 17  # 17x17 = 289 plane waves
    kpoints = _irreducible_kpoints(n_per_edge=5)

    # Initial evaluation
    gap, max_b4, min_b5, field4, field5 = _compute_gap_and_fields(cell, Npw_coarse, kpoints)
    best_gap = gap
    best_cell = cell.copy()

    if time.time() > _DEADLINE - 30:
        return best_cell.tolist()

    # Topological optimization loop
    # Strategy: at each step, compute field intensities, then flip C4v orbits
    # to increase dielectric where band 4 intensity is high and decrease where band 5 is high.
    step = 0
    while time.time() < _DEADLINE - 30:
        step += 1
        if field4 is None or field5 is None:
            break

        # Compute the "desirability" of adding dielectric at each pixel:
        # Add where band 4 (lower band) has high intensity (pushes it up),
        # Remove where band 5 (upper band) has high intensity (pushes it down).
        # Actually: to OPEN the gap, we want to raise band 4 and lower band 5.
        # Adding dielectric raises modes that have high intensity in the dielectric region.
        # So: add dielectric where band 4 intensity is high (raises band 4),
        #     remove dielectric where band 5 intensity is high (lowers band 5).
        add_score = field4.copy()
        remove_score = field5.copy()

        # Normalize
        if add_score.max() > 0:
            add_score /= add_score.max()
        if remove_score.max() > 0:
            remove_score /= remove_score.max()

        # For each C4v orbit, compute net benefit of flipping
        # If pixel is 0 (air), flipping to 1 adds dielectric. Benefit = add_score[i,j] - remove_score[i,j]
        # If pixel is 1 (dielectric), flipping to 0 removes dielectric. Benefit = remove_score[i,j] - add_score[i,j]
        # We want to flip the orbit with maximum positive benefit.

        # Enumerate unique orbits
        visited = set()
        orbit_benefits = []
        for i in range(size):
            for j in range(size):
                if (i, j) in visited:
                    continue
                # Get orbit
                positions = []
                r, c = i, j
                for _ in range(4):
                    for (pr, pc) in [(r, c), (c, r), (size - 1 - r, c), (size - 1 - c, r)]:
                        if (pr, pc) not in visited:
                            visited.add((pr, pc))
                            positions.append((pr, pc))
                    r, c = c, size - 1 - r

                # Compute benefit
                benefit = 0.0
                for (pi, pj) in positions:
                    if cell[pi, pj] == 0:
                        benefit += add_score[pi, pj] - remove_score[pi, pj]
                    else:
                        benefit += remove_score[pi, pj] - add_score[pi, pj]

                orbit_benefits.append((benefit, positions))

        if not orbit_benefits:
            break

        # Sort by benefit descending
        orbit_benefits.sort(key=lambda x: -x[0])

        # Try the top few orbits
        tried = 0
        improved = False
        for benefit, positions in orbit_benefits[:5]:
            if benefit <= 0.01:  # not worth trying
                break
            if time.time() > _DEADLINE - 30:
                break

            # Apply flip
            new_cell = cell.copy()
            for (pi, pj) in positions:
                new_cell[pi, pj] = 1 - new_cell[pi, pj]

            # Evaluate
            new_gap, _, _, _, _ = _compute_gap_and_fields(new_cell, Npw_coarse, kpoints)

            if new_gap > best_gap + 1e-6:
                cell = new_cell
                best_gap = new_gap
                best_cell = cell.copy()
                improved = True
                # Recompute fields with the new best
                _, _, _, field4, field5 = _compute_gap_and_fields(cell, Npw_coarse, kpoints)
                break
            tried += 1

        if not improved and step > 20:
            # Stagnation: try random flips to escape local optimum
            if step % 5 == 0:
                for _ in range(3):
                    if time.time() > _DEADLINE - 30:
                        break
                    ri = np.random.randint(size)
                    rj = np.random.randint(size)
                    new_cell = cell.copy()
                    _apply_c4v_flip(new_cell, ri, rj, 1 - new_cell[ri, rj])
                    new_gap, _, _, _, _ = _compute_gap_and_fields(new_cell, Npw_coarse, kpoints)
                    if new_gap > best_gap:
                        cell = new_cell
                        best_gap = new_gap
                        best_cell = cell.copy()
                        _, _, _, field4, field5 = _compute_gap_and_fields(cell, Npw_coarse, kpoints)

    # Final refinement with more k-points if time allows
    if time.time() < _DEADLINE - 15:
        kpoints_fine = _irreducible_kpoints(n_per_edge=8)
        final_gap, _, _, _, _ = _compute_gap_and_fields(best_cell, Npw_coarse, kpoints_fine)
        # The final cell is best_cell

    return best_cell.tolist()
