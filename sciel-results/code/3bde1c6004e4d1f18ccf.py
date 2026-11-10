"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def design(problem):
    """
    Designs a 32x32 unit cell using pixel-level local search with 4-fold symmetry,
    scored by a fast plane-wave band solver.
    """
    start_time = time.time()
    deadline = start_time + 215.0  # Safety margin before 230s kill

    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)

    def enforce_symmetry(cell):
        """Enforce 4-fold symmetry: 180-degree rotation and transpose."""
        cell = np.array(cell, dtype=int)
        # Enforce 180-degree rotation symmetry
        cell = (cell + cell[::-1, ::-1]) // 2
        # Enforce transpose symmetry (mirror across main diagonal)
        cell = (cell + cell.T) // 2
        # Re-enforce 180 after transpose (should be consistent but safe)
        cell = (cell + cell[::-1, ::-1]) // 2
        return cell

    def compute_fft_inv_eps(cell):
        """Compute 2D FFT of 1/epsilon."""
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        fft_2d = np.fft.fft2(inv_eps) / (size * size)
        return fft_2d

    def solve_bands(cell, n_pw=12, k_pts=None):
        """
        TM band solver using plane-wave expansion.
        Returns gap-midgap ratio for bands 4 and 5 (1-indexed, indices 3 and 4).
        """
        fft_2d = compute_fft_inv_eps(cell)

        def get_coeff(m, n):
            m_idx = m % size
            n_idx = n % size
            return fft_2d[m_idx, n_idx]

        if k_pts is None:
            # Key points in the irreducible BZ: 0 <= ky <= kx <= 1
            k_pts = [
                (0.0, 0.0),
                (0.25, 0.0),
                (0.5, 0.0),
                (0.75, 0.0),
                (1.0, 0.0),
                (0.25, 0.25),
                (0.5, 0.25),
                (0.75, 0.25),
                (1.0, 0.25),
                (0.5, 0.5),
                (0.75, 0.5),
                (1.0, 0.5),
                (0.75, 0.75),
                (1.0, 0.75),
                (1.0, 1.0),
            ]

        # Build list of plane waves within circular cutoff
        g_pairs = []
        for m in range(-n_pw, n_pw + 1):
            for n in range(-n_pw, n_pw + 1):
                if m * m + n * n <= n_pw * n_pw:
                    g_pairs.append((m, n))

        n_pw_actual = len(g_pairs)

        max_omega4 = -1e10
        min_omega5 = 1e10

        for kx, ky in k_pts:
            # Build Hermitian matrix for TM polarization
            # H[G, G'] = eps_inv[G - G'] * |k - G'|^2
            M = np.zeros((n_pw_actual, n_pw_actual), dtype=complex)

            for i, (mi, ni) in enumerate(g_pairs):
                for j, (mj, nj) in enumerate(g_pairs):
                    diff_m = mi - mj
                    diff_n = ni - nj
                    eps_val = get_coeff(diff_m, diff_n)
                    kx_g = kx - mj
                    ky_g = ky - nj
                    M[i, j] = eps_val * (kx_g * kx_g + ky_g * ky_g)

            # Symmetrize to get Hermitian matrix
            H = (M + M.conj().T) / 2.0

            try:
                eigenvalues = np.linalg.eigvalsh(H)
                eigenvalues = np.sort(np.real(eigenvalues))
                if len(eigenvalues) >= 5:
                    omega4 = eigenvalues[3]
                    omega5 = eigenvalues[4]
                    if omega4 > max_omega4:
                        max_omega4 = omega4
                    if omega5 < min_omega5:
                        min_omega5 = omega5
            except:
                pass

        if max_omega4 <= 0 or min_omega5 <= 0 or min_omega5 <= max_omega4:
            return -1.0

        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0

        if midgap <= 0:
            return -1.0

        return gap / midgap

    def get_orbit(i, j, size=32):
        """Get all pixels in the 4-fold symmetric orbit of (i, j)."""
        orbit = set()
        orbit.add((i, j))
        orbit.add((size - 1 - i, size - 1 - j))  # 180 rot
        orbit.add((j, i))  # transpose
        orbit.add((size - 1 - j, size - 1 - i))  # 180 rot of transpose
        return list(orbit)

    def random_symmetric_cell(size=32, fill_prob=0.5):
        """Generate a random 4-fold symmetric cell."""
        cell = np.zeros((size, size), dtype=int)
        # Only iterate over the fundamental domain
        for i in range(size):
            for j in range(size):
                # Only process the first pixel in each orbit
                orbit = get_orbit(i, j, size)
                if (i, j) != min(orbit):
                    continue
                if np.random.random() < fill_prob:
                    for (oi, oj) in orbit:
                        cell[oi, oj] = 1
        return cell

    # Phase 1: Coarse search - try several random starting points
    np.random.seed(42)
    best_score = -1e10
    best_cell = None

    # Try a few different fill probabilities and random starts
    for fill_prob in [0.3, 0.4, 0.5, 0.6, 0.7]:
        if time.time() > deadline - 60:
            break
        cell = random_symmetric_cell(size, fill_prob)
        score = solve_bands(cell, n_pw=10)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()

    # Also try a structured starting point: checkerboard-like pattern
    cell_struct = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if (i // 4 + j // 4) % 2 == 0:
                cell_struct[i, j] = 1
    cell_struct = enforce_symmetry(cell_struct)
    score = solve_bands(cell_struct, n_pw=10)
    if score > best_score:
        best_score = score
        best_cell = cell_struct.copy()

    if best_cell is None:
        best_cell = random_symmetric_cell(size, 0.5)
        best_score = solve_bands(best_cell, n_pw=10)

    # Phase 2: Local search - flip symmetric pixel orbits
    current_cell = best_cell.copy()
    current_score = best_score
    no_improve_count = 0
    max_no_improve = 50

    while time.time() < deadline - 10 and no_improve_count < max_no_improve:
        # Pick a random pixel and flip its entire orbit
        i = np.random.randint(size)
        j = np.random.randint(size)
        orbit = get_orbit(i, j, size)

        # Check if all pixels in orbit have the same value (they should after symmetry)
        val = current_cell[i, j]
        new_val = 1 - val

        # Apply flip
        old_vals = []
        for (oi, oj) in orbit:
            old_vals.append(current_cell[oi, oj])
            current_cell[oi, oj] = new_val

        new_score = solve_bands(current_cell, n_pw=10)

        if new_score > current_score:
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_cell = current_cell.copy()
        else:
            # Revert
            for (oi, oj), ov in zip(orbit, old_vals):
                current_cell[oi, oj] = ov
            no_improve_count += 1

    # Phase 3: If time remains, do a finer local search with more k-points
    if time.time() < deadline - 15 and best_cell is not None:
        current_cell = best_cell.copy()
        current_score = best_score
        no_improve_count = 0
        max_no_improve = 30

        while time.time() < deadline - 5 and no_improve_count < max_no_improve:
            i = np.random.randint(size)
            j = np.random.randint(size)
            orbit = get_orbit(i, j, size)

            val = current_cell[i, j]
            new_val = 1 - val

            old_vals = []
            for (oi, oj) in orbit:
                old_vals.append(current_cell[oi, oj])
                current_cell[oi, oj] = new_val

            new_score = solve_bands(current_cell, n_pw=14)

            if new_score > current_score:
                current_score = new_score
                no_improve_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_cell = current_cell.copy()
            else:
                for (oi, oj), ov in zip(orbit, old_vals):
                    current_cell[oi, oj] = ov
                no_improve_count += 1

    # Final: enforce symmetry and return
    best_cell = enforce_symmetry(best_cell)
    best_cell = np.clip(best_cell, 0, 1).astype(int)

    # Convert to nested list
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(best_cell[i, j]))
        result.append(row)

    return result
# EVOLVE-BLOCK-END
