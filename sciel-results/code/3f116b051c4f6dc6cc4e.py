"""Design a 32x32 symmetric photonic crystal unit cell for maximum TM band gap."""
import numpy as np
import time
from scipy.linalg import eigh

# EVOLVE-BLOCK-START

def design(problem):
    """
    Design a 32x32 unit cell to maximize the TM band gap between bands 4 and 5.
    
    Strategy:
    1. Use a fast internal PWE solver with vectorized matrix construction.
    2. Search over parametric geometries: four-quadrant rods, central features,
       cross-shaped bars, and combinations.
    3. Coarse-to-fine optimization with time budget management.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    
    start_time = time.time()
    deadline = start_time + 210.0  # 20s margin before 230s kill

    def time_left():
        return deadline - time.time()

    # Fast PWE solver for TM modes
    def compute_tm_bands(cell, n_pw=15, n_kpts=20):
        """
        Compute TM band structure using plane-wave expansion.
        Returns omega array of shape (n_kpts, n_bands_needed).
        """
        cell = np.array(cell, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell
        inv_eps = 1.0 / eps
        N = size

        # Compute Fourier coefficients of 1/epsilon
        fft_coeffs = np.fft.fft2(inv_eps)

        def get_coeff(m, n):
            idx_m = m % N
            idx_n = n % N
            return fft_coeffs[idx_m, idx_n] / (N * N)

        # Build G vector list
        half = n_pw // 2
        m_vals = list(range(-half, half + 1))
        if len(m_vals) < n_pw:
            m_vals = m_vals[:n_pw]
        else:
            m_vals = m_vals[-n_pw:]

        g_indices = [(m, n) for m in m_vals for n in m_vals]
        n_total = len(g_indices)

        # Precompute S matrix: S[i,j] = 1/eps(G_i - G_j)
        S = np.zeros((n_total, n_total), dtype=float)
        for i, (mi, ni) in enumerate(g_indices):
            for j, (mj, nj) in enumerate(g_indices):
                S[i, j] = get_coeff(mi - mj, ni - nj).real

        # K-points in irreducible BZ (units of pi/a)
        # Irreducible wedge: 0 <= kx <= ky <= 1
        k_points = []
        n_seg = max(2, n_kpts // 3)
        # Gamma(0,0) to X(0,1): kx=0, ky from 0 to 1
        for t in np.linspace(0, 1, n_seg + 1):
            k_points.append((0.0, float(t)))
        # X(0,1) to M(1,1): ky=1, kx from 0 to 1
        for t in np.linspace(0, 1, n_seg + 1):
            k_points.append((float(t), 1.0))
        # M(1,1) to Gamma(0,0): kx=ky, from 1 to 0
        for t in np.linspace(1, 0, n_seg + 1):
            k_points.append((float(t), float(t)))

        # Remove duplicates
        k_points_unique = []
        for kp in k_points:
            if not any(abs(kp[0]-kp2[0]) < 1e-10 and abs(kp[1]-kp2[1]) < 1e-10 for kp2 in k_points_unique):
                k_points_unique.append(kp)

        n_kpts_actual = len(k_points_unique)
        n_bands_needed = lower_band + 1  # bands 1 through 5
        omega = np.zeros((n_kpts_actual, n_bands_needed))

        # Precompute |G_j + k|^2 for each k-point and G_j
        for ki, (kx, ky) in enumerate(k_points_unique):
            # A[i,j] = S[i,j] * |G_j + k|^2
            # B[i,j] = S[i,j]
            # G_j in units of pi/a is (2*m_j, 2*n_j)
            g_sq = np.zeros(n_total)
            for j, (mj, nj) in enumerate(g_indices):
                gm = 2.0 * mj + kx
                gn = 2.0 * nj + ky
                g_sq[j] = gm * gm + gn * gn

            A = S * g_sq[None, :]  # Broadcast: A[i,j] = S[i,j] * g_sq[j]
            B = S.copy()

            try:
                eigenvalues = eigh(A, B, eigvals_only=True)
                eigenvalues = np.sort(eigenvalues)
                omega[ki, :] = np.sqrt(np.maximum(eigenvalues[:n_bands_needed], 0))
            except Exception:
                omega[ki, :] = np.nan

        return omega

    def score_cell(cell, n_pw=13, n_kpts=15):
        """Score a cell by computing the gap-midgap ratio between bands 4 and 5."""
        try:
            omega = compute_tm_bands(cell, n_pw=n_pw, n_kpts=n_kpts)
            if np.any(np.isnan(omega)):
                return -np.inf
            max_omega_4 = np.max(omega[:, 3])
            min_omega_5 = np.min(omega[:, 4])
            gap = min_omega_5 - max_omega_4
            midgap = (min_omega_5 + max_omega_4) / 2.0
            if midgap > 0:
                return 2.0 * gap / midgap
            return -np.inf
        except Exception:
            return -np.inf

    # Cell construction functions
    def make_four_quadrant_cell(rod_radius, exponent=2.0):
        """Four rods centered in each quadrant."""
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        centers = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
        
        for i in range(size):
            for j in range(size):
                x, y = coords[i], coords[j]
                for (cx, cy) in centers:
                    dx, dy = x - cx, y - cy
                    if rod_radius > 0:
                        val = abs(dx / rod_radius) ** exponent + abs(dy / rod_radius) ** exponent
                    else:
                        val = 0
                    if val <= 1.0 + 1e-9:
                        cell[i, j] = 1
                        break
        return cell.tolist()

    def make_central_cell(r, n=2.0):
        """Central dielectric feature."""
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        for i in range(size):
            for j in range(size):
                x, y = coords[i], coords[j]
                if r > 0:
                    val = abs(x / r) ** n + abs(y / r) ** n
                    if val <= 1.0 + 1e-9:
                        cell[i, j] = 1
        return cell.tolist()

    def make_cross_cell(arm_width, arm_length=0.8):
        """Cross-shaped dielectric bars along axes."""
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        hw = arm_width / 2.0
        hl = arm_length / 2.0
        for i in range(size):
            for j in range(size):
                x, y = coords[i], coords[j]
                # Horizontal bar
                if abs(y) <= hw and abs(x) <= hl:
                    cell[i, j] = 1
                # Vertical bar
                elif abs(x) <= hw and abs(y) <= hl:
                    cell[i, j] = 1
        return cell.tolist()

    def make_combined_cell(rod_radius, rod_exp, cross_width, cross_len):
        """Combine four-quadrant rods with a cross."""
        cell1 = np.array(make_four_quadrant_cell(rod_radius, rod_exp), dtype=int)
        cell2 = np.array(make_cross_cell(cross_width, cross_len), dtype=int)
        cell = np.maximum(cell1, cell2)
        return cell.tolist()

    def make_ring_cell(inner_r, outer_r, n=2.0):
        """Annular ring feature."""
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        for i in range(size):
            for j in range(size):
                x, y = coords[i], coords[j]
                r = np.sqrt(x*x + y*y)
                if inner_r <= r <= outer_r:
                    cell[i, j] = 1
        return cell.tolist()

    # Phase 1: Coarse search over diverse geometries
    candidates = []

    # Four-quadrant circles
    for r in [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]:
        candidates.append(('fq_circle', (r, 2.0)))

    # Four-quadrant squircles
    for r in [0.08, 0.10, 0.12, 0.14, 0.16]:
        for n in [3.0, 4.0, 5.0]:
            candidates.append(('fq_squircle', (r, n)))

    # Central circles
    for r in [0.15, 0.18, 0.20, 0.22, 0.25]:
        candidates.append(('central', (r, 2.0)))

    # Cross shapes
    for w in [0.05, 0.08, 0.10, 0.12, 0.15]:
        for l in [0.6, 0.7, 0.8, 0.9]:
            candidates.append(('cross', (w, l)))

    # Rings
    for inner in [0.10, 0.15, 0.20]:
        for outer in [inner + 0.05, inner + 0.08, inner + 0.12]:
            if outer < 0.45:
                candidates.append(('ring', (inner, outer, 2.0)))

    # Combined: rods + cross
    for r in [0.08, 0.10, 0.12]:
        for w in [0.04, 0.06, 0.08]:
            candidates.append(('combined', (r, 2.0, w, 0.7)))

    best_score = -np.inf
    best_cell = candidates[0][1]
    best_type = candidates[0][0]

    n_pw_coarse = 13
    n_kpts_coarse = 12

    for ctype, params in candidates:
        if time_left() < 15:
            break
        try:
            if ctype == 'fq_circle':
                cell = make_four_quadrant_cell(params[0], params[1])
            elif ctype == 'fq_squircle':
                cell = make_four_quadrant_cell(params[0], params[1])
            elif ctype == 'central':
                cell = make_central_cell(params[0], params[1])
            elif ctype == 'cross':
                cell = make_cross_cell(params[0], params[1])
            elif ctype == 'ring':
                cell = make_ring_cell(params[0], params[1], params[2])
            elif ctype == 'combined':
                cell = make_combined_cell(params[0], params[1], params[2], params[3])
            else:
                continue

            s = score_cell(cell, n_pw=n_pw_coarse, n_kpts=n_kpts_coarse)
            if s > best_score:
                best_score = s
                best_cell = cell
                best_type = ctype
        except Exception:
            continue

    # Phase 2: Local refinement around best candidate
    if time_left() > 30 and best_cell is not None:
        # Determine which type to refine
        if best_type in ('fq_circle', 'fq_squircle'):
            # Refine rod radius and exponent
            base_r = None
            base_n = 2.0
            # Find the base parameters
            for ctype, params in candidates:
                if ctype == best_type:
                    if ctype == 'fq_circle':
                        base_r = params[0]
                        base_n = params[1]
                    elif ctype == 'fq_squircle':
                        base_r = params[0]
                        base_n = params[1]
                    break
            
            if base_r is None:
                base_r = 0.12
            
            # Local search in radius
            for dr in [-0.02, -0.01, 0.0, 0.01, 0.02]:
                if time_left() < 10:
                    break
                r = base_r + dr
                if r < 0.03 or r > 0.25:
                    continue
                cell = make_four_quadrant_cell(r, base_n)
                s = score_cell(cell, n_pw=n_pw_coarse, n_kpts=n_kpts_coarse)
                if s > best_score:
                    best_score = s
                    best_cell = cell

            # Local search in exponent
            for dn in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                if time_left() < 10:
                    break
                n = base_n + dn
                if n < 1.5 or n > 6.0:
                    continue
                cell = make_four_quadrant_cell(base_r, n)
                s = score_cell(cell, n_pw=n_pw_coarse, n_kpts=n_kpts_coarse)
                if s > best_score:
                    best_score = s
                    best_cell = cell

        elif best_type == 'cross':
            base_w = None
            base_l = None
            for ctype, params in candidates:
                if ctype == 'cross':
                    base_w = params[0]
                    base_l = params[1]
                    break
            
            if base_w is None:
                base_w, base_l = 0.08, 0.7

            for dw in [-0.02, -0.01, 0.0, 0.01, 0.02]:
                if time_left() < 10:
                    break
                w = base_w + dw
                if w < 0.02 or w > 0.20:
                    continue
                cell = make_cross_cell(w, base_l)
                s = score_cell(cell, n_pw=n_pw_coarse, n_kpts=n_kpts_coarse)
                if s > best_score:
                    best_score = s
                    best_cell = cell

            for dl in [-0.1, -0.05, 0.0, 0.05, 0.1]:
                if time_left() < 10:
                    break
                l = base_l + dl
                if l < 0.4 or l > 1.0:
                    continue
                cell = make_cross_cell(base_w, l)
                s = score_cell(cell, n_pw=n_pw_coarse, n_kpts=n_kpts_coarse)
                if s > best_score:
                    best_score = s
                    best_cell = cell

        elif best_type == 'combined':
            base_params = None
            for ctype, params in candidates:
                if ctype == 'combined':
                    base_params = params
                    break
            
            if base_params is None:
                base_params = (0.10, 2.0, 0.06, 0.7)

            # Vary rod radius
            for dr in [-0.02, -0.01, 0.0, 0.01, 0.02]:
                if time_left() < 10:
                    break
                r = base_params[0] + dr
                if r < 0.03 or r > 0.25:
                    continue
                cell = make_combined_cell(r, base_params[1], base_params[2], base_params[3])
                s = score_cell(cell, n_pw=n_pw_coarse, n_kpts=n_kpts_coarse)
                if s > best_score:
                    best_score = s
                    best_cell = cell

            # Vary cross width
            for dw in [-0.02, -0.01, 0.0, 0.01, 0.02]:
                if time_left() < 10:
                    break
                w = base_params[2] + dw
                if w < 0.02 or w > 0.15:
                    continue
                cell = make_combined_cell(base_params[0], base_params[1], w, base_params[3])
                s = score_cell(cell, n_pw=n_pw_coarse, n_kpts=n_kpts_coarse)
                if s > best_score:
                    best_score = s
                    best_cell = cell

    # Phase 3: If time remains, do a finer search with more k-points
    if time_left() > 40 and best_cell is not None:
        n_pw_fine = 15
        n_kpts_fine = 20
        
        # Re-evaluate best cell with finer settings
        s_fine = score_cell(best_cell, n_pw=n_pw_fine, n_kpts=n_kpts_fine)
        if s_fine > best_score:
            best_score = s_fine

        # Try a few nearby variations with fine evaluation
        if best_type in ('fq_circle', 'fq_squircle'):
            for r in [0.10, 0.11, 0.12, 0.13, 0.14, 0.15]:
                if time_left() < 15:
                    break
                cell = make_four_quadrant_cell(r, 2.0)
                s = score_cell(cell, n_pw=n_pw_fine, n_kpts=n_kpts_fine)
                if s > best_score:
                    best_score = s
                    best_cell = cell
        elif best_type == 'cross':
            for w in [0.06, 0.07, 0.08, 0.09, 0.10]:
                if time_left() < 15:
                    break
                cell = make_cross_cell(w, 0.75)
                s = score_cell(cell, n_pw=n_pw_fine, n_kpts=n_kpts_fine)
                if s > best_score:
                    best_score = s
                    best_cell = cell

    # Ensure D4 symmetry: the construction functions already produce symmetric cells
    # but let's verify and fix if needed
    best_arr = np.array(best_cell, dtype=int)
    # Check 180-degree rotation symmetry
    best_arr_rot = best_arr[::-1, ::-1]
    if not np.array_equal(best_arr, best_arr_rot):
        # Average to enforce symmetry
        best_arr = ((best_arr + best_arr_rot) / 2).astype(int)
    # Check transpose symmetry
    best_arr_t = best_arr.T
    if not np.array_equal(best_arr, best_arr_t):
        best_arr = ((best_arr + best_arr_t) / 2).astype(int)

    return best_arr.tolist()

# EVOLVE-BLOCK-END
