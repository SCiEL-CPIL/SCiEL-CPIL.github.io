import numpy as np
import time
import math

def design(problem):
    """
    Direct binary pixel-level optimization for TM band gap between bands 4 and 5.
    Uses C4v symmetry (center-symmetric + transpose), fast 7x7 PW evaluator,
    and simulated annealing to explore the full binary design space.
    """
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 230s kill

    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)

    # C4v symmetry: cell[i][j] == cell[size-1-i][size-1-j] (center symmetry)
    # and cell[i][j] == cell[j][i] (transpose symmetry)
    # Independent pixels: those in the upper-left "fundamental domain"
    # For a 32x32 grid with center at (15.5, 15.5):
    # Fundamental domain: i <= 15 and j <= 15 (upper-left quadrant including center lines)
    # Actually for even size 32, center is between pixels.
    # cell[i][j] = cell[31-i][31-j] and cell[i][j] = cell[j][i]
    # So we only need to define pixels where i <= j and i + j < 32 (or some other rule)
    # Simpler: define upper-left triangle where i <= j, then mirror.
    # But we also need center symmetry: cell[i][j] = cell[31-i][31-j]
    # Combined: cell[i][j] = cell[j][i] = cell[31-i][31-j] = cell[31-j][31-i]
    # Fundamental domain: i in [0,15], j in [0,15], with i <= j
    # That gives 16*17/2 = 136 independent pixels.

    def get_independent_indices():
        """Get indices of independent pixels in the fundamental domain."""
        indices = []
        for i in range(size // 2):  # 0 to 15
            for j in range(i, size // 2):  # i to 15
                indices.append((i, j))
        return indices

    indep_indices = get_independent_indices()
    num_indep = len(indep_indices)

    def expand_cell(binary_indep):
        """Expand independent pixels to full 32x32 cell with C4v symmetry."""
        cell = np.zeros((size, size), dtype=int)
        for k, (i, j) in enumerate(indep_indices):
            val = binary_indep[k]
            # Center symmetry: (i,j), (31-i,31-j), (j,i), (31-j,31-i)
            cell[i, j] = val
            cell[size - 1 - i, size - 1 - j] = val
            cell[j, i] = val
            cell[size - 1 - j, size - 1 - i] = val
        return cell

    def cell_to_eps(cell):
        """Convert binary cell to epsilon grid."""
        return np.where(cell == 1, eps_high, eps_low)

    # Fast plane-wave evaluator with 7x7 = 49 PWs
    N_pw = 3  # -3 to 3, giving 7x7 = 49 PWs
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx_grid, Gy_grid = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx_grid.ravel(), Gy_grid.ravel()))
    num_pw = len(G_vectors)  # 49

    # Precompute G differences for the coefficient lookup
    # We need (1/eps)_{G-G'} for G-G' in [-6,6]x[-6,6]

    # K points in irreducible BZ (1/8 of square BZ)
    # Triangle: Gamma(0,0) -> M(0.5,0) -> X(0.5,0.5) -> Gamma
    # Use ~20 k-points for speed
    k_points = []
    # Along Gamma-M: (t, 0)
    for t in np.linspace(0, 0.5, 7):
        k_points.append((t, 0))
    # Along M-X: (0.5, t)
    for t in np.linspace(0, 0.5, 7):
        k_points.append((0.5, t))
    # Along Gamma-X: (t, t)
    for t in np.linspace(0, 0.5, 7):
        k_points.append((t, t))
    # Remove duplicates
    unique_k = []
    seen = set()
    for k in k_points:
        key = (round(k[0], 10), round(k[1], 10))
        if key not in seen:
            seen.add(key)
            unique_k.append(k)

    def evaluate_cell_fast(cell):
        """Fast evaluation using 49 PWs and ~20 k-points."""
        eps_grid = cell_to_eps(cell)
        inv_eps = 1.0 / eps_grid

        # Compute Fourier coefficients of 1/eps via FFT
        fft_inv_eps = np.fft.fft2(inv_eps)
        fft_inv_eps /= (size * size)

        # Build coefficient dictionary for G-G' differences
        coeff = {}
        for dx in range(-2 * N_pw, 2 * N_pw + 1):
            for dy in range(-2 * N_pw, 2 * N_pw + 1):
                ix = dx % size
                iy = dy % size
                coeff[(dx, dy)] = fft_inv_eps[ix, iy]

        min_omega_5 = np.inf
        max_omega_4 = -np.inf

        for kx, ky in unique_k:
            k = np.array([kx, ky])

            # Build matrix M[i,j] = (1/eps)_{Gi-Gj} * |k+Gi|^2
            M = np.zeros((num_pw, num_pw), dtype=complex)
            for i in range(num_pw):
                Gi = G_vectors[i]
                kGi = k + Gi
                w2_i = np.dot(kGi, kGi)
                for j in range(num_pw):
                    Gj = G_vectors[j]
                    dx = int(Gi[0] - Gj[0])
                    dy = int(Gi[1] - Gj[1])
                    c = coeff.get((dx, dy), 0)
                    M[i, j] = c * w2_i

            eigvals = np.linalg.eigvals(M)
            real_ev = np.sort(np.real(eigvals))

            if len(real_ev) >= 5:
                omega_4 = math.sqrt(max(0, real_ev[3]))
                omega_5 = math.sqrt(max(0, real_ev[4]))
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)

        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0

        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap < 1e-10:
            return -1.0
        return 2.0 * gap / midgap

    # Seed: a superlattice pattern known to produce good band gaps
    # Start with a pattern of alternating dielectric/air in a structured way
    # Try a "hole" pattern: dielectric background with air holes
    def make_seed_cell():
        """Create an initial cell with a structured pattern."""
        cell = np.ones((size, size), dtype=int)  # Start with all dielectric
        # Add air holes in a periodic pattern
        # 4 holes per unit cell in a square arrangement
        hole_positions = [(8, 8), (24, 8), (8, 24), (24, 24)]
        hole_radius = 3
        for hx, hy in hole_positions:
            for i in range(size):
                for j in range(size):
                    if (i - hx) ** 2 + (j - hy) ** 2 < hole_radius ** 2:
                        cell[i, j] = 0
        # Enforce C4v symmetry
        cell = np.maximum(cell, cell[::-1, ::-1])  # center symmetry
        cell = np.maximum(cell, cell.T)  # transpose
        # Actually need to enforce properly
        for i in range(size):
            for j in range(size):
                avg = (cell[i, j] + cell[31-i, 31-j] + cell[j, i] + cell[31-j, 31-i]) / 4.0
                cell[i, j] = 1 if avg > 0.5 else 0
        return cell

    def cell_to_indep(cell):
        """Extract independent pixels from full cell."""
        indep = np.zeros(num_indep, dtype=int)
        for k, (i, j) in enumerate(indep_indices):
            indep[k] = cell[i, j]
        return indep

    seed_cell = make_seed_cell()
    current_indep = cell_to_indep(seed_cell)
    current_cell = expand_cell(current_indep)
    current_score = evaluate_cell_fast(current_cell)

    best_indep = current_indep.copy()
    best_cell = current_cell.copy()
    best_score = current_score

    # Simulated annealing
    T_start = 2.0
    T_end = 0.01
    num_steps = 5000
    # Adaptive: run as many steps as time allows

    rng = np.random.default_rng(42)

    for step in range(num_steps):
        if time.time() > deadline - 10:
            break

        # Temperature schedule
        frac = step / num_steps
        T = T_start * (T_end / T_start) ** frac

        # Random flip: pick 1-3 random independent pixels and flip them
        num_flips = rng.integers(1, 4)
        flip_indices = rng.choice(num_indep, size=num_flips, replace=False)

        new_indep = current_indep.copy()
        for idx in flip_indices:
            new_indep[idx] = 1 - new_indep[idx]

        new_cell = expand_cell(new_indep)
        new_score = evaluate_cell_fast(new_cell)

        delta = new_score - current_score

        if delta > 0 or rng.random() < math.exp(delta / max(T, 1e-10)):
            current_indep = new_indep
            current_cell = new_cell
            current_score = new_score

            if new_score > best_score:
                best_score = new_score
                best_indep = new_indep.copy()
                best_cell = new_cell.copy()

    # Return the best cell found
    final_cell = best_cell
    # Ensure strict symmetry
    for i in range(size):
        for j in range(size):
            avg = (final_cell[i, j] + final_cell[31-i, 31-j] + final_cell[j, i] + final_cell[31-j, 31-i]) / 4.0
            final_cell[i, j] = 1 if avg > 0.5 else 0

    # Convert to nested list
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(final_cell[i, j]))
        result.append(row)

    return result
