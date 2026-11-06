"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def _make_symmetric_cell(size, pattern_fn):
    """Create a symmetric 32x32 cell using the symmetry group of the square."""
    # We need cell == cell[::-1] row-wise, cell == cell[::-1] column-wise, cell == cell.T
    # This means 4-fold rotational symmetry + reflections
    # For a 32x32 cell, we can define it on the fundamental domain and mirror
    # The center is at (15.5, 15.5)
    cell = [[0] * size for _ in range(size)]
    # For each pixel in the top-left quadrant (0..15, 0..15), mirror to all 4 quadrants
    # with appropriate reflections
    for i in range(size // 2):
        for j in range(size // 2):
            val = pattern_fn(i, j, size)
            # Top-left
            cell[i][j] = val
            # Top-right: reflect column
            cell[i][size - 1 - j] = val
            # Bottom-left: reflect row
            cell[size - 1 - i][j] = val
            # Bottom-right: reflect both
            cell[size - 1 - i][size - 1 - j] = val
    return cell

def _fourier_coefficients(cell, size, max_order=16):
    """Compute Fourier coefficients of the pixel map up to max_order."""
    N = max_order * 2 + 1
    coeffs = np.zeros((N, N), dtype=complex)
    for gx in range(-max_order, max_order + 1):
        for gy in range(-max_order, max_order + 1):
            s = 0.0
            for i in range(size):
                x = (i + 0.5) / size
                phase_x = math.exp(2j * math.pi * gx * x)
                for j in range(size):
                    y = (j + 0.5) / size
                    phase_y = math.exp(2j * math.pi * gy * y)
                    s += cell[i][j] * phase_x * phase_y
            coeffs[gx + max_order, gy + max_order] = s / (size * size)
    return coeffs

def _compute_1over_eps_fourier(cell, size, eps_high, max_order=16):
    """Compute Fourier coefficients of 1/epsilon(x,y)."""
    # First compute 1/eps at each pixel
    inv_eps = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            eps = eps_high if cell[i][j] == 1 else 1.0
            inv_eps[i, j] = 1.0 / eps
    
    # Now compute Fourier coefficients
    N = max_order * 2 + 1
    coeffs = np.zeros((N, N), dtype=complex)
    for gx in range(-max_order, max_order + 1):
        for gy in range(-max_order, max_order + 1):
            s = 0.0
            for i in range(size):
                x = (i + 0.5) / size
                phase_x = math.exp(2j * math.pi * gx * x)
                for j in range(size):
                    y = (j + 0.5) / size
                    phase_y = math.exp(2j * math.pi * gy * y)
                    s += inv_eps[i, j] * phase_x * phase_y
            coeffs[gx + max_order, gy + max_order] = s / (size * size)
    return coeffs

def _band_structure_tm(g_inv_eps, size, eps_high, kx, ky, max_order=16, n_pw=289):
    """Compute TM band structure using plane-wave Galerkin method."""
    # Plane waves: G = (2*pi/size) * (mx, my) for mx,my in range
    # We need 289 plane waves total. Use a circular cutoff.
    # 289 = 17^2, so use mx, my in [-8, 8] (17 values each)
    max_gx = 8
    max_gy = 8
    G_vectors = []
    for mx in range(-max_gx, max_gx + 1):
        for my in range(-max_gy, max_gy + 1):
            G_vectors.append((mx, my))
    # Sort by |G|^2
    G_vectors.sort(key=lambda v: v[0]**2 + v[1]**2)
    n_pw = min(len(G_vectors), n_pw)
    G_vectors = G_vectors[:n_pw]
    
    N_g = len(G_vectors)
    g_inv = g_inv_eps
    
    # Build Hamiltonian matrix
    # H_{G', G} = (K+G')^2 * g_{G-G'}
    # where g_{G-G'} is the Fourier coefficient of 1/eps at G-G'
    
    H = np.zeros((N_g, N_g), dtype=complex)
    for i, (mx_i, my_i) in enumerate(G_vectors):
        Gx_i = 2.0 * math.pi * mx_i / size
        Gy_i = 2.0 * math.pi * my_i / size
        Kx_i = kx + Gx_i
        Ky_i = ky + Gy_i
        K_sq_i = Kx_i**2 + Ky_i**2
        for j, (mx_j, my_j) in enumerate(G_vectors):
            # G - G' = (mx_i - mx_j, my_i - my_j)
            dx = mx_i - mx_j
            dy = my_i - my_j
            # Check if this is within our Fourier coefficient range
            if abs(dx) <= max_order and abs(dy) <= max_order:
                g_val = g_inv[dx + max_order, dy + max_order]
            else:
                g_val = 0.0
            H[i, j] = K_sq_i * g_val
    
    # Solve eigenvalue problem
    # H is not necessarily Hermitian, but eigenvalues should be real
    try:
        evals = np.linalg.eigvals(H)
        evals = np.sort(np.real(evals))
        # Filter out negative or zero eigenvalues
        evals = evals[evals > 1e-10]
        if len(evals) >= 5:
            return evals[3], evals[4]  # 4th and 5th bands (0-indexed 3, 4)
    except:
        pass
    return None, None

def _evaluate_cell(cell, size, eps_high, max_order=16, n_k_points=30):
    """Evaluate the gap-midgap ratio for a given cell."""
    g_inv_eps = _compute_1over_eps_fourier(cell, size, eps_high, max_order)
    
    # Irreducible Brillouin zone for square lattice
    # BZ: kx, ky in [-pi/size, pi/size]
    # Irreducible wedge: 0 <= ky <= kx, 0 <= kx <= pi/size
    # Plus some interior points
    
    kx_max = math.pi / size
    ky_max = math.pi / size
    
    # Generate 30 k-points in the irreducible BZ
    k_points = []
    # Boundary: ky = 0, kx from 0 to pi/size
    n_boundary = 15
    for i in range(n_boundary):
        kx = kx_max * i / (n_boundary - 1)
        k_points.append((kx, 0.0))
    # Boundary: ky = kx, from (0,0) to (pi/size, pi/size)
    for i in range(n_boundary):
        kx = kx_max * i / (n_boundary - 1)
        k_points.append((kx, kx))
    
    min_omega_5 = float('inf')
    max_omega_4 = float('-inf')
    
    for kx, ky in k_points:
        w4, w5 = _band_structure_tm(g_inv_eps, size, eps_high, kx, ky, max_order)
        if w4 is not None and w5 is not None:
            if w5 < min_omega_5:
                min_omega_5 = w5
            if w4 > max_omega_4:
                max_omega_4 = w4
    
    if min_omega_5 == float('inf') or max_omega_4 == float('-inf'):
        return -1.0
    
    gap = min_omega_5 - max_omega_4
    mid = (min_omega_5 + max_omega_4) / 2.0
    
    if mid == 0:
        return 0.0
    
    return gap / mid

def design(problem):
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    
    # Start with a four-rod pattern at (±a/4, ±a/4)
    # This enhances second-order Fourier components
    def four_rod_pattern(i, j, sz):
        # Positions at (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)
        # In pixel coordinates, center of these positions
        centers = [
            (0.25 * sz, 0.25 * sz),
            (0.75 * sz, 0.25 * sz),
            (0.25 * sz, 0.75 * sz),
            (0.75 * sz, 0.75 * sz)
        ]
        radius = 0.15 * sz  # radius in pixels
        for cx, cy in centers:
            dx = i + 0.5 - cx
            dy = j + 0.5 - cy
            if dx * dx + dy * dy < radius * radius:
                return 1
        return 0
    
    cell = _make_symmetric_cell(size, four_rod_pattern)
    
    # Evaluate to get baseline score
    start_time = time.time()
    deadline = start_time + 100  # 100 second budget for evaluation
    
    # Try a few variations
    best_cell = cell
    best_score = _evaluate_cell(cell, size, eps_high)
    
    # Try different radii
    radii = [0.12, 0.15, 0.18, 0.20, 0.22]
    for r in radii:
        if time.time() > deadline - 30:
            break
        
        def pattern(i, j, sz, rad=r):
            centers = [
                (0.25 * sz, 0.25 * sz),
                (0.75 * sz, 0.25 * sz),
                (0.25 * sz, 0.75 * sz),
                (0.75 * sz, 0.75 * sz)
            ]
            for cx, cy in centers:
                dx = i + 0.5 - cx
                dy = j + 0.5 - cy
                if dx * dx + dy * dy < (rad * sz) ** 2:
                    return 1
            return 0
        
        test_cell = _make_symmetric_cell(size, pattern)
        score = _evaluate_cell(test_cell, size, eps_high)
        if score > best_score:
            best_score = score
            best_cell = test_cell
    
    return best_cell
# EVOLVE-BLOCK-END
