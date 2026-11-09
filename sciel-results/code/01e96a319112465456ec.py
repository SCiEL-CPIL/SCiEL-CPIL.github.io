"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    Strategy:
    1. Represent the dielectric field using a C4v-symmetric trigonometric basis
       (sin/cos terms that respect the square symmetry group).
    2. Optimize the continuous field parameters using simulated annealing.
    3. Use a full 289-plane-wave evaluator matching the scoring method.
    4. Project to discrete pixels and refine with local orbit-flip search.
    """
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin for 200s limit

    # --- 1. Precompute plane-wave setup (289 PWs matching evaluator) ---
    N_pw = 17
    half = N_pw // 2  # 8
    G_vectors = []
    for gx in range(-half, half + 1):
        for gy in range(-half, half + 1):
            if gx * gx + gy * gy <= (half + 0.5) ** 2:  # |G| <= 8.5
                G_vectors.append((gx, gy))
    G_vectors = np.array(G_vectors, dtype=float)
    n_pw = len(G_vectors)
    
    # Precompute 30 k-points in irreducible BZ
    k_points = []
    for i in range(11):
        k_points.append((i * 0.5 / 10.0, 0.0))
    for i in range(1, 11):
        k_points.append((0.5, i * 0.5 / 10.0))
    for i in range(1, 11):
        t = i / 10.0
        k_points.append((0.5 * (1 - t), 0.5 * (1 - t)))
    unique_k = []
    for k in k_points:
        if not any(abs(k[0] - uk[0]) < 1e-9 and abs(k[1] - uk[1]) < 1e-9 for uk in unique_k):
            unique_k.append(k)
    k_points = np.array(unique_k[:30], dtype=float)
    
    # Precompute coefficient index matrix
    coeff_lookup = {}
    for idx, (gx, gy) in enumerate(G_vectors):
        coeff_lookup[(int(gx), int(gy))] = idx
        
    coeff_idx_mat = np.full((n_pw, n_pw), -1, dtype=int)
    for i in range(n_pw):
        for j in range(n_pw):
            dx = int(G_vectors[i, 0] - G_vectors[j, 0])
            dy = int(G_vectors[i, 1] - G_vectors[j, 1])
            key = (dx, dy)
            if key in coeff_lookup:
                coeff_idx_mat[i, j] = coeff_lookup[key]
                
    # Precompute cross product matrices for each k-point
    cross_mats = []
    for kx, ky in k_points:
        kG_x = kx + G_vectors[:, 0]
        kG_y = ky + G_vectors[:, 1]
        cross = np.outer(kG_x, kG_y) - np.outer(kG_y, kG_x)
        cross_mats.append(cross)
    
    # Precompute valid mask for coeff_idx_mat
    valid_mask = coeff_idx_mat >= 0
    valid_indices = np.where(valid_mask)

    def compute_inv_eps_fourier(cell_arr):
        """Compute Fourier coefficients of 1/eps for the given cell."""
        eps = np.where(cell_arr == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        fft_inv = np.fft.fft2(inv_eps)
        c = fft_inv / (size * size)
        coeffs = np.zeros(n_pw, dtype=complex)
        for idx, (gx, gy) in enumerate(G_vectors):
            ix = int(gx) % size
            iy = int(gy) % size
            coeffs[idx] = c[iy, ix]
        return np.real(coeffs)

    def score_from_coeffs(coeffs):
        """Compute gap-midgap ratio directly from Fourier coefficients."""
        max_w4 = 0.0
        min_w5 = float('inf')
        
        for idx_k in range(len(k_points)):
            cross = cross_mats[idx_k]
            C = np.zeros((n_pw, n_pw), dtype=float)
            C[valid_indices] = coeffs[coeff_idx_mat[valid_indices]]
            
            M = C * cross
            M = (M + M.T) / 2.0
            
            eigvals = np.linalg.eigvalsh(M)
            
            if len(eigvals) >= 5:
                w4 = math.sqrt(max(0.0, eigvals[3]))
                w5 = math.sqrt(max(0.0, eigvals[4]))
                if w4 > max_w4:
                    max_w4 = w4
                if w5 < min_w5:
                    min_w5 = w5
        
        if min_w5 <= 0 or max_w4 <= 0:
            return -1.0
        return 2.0 * (min_w5 - max_w4) / (min_w5 + max_w4)

    def score_cell(cell_arr):
        """Compute the gap-midgap ratio for a discrete cell."""
        coeffs = compute_inv_eps_fourier(cell_arr)
        return score_from_coeffs(coeffs)

    # --- 2. C4v-symmetric trigonometric parameterization ---
    # We represent the continuous field f(x,y) as:
    # f(x,y) = a0 + sum_{m,n} a_{mn} * [cos(2*pi*m*x)*cos(2*pi*n*y) + cos(2*pi*n*x)*cos(2*pi*m*y)]
    # with m,n >= 1, and a_{mn} = a_{nm}
    # Plus diagonal terms: a_{mm} * cos(2*pi*m*x)*cos(2*pi*m*y)
    # 
    # For C4v symmetry, we need:
    # - Even in x and y (symmetric about center)
    # - f(x,y) = f(y,x)
    # 
    # Using cos terms automatically gives even symmetry about x=0, but we need
    # symmetry about x=0.5. So we use cos(2*pi*m*(x-0.5)) = cos(2*pi*m*x - pi*m)
    # which is (-1)^m * cos(2*pi*m*x). For even m, this is cos(2*pi*m*x).
    # 
    # Actually, let's use a simpler approach: define the field on the grid and
    # enforce symmetry by construction.
    
    # Grid coordinates (pixel centers)
    x_coords = np.array([(i + 0.5) / size for i in range(size)])
    y_coords = np.array([(j + 0.5) / size for j in range(size)])
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    # Define basis functions with C4v symmetry
    # Basis: cos(2*pi*m*x)*cos(2*pi*n*y) + cos(2*pi*n*x)*cos(2*pi*m*y) for m<n
    #         cos(2*pi*m*x)*cos(2*pi*m*y) for m=n
    # With m,n in {1, 2, 3, 4} (max frequency 4 for good resolution)
    
    max_freq = 4
    param_pairs = []
    for m in range(1, max_freq + 1):
        for n in range(m, max_freq + 1):
            param_pairs.append((m, n))
    num_params = len(param_pairs)
    
    # Precompute basis functions
    basis_funcs = []
    for m, n in param_pairs:
        cos_mx = np.cos(2 * np.pi * m * X)
        cos_ny = np.cos(2 * np.pi * n * Y)
        cos_nx = np.cos(2 * np.pi * n * X)
        cos_my = np.cos(2 * np.pi * m * Y)
        
        if m == n:
            bf = cos_mx * cos_ny
        else:
            bf = cos_mx * cos_ny + cos_nx * cos_my
        
        # Enforce C4v: average over symmetry operations
        # The above construction already ensures f(x,y) = f(y,x)
        # and even symmetry about 0, which combined with the cos terms gives
        # the right symmetry about 0.5 for even frequencies
        basis_funcs.append(bf)
    
    # Also add a constant term (DC component)
    # Total parameters: 1 (DC) + num_params (cos terms)
    total_params = 1 + num_params
    
    def params_to_field(params):
        """Convert parameter vector to a 32x32 field."""
        field = np.zeros((size, size), dtype=float)
        field += params[0]  # DC component
        for idx, bf in enumerate(basis_funcs):
            field += params[idx + 1] * bf
        return field
    
    def field_to_cell(field, threshold=None):
        """Convert continuous field to discrete 0/1 cell with C4v symmetry."""
        if threshold is None:
            threshold = np.mean(field)
        
        # Threshold to binary
        cell = (field > threshold).astype(int)
        
        # Enforce C4v symmetry by averaging over symmetry group
        sym_cell = np.zeros_like(cell)
        for i in range(size):
            for j in range(size):
                vals = [
                    cell[i, j],
                    cell[j, i],
                    cell[size-1-i, size-1-j],
                    cell[size-1-j, size-1-i]
                ]
                avg_val = sum(vals) / len(vals)
                sym_cell[i, j] = 1 if avg_val > 0.5 else 0
        
        return sym_cell

    # --- 3. Simulated annealing optimization ---
    rng = np.random.default_rng(12345)
    
    # Initial parameters: start with a moderate dielectric pattern
    # DC component: choose so that ~50% is dielectric
    # The mean of cos terms is 0, so field ≈ params[0]
    # We want threshold at mean, so params[0] should be around 0
    # The amplitude of cos terms determines contrast
    
    init_params = np.zeros(total_params)
    init_params[0] = 0.0  # DC
    # Initialize with small random values
    for i in range(1, total_params):
        init_params[i] = rng.normal(0, 0.5)
    
    # Simulated annealing
    best_params = init_params.copy()
    best_score = -1.0
    current_params = init_params.copy()
    current_score = -1.0
    
    # Evaluate initial
    field = params_to_field(current_params)
    cell = field_to_cell(field)
    current_score = score_cell(cell)
    if current_score > best_score:
        best_score = current_score
        best_params = current_params.copy()
    
    # Annealing schedule
    T_start = 1.0
    T_end = 0.01
    n_iter = 300  # Number of annealing iterations
    
    for iter_num in range(n_iter):
        if time.time() > deadline - 30.0:
            break
        
        # Temperature schedule (geometric cooling)
        T = T_start * (T_end / T_start) ** (iter_num / n_iter)
        
        # Propose new parameters
        step_size = 0.3 * T + 0.05
        new_params = current_params + rng.normal(0, step_size, size=total_params)
        
        # Evaluate
        field = params_to_field(new_params)
        cell = field_to_cell(field)
        new_score = score_cell(cell)
        
        # Acceptance criterion
        if new_score > current_score:
            current_params = new_params
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_params = new_params.copy()
        else:
            # Metropolis acceptance
            delta = current_score - new_score
            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
                current_params = new_params
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_params = new_params.copy()
    
    # Get best discrete cell
    best_field = params_to_field(best_params)
    best_cell = field_to_cell(best_field)
    best_score = score_cell(best_cell)
    
    # --- 4. Local search refinement on discrete pixels ---
    if time.time() < deadline - 5.0:
        # Precompute symmetry orbits
        orbits = []
        visited = np.zeros((size, size), dtype=bool)
        for i in range(size):
            for j in range(size):
                if not visited[i, j]:
                    orbit = [(i, j), (j, i), (size-1-i, size-1-j), (size-1-j, size-1-i)]
                    unique_orbit = []
                    for p in orbit:
                        if not visited[p[0], p[1]]:
                            unique_orbit.append(p)
                            visited[p[0], p[1]] = True
                    if unique_orbit:
                        orbits.append(unique_orbit)
        
        cell_arr = best_cell.copy()
        local_best_score = best_score
        local_best_cell = best_cell.copy()
        
        # Hill climbing with orbit flips
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < deadline - 2.0 and no_improve_count < max_no_improve:
            # Randomly choose 1-2 orbits to flip
            n_flips = rng.integers(1, 3)
            flipped_orbits = []
            for _ in range(n_flips):
                idx = rng.integers(0, len(orbits))
                orbit = orbits[idx]
                flipped_orbits.append(orbit)
                for (i, j) in orbit:
                    cell_arr[i, j] = 1 - cell_arr[i, j]
            
            new_score = score_cell(cell_arr)
            
            if new_score > local_best_score:
                local_best_score = new_score
                local_best_cell = cell_arr.copy()
                no_improve_count = 0
            else:
                # Revert
                for orbit in flipped_orbits:
                    for (i, j) in orbit:
                        cell_arr[i, j] = 1 - cell_arr[i, j]
                no_improve_count += 1
        
        if local_best_score > best_score:
            best_score = local_best_score
            best_cell = local_best_cell

    return [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]
# EVOLVE-BLOCK-END
