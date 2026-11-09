"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def design(problem):
    """Physics-driven inverse design: optimize discrete Fourier coefficients of eps to maximize TM band-4/5 gap."""
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12.0)
    eps_low = problem.get("eps_low", 1.0)
    deadline = time.time() + 115.0  # safety margin before 120s CPU budget
    
    # Precompute plane-wave vectors (289 = 17x17 grid, |G| <= 8.5)
    N = 17
    half = N // 2  # 8
    G_vectors = []
    for gx in range(-half, half + 1):
        for gy in range(-half, half + 1):
            if gx * gx + gy * gy <= (half + 0.5) ** 2:  # |G| <= 8.5
                G_vectors.append((gx, gy))
    G_vectors = np.array(G_vectors, dtype=float)
    n_pw = len(G_vectors)
    
    # Precompute 30 k-points in irreducible BZ
    k_points = []
    # Along Gamma-X: kx from 0 to 0.5, ky=0
    for i in range(11):
        k_points.append((i * 0.5 / 10.0, 0.0))
    # Along X-M: kx=0.5, ky from 0 to 0.5
    for i in range(1, 11):
        k_points.append((0.5, i * 0.5 / 10.0))
    # Along M-Gamma: kx from 0.5 to 0, ky from 0.5 to 0
    for i in range(1, 11):
        t = i / 10.0
        k_points.append((0.5 * (1 - t), 0.5 * (1 - t)))
    # Remove duplicates
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
                
    # Precompute cross product matrices for each k-point to save time
    cross_mats = []
    for kx, ky in k_points:
        kG_x = kx + G_vectors[:, 0]
        kG_y = ky + G_vectors[:, 1]
        cross = np.outer(kG_x, kG_y) - np.outer(kG_y, kG_x)
        cross_mats.append(cross)

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

    def score_cell(cell_arr):
        """Compute the gap-midgap ratio."""
        coeffs = compute_inv_eps_fourier(cell_arr)
        
        max_w4 = 0.0
        min_w5 = float('inf')
        
        for idx_k, cross in enumerate(cross_mats):
            # Build coefficient matrix using precomputed indices
            C = np.zeros((n_pw, n_pw), dtype=float)
            valid = coeff_idx_mat >= 0
            C[valid] = coeffs[coeff_idx_mat[valid]]
            
            M = C * cross
            M = (M + M.T) / 2.0
            
            eigvals = np.linalg.eigvalsh(M)
            
            if len(eigvals) >= 5:
                w4 = math.sqrt(max(0, eigvals[3]))
                w5 = math.sqrt(max(0, eigvals[4]))
                if w4 > max_w4: max_w4 = w4
                if w5 < min_w5: min_w5 = w5
        
        if min_w5 <= 0:
            return -1.0
        return 2.0 * (min_w5 - max_w4) / (min_w5 + max_w4)

    # Precompute symmetry orbits for discrete constraint handling
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
    
    rng = np.random.default_rng(42)
    
    # Strategy: Optimize in the continuous space of Fourier coefficients, then project to discrete pixels
    # We'll use a hybrid approach: 
    # 1. Start with a good discrete seed
    # 2. Use gradient-based optimization on the continuous Fourier coefficients
    # 3. Project back to discrete pixels and refine with local search
    
    # Define seed patterns
    def make_circle(radius, inverted=False):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            for j in range(size):
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < radius * radius:
                    cell[i, j] = 1 if not inverted else 0
                    if inverted: cell[i, j] = 0
                    else: cell[i, j] = 1
        return cell

    def make_square(half_width, inverted=False):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            for j in range(size):
                y = (j + 0.5) / size - 0.5
                if abs(x) < half_width and abs(y) < half_width:
                    cell[i, j] = 1 if not inverted else 0
                    if inverted: cell[i, j] = 0
                    else: cell[i, j] = 1
        return cell

    def make_checkerboard(block_size=8):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                if (i // block_size + j // block_size) % 2 == 0:
                    cell[i, j] = 1
        return cell

    def make_plus(width=4):
        cell = np.zeros((size, size), dtype=int)
        center = size // 2
        for i in range(size):
            for j in range(size):
                if abs(i - center + 0.5) < width or abs(j - center + 0.5) < width:
                    cell[i, j] = 1
        return cell

    def make_ring(inner_r, outer_r, inverted=False):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            for j in range(size):
                y = (j + 0.5) / size - 0.5
                r = math.sqrt(x*x + y*y)
                if inner_r < r < outer_r:
                    cell[i, j] = 1 if not inverted else 0
                    if inverted: cell[i, j] = 0
                    else: cell[i, j] = 1
        return cell

    seeds = [
        make_circle(0.2, inverted=False),
        make_circle(0.2, inverted=True),
        make_square(0.2, inverted=False),
        make_square(0.2, inverted=True),
        make_checkerboard(8),
        make_plus(4),
        make_ring(0.1, 0.3, inverted=False),
        make_ring(0.1, 0.3, inverted=True)
    ]
    
    best_score = -1.0
    best_cell = seeds[0].copy()
    
    # Phase 1: Evaluate seeds and find the best one
    for seed in seeds:
        if time.time() > deadline - 30.0: break
        s = score_cell(seed)
        if s > best_score:
            best_score = s
            best_cell = seed.copy()
    
    # Phase 2: Gradient-based optimization in Fourier space
    # We'll optimize the Fourier coefficients of 1/eps directly
    # The constraint is that the inverse Fourier transform should be close to 1/eps_high or 1/eps_low
    
    # Get initial Fourier coefficients from best seed
    initial_coeffs = compute_inv_eps_fourier(best_cell)
    
    # Define the optimization variables: Fourier coefficients
    # We'll use L-BFGS-B or similar to maximize the gap
    # Since scipy.optimize might be slow, we'll implement a simple gradient ascent
    
    # Compute gradient of the score with respect to Fourier coefficients
    # This requires automatic differentiation or finite differences
    # For efficiency, we'll use finite differences with a small step
    
    def compute_score_and_grad(coeffs):
        """Compute score and gradient using finite differences."""
        base_score = score_cell_from_coeffs(coeffs)
        
        # Compute gradient using central differences
        n_coeffs = len(coeffs)
        grad = np.zeros(n_coeffs)
        step = 0.01
        
        for i in range(n_coeffs):
            if time.time() > deadline - 20.0:
                break
            
            coeffs_plus = coeffs.copy()
            coeffs_minus = coeffs.copy()
            coeffs_plus[i] += step
            coeffs_minus[i] -= step
            
            score_plus = score_cell_from_coeffs(coeffs_plus)
            score_minus = score_cell_from_coeffs(coeffs_minus)
            
            grad[i] = (score_plus - score_minus) / (2 * step)
        
        return base_score, grad
    
    def score_cell_from_coeffs(coeffs):
        """Compute score directly from Fourier coefficients."""
        max_w4 = 0.0
        min_w5 = float('inf')
        
        for idx_k, cross in enumerate(cross_mats):
            # Build coefficient matrix using precomputed indices
            C = np.zeros((n_pw, n_pw), dtype=float)
            valid = coeff_idx_mat >= 0
            C[valid] = coeffs[coeff_idx_mat[valid]]
            
            M = C * cross
            M = (M + M.T) / 2.0
            
            eigvals = np.linalg.eigvalsh(M)
            
            if len(eigvals) >= 5:
                w4 = math.sqrt(max(0, eigvals[3]))
                w5 = math.sqrt(max(0, eigvals[4]))
                if w4 > max_w4: max_w4 = w4
                if w5 < min_w5: min_w5 = w5
        
        if min_w5 <= 0:
            return -1.0
        return 2.0 * (min_w5 - max_w4) / (min_w5 + max_w4)
    
    # Gradient ascent with line search
    coeffs = initial_coeffs.copy()
    current_score = score_cell_from_coeffs(coeffs)
    
    # Adaptive step size
    step_size = 0.1
    max_iter = 200
    
    for iter_num in range(max_iter):
        if time.time() > deadline - 25.0:
            break
        
        # Compute gradient
        score, grad = compute_score_and_grad(coeffs)
        
        if score > current_score:
            current_score = score
        
        # Update coefficients
        new_coeffs = coeffs + step_size * grad
        
        # Project to valid range (Fourier coefficients of 1/eps should be in [1/eps_high, 1/eps_low])
        new_coeffs = np.clip(new_coeffs, 1.0/eps_high - 0.1, 1.0/eps_low + 0.1)
        
        # Evaluate new score
        new_score = score_cell_from_coeffs(new_coeffs)
        
        if new_score > current_score:
            coeffs = new_coeffs
            current_score = new_score
            # Increase step size slightly if improvement
            step_size = min(step_size * 1.1, 0.5)
        else:
            # Decrease step size
            step_size *= 0.8
            if step_size < 0.001:
                break
    
    # Project back to discrete pixels
    # Inverse Fourier transform of coeffs to get continuous 1/eps field
    # Then threshold to get discrete pixels
    
    def coeffs_to_cell(coeffs):
        """Convert Fourier coefficients to discrete cell."""
        # Create full FFT grid
        fft_grid = np.zeros((size, size), dtype=complex)
        for idx, (gx, gy) in enumerate(G_vectors):
            ix = int(gx) % size
            iy = int(gy) % size
            fft_grid[iy, ix] = coeffs[idx]
        
        # Inverse FFT
        inv_eps_field = np.fft.ifft2(fft_grid * size * size).real
        
        # Threshold to get discrete pixels
        # The field should be close to 1/eps_high or 1/eps_low
        mid_val = (1.0/eps_high + 1.0/eps_low) / 2
        cell = (inv_eps_field < mid_val).astype(int)
        
        # Enforce symmetry
        cell = enforce_symmetry(cell)
        
        return cell
    
    def enforce_symmetry(cell):
        """Enforce C4v symmetry on the cell."""
        sym_cell = np.zeros_like(cell)
        for i in range(size):
            for j in range(size):
                # Average over symmetry group
                vals = [
                    cell[i, j],
                    cell[j, i],
                    cell[size-1-i, size-1-j],
                    cell[size-1-j, size-1-i]
                ]
                # Use majority vote or threshold
                avg_val = sum(vals) / len(vals)
                sym_cell[i, j] = 1 if avg_val > 0.5 else 0
        return sym_cell
    
    # Get discrete cell from optimized coefficients
    optimized_cell = coeffs_to_cell(coeffs)
    optimized_score = score_cell(optimized_cell)
    
    if optimized_score > best_score:
        best_score = optimized_score
        best_cell = optimized_cell
    
    # Phase 3: Local search refinement on the best discrete cell
    if time.time() < deadline - 5.0:
        cell_arr = best_cell.copy()
        local_best_score = best_score
        local_best_cell = best_cell.copy()
        
        # Hill climbing with multiple orbit flips
        while time.time() < deadline - 2.0:
            # Randomly choose 1 to 3 orbits to flip
            n_flips = rng.integers(1, 4)
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
            else:
                # Revert
                for orbit in flipped_orbits:
                    for (i, j) in orbit:
                        cell_arr[i, j] = 1 - cell_arr[i, j]
                        
        if local_best_score > best_score:
            best_score = local_best_score
            best_cell = local_best_cell

    return [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]
# EVOLVE-BLOCK-END
