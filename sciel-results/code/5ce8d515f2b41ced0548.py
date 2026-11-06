"""Photonic crystal unit cell design for large TM 4-5 band gap."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def design(problem):
    """Design a symmetric 32x32 unit cell maximizing TM 4-5 band gap.
    
    Uses Evolutionary Strategy (ES) on symmetry-constrained binary pixels.
    """
    t_start = time.time()
    DEADLINE = 215.0  # Safety margin before 230s kill
    
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    eps_low = problem["eps_low"]
    half_size = size // 2  # 16

    # --- Fast TM plane-wave solver setup ---
    n_pw = 17
    half_pw = n_pw // 2
    G_vals = np.arange(-half_pw, half_pw + 1)
    Gx_grid, Gy_grid = np.meshgrid(G_vals, G_vals, indexing='ij')
    Gx = Gx_grid.ravel()
    Gy = Gy_grid.ravel()
    n_pw_total = n_pw * n_pw  # 289

    # K-points in irreducible BZ (30 points)
    pi = np.pi
    k_points = []
    # Gamma to M: (0,0) to (pi, 0)
    for i in range(10):
        t = i / 9.0
        k_points.append((t * pi, 0.0))
    # M to X: (pi, 0) to (pi, pi) -> actually boundary of IBZ is (pi,0) to (pi/2, pi/2)
    # Standard IBZ for square lattice: triangle (0,0), (pi,0), (pi/2, pi/2)
    # Let's use the standard 30 points covering the triangle.
    # Path 1: Gamma (0,0) to M (pi, 0)
    # Path 2: M (pi, 0) to X (pi/2, pi/2)
    # Path 3: X (pi/2, pi/2) to Gamma (0,0)
    
    # Re-defining k_points to strictly cover the IBZ triangle
    k_points = []
    # Gamma to M
    for i in range(10):
        t = i / 9.0
        k_points.append((t * pi, 0.0))
    # M to X
    for i in range(1, 10):
        t = i / 9.0
        k_points.append((pi - t * pi/2, t * pi/2))
    # X to Gamma
    for i in range(1, 10):
        t = i / 9.0
        k_points.append(((1-t) * pi/2, (1-t) * pi/2))
        
    k_points = np.array(k_points)

    # Precompute (k+G)^2
    k2_all = np.zeros((len(k_points), n_pw_total))
    for i, (kx, ky) in enumerate(k_points):
        k2_all[i] = (kx + Gx) ** 2 + (ky + Gy) ** 2

    # Precompute difference indices for Fourier coupling
    # H_{ij} = (k+G_i)^2 * eps^{-1}_{G_i - G_j}
    diff_idx = np.zeros((n_pw_total, n_pw_total), dtype=int)
    valid_mask = np.zeros((n_pw_total, n_pw_total), dtype=bool)
    for i in range(n_pw_total):
        for j in range(n_pw_total):
            dGx = int(Gx[i]) - int(Gx[j])
            dGy = int(Gy[i]) - int(Gy[j])
            idx_x = dGx + half_pw
            idx_y = dGy + half_pw
            if 0 <= idx_x < n_pw and 0 <= idx_y < n_pw:
                diff_idx[i, j] = idx_x * n_pw + idx_y
                valid_mask[i, j] = True
            else:
                diff_idx[i, j] = 0
    valid_rows, valid_cols = np.where(valid_mask)
    valid_diff_indices = diff_idx[valid_rows, valid_cols]

    def get_fourier_coeffs(cell_quadrant):
        """Compute Fourier coefficients of 1/eps using FFT.
        cell_quadrant is 16x16.
        """
        # Expand to full 32x32
        cell_full = np.zeros((size, size), dtype=int)
        cell_full[:half_size, :half_size] = cell_quadrant
        cell_full[half_size:, :half_size] = cell_quadrant[::-1, :]
        cell_full[:half_size, half_size:] = cell_quadrant[:, ::-1]
        cell_full[half_size:, half_size:] = cell_quadrant[::-1, ::-1]
        
        eps_map = np.where(cell_full == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_map
        fft_2d = np.fft.fft2(inv_eps)
        
        coeffs = np.zeros(n_pw_total, dtype=complex)
        # We only need to fill the valid_diff_indices part, but get_fourier_coeffs
        # returns the full array indexed by Gx, Gy.
        for idx in range(n_pw_total):
            gx = int(Gx[idx])
            gy = int(Gy[idx])
            fft_ix = gx % size
            fft_iy = gy % size
            coeffs[idx] = fft_2d[fft_ix, fft_iy] / (size * size)
        return coeffs

    def compute_band_ratio(coeffs):
        """Compute gap-midgap ratio using all 30 k-points and 289 PWs."""
        max_band4 = -np.inf
        min_band5 = np.inf

        for k_idx in range(len(k_points)):
            k2 = k2_all[k_idx]
            # H is 289x289 complex
            # H_{ij} = k2_i * coeffs_{G_i - G_j}
            # We can construct H efficiently
            H_vals = k2[valid_rows] * coeffs[valid_diff_indices]
            
            # Build sparse-like structure or dense
            # Since 289 is small, dense is fine.
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            H[valid_rows, valid_cols] = H_vals
            H = (H + H.conj().T) / 2.0
            
            # Eigendecomposition
            eigvals = np.linalg.eigvalsh(H)
            # eigvals are sorted
            if len(eigvals) >= 5:
                b4 = eigvals[3]
                b5 = eigvals[4]
                if b4 > max_band4: max_band4 = b4
                if b5 < min_band5: min_band5 = b5

        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        if midgap > 1e-10:
            return gap / midgap
        return 0.0

    def evaluate_cell_quadrant(cell_q):
        coeffs = get_fourier_coeffs(cell_q)
        return compute_band_ratio(coeffs)

    # --- Initial Candidates ---
    
    def make_quadrant_from_parametric(type_id, params):
        """Generate a 16x16 quadrant from parametric shapes."""
        q = np.zeros((half_size, half_size), dtype=int)
        for i in range(half_size):
            for j in range(half_size):
                # Coordinates in full cell [0, 32)
                x = (j + 0.5) / size - 0.5
                y = (i + 0.5) / size - 0.5
                val = 0
                if type_id == 0: # Four rods
                    r, off = params
                    for sx in [1, -1]:
                        for sy in [1, -1]:
                            dx = x - sx * off
                            dy = y - sy * off
                            if dx*dx + dy*dy < r*r:
                                val = 1
                elif type_id == 1: # Single rod
                    r = params[0]
                    if x*x + y*y < r*r:
                        val = 1
                elif type_id == 2: # Ring
                    ri, ro = params
                    r2 = x*x + y*y
                    if ri*ri < r2 < ro*ro:
                        val = 1
                elif type_id == 3: # Moire
                    f1, f2, p1, p2, thr = params
                    v1 = np.sin(2 * np.pi * f1 * (x + y) + p1)
                    v2 = np.sin(2 * np.pi * f2 * (x - y) + p2)
                    val = 1 if (v1 > 0 and v2 > 0) else 0
                    if val and (v1*v2 < thr): # Threshold logic adjustment
                         pass # Keeping it simple: AND of square waves
                q[i, j] = val
        return q

    # Generate initial population
    pop_size = 40
    population = []
    scores = []
    
    # Add some parametric seeds
    seed_configs = [
        (0, (0.08, 0.25)), (0, (0.10, 0.22)), (0, (0.06, 0.28)),
        (1, (0.20)), (1, (0.25)),
        (2, (0.10, 0.20)), (2, (0.15, 0.25)),
        (3, (4, 5, 0, 0, 0.5)), (3, (3, 4, np.pi/2, 0, 0.5))
    ]
    
    for cfg in seed_configs:
        q = make_quadrant_from_parametric(cfg[0], cfg[1])
        population.append(q)
        
    # Fill rest with random
    while len(population) < pop_size:
        q = np.random.randint(0, 2, (half_size, half_size))
        population.append(q)

    # Evaluate initial population
    for q in population:
        s = evaluate_cell_quadrant(q)
        scores.append(s)
        
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_q = population[best_idx].copy()
    
    # --- ES Loop ---
    sigma = 0.1 # Probability of flipping a pixel in the quadrant
    
    while time.time() < DEADLINE:
        # Generate offspring
        offspring = []
        for _ in range(pop_size):
            # Pick parent (tournament selection)
            p1_idx = np.random.randint(0, pop_size)
            p2_idx = np.random.randint(0, pop_size)
            if scores[p1_idx] > scores[p2_idx]:
                parent = population[p1_idx]
            else:
                parent = population[p2_idx]
                
            # Mutate
            child = parent.copy()
            # Flip random pixels
            n_mut = np.random.binomial(half_size * half_size, sigma)
            if n_mut > 0:
                flat_idx = np.random.choice(half_size * half_size, n_mut, replace=False)
                child_flat = child.ravel()
                child_flat[flat_idx] = 1 - child_flat[flat_idx]
                child = child_flat.reshape((half_size, half_size))
            offspring.append(child)
        
        # Evaluate offspring
        off_scores = []
        for q in offspring:
            s = evaluate_cell_quadrant(q)
            off_scores.append(s)
            
        # Update population (Elitism: keep best from parent and offspring)
        # Combine
        all_q = population + offspring
        all_s = scores + off_scores
        
        # Sort
        sort_idx = np.argsort(all_s)[::-1]
        
        # Keep top pop_size
        new_population = [all_q[i] for i in sort_idx[:pop_size]]
        new_scores = [all_s[i] for i in sort_idx[:pop_size]]
        
        population = new_population
        scores = new_scores
        
        if scores[0] > best_score:
            best_score = scores[0]
            best_q = population[0].copy()
            
        # Adaptive sigma? 
        # If best improved, maybe decrease sigma? Or keep constant?
        # Let's keep constant for simplicity, maybe slight decay.
        # sigma *= 0.99

    # Finalize
    # Expand best_q to full cell
    cell_full = np.zeros((size, size), dtype=int)
    cell_full[:half_size, :half_size] = best_q
    cell_full[half_size:, :half_size] = best_q[::-1, :]
    cell_full[:half_size, half_size:] = best_q[:, ::-1]
    cell_full[half_size:, half_size:] = best_q[::-1, ::-1]
    
    # Ensure symmetry is exact (it should be by construction)
    # cell_full = (cell_full + cell_full[::-1, ::-1] + cell_full[:, ::-1] + cell_full[::-1, :])
    # cell_full = (cell_full >= 2).astype(int)
    
    result = [[int(cell_full[i][j]) for j in range(size)] for i in range(size)]
    return result

# EVOLVE-BLOCK-END
