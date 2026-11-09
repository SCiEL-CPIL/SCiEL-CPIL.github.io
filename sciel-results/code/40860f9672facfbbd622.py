"""Photonic crystal design with local search and fast plane-wave evaluation."""
import numpy as np
import time
import math

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    start_time = time.time()
    time_limit = 210.0  # Safety margin before 230s kill
    
    # Initialize with a square lattice of dielectric rods (common photonic crystal)
    # Use a pattern that might have band gaps
    cell = np.zeros((size, size), dtype=int)
    
    # Create a pattern: dielectric rods at (0,0), (0.5,0), (0,0.5), (0.5,0.5) in unit cell
    # This is a 2x2 array of rods per unit cell
    rod_radius = 0.15  # Fraction of lattice constant
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            # Check distance to nearest rod center
            # Rods at (0,0), (0.5,0), (0,0.5), (0.5,0.5)
            dx0 = abs(x)
            dy0 = abs(y)
            dx1 = abs(x - 0.5)
            dy1 = abs(y - 0.5)
            dx2 = abs(x + 0.5)
            dy2 = abs(y + 0.5)
            # Nearest rod in 2x2 grid
            d0 = dx0 * dx0 + dy0 * dy0
            d1 = dx1 * dx1 + dy0 * dy0
            d2 = dx0 * dx0 + dy1 * dy1
            d3 = dx1 * dx1 + dy1 * dy1
            min_d = min(d0, d1, d2, d3)
            if min_d < rod_radius * rod_radius:
                cell[i, j] = 1
    
    # Fast plane-wave solver
    # 289 = 17^2, so Gx, Gy in [-8, 8]
    N_G = 17
    g_range = np.arange(-N_G//2, N_G//2 + 1)
    Gx, Gy = np.meshgrid(g_range, g_range, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    N_pw = len(G_vectors)
    
    # Precompute G^2
    G_sq = np.sum(G_vectors**2, axis=1)
    
    # K points in irreducible BZ (simplified set for speed)
    # Irreducible BZ: 0 <= ky <= kx, kx in [0, 1], ky in [0, 1]
    k_points = []
    # Boundary points
    for i in range(6):
        k_points.append([i/5.0, 0.0])
        k_points.append([0.5, i/5.0])
        k_points.append([i/5.0, i/5.0])
    # Interior points
    k_points.append([0.25, 0.1])
    k_points.append([0.4, 0.2])
    k_points.append([0.3, 0.15])
    k_points = np.array(k_points)
    
    def get_fourier_eps_inv(cell_arr):
        """Get Fourier coefficients of 1/epsilon using FFT."""
        eps = np.where(cell_arr == 1, eps_high, eps_low)
        eps_inv = 1.0 / eps
        coeffs = np.fft.fft2(eps_inv) / (size * size)
        # Map G_vectors to FFT grid indices
        offset = size // 2
        Gx_grid = G_vectors[:, 0] + offset
        Gy_grid = G_vectors[:, 1] + offset
        return coeffs[Gx_grid, Gy_grid]
    
    def evaluate(cell_arr):
        """Evaluate the gap-midgap ratio for bands 4 and 5."""
        eps_inv_coeffs = get_fourier_eps_inv(cell_arr)
        
        min_omega_5 = float('inf')
        max_omega_4 = float('-inf')
        
        for k in k_points:
            k_vec = np.array(k)
            # Precompute (k+G_j)^2
            k_plus_G = k_vec[None, :] + G_vectors
            k_plus_G_sq = np.sum(k_plus_G**2, axis=1)
            
            # Build H matrix: H[i, j] = eps_inv_coeffs[G_i - G_j] * (k+G_j)^2
            H = np.zeros((N_pw, N_pw), dtype=complex)
            
            for i in range(N_pw):
                Gi = G_vectors[i]
                diffs = Gi[None, :] - G_vectors
                dx = diffs[:, 0]
                dy = diffs[:, 1]
                mask = (np.abs(dx) <= N_G//2) & (np.abs(dy) <= N_G//2)
                if np.any(mask):
                    idxs = (dx[mask] + N_G//2) + (dy[mask] + N_G//2) * N_G
                    coeffs = eps_inv_coeffs[idxs]
                    valid_j = np.where(mask)[0]
                    H[i, valid_j] = coeffs * k_plus_G_sq[valid_j]
            
            # Solve eigenvalue problem
            evals = np.linalg.eigvals(H)
            evals = np.sort(np.real(evals))
            
            if len(evals) >= 5:
                w4 = evals[3]
                w5 = evals[4]
                max_omega_4 = max(max_omega_4, w4)
                min_omega_5 = min(min_omega_5, w5)
        
        if min_omega_5 == float('inf') or max_omega_4 == float('-inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap
    
    # Initial evaluation
    best_score = evaluate(cell)
    best_cell = cell.copy()
    
    # Local search
    def get_orbit(i, j):
        """Returns list of unique (r, c) in the 4-fold symmetric orbit."""
        N = size
        points = [
            (i, j),
            (j, N-1-i),
            (N-1-i, N-1-j),
            (N-1-j, i),
            (i, N-1-j),
            (N-1-i, j),
            (j, i),
            (N-1-j, N-1-i)
        ]
        return list(set(points))
    
    iteration = 0
    while time.time() - start_time < time_limit:
        iteration += 1
        # Pick random pixel
        i = np.random.randint(size)
        j = np.random.randint(size)
        
        orbit = get_orbit(i, j)
        
        # Flip
        for (r, c) in orbit:
            cell[r, c] = 1 - cell[r, c]
        
        score = evaluate(cell)
        
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
        else:
            # Revert
            for (r, c) in orbit:
                cell[r, c] = 1 - cell[r, c]
        
        # Check time more frequently
        if time.time() - start_time > time_limit - 5:
            break
    
    # Return best_cell as nested list
    return best_cell.tolist()
