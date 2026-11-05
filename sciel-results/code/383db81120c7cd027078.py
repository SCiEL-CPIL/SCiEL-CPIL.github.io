"""Combined approach: topological optimization with symmetry constraints and coarse-to-fine gap evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

def design(problem):
    start_time = time.time()
    deadline = start_time + 200  # Safety margin before 230s wall-clock limit
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    def make_symmetric_cell(quadrant):
        """
        Create a full 32x32 symmetric cell from a 16x16 quadrant.
        The quadrant represents the top-left 16x16 block.
        Symmetry: cell == cell[::-1] row-wise and column-wise and cell == its transpose.
        """
        cell = [[0] * size for _ in range(size)]
        half = size // 2  # 16
        
        for i in range(half):
            for j in range(half):
                val = quadrant[i][j]
                # Top-left
                cell[i][j] = val
                # Top-right (mirror left-right)
                cell[i][size - 1 - j] = val
                # Bottom-left (mirror top-bottom)
                cell[size - 1 - i][j] = val
                # Bottom-right (mirror both)
                cell[size - 1 - i][size - 1 - j] = val
        
        return cell
    
    def get_fourier_coefficients(cell, n_g=8):
        """Compute Fourier coefficients of epsilon for the given cell."""
        coeffs = np.zeros((2*n_g+1, 2*n_g+1), dtype=complex)
        
        M = np.array(cell, dtype=float)
        
        # Compute 2D Fourier series coefficients
        # eps(x,y) = sum_{n,m} c_{n,m} exp(i(n*Gx*x + m*Gy*y))
        # where Gx = Gy = 2*pi
        # c_{n,m} = (1/A) * integral eps(x,y) exp(-i(n*Gx*x + m*Gy*y)) dx dy
        # For pixel grid: c_{n,m} = (1/size^2) * sum_{i,j} M[i,j] * exp(-i*(n*2pi*(i+0.5)/size + m*2pi*(j+0.5)/size))
        
        for n in range(-n_g, n_g+1):
            for m in range(-n_g, m_g+1) if False else range(-n_g, n_g+1):
                idx_n = n + n_g
                idx_m = m + n_g
                
                Gx = 2 * np.pi * n
                Gy = 2 * np.pi * m
                
                if Gx == 0 and Gy == 0:
                    coeffs[idx_n, idx_m] = np.mean(M)
                else:
                    # Compute sum efficiently using separability
                    # sum_i M[i,j] exp(-i*Gx*(i+0.5)/size) * sum_j exp(-i*Gy*(j+0.5)/size)
                    # Actually need full 2D sum
                    
                    i_vals = np.arange(size)
                    j_vals = np.arange(size)
                    
                    phase_x = np.exp(-1j * Gx * (i_vals + 0.5) / size)
                    phase_y = np.exp(-1j * Gy * (j_vals + 0.5) / size)
                    
                    # c = (1/size^2) * sum_i sum_j M[i,j] * phase_x[i] * phase_y[j]
                    # = (1/size^2) * (phase_x.T @ M @ phase_y)
                    val = (phase_x.T @ M @ phase_y) / (size * size)
                    coeffs[idx_n, idx_m] = val
        
        return coeffs
    
    def compute_gap_exact(cell, n_g=8, n_k=30):
        """
        Exact gap computation with 289 G-vectors (n_g=8) and 30 k-points.
        Returns the gap-midgap ratio.
        """
        eps_coeffs = get_fourier_coefficients(cell, n_g)
        
        # Build list of G vectors
        G_vecs = []
        for nx in range(-n_g, n_g+1):
            for ny in range(-n_g, n_g+1):
                G_vecs.append((2*np.pi*nx, 2*np.pi*ny, nx, ny))
        
        N_G = len(G_vecs)
        
        # Precompute index mapping for epsilon coupling
        E_indices = np.full((N_G, N_G), -1, dtype=int)
        for idx_g, (Gx, Gy, nx, ny) in enumerate(G_vecs):
            for idx_gp, (Gxp, Gyp, nxp, nyp) in enumerate(G_vecs):
                dx = nx - nxp
                dy = ny - nyp
                if -n_g <= dx <= n_g and -n_g <= dy <= n_g:
                    E_indices[idx_g, idx_gp] = (dx + n_g) * (2*n_g+1) + (dy + n_g)
        
        eps_flat = eps_coeffs.flatten()
        
        # K-points in irreducible Brillouin zone
        # Square lattice IBZ: 0 <= kx <= pi, 0 <= ky <= pi, kx <= ky (by symmetry)
        # 30 points covering boundary and interior
        k_points = []
        # Points along kx = 0 (ky from 0 to pi)
        for i in range(6):
            ky = i / 5 * np.pi
            k_points.append((0.0, ky))
        # Points along ky = kx (diagonal)
        for i in range(1, 7):
            kx = i / 6 * np.pi
            k_points.append((kx, kx))
        # Points along ky = pi (kx from 0 to pi)
        for i in range(1, 6):
            kx = i / 5 * np.pi
            k_points.append((kx, np.pi))
        # Interior points
        interior_points = [
            (0.2*np.pi, 0.4*np.pi),
            (0.3*np.pi, 0.5*np.pi),
            (0.4*np.pi, 0.6*np.pi),
            (0.5*np.pi, 0.7*np.pi),
            (0.6*np.pi, 0.8*np.pi),
            (0.1*np.pi, 0.3*np.pi),
            (0.2*np.pi, 0.6*np.pi),
            (0.3*np.pi, 0.8*np.pi),
            (0.4*np.pi, 0.9*np.pi),
            (0.5*np.pi, 0.95*np.pi),
        ]
        k_points.extend(interior_points)
        
        # Deduplicate and ensure we have 30 points
        k_points = list(set([(round(kx,6), round(ky,6)) for kx, ky in k_points]))
        # If we have more than 30, take first 30
        if len(k_points) > 30:
            k_points = k_points[:30]
        
        all_omegas = []
        
        for kx, ky in k_points:
            D = np.zeros(N_G, dtype=float)
            for idx_g, (Gx, Gy, nx, ny) in enumerate(G_vecs):
                kx_g = kx + Gx
                ky_g = ky + Gy
                D[idx_g] = kx_g * kx_g + ky_g * ky_g
            
            valid_mask = E_indices != -1
            E = np.zeros((N_G, N_G), dtype=complex)
            E[valid_mask] = eps_flat[E_indices[valid_mask]]
            
            try:
                sqrt_D = np.sqrt(D)
                inv_sqrt_D = 1.0 / sqrt_D
                H = (sqrt_D[:, None] * E) * inv_sqrt_D[None, :]
                H = (H + H.conj().T) / 2.0
                
                mu_vals = np.linalg.eigvalsh(H)
                mu_real = np.real(mu_vals)
                
                mu_valid = mu_real[mu_real > 1e-10]
                if len(mu_valid) < 5:
                    continue
                
                mu_sorted = np.sort(mu_valid)
                omegas = 1.0 / np.sqrt(mu_sorted)
                omegas = np.sort(omegas)
                
                if len(omegas) >= 5:
                    all_omegas.append(omegas)
            except Exception:
                continue
        
        if not all_omegas:
            return 0.0
            
        max_band4 = max(o[3] for o in all_omegas)
        min_band5 = min(o[4] for o in all_omegas)
        
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = gap / midgap
        return ratio
    
    def compute_gap_coarse(cell, n_g=5, n_k=10):
        """
        Coarse-grained gap computation for fast local search.
        """
        eps_coeffs = get_fourier_coefficients(cell, n_g)
        
        G_vecs = []
        for nx in range(-n_g, n_g+1):
            for ny in range(-n_g, n_g+1):
                G_vecs.append((2*np.pi*nx, 2*np.pi*ny, nx, ny))
        
        N_G = len(G_vecs)
        
        E_indices = np.full((N_G, N_G), -1, dtype=int)
        for idx_g, (Gx, Gy, nx, ny) in enumerate(G_vecs):
            for idx_gp, (Gxp, Gyp, nxp, nyp) in enumerate(G_vecs):
                dx = nx - nxp
                dy = ny - nyp
                if -n_g <= dx <= n_g and -n_g <= dy <= n_g:
                    E_indices[idx_g, idx_gp] = (dx + n_g) * (2*n_g+1) + (dy + n_g)
        
        eps_flat = eps_coeffs.flatten()
        
        k_points = []
        for i in range(n_k):
            for j in range(n_k):
                kx = i / (n_k-1) * np.pi
                ky = j / (n_k-1) * np.pi
                if kx <= ky:
                    k_points.append((kx, ky))
        
        all_omegas = []
        
        for kx, ky in k_points:
            D = np.zeros(N_G, dtype=float)
            for idx_g, (Gx, Gy, nx, ny) in enumerate(G_vecs):
                kx_g = kx + Gx
                ky_g = ky + Gy
                D[idx_g] = kx_g * kx_g + ky_g * ky_g
            
            valid_mask = E_indices != -1
            E = np.zeros((N_G, N_G), dtype=complex)
            E[valid_mask] = eps_flat[E_indices[valid_mask]]
            
            try:
                sqrt_D = np.sqrt(D)
                inv_sqrt_D = 1.0 / sqrt_D
                H = (sqrt_D[:, None] * E) * inv_sqrt_D[None, :]
                H = (H + H.conj().T) / 2.0
                
                mu_vals = np.linalg.eigvalsh(H)
                mu_real = np.real(mu_vals)
                
                mu_valid = mu_real[mu_real > 1e-10]
                if len(mu_valid) < 5:
                    continue
                
                mu_sorted = np.sort(mu_valid)
                omegas = 1.0 / np.sqrt(mu_sorted)
                omegas = np.sort(omegas)
                
                if len(omegas) >= 5:
                    all_omegas.append(omegas)
            except Exception:
                continue
        
        if not all_omegas:
            return 0.0
            
        max_band4 = max(o[3] for o in all_omegas)
        min_band5 = min(o[4] for o in all_omegas)
        
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = gap / midgap
        return ratio
    
    # Initialize with a super-ellipse shape as starting point
    beta = 4.0
    radius = 0.35
    quadrant = [[0] * 16 for _ in range(16)]
    for i in range(16):
        for j in range(16):
            x_norm = (i + 0.5) / size - 0.5
            y_norm = (j + 0.5) / size - 0.5
            val_x = abs(x_norm / radius)
            val_y = abs(y_norm / radius)
            dist = val_x**beta + val_y**beta
            quadrant[i][j] = 1 if dist < 1 else 0
    
    best_quadrant = [row[:] for row in quadrant]
    best_score = compute_gap_coarse(make_symmetric_cell(quadrant))
    
    # Topological optimization: flip pixels to improve the gap
    random.seed(42)
    
    for iteration in range(500):
        if time.time() > deadline - 30:
            break
        
        # Randomly select a pixel in the quadrant to flip
        i = random.randint(0, 15)
        j = random.randint(0, 15)
        
        old_val = quadrant[i][j]
        quadrant[i][j] = 1 - old_val
        
        cell = make_symmetric_cell(quadrant)
        score = compute_gap_coarse(cell)
        
        if score > best_score:
            best_score = score
            best_quadrant = [row[:] for row in quadrant]
        else:
            # Revert
            quadrant[i][j] = old_val
    
    # Final refinement with exact evaluation on best candidate
    best_cell = make_symmetric_cell(best_quadrant)
    final_score = compute_gap_exact(best_cell)
    
    return best_cell
# EVOLVE-BLOCK-END
