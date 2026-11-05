"""Combined: super-ellipse topology with parameter search for TM band gap maximization."""
import numpy as np
import time
import math

def design(problem):
    start_time = time.time()
    deadline = start_time + 180  # Safety margin before 200s limit
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    def make_cell(beta, r_beta):
        """Generate a symmetric super-ellipse cell."""
        cell = [[0] * size for _ in range(size)]
        for i in range(size):
            x_norm = (i + 0.5) / size - 0.5
            for j in range(size):
                y_norm = (j + 0.5) / size - 0.5
                val_x = abs(x_norm / r_beta)
                val_y = abs(y_norm / r_beta)
                dist = val_x**beta + val_y**beta
                if dist < 1.0:
                    cell[i][j] = 1
        return cell
    
    def compute_fourier_coefficients(cell):
        """Compute exact Fourier coefficients for the pixel map."""
        eps = np.array(cell, dtype=float)
        coeffs = {}
        # Use DFT as approximation for speed; evaluator uses exact integration
        # But for surrogate, DFT is close enough for relative comparisons
        eps_2d = eps_low + (eps_high - eps_low) * eps
        F = np.fft.fft2(eps_2d) / (size * size)
        # Shift to center zero frequency
        F = np.fft.fftshift(F)
        # G vectors: indices -16 to 15 (for 32x32)
        # But we only need up to |G| <= 8 for 289 plane waves (17x17)
        for nx in range(-8, 9):
            for ny in range(-8, 9):
                # Map to fftshifted array index
                ix = nx + 16
                iy = ny + 16
                if 0 <= ix < size and 0 <= iy < size:
                    coeffs[(nx, ny)] = F[ix, iy]
                else:
                    coeffs[(nx, ny)] = 0.0 + 0.0j
        return coeffs
    
    def compute_gap_surrogate(cell):
        """Fast surrogate: compute gap between band 4 and 5 using DFT-based PWE."""
        coeffs = compute_fourier_coefficients(cell)
        
        # Build inverse epsilon matrix for TM mode
        # For TM: (1/eps)_{G,G'} = delta_{G,G'} * (1/eps_{G-G'})
        # Actually for TM, the eigenvalue problem is:
        # sum_{G'} (1/eps_{G-G'}) * |G'|^2 * c_{G'} = omega^2 * c_G
        # where 1/eps_{G} are Fourier coefficients of 1/epsilon(r)
        
        # Compute 1/eps Fourier coefficients
        inv_eps = np.zeros((size, size), dtype=float)
        for i in range(size):
            for j in range(size):
                eps_val = eps_low if cell[i][j] == 0 else eps_high
                inv_eps[i, j] = 1.0 / eps_val
        
        inv_eps_2d = inv_eps
        F_inv = np.fft.fft2(inv_eps_2d) / (size * size)
        F_inv = np.fft.fftshift(F_inv)
        
        inv_eps_coeffs = {}
        for nx in range(-8, 9):
            for ny in range(-8, 9):
                ix = nx + 16
                iy = ny + 16
                if 0 <= ix < size and 0 <= iy < size:
                    inv_eps_coeffs[(nx, ny)] = F_inv[ix, iy]
                else:
                    inv_eps_coeffs[(nx, ny)] = 0.0 + 0.0j
        
        # G vectors list
        G_list = [(nx, ny) for nx in range(-8, 9) for ny in range(-8, 9)]
        G_mag_sq = {}
        for G in G_list:
            G_mag_sq[G] = (2 * np.pi * G[0])**2 + (2 * np.pi * G[1])**2
        
        # Sample k-points in irreducible BZ
        # Irreducible BZ for square lattice: 0 <= kx <= ky, kx+ky <= pi/a
        # Use a few representative k-points
        k_points = []
        # Gamma point
        k_points.append((0.0, 0.0))
        # X point (pi, 0)
        k_points.append((np.pi, 0.0))
        # M point (pi, pi)
        k_points.append((np.pi, np.pi))
        # Midpoint of Gamma-X
        k_points.append((np.pi/2, 0.0))
        # Midpoint of X-M
        k_points.append((np.pi, np.pi/2))
        # Midpoint of Gamma-M (diagonal)
        k_points.append((np.pi/2, np.pi/2))
        # Additional interior points
        k_points.append((np.pi/3, np.pi/6))
        k_points.append((np.pi/4, np.pi/8))
        k_points.append((2*np.pi/5, np.pi/5))
        k_points.append((np.pi/2.5, np.pi/5))
        
        # For each k-point, solve eigenvalue problem
        min_omega5 = float('inf')
        max_omega4 = 0.0
        
        for kx, ky in k_points:
            # Build matrix A where A_{G,G'} = (1/eps_{G-G'}) * (k+G')^2
            nG = len(G_list)
            A = np.zeros((nG, nG), dtype=complex)
            
            for gi, Gi in enumerate(G_list):
                for gj, Gj in enumerate(G_list):
                    # G - G'
                    dGx = Gi[0] - Gj[0]
                    dGy = Gi[1] - Gj[1]
                    # (k + G')^2
                    kpx = kx + 2 * np.pi * Gj[0]
                    kpy = ky + 2 * np.pi * Gj[1]
                    kp_sq = kpx**2 + kpy**2
                    # 1/eps_{G-G'}
                    if dGx == 0 and dGy == 0:
                        inv_eps_val = inv_eps_coeffs[(0, 0)]
                    else:
                        inv_eps_val = inv_eps_coeffs.get((dGx, dGy), 0.0 + 0.0j)
                    A[gi, gj] = inv_eps_val * kp_sq
            
            # Solve eigenvalue problem
            try:
                eigvals = np.linalg.eigvalsh(A)
                eigvals = np.real(eigvals)
                eigvals = np.sort(eigvals)
                # Take positive eigenvalues only
                eigvals = eigvals[eigvals > 0]
                if len(eigvals) >= 5:
                    omega4 = np.sqrt(eigvals[3])  # 4th band (0-indexed: 3)
                    omega5 = np.sqrt(eigvals[4])  # 5th band (0-indexed: 4)
                    min_omega5 = min(min_omega5, omega5)
                    max_omega4 = max(max_omega4, omega4)
            except:
                continue
        
        if min_omega5 == float('inf') or max_omega4 == 0:
            return -1.0
        
        gap = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
        return gap
    
    # Local search over beta and r_beta
    best_gap = -1.0
    best_beta = 4.0
    best_r = 0.2
    
    # Initial grid search
    beta_values = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    r_values = [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]
    
    # Coarse search
    for beta in beta_values:
        for r in r_values:
            if time.time() > deadline:
                break
            cell = make_cell(beta, r)
            gap = compute_gap_surrogate(cell)
            if gap > best_gap:
                best_gap = gap
                best_beta = beta
                best_r = r
        if time.time() > deadline:
            break
    
    # Fine search around best
    if time.time() < deadline:
        beta_range = np.linspace(best_beta - 1.0, best_beta + 1.0, 7)
        r_range = np.linspace(best_r - 0.03, best_r + 0.03, 7)
        
        for beta in beta_range:
            if beta < 1.0:
                continue
            for r in r_range:
                if r < 0.1 or r > 0.4:
                    continue
                if time.time() > deadline:
                    break
                cell = make_cell(beta, r)
                gap = compute_gap_surrogate(cell)
                if gap > best_gap:
                    best_gap = gap
                    best_beta = beta
                    best_r = r
            if time.time() > deadline:
                break
    
    # Even finer search
    if time.time() < deadline - 10:
        beta_range = np.linspace(best_beta - 0.3, best_beta + 0.3, 5)
        r_range = np.linspace(best_r - 0.01, best_r + 0.01, 5)
        
        for beta in beta_range:
            if beta < 1.0:
                continue
            for r in r_range:
                if r < 0.1 or r > 0.4:
                    continue
                if time.time() > deadline:
                    break
                cell = make_cell(beta, r)
                gap = compute_gap_surrogate(cell)
                if gap > best_gap:
                    best_gap = gap
                    best_beta = beta
                    best_r = r
            if time.time() > deadline:
                break
    
    # Generate final cell with best parameters
    best_cell = make_cell(best_beta, best_r)
    
    return best_cell
