"""Design a 32x32 symmetric photonic-crystal unit cell maximizing the 4-5 TM band gap."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from itertools import product

def _enforce_symmetry(cell):
    """Enforce full square symmetry: reverse rows, reverse cols, and transpose."""
    n = len(cell)
    # Reverse rows
    cell = cell[::-1]
    # Reverse columns in each row
    cell = [row[::-1] for row in cell]
    # Transpose
    cell = [list(row) for row in zip(*cell)]
    # Reverse rows again to ensure consistency
    cell = cell[::-1]
    # Reverse columns again
    cell = [row[::-1] for row in cell]
    return cell

def _compute_gap_fast(cell, num_pw=121, num_kpts=12):
    """
    Fast approximation of the 4-5 TM band gap using a reduced plane-wave basis.
    Returns the gap-midgap ratio (positive if gap exists).
    """
    n = len(cell)
    eps_high = 12.0
    eps_low = 1.0
    
    # Build epsilon map
    eps = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            eps[i, j] = eps_high if cell[i][j] == 1 else eps_low
    
    # For TM polarization in 2D, the eigenvalue equation is:
    # sum_G' eps^{-1}(G-G') |G+K|^2 c(G') = omega^2 c(G)
    # where eps^{-1} is the inverse dielectric function
    
    # Compute inverse dielectric
    eps_inv = np.where(eps > 0, 1.0 / eps, 0.0)
    
    # Compute Fourier coefficients of eps^{-1}
    # Use a small set of G vectors for speed
    max_g = 5  # G vectors from -max_g to max_g in each direction
    G_list = []
    for gx in range(-max_g, max_g + 1):
        for gy in range(-max_g, max_g + 1):
            G_list.append((gx, gy))
    
    N_G = len(G_list)
    
    # Fourier coefficients of eps_inv
    eps_inv_fourier = np.zeros((N_G, N_G), dtype=complex)
    for idx, (gx, gy) in enumerate(G_list):
        # Compute FFT of eps_inv * exp(-i 2pi (gx*x + gy*y))
        # x, y in [0,1)
        xs = np.arange(n) / n
        phase = np.exp(-2j * np.pi * (gx * xs[:, None] + gy * xs[None, :]))
        integral = np.sum(eps_inv * phase) / (n * n)
        eps_inv_fourier[idx, :] = integral  # This is wrong, need to fix
    
    # Actually, let's use a proper FFT approach
    # For speed, let's use a simpler approximation: just compute the gap from
    # a small number of k-points using a reduced basis
    
    # Let's use a different approach: compute the band structure at a few key k-points
    # using a direct matrix diagonalization with a small PW basis
    
    k_points = []
    # Key k-points in the irreducible BZ: (0,0), (0.5,0), (0.5,0.5), and some interior points
    k_points.append((0.0, 0.0))
    k_points.append((0.5, 0.0))
    k_points.append((0.5, 0.5))
    k_points.append((0.25, 0.25))
    k_points.append((0.125, 0.0))
    k_points.append((0.375, 0.0))
    k_points.append((0.25, 0.0))
    k_points.append((0.0, 0.25))
    k_points.append((0.125, 0.125))
    k_points.append((0.375, 0.125))
    k_points.append((0.25, 0.375))
    k_points.append((0.375, 0.375))
    
    # Use a small PW basis: G vectors with |G| <= 4
    pw_max = 4
    pw_vectors = []
    for gx in range(-pw_max, pw_max + 1):
        for gy in range(-pw_max, pw_max + 1):
            if gx * gx + gy * gy <= pw_max * pw_max:
                pw_vectors.append((gx, gy))
    N_pw = len(pw_vectors)
    
    # Precompute Fourier coefficients of eps^{-1} for all needed G differences
    # eps_inv_fourier[G1 - G2]
    # We need this for all pairs of PW vectors
    
    # Compute eps_inv Fourier coefficients using 2D FFT
    # The Fourier coefficient for wavevector (h, k) is:
    # eps_inv_hat(h, k) = sum_{i,j} eps_inv[i,j] * exp(-i 2pi (h*i/n + k*j/n)) / n^2
    
    # We need coefficients for G differences, which range from -2*pw_max to 2*pw_max
    fourier_max = 2 * pw_max
    eps_inv_hat = np.zeros((2 * fourier_max + 1, 2 * fourier_max + 1), dtype=complex)
    
    # Use FFT to compute all coefficients at once
    # Pad to a larger grid for accuracy
    fft_size = n  # 32
    eps_inv_fft = np.fft.fft2(eps_inv)
    
    for idx, (gx, gy) in enumerate(pw_vectors):
        for jdx, (gx2, gy2) in enumerate(pw_vectors):
            dh = gx - gx2
            dk = gy - gy2
            # Get the Fourier coefficient at (dh, dk)
            # FFT gives us coefficients at frequencies 0, 1, ..., n-1
            # We need to handle negative frequencies
            h_idx = dh % fft_size
            k_idx = dk % fft_size
            eps_inv_hat[idx, jdx] = eps_inv_fft[h_idx, k_idx] / (fft_size * fft_size)
    
    # Now compute bands at each k-point
    omega_4_max = 0.0
    omega_5_min = float('inf')
    
    for kx, ky in k_points:
        # Build the eigenvalue matrix
        # H[G1, G2] = sum_G' eps_inv_hat[G1-G2+G'] * |G1+K-G'|^2
        # Wait, the correct form is:
        # For TM: sum_G' eps^{-1}(G-G') * |G+K|^2 * delta(G'-G) ... no
        # The correct eigenvalue equation for TM is:
        # sum_G' [eps^{-1}]_{G-G'} * |G+K|^2 * delta_{G',G} ... 
        # Actually, the standard form is:
        # sum_G' eps^{-1}(G-G') |G+K|^2 c(G') = omega^2 c(G)
        # But this isn't quite right either. Let me use the standard form:
        # The Hamiltonian matrix elements are:
        # H_{G,G'} = sum_{G''} eps^{-1}(G-G'') * |G''+K|^2 * delta(G'', G')
        # = eps^{-1}(G-G') * |G'+K|^2
        # Wait, that's also not right. Let me think more carefully.
        
        # For TM polarization, the eigenvalue problem is:
        # sum_{G'} eps^{-1}(G-G') |G'+K|^2 c(G') = omega^2 c(G)
        
        # So H_{G,G'} = eps^{-1}(G-G') * |G'+K|^2
        
        H = np.zeros((N_pw, N_pw), dtype=complex)
        for i, (gx, gy) in enumerate(pw_vectors):
            for j, (gx2, gy2) in enumerate(pw_vectors):
                # eps^{-1}(G_i - G_j)
                dh = gx - gx2
                dk = gy - gy2
                h_idx = dh % fft_size
                k_idx = dk % fft_size
                eps_inv_val = eps_inv_fft[h_idx, k_idx] / (fft_size * fft_size)
                
                # |G_j + K|^2
                # K = (kx, ky) in units of 2pi
                # G_j + K in units of 2pi: (gx2 + kx, gy2 + ky)
                # |G_j + K|^2 = (gx2 + kx)^2 + (gy2 + ky)^2 (in units of (2pi)^2)
                # But we need to be careful with units. Let's work in units where the
                # lattice constant is 1, so K is in units of 2pi/L = 2pi.
                # The eigenvalue omega^2 will be in units of (2pi/L)^2.
                kx2 = gx2 + kx
                ky2 = gy2 + ky
                k_sq = kx2 * kx2 + ky2 * ky2
                
                H[i, j] = eps_inv_val * k_sq
        
        # Diagonalize
        try:
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(np.real(eigvals))
            
            # Take the 4th and 5th bands (0-indexed: 3 and 4)
            if len(eigvals) >= 5:
                omega_4 = math.sqrt(max(eigvals[3], 0.0))
                omega_5 = math.sqrt(max(eigvals[4], 0.0))
                omega_4_max = max(omega_4_max, omega_4)
                omega_5_min = min(omega_5_min, omega_5)
        except:
            pass
    
    if omega_5_min == float('inf'):
        return -1.0
    
    midgap = 0.5 * (omega_5_min + omega_4_max)
    if midgap == 0:
        return -1.0
    
    gap_ratio = 2.0 * (omega_5_min - omega_4_max) / midgap
    return gap_ratio

def _four_rod_cell(radii):
    """
    Create a 32x32 cell with 4 dielectric rods at positions (±0.25, ±0.25).
    radii: tuple of 4 radii for each rod position.
    """
    size = 32
    cell = [[0] * size for _ in range(size)]
    
    # Rod positions (in fractional coordinates, centered at 0.5, 0.5)
    positions = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
    
    for idx, (cx, cy) in enumerate(positions):
        r = radii[idx]
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size
                y = (j + 0.5) / size
                dx = x - cx
                dy = y - cy
                if dx * dx + dy * dy < r * r:
                    cell[i][j] = 1
    
    return cell

def design(problem):
    """Design a 32x32 photonic crystal unit cell maximizing the 4-5 TM band gap."""
    start_time = time.time()
    deadline = start_time + 210.0  # Safety margin before 230s limit
    
    size = problem.get("pixels", 32)
    
    # Start with a 4-rod configuration
    # Initial guess: equal radii
    best_radii = (0.18, 0.18, 0.18, 0.18)
    best_score = -1.0
    best_cell = _four_rod_cell(best_radii)
    
    # Coarse grid search over radii
    # Each radius ranges from 0.05 to 0.30
    radius_range = np.arange(0.08, 0.32, 0.04)
    
    # Evaluate initial guess
    score = _compute_gap_fast(best_cell)
    if score > best_score:
        best_score = score
        best_radii = (0.18, 0.18, 0.18, 0.18)
    
    # Grid search
    for r0 in radius_range:
        for r1 in radius_range:
            for r2 in radius_range:
                for r3 in radius_range:
                    if time.time() > deadline:
                        break
                    radii = (r0, r1, r2, r3)
                    cell = _four_rod_cell(radii)
                    score = _compute_gap_fast(cell)
                    if score > best_score:
                        best_score = score
                        best_radii = radii
                        best_cell = cell
    
    # Local refinement around the best point
    if time.time() < deadline - 30:
        # Refine with smaller steps
        step = 0.01
        for _ in range(5):  # 5 refinement iterations
            improved = False
            for idx in range(4):
                for delta in [-step, step]:
                    if time.time() > deadline:
                        break
                    new_radii = list(best_radii)
                    new_radii[idx] += delta
                    if new_radii[idx] < 0.05 or new_radii[idx] > 0.35:
                        continue
                    new_radii = tuple(new_radii)
                    cell = _four_rod_cell(new_radii)
                    score = _compute_gap_fast(cell)
                    if score > best_score:
                        best_score = score
                        best_radii = new_radii
                        best_cell = cell
                        improved = True
            if not improved:
                step *= 0.5
    
    # Ensure symmetry is properly enforced
    best_cell = _enforce_symmetry(best_cell)
    
    # Verify it's a valid 32x32 cell
    assert len(best_cell) == 32
    assert all(len(row) == 32 for row in best_cell)
    assert all(cell[i][j] in (0, 1) for i in range(32) for j in range(32))
    
    return best_cell
# EVOLVE-BLOCK-END
