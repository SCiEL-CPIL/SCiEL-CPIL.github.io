"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize
import itertools

# EVOLVE-BLOCK-START
def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    Strategy:
    1. Parameterize the cell in Fourier space using C4v symmetry.
    2. Use a fast, accurate plane-wave evaluator (15x15 basis, dense k-points).
    3. Optimize the Fourier coefficients using Nelder-Mead.
    4. Return the best design found within the time budget.
    """
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Time management
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin for 200s limit

    # --- 1. Symmetry and Fourier Parameterization ---
    # C4v symmetry: f(x,y) = f(-x,y) = f(x,-y) = f(y,x) = ...
    # In a discrete 32x32 grid centered at (15.5, 15.5), we can define independent pixels.
    # However, it's easier to work with Fourier coefficients.
    # For C4v, the Fourier coefficients satisfy:
    # c_{h,k} = c_{-h,k} = c_{h,-k} = c_{k,h} = c_{-k,-h} = c_{-h,-k} = c_{k,-h} = c_{-k,h}
    # And they must be real for a real-valued function.
    
    # We will optimize a set of independent real Fourier coefficients.
    # Let's pick a subset of (h,k) pairs that are unique under C4v.
    # h, k range from -15 to 15 (since size=32, max freq is 15.5, but integer freqs are -15..15).
    
    # Identify unique orbits of (h,k) under C4v
    unique_coeffs = []
    seen = set()
    for h in range(-15, 16):
        for k in range(-15, 16):
            # Generate all symmetric equivalents
            equivs = [
                (h, k), (-h, k), (h, -k), (-h, -k),
                (k, h), (-k, h), (k, -h), (-k, -h)
            ]
            # Normalize to a canonical form for set lookup
            # Sort the tuples, then take the lexicographically smallest
            equivs_sorted = sorted(equips)
            key = equivs_sorted[0]
            if key not in seen:
                seen.add(key)
                unique_coeffs.append((h, k))
                
    # Sort unique coefficients by magnitude for better optimization landscape
    unique_coeffs.sort(key=lambda x: x[0]**2 + x[1]**2)
    
    # We will optimize the first N coefficients. Let's start with a reasonable number.
    # 32x32 grid has 1024 pixels. C4v reduces this to ~400 unique pixels.
    # Fourier space is sparse. Let's use the first 50 most significant coefficients.
    num_coeffs = 60
    
    # --- 2. Fast Evaluator ---
    # Plane wave basis
    pw_half = 7  # 15x15 = 225 modes
    G_vectors = np.array([(h, k) for h in range(-pw_half, pw_half + 1) 
                          for k in range(-pw_half, pw_half + 1)], dtype=float)
    N_pw = len(G_vectors)
    
    # Precompute G vector arrays
    Gx_arr = G_vectors[:, 0]
    Gy_arr = G_vectors[:, 1]
    
    # Precompute G differences and indices
    G_diffs_h = np.zeros((N_pw, N_pw), dtype=int)
    G_diffs_k = np.zeros((N_pw, N_pw), dtype=int)
    for i in range(N_pw):
        for j in range(N_pw):
            G_diffs_h[i, j] = (G_vectors[i, 0] - G_vectors[j, 0]) % size
            G_diffs_k[i, j] = (G_vectors[i, 1] - G_vectors[j, 1]) % size
    
    # K-points in irreducible BZ
    # Irreducible wedge: 0 <= kx <= ky, kx + ky <= 0.5
    k_points = []
    # Gamma
    k_points.append((0.0, 0.0))
    # M point (0.5, 0)
    k_points.append((0.5, 0.0))
    # X point (0.5, 0.5)
    k_points.append((0.5, 0.5))
    # Along Gamma-M: (t, 0)
    for t in [0.1, 0.2, 0.3, 0.4]:
        k_points.append((t, 0.0))
    # Along M-X: (0.5, t)
    for t in [0.1, 0.2, 0.3, 0.4]:
        k_points.append((0.5, t))
    # Along Gamma-X: (t, t)
    for t in [0.1, 0.2, 0.3, 0.4]:
        k_points.append((t, t))
    # Interior points
    k_points.append((0.25, 0.1))
    k_points.append((0.3, 0.15))
    k_points.append((0.4, 0.2))
    k_points.append((0.2, 0.25))
    
    def evaluate_gap(coeffs_vec):
        """
        Computes the approximate band gap between bands 4 and 5.
        coeffs_vec: array of independent Fourier coefficients
        """
        # Reconstruct the full 2D Fourier coefficient array
        fft_coeffs = np.zeros((size, size), dtype=complex)
        
        # Set the independent coefficients and their symmetric equivalents
        for idx, (h, k) in enumerate(unique_coeffs[:num_coeffs]):
            val = coeffs_vec[idx]
            # Add all symmetric equivalents
            equivs = [
                (h, k), (-h, k), (h, -k), (-h, -k),
                (k, h), (-k, h), (k, -h), (-k, -h)
            ]
            # Avoid double counting if h==0 or k==0 or h==k
            # We already ensured unique_coeffs are unique orbits, 
            # but when populating the array, we need to be careful.
            # Actually, the list 'unique_coeffs' contains one representative per orbit.
            # So we just need to add the value to all positions in the orbit.
            for hh, kk in equivs:
                fft_coeffs[hh % size, kk % size] += val
        
        # The Fourier transform convention:
        # f(x) = sum c_k exp(i k x)
        # c_k = (1/N) sum f(x) exp(-i k x)
        # Our fft_coeffs are c_k.
        # To get the real-space grid, we do inverse FFT.
        # numpy.fft.ifft2 expects the array in standard order (0..N-1).
        # Our fft_coeffs is in standard order (0..31).
        # But wait, the DC term (0,0) is included.
        
        # Reconstruct real-space dielectric function
        # ifft2 gives (1/N) * sum. We want sum c_k exp(i k x).
        # So we multiply by N^2.
        eps_grid_complex = np.fft.ifft2(fft_coeffs) * (size * size)
        eps_grid = np.real(eps_grid_complex)
        
        # Threshold to binary 0/1
        # We want a specific fill fraction. Let's aim for 50% dielectric.
        # Sort the grid and pick the median.
        threshold = np.median(eps_grid)
        cell_grid = (eps_grid > threshold).astype(float)
        
        # Ensure exact 50% fill if possible, or close to it
        # This is a soft constraint, the optimizer will adjust coefficients to maintain symmetry
        # and the thresholding will handle the rest.
        
        # Build inverse epsilon grid
        inv_eps = np.where(cell_grid == 1, 1.0/eps_high, 1.0/eps_low)
        
        # FFT of inverse epsilon
        fft_1eps = np.fft.fft2(inv_eps)
        # Note: numpy.fft.fft2 computes sum f(x) exp(-i k x).
        # The coefficients needed for the Hamiltonian are c_{G_i - G_j} = (1/N^2) sum inv_eps(x) exp(-i (G_i-G_j) x).
        # So coeffs = fft_1eps / N^2.
        coeffs = fft_1eps[G_diffs_h, G_diffs_k] / (size * size)
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            Kx = Gx_arr + kx
            Ky = Gy_arr + ky
            
            # H = (K_i · K_j) * coeffs_{G_i - G_j}
            dots = np.outer(Kx, Kx) + np.outer(Ky, Ky)
            H = dots * coeffs
            H_real = H.real
            
            # Get eigenvalues
            eigenvalues = np.linalg.eigvalsh(H_real)
            
            if len(eigenvalues) >= 5:
                w4_sq = eigenvalues[3]
                w5_sq = eigenvalues[4]
                
                w4 = np.sqrt(max(0, w4_sq))
                w5 = np.sqrt(max(0, w5_sq))
                
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
        
        if min_omega_5 == float('inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = 2 * gap / midgap
        return ratio

    # --- 3. Optimization ---
    best_score = -float('inf')
    best_coeffs = np.zeros(num_coeffs)
    
    # Initial guess: random small coefficients
    np.random.seed(42)
    
    # Run multiple restarts
    max_restarts = 10
    restart_time_budget = (deadline - start_time) / max_restarts
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize coefficients
        init_coeffs = np.random.normal(0, 0.1, num_coeffs)
        
        # Objective function for minimizer (we want to maximize gap, so minimize -gap)
        def objective(coeffs_vec):
            gap = evaluate_gap(coeffs_vec)
            return -gap
        
        # Use Nelder-Mead
        try:
            result = minimize(objective, init_coeffs, method='Nelder-Mead', 
                              options={'xatol': 1e-3, 'fatol': 1e-3, 'maxiter': 200, 'adaptive': True})
            
            if result.success or result.fun < -best_score:
                current_score = -result.fun
                if current_score > best_score:
                    best_score = current_score
                    best_coeffs = result.x.copy()
                    
        except Exception as e:
            pass
            
    # --- 4. Finalize ---
    # Reconstruct the best cell
    fft_coeffs = np.zeros((size, size), dtype=complex)
    for idx, (h, k) in enumerate(unique_coeffs[:num_coeffs]):
        val = best_coeffs[idx]
        equivs = [
            (h, k), (-h, k), (h, -k), (-h, -k),
            (k, h), (-k, h), (k, -h), (-k, -h)
        ]
        for hh, kk in equivs:
            fft_coeffs[hh % size, kk % size] += val
            
    eps_grid_complex = np.fft.ifft2(fft_coeffs) * (size * size)
    eps_grid = np.real(eps_grid_complex)
    threshold = np.median(eps_grid)
    cell_grid = (eps_grid > threshold).astype(int)
    
    # Enforce exact symmetry on the final grid just in case of numerical noise
    # The Fourier construction should be symmetric, but let's double check.
    sym_cell = np.zeros_like(cell_grid)
    for i in range(size):
        for j in range(size):
            points = [
                (i, j),
                (size - 1 - i, j),
                (i, size - 1 - j),
                (size - 1 - i, size - 1 - j),
                (j, i),
                (size - 1 - j, j),
                (j, size - 1 - i),
                (size - 1 - j, size - 1 - i)
            ]
            s = sum(cell_grid[pi, pj] for pi, pj in points)
            sym_cell[i, j] = 1 if s >= 4 else 0
            
    return sym_cell.tolist()

# EVOLVE-BLOCK-END
