"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def design(problem):
    """
    Design a 32x32 photonic crystal unit cell with D4 symmetry to maximize
    the gap between TM bands 4 and 5.
    
    Strategy:
    1. Generate a few candidate designs (circular rods with different radii).
    2. Use a fast approximate plane-wave method to estimate the gap between bands 4 and 5.
    3. Return the design with the largest estimated gap.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    def generate_cell(radius_frac):
        """Generate a 32x32 cell with a central circular rod of given radius fraction."""
        cell = [[0] * size for _ in range(size)]
        r_px = radius_frac * size
        cx = size / 2.0
        cy = size / 2.0
        for i in range(size):
            for j in range(size):
                x = (j + 0.5) - cx
                y = (i + 0.5) - cy
                if x * x + y * y < r_px * r_px:
                    cell[i][j] = 1
        return cell
    
    def compute_fourier_coefficients(cell, max_order=4):
        """
        Compute Fourier coefficients of 1/epsilon for G vectors with |m|,|n| <= max_order.
        """
        eps = np.array([[eps_high if cell[i][j] == 1 else eps_low for j in range(size)] for i in range(size)])
        inv_eps = 1.0 / eps
        
        coeffs = {}
        for m in range(-max_order, max_order + 1):
            for n in range(-max_order, max_order + 1):
                # Calculate Fourier coefficient: (1/N^2) * sum_{x,y} f(x,y) exp(-i 2pi (mx + ny) / N)
                # Since the cell is discrete pixels, we sum over pixels.
                # x, y are pixel indices 0..N-1.
                # The basis function is exp(-i 2pi (m*x/N + n*y/N))
                val = 0.0
                for i in range(size):
                    for j in range(size):
                        val += inv_eps[i, j] * np.exp(-1j * 2 * np.pi * (m * i / size + n * j / size))
                coeffs[(m, n)] = val / (size * size)
        return coeffs
    
    def approximate_gap_ratio(cell, max_order=4, k_points=[(0,0), (0.5,0), (0.5,0.5)]):
        """
        Approximate the gap ratio between bands 4 and 5.
        Uses a small basis set of plane waves.
        """
        coeffs = compute_fourier_coefficients(cell, max_order)
        
        # Define the set of G vectors to use in the basis
        # We need enough G vectors to resolve the bands up to index 5.
        # A 5x5 grid of G vectors (m,n in -2..2) gives 25 states.
        # Let's use m,n in -2..2.
        g_vectors = [(m, n) for m in range(-2, 3) for n in range(-2, 3)]
        
        # Build the Hamiltonian matrix for a specific k-point
        # H_{G', G} = |k+G'|^2 * (1/epsilon)_{G'-G} for TM
        # Wait, the matrix element is (k+G')^2 * (1/epsilon)_{G'-G}
        # Note: (1/epsilon) is the Fourier transform of the inverse dielectric function.
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            # Build matrix
            N = len(g_vectors)
            H = np.zeros((N, N), dtype=complex)
            
            for i, Gp in enumerate(g_vectors):
                for j, G in enumerate(g_vectors):
                    # G_diff = Gp - G
                    m_diff = Gp[0] - G[0]
                    n_diff = Gp[1] - G[1]
                    
                    # Check if (m_diff, n_diff) is in coeffs
                    if (m_diff, n_diff) in coeffs:
                        eps_inv_G = coeffs[(m_diff, n_diff)]
                    else:
                        # If not computed, assume 0 (valid for high order if max_order is sufficient)
                        # For max_order=4, diff can be up to 4, so it should be there.
                        eps_inv_G = 0.0
                    
                    # k + Gp
                    kx_total = kx + Gp[0]
                    ky_total = ky + Gp[1]
                    k_sq = kx_total**2 + ky_total**2
                    
                    H[i, j] = k_sq * eps_inv_G
            
            # Solve eigenvalue problem
            # H is Hermitian
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals.real)
            
            # We are interested in bands 4 and 5 (indices 3 and 4 in 0-based)
            if len(eigvals) >= 5:
                omega_4 = eigvals[3]
                omega_5 = eigvals[4]
                
                if omega_5 < min_omega_5:
                    min_omega_5 = omega_5
                if omega_4 > max_omega_4:
                    max_omega_4 = omega_4
        
        if min_omega_5 == float('inf') or max_omega_4 == 0:
            return -1.0
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap <= 0:
            return -1.0
        
        return gap / midgap
    
    # Candidate radii
    radii = [0.20, 0.22, 0.25, 0.28, 0.30]
    
    best_score = -1.0
    best_cell = generate_cell(0.20)
    
    for r in radii:
        if time.time() - start_time > deadline:
            break
        cell = generate_cell(r)
        score = approximate_gap_ratio(cell)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Enforce D4 symmetry strictly (just in case)
    final_cell = best_cell
    for i in range(size):
        for j in range(size):
            vals = [
                final_cell[i][j],
                final_cell[size-1-i][j],
                final_cell[i][size-1-j],
                final_cell[size-1-i][size-1-j],
                final_cell[j][i],
                final_cell[size-1-j][size-1-i],
                final_cell[j][size-1-i],
                final_cell[size-1-j][i],
            ]
            avg = sum(vals) / len(vals)
            final_cell[i][j] = 1 if avg >= 0.5 else 0
    
    return final_cell

# EVOLVE-BLOCK-END
