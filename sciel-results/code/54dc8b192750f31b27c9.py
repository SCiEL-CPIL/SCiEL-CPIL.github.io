"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def design(problem):
    """Return a symmetric 32x32 photonic crystal unit cell designed for a TM band gap."""
    size = 32
    eps_low = problem.get('eps_low', 1)
    eps_high = problem.get('eps_high', 12)
    
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    # Fast band gap evaluation using plane wave method
    # We use a reduced basis for speed during optimization, 
    # but the logic mirrors the exact method.
    
    # For TM polarization, we solve:
    # (1/eps(k)) * |k|^2 * a = omega^2 * a
    # In matrix form: sum_{G'} (k+G)^2 * (1/eps)_{G-G'} * a_{G'} = omega^2 * a_G
    
    # Let's precompute the G vectors for a truncated basis.
    # The evaluator uses 289 plane waves, which is (2*8+1)^2 = 17^2.
    # We will use the same basis for accuracy in our local search.
    
    npw = 8
    G_list = []
    for mx in range(-npw, npw + 1):
        for my in range(-npw, npw + 1):
            G_list.append((2 * np.pi * mx, 2 * np.pi * my))
    G = np.array(G_list)
    nG = len(G)
    
    # Precompute k-points in the irreducible Brillouin zone
    # The irreducible BZ for a square lattice is the triangle with vertices
    # Gamma (0,0), M (pi,0), and X (pi/2, pi/2).
    # We sample points on the boundaries and some interior points.
    
    def get_k_points():
        points = []
        # Gamma to M
        for t in np.linspace(0, 1, 5):
            points.append((np.pi * t, 0))
        # M to X
        for t in np.linspace(0, 1, 5):
            points.append((np.pi - np.pi/2 * t, np.pi/2 * t))
        # X to Gamma
        for t in np.linspace(0, 1, 5):
            points.append((np.pi/2 * (1 - t), np.pi/2 * (1 - t)))
        # Some interior points
        points.append((np.pi/3, np.pi/6))
        points.append((np.pi/4, np.pi/8))
        return points
    
    k_points = get_k_points()
    
    def compute_gap(cell_2d):
        """
        Compute the gap-midgap ratio between TM bands 4 and 5.
        """
        eps = np.where(np.array(cell_2d) == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        
        # Compute Fourier coefficients of 1/eps
        # (1/eps)_G = (1/N^2) sum_{i,j} (1/eps)_{ij} exp(-i G . r_{ij})
        N = size
        x = (np.arange(N) + 0.5) / N
        y = (np.arange(N) + 0.5) / N
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        inv_eps_coeffs = np.zeros(nG, dtype=complex)
        for idx, (gx, gy) in enumerate(G):
            inv_eps_coeffs[idx] = np.sum(inv_eps * np.exp(-1j * (gx * X + gy * Y))) / (N * N)
        
        max_omega4_sq = 0.0
        min_omega5_sq = np.inf
        
        for kx, ky in k_points:
            kvec = np.array([kx, ky])
            k_plus_G = kvec + G  # shape (nG, 2)
            k_sq = np.sum(k_plus_G ** 2, axis=1)
            
            # Build the eigenvalue matrix
            # M[G, G'] = (k+G)^2 * (1/eps)_{G-G'}
            # Wait, the standard formulation is:
            # sum_{G'} (k+G)·(k+G') / eps_{G-G'} * a_{G'} = omega^2 * a_G
            # No, for TM (Ez), the equation is:
            # (1/eps(r)) * (-nabla^2) * Ez = omega^2 * Ez
            # In Fourier space:
            # sum_{G'} (k+G)^2 * (1/eps)_{G-G'} * a_{G'} = omega^2 * a_G
            # Note: (k+G)^2 is the magnitude squared of the wave vector.
            
            # We need the matrix A where A[i, j] = (1/eps)_{G_i - G_j}
            # Then the eigenvalue problem is: diag(k_sq) @ A @ a = omega^2 @ a
            # Wait, this is not symmetric. 
            # Actually, the operator is (1/eps) * (-nabla^2).
            # Let's use the standard symmetric formulation if possible, or just solve the non-symmetric one.
            # Since (1/eps) is real and symmetric in G-space (if the structure is symmetric),
            # and diag(k_sq) is real and symmetric, the product is not necessarily symmetric.
            # However, we can solve the generalized eigenvalue problem or just use eigh on a transformed matrix.
            # For speed, let's just use np.linalg.eigvals.
            
            # Construct matrix M
            # M[i, j] = k_sq[i] * inv_eps_coeffs[G_i - G_j]
            
            # To find inv_eps_coeffs[G_i - G_j], we need to map (G_i - G_j) to an index.
            # G_i - G_j = 2*pi * (mx_i - mx_j, my_i - my_j)
            # The difference indices range from -2*npw to 2*npw.
            # We only computed inv_eps_coeffs for G in [-npw, npw].
            # If G_i - G_j is outside this range, the coefficient is 0 (due to truncation).
            
            # Let's create a lookup dictionary for the coefficients
            coeff_dict = {}
            for idx, (gx, gy) in enumerate(G):
                coeff_dict[(gx, gy)] = inv_eps_coeffs[idx]
            
            M = np.zeros((nG, nG), dtype=complex)
            for i in range(nG):
                for j in range(nG):
                    dgx = G[i, 0] - G[j, 0]
                    dgy = G[i, 1] - G[j, 1]
                    c = coeff_dict.get((dgx, dgy), 0.0)
                    M[i, j] = k_sq[i] * c
            
            # Solve eigenvalue problem
            # M a = omega^2 a
            # Since M is not symmetric, we use eigvals
            eigvals = np.linalg.eigvals(M)
            # Take the real parts (should be real for physical systems)
            eigvals_real = np.real(eigvals)
            # Sort
            eigvals_real = np.sort(eigvals_real)
            
            # Find bands 4 and 5 (indices 3 and 4)
            # We need to be careful about degeneracies.
            # The problem asks for the gap between the 4th and 5th bands.
            # This implies distinct bands.
            
            if len(eigvals_real) >= 5:
                omega4_sq = eigvals_real[3]
                omega5_sq = eigvals_real[4]
                max_omega4_sq = max(max_omega4_sq, omega4_sq)
                min_omega5_sq = min(min_omega5_sq, omega5_sq)
        
        if min_omega5_sq == np.inf or max_omega4_sq == 0.0:
            return 0.0
        
        omega4 = np.sqrt(max_omega4_sq)
        omega5 = np.sqrt(min_omega5_sq)
        
        gap = 2 * (omega5 - omega4) / (omega5 + omega4)
        return gap

    def quadrant_to_full(q):
        """Convert a 16x16 quadrant to a full 32x32 symmetric cell."""
        q = np.array(q, dtype=int)
        top = np.hstack([q, q[:, ::-1]])
        bottom = top[::-1, :]
        return np.vstack([top, bottom])

    def full_to_quadrant(cell):
        """Extract the top-left 16x16 quadrant from a full 32x32 cell."""
        return np.array(cell[:16, :16], dtype=int)

    # Initial candidate: A central dielectric disk
    # Let's start with a disk of radius ~0.3 in a 16x16 quadrant
    # Center of quadrant is (15.5, 15.5) in pixel coords? No, the quadrant is 16x16.
    # The center of the full cell is at (15.5, 15.5) in pixel indices (0-based).
    # In the quadrant (0..15, 0..15), the center is at (15.5, 15.5).
    
    def create_disk_quadrant(radius):
        q = np.zeros((16, 16), dtype=int)
        # Pixel centers in quadrant
        i_idx = np.arange(16)
        j_idx = np.arange(16)
        I, J = np.meshgrid(i_idx, j_idx, indexing='ij')
        
        # Distance from center (15.5, 15.5)
        # Note: The full cell center is between pixels 15 and 16.
        # So in the quadrant, the center is at (15.5, 15.5).
        dist_sq = (I - 15.5) ** 2 + (J - 15.5) ** 2
        
        # Radius in pixels. The full cell size is 32.
        # If we want a physical radius r (0 to 0.5 of cell), 
        # in pixels it is r * 32.
        # But wait, the quadrant is 16x16, which is half the cell.
        # So a physical radius r corresponds to r * 32 pixels in the full cell.
        # In the quadrant, the distance is in pixels.
        
        # Let's just iterate over possible radii in pixels
        q[dist_sq < radius ** 2] = 1
        return q

    best_score = -1.0
    best_cell = None
    
    # Strategy 1: Try different disk radii
    for r_pix in np.arange(2, 16, 0.5):
        q = create_disk_quadrant(r_pix)
        cell = quadrant_to_full(q)
        score = compute_gap(cell)
        if score > best_score:
            best_score = score
            best_cell = cell
        if time.time() > deadline - 20:
            break

    # Strategy 2: Local search on the best cell found
    if best_cell is not None:
        q_best = full_to_quadrant(best_cell)
        
        # Random hill climbing
        for step in range(200):
            if time.time() > deadline - 5:
                break
            
            # Pick a random pixel in the quadrant
            i = random.randint(0, 15)
            j = random.randint(0, 15)
            
            # Flip it
            q_new = q_best.copy()
            q_new[i, j] = 1 - q_new[i, j]
            
            cell_new = quadrant_to_full(q_new)
            score_new = compute_gap(cell_new)
            
            if score_new > best_score:
                best_score = score_new
                best_cell = cell_new
                q_best = q_new

    # Return the best cell as a nested list
    if best_cell is None:
        # Fallback
        best_cell = np.zeros((32, 32), dtype=int)
    
    return best_cell.tolist()
# EVOLVE-BLOCK-END
