"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """Optimized photonic crystal design for maximum TM 4-5 band gap."""
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 110.0  # 110s budget, leaving 10s margin
    
    # Plane Wave Method Setup
    # The evaluator uses 289 plane waves. 289 = 17 * 17.
    # So N = 17. The G vectors range from -8 to 8.
    N = 17
    half_N = N // 2
    g_indices = np.arange(-half_N, half_N + 1)
    # G vectors are (m, n) * 2*pi. We work in units where 2*pi = 1 for simplicity in k-space scaling?
    # No, let's stick to standard units.
    # The Hamiltonian for TM is H_{G, G'} = (k+G) dot (k+G') * eps^{-1}_{G-G'}
    # where eps^{-1} is the Fourier coefficient of 1/epsilon(r).
    
    def get_g_vectors():
        """Generate list of G vectors (m, n) for N x N grid."""
        gs = []
        for m in g_indices:
            for n in g_indices:
                gs.append((m, n))
        return np.array(gs)
    
    G_vecs = get_g_vectors()
    n_g = len(G_vecs)
    
    # Pre-compute G-G' differences
    # G_diff[i, j] = G_vecs[i] - G_vecs[j]
    # We need the Fourier coefficient of 1/eps at this difference.
    # The Fourier coefficient c_{G} = (1/a^2) * int exp(-i G r) (1/eps(r)) dr
    # For a pixelated map, c_{G} = (1/S^2) * sum_{pixels} (1/eps_{pixel}) * exp(-i G * r_{pixel})
    # Note: The sum is over the unit cell.
    
    def compute_fourier_coeffs(cell):
        """
        Compute Fourier coefficients of 1/eps for the given cell.
        cell is a 2D list of 0s and 1s.
        Returns a dictionary mapping (dm, dn) to coefficient.
        """
        eps = np.array(cell, dtype=float)
        eps = eps_low + (eps_high - eps_low) * eps
        inv_eps = 1.0 / eps
        
        # We need coefficients for G = (dm, dn) where dm, dn in [-half_N, half_N]
        # c_{dm, dn} = (1/S^2) * sum_{i,j} inv_eps[i,j] * exp(-i * 2*pi * (dm*i/S + dn*j/S))
        # This is just the FFT of inv_eps, normalized.
        # np.fft.fft2 gives sum_{i,j} f[i,j] exp(-i 2 pi (k i / S + l j / S))
        # So c_{k,l} = fft2(inv_eps)[k, l] / S^2
        # We need to map indices dm, dn (which can be negative) to fft indices.
        # fft index k corresponds to frequency k if k <= S/2, else k-S.
        
        fft_inv_eps = np.fft.fft2(inv_eps)
        fft_inv_eps /= (size * size)
        
        # Create a lookup table for coefficients
        # We only need coefficients for differences within [-half_N, half_N]
        coeffs = {}
        for dm in g_indices:
            for dn in g_indices:
                # Map dm, dn to fft indices
                idx_m = dm % size
                idx_n = dn % size
                coeffs[(dm, dn)] = fft_inv_eps[idx_m, idx_n]
        return coeffs

    def compute_bands(cell, k_points):
        """
        Compute bands for a given cell and list of k-points.
        Returns list of (max_omega_4, min_omega_5) tuples for each k-point?
        Actually, we want the global max of band 4 and global min of band 5.
        """
        coeffs = compute_fourier_coeffs(cell)
        
        # Pre-compute the A matrix part: A[i, j] = coeffs[G_i - G_j]
        # This depends only on the cell, not on k.
        A = np.zeros((n_g, n_g), dtype=complex)
        for i in range(n_g):
            for j in range(n_g):
                dm = G_vecs[i, 0] - G_vecs[j, 0]
                dn = G_vecs[i, 1] - G_vecs[j, 1]
                A[i, j] = coeffs[(int(dm), int(dn))]
        
        max_omega4 = 0.0
        min_omega5 = float('inf')
        
        for kx, ky in k_points:
            # k is in units of 2*pi/a. So k_vector = (kx, ky) * 2*pi?
            # In the formula H = (k+G) dot (k+G') / eps, if k and G are in reciprocal space units (1/length),
            # and we set a=1, then G = 2*pi*(m,n).
            # Let's use k in units of 2*pi. So actual k_vec = (kx, ky) * 2*pi.
            # G_vec = (m, n) * 2*pi.
            # (k+G) dot (k+G') = (2*pi)^2 * (kx+m, ky+n) dot (kx+m', ky+n')
            # We can factor out (2*pi)^2 from the eigenvalues.
            # omega^2 = lambda * (2*pi)^2? No.
            # The equation is: sum (k+G) dot (k+G') (1/eps)_{G-G'} E_{G'} = omega^2 E_G
            # If k and G are in units of 2*pi/a, let's just use the numbers directly and multiply by (2*pi)^2 at the end if needed for frequency, but the ratio is scale invariant.
            # Let's define K_vecs = k + G_vecs.
            # H[i, j] = K_vecs[i] dot K_vecs[j] * A[i, j]
            
            K_vecs = G_vecs + np.array([kx, ky])
            
            # Compute dot products
            # Dot[i, j] = K_vecs[i,0]*K_vecs[j,0] + K_vecs[i,1]*K_vecs[j,1]
            # This is K_vecs @ K_vecs.T
            dots = np.sum(K_vecs[:, np.newaxis, :] * K_vecs[np.newaxis, :, :], axis=2)
            
            H = dots * A
            
            # H should be Hermitian. Due to numerical errors, symmetrize.
            H = (H + H.conj().T) / 2.0
            
            try:
                eigvals = np.linalg.eigvalsh(H)
                eigvals = np.real(eigvals)
                # Filter out negative values (numerical noise)
                eigvals = np.maximum(eigvals, 0)
                # Sort
                eigvals = np.sort(eigvals)
                
                if len(eigvals) >= 5:
                    # Bands are 1-indexed in task, so 4 and 5 are indices 3 and 4.
                    omega4 = math.sqrt(eigvals[3])
                    omega5 = math.sqrt(eigvals[4])
                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
            except Exception:
                continue
        
        if min_omega5 == float('inf') or max_omega4 == 0:
            return -1.0
        
        gap = min_omega5 - max_omega4
        midgap = 0.5 * (min_omega5 + max_omega4)
        if midgap == 0:
            return -1.0
        return gap / midgap

    # Define K-points in the irreducible Brillouin Zone
    # Square lattice BZ: -pi/a < kx, ky < pi/a.
    # Irreducible wedge: 0 <= kx <= ky, kx^2 + ky^2 <= (pi/a)^2? No, that's circular.
    # Square BZ is a square. Irreducible 1/8th triangle:
    # 0 <= kx <= ky <= pi/a.
    # In units of 2*pi/a (so BZ is -0.5 to 0.5):
    # 0 <= kx <= ky <= 0.5.
    
    k_points = []
    # Add points along the boundaries and interior
    # Gamma (0,0)
    k_points.append((0.0, 0.0))
    # X (0.5, 0)
    k_points.append((0.5, 0.0))
    # M (0.5, 0.5)
    k_points.append((0.5, 0.5))
    
    # Grid in the triangle
    # kx from 0 to 0.5
    # ky from kx to 0.5
    n_k = 6
    for i in range(n_k):
        kx = 0.5 * i / (n_k - 1)
        for j in range(n_k - i):
            ky = kx + (0.5 - kx) * j / max(1, n_k - 1 - i) # This logic is messy, let's just use a simple grid and filter
            
    k_points = []
    n_grid = 7
    for i in range(n_grid):
        kx = 0.5 * i / (n_grid - 1)
        for j in range(n_grid - i):
            # ky ranges from kx to 0.5
            if n_grid - 1 - i == 0:
                ky = 0.5
            else:
                ky = kx + (0.5 - kx) * j / (n_grid - 1 - i)
            k_points.append((kx, ky))
    
    # Remove duplicates
    k_points = list(set(k_points))

    def make_cell_circle_hole(radius):
        """Dielectric background with circular air hole at center."""
        cell = [[1]*size for _ in range(size)]
        center = size / 2.0
        for i in range(size):
            for j in range(size):
                dx = (i + 0.5) - center
                dy = (j + 0.5) - center
                r = math.sqrt(dx*dx + dy*dy)
                if r < radius * size:
                    cell[i][j] = 0
        return cell

    def make_cell_square_hole(half_side):
        """Dielectric background with square air hole at center."""
        cell = [[1]*size for _ in range(size)]
        center = size / 2.0
        for i in range(size):
            for j in range(size):
                dx = abs((i + 0.5) - center)
                dy = abs((j + 0.5) - center)
                if dx < half_side * size and dy < half_side * size:
                    cell[i][j] = 0
        return cell

    def make_cell_diamond_hole(half_diag):
        """Dielectric background with diamond (rotated square) air hole."""
        cell = [[1]*size for _ in range(size)]
        center = size / 2.0
        for i in range(size):
            for j in range(size):
                dx = abs((i + 0.5) - center)
                dy = abs((j + 0.5) - center)
                if dx + dy < half_diag * size:
                    cell[i][j] = 0
        return cell

    best_score = -1.0
    best_cell = None

    # Search parameters
    # Radius of hole in units of cell size (0 to 0.5)
    radii = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    
    # 1. Circular Holes
    for r in radii:
        if time.time() > deadline:
            break
        cell = make_cell_circle_hole(r)
        score = compute_bands(cell, k_points)
        if score > best_score:
            best_score = score
            best_cell = cell

    # 2. Square Holes
    for h in radii:
        if time.time() > deadline:
            break
        cell = make_cell_square_hole(h)
        score = compute_bands(cell, k_points)
        if score > best_score:
            best_score = score
            best_cell = cell

    # 3. Diamond Holes
    for d in radii:
        if time.time() > deadline:
            break
        cell = make_cell_diamond_hole(d)
        score = compute_bands(cell, k_points)
        if score > best_score:
            best_score = score
            best_cell = cell

    # Refine the best candidate
    if best_cell is not None and time.time() < deadline - 20:
        # Determine which shape was best by checking the cell content roughly
        # Just refine all shapes around the best score's likely parameter?
        # Simpler: Just do a finer grid for all shapes.
        fine_radii = [0.15, 0.2, 0.25, 0.3, 0.35]
        
        for r in fine_radii:
            if time.time() > deadline:
                break
            cell = make_cell_circle_hole(r)
            score = compute_bands(cell, k_points)
            if score > best_score:
                best_score = score
                best_cell = cell

        for h in fine_radii:
            if time.time() > deadline:
                break
            cell = make_cell_square_hole(h)
            score = compute_bands(cell, k_points)
            if score > best_score:
                best_score = score
                best_cell = cell

        for d in fine_radii:
            if time.time() > deadline:
                break
            cell = make_cell_diamond_hole(d)
            score = compute_bands(cell, k_points)
            if score > best_score:
                best_score = score
                best_cell = cell

    # Fallback: Return a valid symmetric cell if nothing found
    if best_cell is None:
        best_cell = make_cell_circle_hole(0.2)

    return best_cell
# EVOLVE-BLOCK-END
