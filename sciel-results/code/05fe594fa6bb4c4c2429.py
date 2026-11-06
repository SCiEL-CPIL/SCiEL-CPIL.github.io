# EVOLVE-BLOCK-START
import time
import numpy as np

def design(problem):
    """Return a symmetric 32x32 photonic crystal unit cell designed for a TM band gap."""
    size = 32
    t_start = time.time()
    deadline = t_start + 110.0  # safety margin before 120s limit
    
    # We will use a fast approximate solver to find the best geometry
    # among a set of candidates.
    
    # Candidate generators
    def make_rod_cell(radius):
        cell = np.zeros((size, size), dtype=int)
        yy, xx = np.mgrid[0:size, 0:size]
        # Center is (15.5, 15.5)
        # Distance from center
        dist = np.sqrt((xx - 15.5)**2 + (yy - 15.5)**2)
        # Radius in pixels
        r_pix = radius * size
        cell[dist < r_pix] = 1
        return cell.tolist()

    def make_hole_cell(radius):
        cell = np.ones((size, size), dtype=int)
        yy, xx = np.mgrid[0:size, 0:size]
        dist = np.sqrt((xx - 15.5)**2 + (yy - 15.5)**2)
        r_pix = radius * size
        cell[dist < r_pix] = 0
        return cell.tolist()

    def make_four_rod_cell(radius, offset_frac):
        cell = np.zeros((size, size), dtype=int)
        yy, xx = np.mgrid[0:size, 0:size]
        # Centers at (offset, offset), (offset, -offset), etc relative to center
        # In pixel coords:
        c = 15.5
        off = offset_frac * size
        centers = [(c-off, c-off), (c+off, c-off), (c-off, c+off), (c+off, c+off)]
        r_pix = radius * size
        for cx, cy in centers:
            dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            cell[dist < r_pix] = 1
        return cell.tolist()

    def make_four_hole_cell(radius, offset_frac):
        cell = np.ones((size, size), dtype=int)
        yy, xx = np.mgrid[0:size, 0:size]
        c = 15.5
        off = offset_frac * size
        centers = [(c-off, c-off), (c+off, c-off), (c-off, c+off), (c+off, c+off)]
        r_pix = radius * size
        for cx, cy in centers:
            dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            cell[dist < r_pix] = 0
        return cell.tolist()

    # Fast TM Band Gap Estimator
    def estimate_gap(cell_list):
        # Convert to numpy array
        cell = np.array(cell_list)
        eps = 1 + (12 - 1) * cell  # 1 for air, 12 for dielectric
        
        # Compute inverse epsilon
        inv_eps = 1.0 / eps
        
        # Compute Fourier coefficients of inv_eps
        # We need coefficients for G-G' where G, G' are in the truncation set
        # Truncation: N=9 (indices -4 to 4)
        N = 9
        G_vals = np.arange(-N//2, N//2 + 1)
        G_vecs = [(gx, gy) for gx in G_vals for gy in G_vals]
        n_pw = len(G_vecs)
        
        # FFT of inv_eps
        # Note: The lattice constant a=1.
        # Fourier coeff c_G = (1/a^2) sum_r exp(-i G.r) f(r)
        # Using FFT:
        # Shift to center? The pixel grid is 0..31.
        # The function is periodic.
        # Standard FFT gives coefficients for frequencies 0..31.
        # We need frequencies -16..15 mapped to -4..4?
        # Actually, the pixel grid represents one unit cell.
        # The discrete FFT of the 32x32 grid gives the coefficients for the discrete frequencies.
        # Since the grid is 32x32, the fundamental frequency is 2*pi/32?
        # No, the unit cell is size 1. The pixels are sub-cells.
        # The Fourier series is exp(i 2 pi (n x + m y)).
        # The pixel values are samples of the function.
        # We can use np.fft.fft2.
        
        fft_inv_eps = np.fft.fft2(inv_eps)
        # The FFT output is ordered 0..31.
        # We need to map these to the plane wave indices.
        # The plane wave vector G = (2*pi/size) * (nx, ny)?
        # No, the plane waves are defined in reciprocal space of the unit cell.
        # G = (2*pi) * (m, n).
        # The pixel grid samples the function at x = (i+0.5)/32, y = (j+0.5)/32.
        # The FFT of the sampled data approximates the Fourier coefficients.
        # Coefficient c_{mn} ~ fft[i,j] / 32^2.
        # We need c_{mn} for m,n in -4..4.
        
        # fft2 output index i corresponds to frequency m = i if i < 16, else i-32.
        # Let's create a lookup or just extract the needed part.
        # Indices -4..4 correspond to:
        # 0..4 and 28..31 (since -4 is 28, -3 is 29, -2 is 30, -1 is 31)
        
        # Let's build the matrix of coefficients C[G_idx, Gp_idx] = inv_eps_coeff[G - Gp]
        # G is a vector (m,n).
        
        # First, get the coefficients for all m,n in -16..15
        # fft_inv_eps is 32x32.
        # c[m,n] = fft_inv_eps[m_mod, n_mod] / (32*32)
        # where m_mod = m % 32.
        
        # We need C_{k, l} = c_{G_k - G_l}
        # G_k and G_l are from G_vecs.
        
        C = np.zeros((n_pw, n_pw), dtype=complex)
        for i, g1 in enumerate(G_vecs):
            for j, g2 in enumerate(G_vecs):
                dm = g1[0] - g2[0]
                dn = g1[1] - g2[1]
                # Get coefficient for (dm, dn)
                idx_m = dm % 32
                idx_n = dn % 32
                C[i, j] = fft_inv_eps[idx_m, idx_n] / (32.0 * 32.0)
        
        # Now, for a given k-point, build the matrix A
        # A_{i, j} = C_{i, j} * |k + G_i|^2
        # Wait, the eigenvalue equation is:
        # sum_j C_{i,j} (k+G_i)^2 E_j = w^2 E_i ?
        # No.
        # The equation is: (k+G)^2 sum_{G'} eps^{-1}(G-G') E_{G'} = w^2 E_G
        # So A_{G, G'} = eps^{-1}(G-G') (k+G)^2
        # This is not symmetric if we just multiply by (k+G)^2.
        # However, we can transform it to a symmetric generalized eigenvalue problem or a standard one.
        # Let F_G = E_G.
        # A_{G, G'} = C_{G, G'} (k+G)^2
        # This matrix is not Hermitian.
        # We can use the symmetric form:
        # Let D_G = E_G / sqrt(eps^{-1}(0))? No.
        # Standard method:
        # H_{G, G'} = sqrt(eps^{-1}(0))^{-1} ...
        
        # Actually, for TM, the operator is self-adjoint in the inner product weighted by eps?
        # Let's use the non-symmetric matrix and find eigenvalues.
        # Or, use the symmetric transformation:
        # A = C * diag(|k+G|^2)
        # Eigenvalues of A are the same as eigenvalues of C^{1/2} A C^{-1/2}?
        # Let's just use np.linalg.eigvals on the non-symmetric matrix.
        # It should be fine for small N.
        
        k_points = []
        # Gamma (0,0)
        k_points.append((0.0, 0.0))
        # X (pi, 0)
        k_points.append((np.pi, 0.0))
        # M (pi, pi)
        k_points.append((np.pi, np.pi))
        # Midpoint Gamma-X
        k_points.append((np.pi/2, 0.0))
        # Midpoint X-M
        k_points.append((np.pi, np.pi/2))
        
        min_w5 = np.inf
        max_w4 = -np.inf
        
        for kx, ky in k_points:
            k_vec = np.array([kx, ky])
            A = np.zeros((n_pw, n_pw), dtype=complex)
            for i, g1 in enumerate(G_vecs):
                G1 = np.array(g1) * 2 * np.pi
                k_plus_G = k_vec + G1
                k_sq = np.dot(k_plus_G, k_plus_G)
                for j, g2 in enumerate(G_vecs):
                    A[i, j] = C[i, j] * k_sq
            
            # Find eigenvalues
            try:
                w2 = np.linalg.eigvals(A)
                # Filter real positive eigenvalues
                w2_real = np.real(w2)
                w2_pos = w2_real[w2_real > 1e-6]
                w2_pos.sort()
                
                if len(w2_pos) >= 5:
                    w4 = np.sqrt(w2_pos[3])
                    w5 = np.sqrt(w2_pos[4])
                    max_w4 = max(max_w4, w4)
                    min_w5 = min(min_w5, w5)
            except Exception:
                pass
        
        if min_w5 == np.inf or max_w4 == -np.inf:
            return -1.0
            
        gap = 2 * (min_w5 - max_w4) / (min_w5 + max_w4)
        return gap

    best_cell = None
    best_score = -1.0
    
    # List of candidates (name, generator_func, params)
    candidates = []
    
    # Central Rods
    for r in [0.25, 0.30, 0.35, 0.40]:
        candidates.append(("rod", make_rod_cell, r))
        
    # Central Holes
    for r in [0.25, 0.30, 0.35, 0.40]:
        candidates.append(("hole", make_hole_cell, r))
        
    # Four Rods
    for r in [0.15, 0.20, 0.25]:
        for off in [0.20, 0.25, 0.30]:
            candidates.append(("4rod", make_four_rod_cell, (r, off)))
            
    # Four Holes
    for r in [0.15, 0.20, 0.25]:
        for off in [0.20, 0.25, 0.30]:
            candidates.append(("4hole", make_four_hole_cell, (r, off)))

    for name, gen_func, param in candidates:
        if time.time() > deadline:
            break
            
        if isinstance(param, tuple):
            cell = gen_func(*param)
        else:
            cell = gen_func(param)
            
        # Check symmetry (should be symmetric by construction, but verify)
        if not _check_symmetry(cell, size):
            continue
            
        score = estimate_gap(cell)
        
        if score > best_score:
            best_score = score
            best_cell = cell
            
    # Fallback
    if best_cell is None:
        best_cell = make_rod_cell(0.3)
        
    return best_cell

def _check_symmetry(cell, size):
    """Check if cell has square symmetry."""
    # Check 180-degree rotation
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[size - 1 - i][size - 1 - j]:
                return False
    # Check transpose
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[j][i]:
                return False
    # Check horizontal reflection
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[size - 1 - i][j]:
                return False
    # Check vertical reflection
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[i][size - 1 - j]:
                return False
    return True
# EVOLVE-BLOCK-END
