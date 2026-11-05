import numpy as np
import time
# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell to maximize the TM band gap between bands 4 and 5.
    
    Strategy:
    1. Implement a fast internal PWE solver using the standard Galerkin method.
    2. Generate candidate cells using four-quadrant rod configurations with various shapes.
    3. Score each candidate with the internal solver and return the best.
    4. Maintain D4 symmetry and stay within the time budget.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    
    # Time budget: 230s limit, keep 20s margin for overhead
    start_time = time.time()
    deadline = start_time + 210.0
    
    # Fast internal PWE solver for TM modes
    def compute_tm_bands(cell, n_pw=21, n_kpts=8):
        """
        Compute TM band structure using plane-wave expansion.
        
        Parameters:
        -----------
        cell : 2D array of 0/1 (0=air, 1=dielectric)
        n_pw : number of plane waves in each direction (total = n_pw^2)
        n_kpts : number of k-points along the irreducible BZ path
        
        Returns:
        --------
        omega : array of shape (n_kpts, n_bands) with band frequencies
        """
        cell = np.array(cell, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell  # epsilon map
        inv_eps = 1.0 / eps
        
        # Compute Fourier coefficients of 1/epsilon
        # We need coefficients for G vectors in a supercell basis.
        # The unit cell is 32x32 pixels. Let's assume the lattice constant a=1.
        # G = 2*pi * (m, n) where m, n are integers.
        # The Fourier coefficient of inv_eps is:
        # inv_eps(G) = (1/N^2) * sum_{x,y} inv_eps(x,y) * exp(-i G . r)
        
        # To speed up, we compute the full FFT and extract the needed coefficients.
        # The FFT gives coefficients for frequencies k = 0, ..., N-1.
        # We need frequencies corresponding to G = 2*pi*m/N? No, G = 2*pi*m/a.
        # If we map pixels to [0, 1], then G = 2*pi * (m, n).
        # The FFT index k corresponds to frequency k/N cycles per unit, or 2*pi*k/N rad/unit.
        # So G_index = 2*pi * (k_x, k_y) / N ?
        # Wait, if the domain is [0, 1] and we have N points, the fundamental frequency is 2*pi.
        # So G_m = 2*pi * m.
        # The FFT of f(x) sampled at N points gives coefficients c_k such that
        # f(x) = sum_k c_k exp(i 2*pi k x / N).
        # So the frequency is 2*pi k / N.
        # We want G = 2*pi m. So k = m * N.
        # This implies that the "Fourier coefficients" in the PWE sense (for G=2*pi*m)
        # are related to the FFT coefficients at index k = m*N.
        # However, since the function is periodic with period 1, and we sample N points,
        # the FFT coefficients c_k for k=0..N-1 represent the frequencies 2*pi*k/N.
        # The PWE method usually sums over G vectors.
        # Let's stick to the standard definition:
        # inv_eps(G) = integral inv_eps(r) exp(-i G r) dr
        # Approximated by sum_{i,j} inv_eps(i,j) exp(-i G (i,j)/N) / N^2
        # If G = 2*pi * (m, n), then exp(-i 2*pi (m i + n j)/N).
        # This is exactly the DFT definition.
        # So inv_eps(m, n) = fft2(inv_eps)[m % N, n % N] / N^2.
        
        fft_coeffs = np.fft.fft2(inv_eps) / (size * size)
        
        def get_coeff(m, n):
            """Get Fourier coefficient for G = 2*pi*(m, n)"""
            idx_m = m % size
            idx_n = n % size
            return fft_coeffs[idx_m, idx_n]
        
        # Define the set of G vectors to include in the basis.
        # We want a circular cutoff or square cutoff.
        # n_pw is the half-width of the square box in G-space.
        # G = (2*pi*m, 2*pi*n).
        # Let's use m, n in range [-n_pw//2, n_pw//2 - 1] if n_pw is odd?
        # Or just center at 0.
        # Let's define the indices m, n such that |m| <= M and |n| <= M.
        # Total number of plane waves ~ (2M+1)^2.
        # Let's set M = n_pw // 2.
        M = n_pw // 2
        m_vals = list(range(-M, M + 1))
        if len(m_vals) < n_pw:
            # Ensure we have at least n_pw values? No, usually n_pw is the count.
            # Let's just use the range defined by M.
            pass
        
        g_indices = [(m, n) for m in m_vals for n in m_vals]
        n_total = len(g_indices)
        
        # Build the S matrix: S_ij = inv_eps(G_i - G_j)
        S = np.zeros((n_total, n_total), dtype=complex)
        for i, (mi, ni) in enumerate(g_indices):
            for j, (mj, nj) in enumerate(g_indices):
                gm = mi - mj
                gn = ni - nj
                S[i, j] = get_coeff(gm, gn)
        
        # Define k-points in the irreducible Brillouin zone
        # High symmetry points: Gamma(0,0), X(0.5,0), M(0.5,0.5)
        # Irreducible wedge: 0 <= kx <= ky, kx + ky <= 0.5 (in units of 1/(2*pi)? No, k is in radians?)
        # Standard BZ for square lattice: [-pi, pi] x [-pi, pi] if a=1?
        # If G = 2*pi*m, then BZ is [-pi, pi] x [-pi, pi].
        # High symmetry points:
        # Gamma: (0, 0)
        # X: (pi, 0) or (0, pi)
        # M: (pi, pi)
        # Let's work in units where pi = 1 for simplicity in k-point definition,
        # but remember G = 2*pi*m.
        # Actually, let's keep k in radians.
        # Gamma: (0, 0)
        # X: (np.pi, 0)
        # M: (np.pi, np.pi)
        
        # Path: Gamma -> X -> M -> Gamma
        # We will sample n_kpts points along this path.
        
        k_points = []
        # Gamma to X: (t, 0) for t in [0, pi]
        for t in np.linspace(0, np.pi, n_kpts//3 + 1):
            k_points.append((t, 0))
        # X to M: (pi, t) for t in [0, pi]
        for t in np.linspace(0, np.pi, n_kpts//3 + 1):
            k_points.append((np.pi, t))
        # M to Gamma: (t, t) for t in [pi, 0]
        for t in np.linspace(np.pi, 0, n_kpts//3 + 1):
            k_points.append((t, t))
        
        # Remove duplicates
        k_points_unique = []
        for kp in k_points:
            if not any(np.allclose(kp, kp2, atol=1e-10) for kp2 in k_points_unique):
                k_points_unique.append(kp)
        
        n_kpts_actual = len(k_points_unique)
        omega = np.zeros((n_kpts_actual, n_total))
        
        for ki, (kx, ky) in enumerate(k_points_unique):
            # Build the eigenvalue matrix for this k-point.
            # The equation is:
            # sum_{G'} S(G_i - G') * |G' + k|^2 * c_{G'} = omega^2 * c_{G_i}
            # Wait, the standard PWE equation for TM is:
            # sum_{G'} (1/eps(G_i - G')) * c_{G'} = (omega/c)^2 * (1/eps_avg) * c_{G_i} ??
            # No, that's for a specific formulation.
            
            # Correct formulation:
            # (1/eps(r)) * (nabla_perp^2 + k^2) E_z = 0
            # E_z = sum c_G exp(i (G+k) r)
            # nabla^2 E_z = sum c_G |G+k|^2 exp(i (G+k) r)
            # (1/eps(r)) * sum c_G |G+k|^2 exp(i (G+k) r) + k^2 E_z = 0
            # Multiply by exp(-i (G_i+k) r) and integrate over cell:
            # sum_{G'} c_{G'} |G'+k|^2 * (1/eps(G_i - G')) + k^2 c_{G_i} * (1/eps(0)) = 0 ??
            # No, the k^2 term is also multiplied by 1/eps(r).
            # So: sum_{G'} c_{G'} |G'+k|^2 * (1/eps(G_i - G')) + k^2 sum_{G'} c_{G'} * (1/eps(G_i - G')) = 0
            # This doesn't look like a standard eigenvalue problem for omega^2.
            
            # Let's use the alternative form often cited:
            # sum_{G'} S(G_i - G') c_{G'} = lambda c_{G_i}
            # where S(G) = 1/eps(G) and lambda = (omega a / 2pi)^2 ?
            # This form assumes that the operator is 1/eps * (-nabla^2).
            # -nabla^2 E = sum c_G |G+k|^2 exp(i(G+k)r).
            # So the matrix element is S(G_i - G') * |G' + k|^2.
            # And the eigenvalue is (omega/c)^2?
            # If we solve: sum S(G_i - G') |G' + k|^2 c_{G'} = (omega/c)^2 c_{G_i}
            # This is a generalized eigenvalue problem if S is not identity.
            # But S is just a matrix.
            # So we construct M_{ij} = S(G_i - G_j) * |G_j + k|^2.
            # And solve M c = lambda c.
            # lambda = (omega/c)^2.
            
            # Note: |G_j + k|^2 = (G_jx + kx)^2 + (G_jy + ky)^2.
            # G_j = 2*pi * (m_j, n_j).
            
            M = np.zeros((n_total, n_total), dtype=complex)
            for i, (mi, ni) in enumerate(g_indices):
                Gi_x = 2.0 * np.pi * mi
                Gi_y = 2.0 * np.pi * ni
                for j, (mj, nj) in enumerate(g_indices):
                    Gj_x = 2.0 * np.pi * mj
                    Gj_y = 2.0 * np.pi * nj
                    G_diff_x = Gi_x - Gj_x
                    G_diff_y = Gi_y - Gj_y
                    # S(G_i - G_j)
                    s_val = get_coeff(int(round(G_diff_x / (2.0 * np.pi))), int(round(G_diff_y / (2.0 * np.pi))))
                    
                    # |G_j + k|^2
                    kx_total = Gj_x + kx
                    ky_total = Gj_y + ky
                    k_mag_sq = kx_total**2 + ky_total**2
                    
                    M[i, j] = s_val * k_mag_sq
            
            # Solve eigenvalue problem
            # Since M is Hermitian (S is real and symmetric, k_mag_sq is real),
            # we can use eigh.
            # However, M might not be positive definite?
            # k_mag_sq is positive. S(0) is positive.
            # Let's just use eig and take real parts.
            try:
                eigvals, eigvecs = np.linalg.eigh(M)
                # Sort eigenvalues
                idx = np.argsort(eigvals)
                eigvals = eigvals[idx]
                # Take the first n_total bands
                omega[ki, :] = np.sqrt(np.real(eigvals))
            except Exception:
                omega[ki, :] = np.inf
        
        return omega

    # Generate candidate cells
    def make_four_quadrant_cell(rod_radius, exponent=2.0):
        """
        Create a cell with four rods centered in each quadrant.
        """
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        
        # Rod centers in normalized coordinates [-0.5, 0.5]
        centers = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
        
        # Vectorized generation for speed
        X, Y = np.meshgrid(coords, coords, indexing='ij')
        
        for cx, cy in centers:
            DX = X - cx
            DY = Y - cy
            # Superellipse distance
            if rod_radius > 0:
                val = np.abs(DX/rod_radius)**exponent + np.abs(DY/rod_radius)**exponent
            else:
                val = np.zeros_like(X)
            cell[val <= 1.0 + 1e-9] = 1
        
        return cell.tolist()

    # Search parameters
    best_score = -np.inf
    best_cell = None
    
    # Grid search over radius and exponent
    radii = np.linspace(0.08, 0.16, 17)
    exponents = [2.0, 2.5, 3.0, 4.0]
    
    # Pre-compute some candidates to avoid redundant work if needed?
    # Just loop.
    
    for r in radii:
        if time.time() > deadline - 10:
            break
        for exp in exponents:
            if time.time() > deadline - 5:
                break
            
            cell = make_four_quadrant_cell(r, exp)
            
            # Quick heuristic check: fill fraction
            fill = np.sum(np.array(cell)) / (size * size)
            # Optimal fill for TM gap is often around 0.3-0.5 for high eps?
            # With eps=12, optimal fill might be different.
            # Let's just compute.
            
            try:
                bands = compute_tm_bands(cell, n_pw=17, n_kpts=6)
                # Extract bands 4 and 5 (index 3 and 4)
                # bands shape: (n_kpts, n_bands)
                # We need min omega_5 and max omega_4.
                # omega_4 is the 4th band (index 3).
                # omega_5 is the 5th band (index 4).
                
                omega_4 = bands[:, 3]
                omega_5 = bands[:, 4]
                
                max_omega_4 = np.max(omega_4)
                min_omega_5 = np.min(omega_5)
                
                if min_omega_5 > max_omega_4:
                    gap = min_omega_5 - max_omega_4
                    midgap = (min_omega_5 + max_omega_4) / 2.0
                    ratio = 2.0 * gap / (min_omega_5 + max_omega_4)
                else:
                    ratio = -1.0
                
                if ratio > best_score:
                    best_score = ratio
                    best_cell = cell
            except Exception:
                continue
    
    # If no cell was found or all failed, return the default
    if best_cell is None:
        best_cell = make_four_quadrant_cell(0.12, 2.0)
        
    return best_cell

# EVOLVE-BLOCK-END
