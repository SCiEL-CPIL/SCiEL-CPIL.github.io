import numpy as np
import time
import scipy.linalg as sla

def design(problem):
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    
    start_time = time.time()
    deadline = start_time + 210.0  # 20s margin for safety

    def check_time():
        return time.time() > deadline - 5.0

    # --- PWE Solver ---
    
    def compute_bands_and_fields(cell, n_pw=17, n_kpts=40):
        """
        Computes TM bands and returns:
        - gap_ratio: 2*(min_w5 - max_w4) / (min_w5 + max_w4)
        - max_w4: Maximum frequency of band 4
        - min_w5: Minimum frequency of band 5
        - I4: Field intensity of band 4 at k where w4 is max
        - I5: Field intensity of band 5 at k where w5 is min
        """
        cell = np.array(cell, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell
        inv_eps = 1.0 / eps
        
        # Fourier coefficients of 1/epsilon
        # fft2 computes sum_{x,y} f(x,y) exp(-i 2 pi (m x + n y) / N)
        # We need c_G = 1/N^2 sum f(x,y) exp(-i G . r)
        # G = 2 pi (m, n) / a. In our discrete grid, a=size, so G = 2 pi (m, n)/size.
        # Standard FFT gives coefficients for frequencies m, n in [0, N-1].
        # We need frequencies in range [-M, M].
        # fftshift moves negative frequencies to the left.
        fft_coeffs = np.fft.fftshift(np.fft.fft2(inv_eps)) / (size * size)
        
        # Determine G vector indices
        # n_pw is odd, so M = n_pw // 2
        M = n_pw // 2
        m_vals = np.arange(-M, M + 1)
        g_indices = [(m, n) for m in m_vals for n in m_vals]
        n_total = len(g_indices)
        
        # Map g_indices to fft_coeffs indices
        # fft_coeffs is shifted, so index 0 corresponds to -M.
        # g_index m is in [-M, M]. Offset by M to get [0, 2M].
        idx_map = np.array([m + M for m, n in g_indices])
        idx_map_n = np.array([n + M for m, n in g_indices])
        
        # Build S matrix (inverse permittivity matrix)
        # S_ij = c_{G_i - G_j}
        # We can vectorize this by computing all differences.
        g_m = np.array([g[0] for g in g_indices])
        g_n = np.array([g[1] for g in g_indices])
        
        dm = g_m[:, None] - g_m[None, :]
        dn = g_n[:, None] - g_n[None, :]
        
        # Map differences to fft_coeffs indices
        # Difference d is in [-2M, 2M]. fft_coeffs covers [-M, M].
        # Wait, G_i - G_j can be outside [-M, M].
        # However, c_G is periodic with period N (size).
        # So c_G = c_{G mod N}.
        # Since N=32 and 2M = 16 (for n_pw=17), the difference is in [-16, 16].
        # This fits within [-16, 16], which is within [-31, 31].
        # So we just need to map d to [0, 31] or [-16, 15] depending on fft_coeffs layout.
        # fft_coeffs is fftshifted, so it represents frequencies [-16, 15] for N=32.
        # Our differences are in [-16, 16].
        # Index in fft_coeffs: d + M (if d is in [-M, M]).
        # If d is outside, we wrap.
        idx_dm = (dm + M) % size
        idx_dn = (dn + M) % size
        
        S = fft_coeffs[idx_dm, idx_dn]
        
        # Define k-points in the irreducible Brillouin zone
        # Square lattice IBZ: Triangle (0,0) -> (0.5, 0) -> (0.5, 0.5) -> (0,0)
        # In units of 2*pi/a.
        k_points = []
        n_per_seg = max(1, n_kpts // 3)
        
        # Gamma (0,0) to X (0.5, 0)
        for t in np.linspace(0, 0.5, n_per_seg + 1):
            k_points.append((t, 0.0))
        # X (0.5, 0) to M (0.5, 0.5)
        for t in np.linspace(0, 0.5, n_per_seg + 1):
            k_points.append((0.5, t))
        # M (0.5, 0.5) to Gamma (0,0)
        for t in np.linspace(0.5, 0, n_per_seg + 1):
            k_points.append((t, t))
        
        # Remove duplicates
        k_points_unique = []
        for kp in k_points:
            if not any(np.allclose(kp, kp2, atol=1e-10) for kp2 in k_points_unique):
                k_points_unique.append(kp)
        
        max_w4 = -np.inf
        min_w5 = np.inf
        best_k_4 = None
        best_k_5 = None
        best_vec_4 = None
        best_vec_5 = None
        
        for kx, ky in k_points_unique:
            if check_time():
                break
            
            # Build eigenvalue matrix
            # The equation is: sum_{G'} S(G_i - G') |G' + k|^2 c_{G'} = omega^2 c_{G_i}
            # This is a generalized eigenvalue problem? No, standard form if we multiply by S^{-1}?
            # Actually, the standard PWE equation for TM is:
            # sum_{G'} \kappa(G_i - G') (G_j + k) \cdot (G_i + k) c_{G'} = omega^2 c_{G_i}
            # where \kappa is 1/epsilon.
            # This is a standard eigenvalue problem: M c = omega^2 c
            # M_{ij} = S_{ij} * ((G_i+k) \cdot (G_j+k)) ? 
            # Wait. The operator is (G+k) \cdot \nabla (1/epsilon) \nabla \cdot (G+k) ?
            # For TM, H is in plane, E_z is out.
            # \nabla \times (1/epsilon \nabla \times E) = (omega/c)^2 E
            # Leads to: sum_{G'} \kappa(G_i - G') (G_i + k) \cdot (G_j + k) c_{G'} = omega^2 c_{G_i}
            # Note: (G_i+k) \cdot (G_j+k) is a scalar.
            
            Gx = 2.0 * np.pi * g_m
            Gy = 2.0 * np.pi * g_n
            
            kx_total = Gx + 2.0 * np.pi * kx
            ky_total = Gy + 2.0 * np.pi * ky
            
            # Dot product (G_i+k) . (G_j+k)
            # We need a matrix D_{ij} = (G_i+k) \cdot (G_j+k)
            # D_{ij} = (Gx_i + kx)(Gx_j + kx) + (Gy_i + ky)(Gy_j + ky)
            # D = outer(kx_total, kx_total) + outer(ky_total, ky_total)
            
            D = np.outer(kx_total, kx_total) + np.outer(ky_total, ky_total)
            
            M = S * D
            
            # M should be Hermitian (Real Symmetric if S is real symmetric)
            # S is real because eps is real.
            # Check symmetry: S_{ij} = c_{G_i - G_j}. S_{ji} = c_{G_j - G_i} = c_{-(G_i - G_j)} = c_{G_i - G_j} (since c is real/even? No, c is complex generally, but for real eps, c_{-G} = c_G^*. Since eps is real, c_G is real? No.
            # If eps(x,y) is real, then c_G = 1/N^2 sum eps(x,y) exp(-i G.r).
            # c_{-G} = 1/N^2 sum eps(x,y) exp(i G.r) = c_G^*.
            # So S_{ij} = c_{G_i - G_j}. S_{ji} = c_{G_j - G_i} = c_{-(G_i - G_j)} = c_{G_i - G_j}^* = S_{ij}^*.
            # So M is Hermitian.
            
            try:
                eigvals, eigvecs = sla.eigh(M)
                eigvals = np.real(eigvals)
                omega = np.sqrt(np.maximum(eigvals, 0))
                
                # Get bands 4 and 5 (indices 3 and 4)
                if len(omega) >= 5:
                    w4 = omega[3]
                    w5 = omega[4]
                    
                    if w4 > max_w4:
                        max_w4 = w4
                        best_k_4 = (kx, ky)
                        best_vec_4 = eigvecs[:, 3]
                        
                    if w5 < min_w5:
                        min_w5 = w5
                        best_k_5 = (kx, ky)
                        best_vec_5 = eigvecs[:, 4]
                        
            except Exception:
                pass
        
        if max_w4 == -np.inf or min_w5 == np.inf:
            return -1.0, 0.0, 0.0, np.zeros((size, size)), np.zeros((size, size))
        
        gap = min_w5 - max_w4
        midgap = (min_w5 + max_w4) / 2.0
        ratio = 2.0 * gap / midgap if midgap > 0 else -1.0
        
        # Compute field intensities
        def compute_field_int(vec, k_point):
            if vec is None or k_point is None:
                return np.zeros((size, size))
            kx, ky = k_point
            coords = (np.arange(size) + 0.5) / size
            X, Y = np.meshgrid(coords, coords, indexing='ij')
            
            field = np.zeros((size, size), dtype=complex)
            for idx, (m, n) in enumerate(g_indices):
                Gx = 2.0 * np.pi * m
                Gy = 2.0 * np.pi * n
                phase = np.exp(1j * ((Gx + 2.0 * np.pi * kx) * X + (Gy + 2.0 * np.pi * ky) * Y))
                field += vec[idx] * phase
            return np.abs(field)**2
        
        I4 = compute_field_int(best_vec_4, best_k_4)
        I5 = compute_field_int(best_vec_5, best_k_5)
        
        return ratio, max_w4, min_w5, I4, I5

    def make_four_quadrant_cell(rod_radius, exponent=2.0):
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        x_grid, y_grid = np.meshgrid(coords, coords, indexing='ij')
        
        centers = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
        
        for (cx, cy) in centers:
            dx = x_grid - cx
            dy = y_grid - cy
            if rod_radius > 0:
                val = np.abs(dx/rod_radius)**exponent + np.abs(dy/rod_radius)**exponent
                cell[val <= 1.0 + 1e-9] = 1
        
        # Enforce D4 symmetry
        cell = ((cell + cell.T) // 2).astype(int)
        cell = ((cell + cell[::-1, ::-1]) // 2).astype(int)
        return cell.tolist()

    def get_orbit(i, j):
        return [(i, j), (j, i), (size-1-i, size-1-j), (size-1-j, size-1-i)]

    def apply_orbit(cell, i, j, val):
        for p, q in get_orbit(i, j):
            cell[p, q] = val
        return cell

    # --- Main Optimization Loop ---
    
    # Start with a good initial guess
    best_cell = make_four_quadrant_cell(0.12, 2.0)
    best_score = -np.inf
    
    current_cell = np.array(best_cell)
    
    # Precompute orbits for all pixels to speed up search
    # We only need to iterate over unique orbits
    # An orbit is unique if we pick the "first" pixel in some ordering
    unique_orbits = []
    visited = set()
    for i in range(size):
        for j in range(size):
            orbit = get_orbit(i, j)
            # Canonical representative: min lexicographic
            rep = min(orbit)
            if rep not in visited:
                visited.add(rep)
                unique_orbits.append(orbit)
                
    # Shuffle orbits to avoid bias
    np.random.shuffle(unique_orbits)
    
    # Optimization
    for iteration in range(1000):
        if check_time():
            break
            
        # Evaluate current cell
        score, w4, w5, I4, I5 = compute_bands_and_fields(current_cell.tolist())
        
        if score > best_score:
            best_score = score
            best_cell = current_cell.tolist()
            
        # If we have a valid gap, use field intensities to guide search
        if best_score > -1.0:
            # Identify pixels to flip
            # We want to increase score.
            # Score increases if w4 decreases or w5 increases.
            # w4 decreases if we add dielectric (1) where I4 is high.
            # w5 increases if we add air (0) where I5 is high.
            
            # Create a "desirability" map
            # D[i,j] = I4[i,j] * (1 - cell[i,j]) + I5[i,j] * cell[i,j]
            # If cell is 0 (Air) and I4 is high -> D is high -> want to set to 1.
            # If cell is 1 (Dielectric) and I5 is high -> D is high -> want to set to 0.
            
            D = I4 * (1.0 - current_cell) + I5 * current_cell
            
            # Find the orbit with the highest "potential" improvement
            # We sum D over the orbit.
            # But we must flip the whole orbit.
            # If we flip 0->1, we gain I4 and lose I5?
            # Change in score is complex. Let's just try flipping the orbit with max D.
            
            best_orbit_idx = -1
            best_D = 0
            for idx, orbit in enumerate(unique_orbits):
                # Check if the orbit is consistent (all same value)
                vals = [current_cell[p, q] for p, q in orbit]
                if len(set(vals)) != 1:
                    continue
                
                val = vals[0]
                # Calculate potential
                d_val = 0
                for p, q in orbit:
                    d_val += D[p, q]
                
                if d_val > best_D:
                    best_D = d_val
                    best_orbit_idx = idx
            
            if best_orbit_idx != -1 and best_D > 1e-3:
                orbit = unique_orbits[best_orbit_idx]
                i, j = orbit[0]
                new_val = 1 - current_cell[i, j]
                
                # Apply flip
                new_cell = current_cell.copy()
                apply_orbit(new_cell, i, j, new_val)
                
                # Evaluate new cell
                new_score, _, _, _, _ = compute_bands_and_fields(new_cell.tolist())
                
                if new_score > score:
                    current_cell = new_cell
                else:
                    # If no improvement, try a random flip or keep going
                    # To escape local minima, occasionally accept random flips
                    if np.random.rand() < 0.1:
                        rand_idx = np.random.randint(0, len(unique_orbits))
                        orbit = unique_orbits[rand_idx]
                        i, j = orbit[0]
                        new_val = 1 - current_cell[i, j]
                        new_cell = current_cell.copy()
                        apply_orbit(new_cell, i, j, new_val)
                        new_score, _, _, _, _ = compute_bands_and_fields(new_cell.tolist())
                        if new_score > score * 0.9: # Accept if not too bad
                            current_cell = new_cell

    return best_cell

# EVOLVE-BLOCK-END
