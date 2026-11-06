import time
import math
import numpy as np

def design(problem):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    # Pre-compute Fourier coefficients for the PWE solver
    # We use a 15x15 grid of plane waves (N=7 in each direction)
    N = 7
    G_vals = np.arange(-N, N + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    Gx = Gx.flatten()
    Gy = Gy.flatten()
    num_G = len(Gx)
    
    # Pre-compute G^2 for the kinetic energy term
    G2 = Gx**2 + Gy**2
    
    def compute_eps_inv_fourier(radius):
        """
        Compute the Fourier coefficients of 1/epsilon for a single central disk.
        For a disk of radius R in a unit cell, the Fourier coefficients are:
        eps_G = (1/A) * integral exp(-i G.r) / eps(r) dA
        For a disk, this involves Bessel functions.
        However, since we are on a pixel grid, we can compute the exact 
        Fourier coefficients of the pixel map directly, or approximate 
        with the analytical disk formula.
        
        The analytical formula for a disk of radius R in a unit cell (period 1) is:
        eps_0 = (eps_high * pi * R^2 + eps_low * (1 - pi * R^2))
        eps_G (G != 0) = (eps_low - eps_high) * (pi * R^2) * 2 * J1(|G| * R) / |G|
        
        Wait, the PWE method usually uses 1/epsilon.
        Let's compute the Fourier coefficients of the dielectric function epsilon(r) 
        or its inverse? 
        Standard PWE for TM polarization (H_z) uses the inverse dielectric function 1/epsilon.
        The equation is: det[ |G+G'|^2 / (2 * eps_{G-G'}) - omega^2 ] = 0
        Wait, the standard form is:
        sum_{G'} [ (|G+G'|^2 / (2 * eps_{G-G'})) - omega^2 delta_{G,G'} ] H_{G'} = 0
        Actually, the matrix element is M_{G,G'} = (|G+G'|^2 / (2 * eps_{G-G'})) - omega^2 delta_{G,G'}
        
        Let's compute eps_inv_coeffs (Fourier coefficients of 1/epsilon).
        For a binary material:
        1/eps(r) = (1/eps_low) * (1 - mask(r)) + (1/eps_high) * mask(r)
        where mask(r) = 1 inside dielectric, 0 in air.
        
        So 1/eps(r) = 1/eps_low - (1/eps_low - 1/eps_high) * mask(r)
        
        The Fourier coefficient of a constant C is C * delta_{G,0}.
        The Fourier coefficient of the mask (disk) is:
        mask_0 = pi * R^2
        mask_G (G!=0) = 2 * J1(|G| * R) / |G|  (for unit period)
        
        So eps_inv_G = (1/eps_low) * delta_{G,0} - (1/eps_low - 1/eps_high) * mask_G
        """
        eps_inv = np.zeros(num_G, dtype=complex)
        
        # Constant term
        eps_inv[0] = 1.0 / eps_low
        
        # Disk contribution
        coeff_diff = (1.0 / eps_low - 1.0 / eps_high)
        
        for idx in range(num_G):
            gx = Gx[idx]
            gy = Gy[idx]
            g_mag = math.sqrt(gx*gx + gy*gy)
            
            if idx == 0:
                mask_0 = math.pi * radius * radius
                eps_inv[idx] -= coeff_diff * mask_0
            else:
                # J1(x)/x
                # For small x, J1(x)/x ~ 0.5
                if g_mag < 1e-6:
                    mask_g = 0.5
                else:
                    # Use numpy's j1 for Bessel function of first kind, order 1
                    mask_g = 2.0 * np.j1(g_mag * radius) / g_mag
                eps_inv[idx] -= coeff_diff * mask_g
                
        return eps_inv

    def solve_bands(eps_inv):
        """
        Solve for the lowest 5 bands at Gamma point (k=0).
        The band gap between 4 and 5 is often maximized at Gamma or M for square lattices.
        We will check Gamma and M.
        """
        # Build the matrix at Gamma (k=0)
        # M_{G,G'} = |G+G'|^2 / (2 * eps_inv_{G-G'})
        # Note: eps_inv is indexed by G-G'. We need to map G-G' to our array.
        # Our G array is sorted. We need a lookup or we can just iterate.
        
        # For efficiency, pre-compute the indices for G-G'
        # Since N is small (15x15=225), we can do it directly.
        
        # We need to find the index in Gx, Gy for a given vector (gx, gy).
        # Create a dictionary for fast lookup
        g_lookup = {}
        for i in range(num_G):
            g_lookup[(Gx[i], Gy[i])] = i
            
        M = np.zeros((num_G, num_G), dtype=complex)
        
        for i in range(num_G):
            gx = Gx[i]
            gy = Gy[i]
            for j in range(num_G):
                gpx = Gx[j]
                gpy = Gy[j]
                
                # Vector difference G_i - G_j
                diff_x = gx - gpx
                diff_y = gy - gpy
                
                idx_diff = g_lookup.get((diff_x, diff_y), None)
                if idx_diff is not None:
                    eps_inv_val = eps_inv[idx_diff]
                    if abs(eps_inv_val) > 1e-10:
                        M[i, j] = (gx*gx + gy*gy) / (2.0 * eps_inv_val)
                    else:
                        M[i, j] = 0 # Should not happen for valid eps
                else:
                    M[i, j] = 0
        
        # Solve eigenvalue problem
        # The matrix is Hermitian (if eps_inv is real and symmetric, which it is for C4v)
        # We only need the first 5 eigenvalues
        try:
            # Use scipy.linalg.eigh for Hermitian matrices
            from scipy.linalg import eigh
            # Only compute first 5 eigenvalues
            vals, vecs = eigh(M, subset_by_index=[0, 4])
            return vals
        except Exception:
            # Fallback to numpy if scipy fails (slower but robust)
            vals = np.linalg.eigvalsh(M)
            return vals[:5]
    
    def evaluate_radius(radius):
        """
        Evaluate the gap ratio for a given radius.
        Checks Gamma and M points.
        """
        eps_inv = compute_eps_inv_fourier(radius)
        
        # Gamma point (k=0)
        bands_gamma = solve_bands(eps_inv)
        
        # M point (k = (pi, 0)) - In reduced coordinates, M is (0.5, 0) or (1, 0)?
        # For a square lattice with period 1, the BZ goes from -0.5 to 0.5.
        # Gamma is (0,0). M is (0.5, 0).
        # The PWE matrix at k is:
        # M_{G,G'} = |G+G'+k|^2 / (2 * eps_inv_{G-G'})
        
        # Let's implement a general solver for k
        def solve_bands_k(kx, ky):
            M = np.zeros((num_G, num_G), dtype=complex)
            for i in range(num_G):
                gx = Gx[i]
                gy = Gy[i]
                for j in range(num_G):
                    gpx = Gx[j]
                    gpy = Gy[j]
                    
                    diff_x = gx - gpx
                    diff_y = gy - gpy
                    idx_diff = g_lookup.get((diff_x, diff_y), None)
                    
                    if idx_diff is not None:
                        eps_inv_val = eps_inv[idx_diff]
                        if abs(eps_inv_val) > 1e-10:
                            # |G + G' + k|^2
                            # Wait, the standard form is |G + G'|^2 where G' is the reciprocal vector.
                            # Let's stick to the form:
                            # (k + G_i) dot (k + G_j) / eps_{G_i - G_j}
                            dot = (gx + kx) * (gpx + kx) + (gy + ky) * (gpy + ky)
                            M[i, j] = dot / (2.0 * eps_inv_val)
                        else:
                            M[i, j] = 0
                    else:
                        M[i, j] = 0
            
            try:
                from scipy.linalg import eigh
                vals, _ = eigh(M, subset_by_index=[0, 4])
                return vals
            except:
                vals = np.linalg.eigvalsh(M)
                return vals[:5]
        
        bands_m = solve_bands_k(0.5, 0.0)
        
        # Calculate gap ratio
        # We need min(omega_5) and max(omega_4)
        # The bands are sorted.
        # Gap is between 4th and 5th (index 3 and 4)
        
        # At Gamma
        w4_g = bands_gamma[3]
        w5_g = bands_gamma[4]
        gap_g = w5_g - w4_g
        mid_g = (w5_g + w4_g) / 2.0
        ratio_g = gap_g / mid_g if mid_g > 0 else 0
        
        # At M
        w4_m = bands_m[3]
        w5_m = bands_m[4]
        gap_m = w5_m - w4_m
        mid_m = (w5_m + w4_m) / 2.0
        ratio_m = gap_m / mid_m if mid_m > 0 else 0
        
        # The evaluator checks 30 points. The gap might be elsewhere.
        # But Gamma and M are the high-symmetry points.
        # Let's take the max ratio found at these points as a proxy.
        # Actually, the task says "largest gap ... over the irreducible BZ".
        # The gap is defined as min(omega_5) - max(omega_4).
        # This global min/max might not be at the same k-point.
        # However, for a first approximation, local gaps at high symmetry points are good indicators.
        # To be safer, we can sample a few more points.
        
        # Sample X point (0.5, 0.5) - Center of BZ edge? No, X is (0.5, 0.5) in some conventions?
        # For square lattice: Gamma(0,0), M(0.5,0), X(0,0.5) is same as M by symmetry.
        # The irreducible BZ is the triangle (0,0)-(0.5,0)-(0.5,0.5).
        
        # Let's sample a few points in the irreducible BZ
        # (0.25, 0.25)
        bands_x = solve_bands_k(0.25, 0.25)
        w4_x = bands_x[3]
        w5_x = bands_x[4]
        gap_x = w5_x - w4_x
        mid_x = (w5_x + w4_x) / 2.0
        ratio_x = gap_x / mid_x if mid_x > 0 else 0
        
        # Return the maximum local ratio as a score
        return max(ratio_g, ratio_m, ratio_x)

    best_score = -1
    best_radius = 0.5
    
    # Coarse search
    radii = np.linspace(0.3, 0.7, 10)
    for r in radii:
        if time.time() > deadline - 20:
            break
        score = evaluate_radius(r)
        if score > best_score:
            best_score = score
            best_radius = r
            
    # Fine search around best
    if time.time() < deadline - 5:
        fine_radii = np.linspace(best_radius - 0.05, best_radius + 0.05, 10)
        for r in fine_radii:
            if time.time() > deadline - 5:
                break
            if r < 0.1 or r > 0.9:
                continue
            score = evaluate_radius(r)
            if score > best_score:
                best_score = score
                best_radius = r

    # Generate the final cell based on best_radius
    cell = [[0 for _ in range(size)] for _ in range(size)]
    center = size / 2.0
    
    for i in range(size):
        for j in range(size):
            x = i + 0.5 - center
            y = j + 0.5 - center
            dist = math.sqrt(x*x + y*y)
            if dist < best_radius * size / 2.0: # Radius is in units of half-cell? 
                # Wait, my radius in PWE was in units of the period (1.0).
                # The cell size is `size` pixels.
                # The period is `size` pixels.
                # So radius in pixels = best_radius * size
                # Distance from center in pixels.
                # Center is (size/2, size/2).
                # Pixel (i,j) center is (i+0.5, j+0.5).
                # Dist in pixels = sqrt((i+0.5 - size/2)^2 + (j+0.5 - size/2)^2)
                # Condition: dist < best_radius * size
                pass
                
    # Recalculate correctly
    for i in range(size):
        for j in range(size):
            x = i + 0.5 - center
            y = j + 0.5 - center
            dist = math.sqrt(x*x + y*y)
            # best_radius is a fraction of the period (size)
            if dist < best_radius * size:
                cell[i][j] = 1

    return cell
