import numpy as np
from scipy.optimize import minimize

def design(problem):
    size = problem["pixels"]
    L = size
    eps_high = problem["eps_high"]
    eps_low = problem["eps_low"]
    
    # 1. Generate initial 4-rod design
    centers = [(8.0, 8.0), (24.0, 8.0), (8.0, 24.0), (24.0, 24.0)]
    radius = 5.3
    cell_init = np.zeros((L, L), dtype=int)
    for i in range(L):
        for j in range(L):
            y_pix = i + 0.5
            x_pix = j + 0.5
            inside = False
            for cx, cy in centers:
                dx = x_pix - cx
                dy = y_pix - cy
                if dx*dx + dy*dy < radius*radius:
                    inside = True
                    break
            if inside:
                cell_init[i, j] = 1
    
    # 2. Compute Fourier coefficients of the initial design
    # We use a grid of G vectors: n_x, n_y in [-8, 8]
    N_G = 17
    n_max = 8
    G_vectors = []
    for nx in range(-n_max, n_max + 1):
        for ny in range(-n_max, n_max + 1):
            G_vectors.append((nx, ny))
    G_vectors = np.array(G_vectors)
    
    # Compute epsilon_G for the initial binary map
    # epsilon(x,y) is 1 or 0? No, epsilon is eps_low or eps_high.
    # The operator uses 1/epsilon.
    # Let's compute the Fourier coefficients of the 1/epsilon map?
    # Or the epsilon map?
    # The matrix element is 1/epsilon_{G-G'}.
    # So we need the Fourier coefficients of 1/epsilon(x,y).
    
    eps_map = np.where(cell_init == 1, eps_high, eps_low)
    inv_eps_map = 1.0 / eps_map
    
    # Compute FFT of inv_eps_map
    # The grid is 32x32.
    # G vectors correspond to frequencies n/32.
    # np.fft.fft2 returns frequencies 0..31.
    # We need to map them to -8..8.
    inv_eps_fft = np.fft.fft2(inv_eps_map)
    # Shift so that zero frequency is in the center
    inv_eps_fft_shifted = np.fft.fftshift(inv_eps_fft)
    
    # Extract coefficients for n in [-8, 8]
    # The shifted FFT has index 16 corresponding to n=0.
    # Indices 8..24 correspond to n=-8..8.
    coeff_grid = np.zeros((N_G, N_G), dtype=complex)
    for i, nx in enumerate(range(-n_max, n_max + 1)):
        for j, ny in enumerate(range(-n_max, n_max + 1)):
            # Map nx to FFT index
            # n=0 is index 16.
            # n=-1 is index 15 (or 31 in unshifted).
            # In shifted array, index 16 is n=0.
            # Index 16 + nx?
            # If nx=-8, index 8. If nx=8, index 24.
            idx_x = 16 + nx
            idx_y = 16 + ny
            coeff_grid[i, j] = inv_eps_fft_shifted[idx_y, idx_x] / (L * L)
            
    # We will optimize the real and imaginary parts?
    # Due to symmetry, inv_eps is real and symmetric.
    # So coefficients are real and symmetric: C_{-nx, -ny} = C_{nx, ny}.
    # And C_{nx, ny} = C_{ny, nx}.
    # We can just optimize the unique real coefficients.
    # Unique coefficients: nx >= 0, ny >= 0.
    # Total unique: (9*9 + 9)/2 = 45?
    # Let's just optimize the full 17x17 grid of real coefficients,
    # but enforce symmetry by averaging or just using a subset.
    
    # For simplicity and robustness, let's optimize the 10 low-order coefficients
    # c_{mn} for m,n in {0,1,2,3} with m<=n.
    # c_{00}, c_{10}, c_{11}, c_{20}, c_{21}, c_{22}, c_{30}, c_{31}, c_{32}, c_{33}
    
    # Extract these from the initial coeff_grid
    # Note: coeff_grid is indexed by i,j corresponding to nx, ny.
    # nx = -8 + i.
    
    def get_coeff(nx, ny):
        i = nx + 8
        j = ny + 8
        return coeff_grid[i, j].real

    c00 = get_coeff(0, 0)
    c10 = get_coeff(1, 0)
    c11 = get_coeff(1, 1)
    c20 = get_coeff(2, 0)
    c21 = get_coeff(2, 1)
    c22 = get_coeff(2, 2)
    c30 = get_coeff(3, 0)
    c31 = get_coeff(3, 1)
    c32 = get_coeff(3, 2)
    c33 = get_coeff(3, 3)
    
    x0 = np.array([c00, c10, c11, c20, c21, c22, c30, c31, c32, c33])
    
    # 3. Define objective function
    def objective(x):
        # Reconstruct inv_eps field from coefficients
        # inv_eps(x,y) = sum c_{mn} cos(2 pi m x / L) cos(2 pi n y / L)
        # Note: This assumes the field is represented by these low-order terms.
        # The actual field is binary, so this is an approximation.
        # We generate the field, threshold it to get binary pixels,
        # then compute the EXACT Fourier coefficients of the binary map
        # for the solver.
        
        # Map x to coeffs
        c00, c10, c11, c20, c21, c22, c30, c31, c32, c33 = x
        
        # Build field
        xs = np.arange(L) + 0.5
        ys = np.arange(L) + 0.5
        XX, YY = np.meshgrid(xs, ys, indexing='xy')
        
        # Normalize coordinates to [0, 1) for Fourier basis?
        # Basis: cos(2 pi m x / L).
        # x is in pixels [0, L).
        
        field = np.zeros((L, L))
        field += c00
        field += c10 * np.cos(2 * np.pi * 1 * XX / L)
        field += c10 * np.cos(2 * np.pi * 1 * YY / L) # Wait, symmetry x<->y
        # Actually, the term c10 corresponds to cos(2 pi 1 x/L) cos(2 pi 0 y/L) + cos(2 pi 0 x/L) cos(2 pi 1 y/L)?
        # My basis definition: c_{mn} cos(2 pi m x/L) cos(2 pi n y/L).
        # For m=1, n=0: cos(2 pi x/L).
        # For m=0, n=1: cos(2 pi y/L).
        # Due to symmetry c_{10} = c_{01}.
        # So term is c10 (cos(2 pi x/L) + cos(2 pi y/L)).
        
        # Let's just sum all unique terms properly.
        # Terms:
        # (0,0): c00
        # (1,0) & (0,1): c10 (cos(2 pi x/L) + cos(2 pi y/L))
        # (1,1): c11 cos(2 pi x/L) cos(2 pi y/L)
        # (2,0) & (0,2): c20 (cos(4 pi x/L) + cos(4 pi y/L))
        # (2,1) & (1,2): c21 (cos(4 pi x/L)cos(2 pi y/L) + cos(2 pi x/L)cos(4 pi y/L))
        # (2,2): c22 cos(4 pi x/L) cos(4 pi y/L)
        # ...
        
        # Simplified reconstruction:
        field = np.zeros((L, L))
        for m in range(4):
            for n in range(m + 1):
                if m == 0 and n == 0:
                    coeff = c00
                elif m == 1 and n == 0:
                    coeff = c10
                elif m == 1 and n == 1:
                    coeff = c11
                elif m == 2 and n == 0:
                    coeff = c20
                elif m == 2 and n == 1:
                    coeff = c21
                elif m == 2 and n == 2:
                    coeff = c22
                elif m == 3 and n == 0:
                    coeff = c30
                elif m == 3 and n == 1:
                    coeff = c31
                elif m == 3 and n == 2:
                    coeff = c32
                elif m == 3 and n == 3:
                    coeff = c33
                else:
                    continue
                
                term = np.cos(2 * np.pi * m * XX / L) * np.cos(2 * np.pi * n * YY / L)
                if m != n:
                    term += np.cos(2 * np.pi * n * XX / L) * np.cos(2 * np.pi * m * YY / L)
                
                field += coeff * term
        
        # Threshold to binary
        # The field represents 1/epsilon.
        # 1/eps_high = 1/12 ~ 0.083
        # 1/eps_low = 1/1 = 1.0
        # So high value of field -> low epsilon (air).
        # Low value of field -> high epsilon (dielectric).
        # We want to map this to 0/1 pixels where 1 is dielectric.
        # So if field < threshold, pixel is 1 (dielectric).
        # If field > threshold, pixel is 0 (air).
        
        # Determine threshold
        # The mean value of 1/eps in the initial design:
        # Fill fraction ~0.35 dielectric.
        # Mean 1/eps = 0.35 * (1/12) + 0.65 * 1 = 0.029 + 0.65 = 0.679.
        # Let's use the median of the field as threshold.
        threshold = np.median(field)
        
        cell_bin = (field < threshold).astype(int)
        
        # Ensure symmetry?
        # The field is symmetric. The threshold is global.
        # Floating point errors might break symmetry slightly.
        # We can enforce symmetry by averaging with its reflections.
        # cell_bin = (cell_bin + cell_bin[::-1, ::-1] + cell_bin[:, ::-1] + cell_bin[::-1, :]) / 4
        # Then threshold again?
        # Simpler: just compute the exact Fourier coeffs of the binary map.
        # The solver will handle it.
        
        # Compute exact Fourier coefficients of the binary map's 1/eps
        eps_map_bin = np.where(cell_bin == 1, eps_high, eps_low)
        inv_eps_bin = 1.0 / eps_map_bin
        
        inv_eps_fft_bin = np.fft.fft2(inv_eps_bin)
        inv_eps_fft_bin_shifted = np.fft.fftshift(inv_eps_fft_bin)
        
        # Build the matrix M
        # M_{G, G'} = |G+G'|^2 / (4 * eps_{G-G'})
        # G vectors are in units of 2 pi / L.
        # G = (nx, ny) * 2 pi / L.
        # |G|^2 = (nx^2 + ny^2) * (2 pi / L)^2.
        
        N_G_local = 17
        M = np.zeros((N_G_local * N_G_local, N_G_local * N_G_local), dtype=float)
        
        # Precompute G magnitudes squared
        G_sq = np.zeros((N_G_local, N_G_local))
        for i in range(N_G_local):
            for j in range(N_G_local):
                nx = i - 8
                ny = j - 8
                G_sq[i, j] = (nx * nx + ny * ny) * (2 * np.pi / L) ** 2
        
        # Loop over G, G'
        # This is 289 * 289 = 83521 iterations.
        # Inside, we need eps_{G-G'}.
        # G-G' has components (nx - nx', ny - ny').
        # The index in coeff_grid_shifted for (kx, ky) is (16+kx, 16+ky).
        
        for i in range(N_G_local):
            for j in range(N_G_local):
                idx_1 = i * N_G_local + j
                nx1 = i - 8
                ny1 = j - 8
                for k in range(N_G_local):
                    for l in range(N_G_local):
                        idx_2 = k * N_G_local + l
                        nx2 = k - 8
                        ny2 = l - 8
                        
                        # G + G'
                        nx_sum = nx1 + nx2
                        ny_sum = ny1 + ny2
                        # |G+G'|^2
                        # (nx_sum^2 + ny_sum^2) * (2 pi / L)^2
                        mag_sq = (nx_sum * nx_sum + ny_sum * ny_sum) * (2 * np.pi / L) ** 2
                        
                        # G - G'
                        nx_diff = nx1 - nx2
                        ny_diff = ny1 - ny2
                        
                        # Get eps_{G-G'}
                        # If nx_diff or ny_diff is outside [-8, 8], we don't have the coefficient.
                        # The evaluator uses "exact coefficients".
                        # If the difference is outside the computed range, what do we do?
                        # The pixel map is periodic. The Fourier series is infinite.
                        # The evaluator likely truncates or uses the available coefficients.
                        # If nx_diff is e.g. 9, we don't have it.
                        # We can assume it's 0? Or use the value from the periodic extension?
                        # Since we only computed up to 8, and the map is 32x32,
                        # the Fourier coefficients decay.
                        # For safety, if out of bounds, set eps to mean? Or 0?
                        # If eps is 0, division by zero.
                        # Let's assume the evaluator handles this or the coefficients are sufficient.
                        # Given 289 waves and 32x32 pixels, the "Nyquist" is 16.
                        # So differences up to 16 are possible.
                        # We only have coeffs up to 8.
                        # This is a problem.
                        
                        # To match the evaluator, I should compute MORE coefficients.
                        # Let's compute up to n=16.
                        # N_G would be 33x33 = 1089 waves.
                        # But the evaluator uses 289 waves for the *basis*.
                        # The matrix size is 289x289.
                        # The coefficients needed are for G-G' where G, G' are in the basis.
                        # Max diff is 16.
                        # So I need Fourier coefficients up to n=16.
                        
                        # Let's re-compute inv_eps_fft with more resolution?
                        # The pixel map is 32x32. The FFT has 32 points.
                        # Frequencies 0..31.
                        # Shifted: -16..15.
                        # So I have coefficients for -16..15.
                        # This covers all differences!
                        
                        # So I should use the full 32x32 FFT coefficients.
                        
                        if -16 <= nx_diff <= 15 and -16 <= ny_diff <= 15:
                            idx_dx = 16 + nx_diff
                            idx_dy = 16 + ny_diff
                            # Wrap around if needed?
                            # FFT indices 0..31.
                            # nx_diff=15 -> idx 31.
                            # nx_diff=-16 -> idx 0.
                            # So it's always valid.
                            eps_val = inv_eps_fft_bin_shifted[idx_dy, idx_dx].real / (L * L)
                            if eps_val == 0:
                                eps_val = 1e-10
                            M[idx_1, idx_2] = mag_sq / (4.0 * eps_val)
                        else:
                            M[idx_1, idx_2] = 0.0
        
        # Solve eigenvalue problem
        # We need the smallest 5 eigenvalues.
        # Use np.linalg.eigvalsh
        try:
            eigenvalues = np.linalg.eigvalsh(M)
        except:
            return 0.0
        
        # Sort
        eigenvalues = np.sort(eigenvalues)
        
        # Get bands 4 and 5 (indices 3 and 4)
        # The bands are indexed 1..N.
        # Band 4 is index 3. Band 5 is index 4.
        # But we need the *minimum* omega_5 and *maximum* omega_4 over the BZ?
        # The evaluator computes bands on 30 wave vectors.
        # My solver computes bands for ONE wave vector?
        # No, the plane-wave method solves for a specific wave vector k.
        # The matrix M depends on k?
        # No, for a periodic system, the Hamiltonian in plane waves is the same for all k?
        # No, the boundary conditions couple G to G+k?
        # Standard plane-wave method:
        # Expand field as sum E_G exp(i (G+k) r).
        # The matrix elements depend on k.
        # M_{G,G'}(k) = |G+k|^2 / (4 eps_{G-G'}).
        
        # I need to compute the gap over the Brillouine Zone.
        # The evaluator uses 30 wave vectors.
        # I should sample the BZ similarly.
        
        # Define 30 wave vectors in the irreducible BZ.
        # Irreducible BZ of square lattice: triangle with vertices (0,0), (pi/L, 0), (pi/L, pi/L).
        # Or (0,0), (pi/2, 0), (pi/2, pi/2) in units of 1/L?
        # G = 2 pi / L.
        # k is in [0, pi/L] x [0, pi/L] reduced by symmetry.
        
        # Let's use a grid of points in the irreducible triangle.
        # Or just a few key points: Gamma (0,0), M (pi/L, 0), X (pi/L, pi/L).
        # And some interior points.
        
        k_points = []
        # Gamma
        k_points.append((0.0, 0.0))
        # M (1, 0) * pi/L
        k_points.append((1.0, 0.0))
        # X (1, 1) * pi/L
        k_points.append((1.0, 1.0))
        # Center of M-X edge? (1, 0.5)
        k_points.append((1.0, 0.5))
        # Center of Gamma-M? (0.5, 0)
        k_points.append((0.5, 0.0))
        # Center of Gamma-X? (0.5, 0.5)
        k_points.append((0.5, 0.5))
        
        # Add more points to approximate the 30 used by evaluator.
        # Just use a 6x6 grid in the triangle?
        # For now, use these 6 points.
        
        max_omega4 = 0.0
        min_omega5 = float('inf')
        
        for kx, ky in k_points:
            # k in units of pi/L
            # G in units of 2 pi / L.
            # G + k:
            # G = (nx, ny) * 2 pi / L.
            # k = (kx, ky) * pi / L = (kx/2, ky/2) * 2 pi / L.
            
            M_k = np.zeros((N_G_local * N_G_local, N_G_local * N_G_local), dtype=float)
            
            for i in range(N_G_local):
                for j in range(N_G_local):
                    idx_1 = i * N_G_local + j
                    nx1 = i - 8
                    ny1 = j - 8
                    
                    # G1 + k
                    # G1 = (nx1, ny1) * 2 pi / L
                    # k = (kx, ky) * pi / L
                    # Sum = (2*nx1 + kx, 2*ny1 + ky) * pi / L
                    # |Sum|^2 = pi^2 / L^2 * ( (2*nx1+kx)^2 + (2*ny1+ky)^2 )
                    mag_sq = (np.pi / L) ** 2 * ( (2*nx1 + kx)**2 + (2*ny1 + ky)**2 )
                    
                    for k_idx in range(N_G_local):
                        for l_idx in range(N_G_local):
                            idx_2 = k_idx * N_G_local + l_idx
                            nx2 = k_idx - 8
                            ny2 = l_idx - 8
                            
                            nx_diff = nx1 - nx2
                            ny_diff = ny1 - ny2
                            
                            if -16 <= nx_diff <= 15 and -16 <= ny_diff <= 15:
                                idx_dx = 16 + nx_diff
                                idx_dy = 16 + ny_diff
                                eps_val = inv_eps_fft_bin_shifted[idx_dy, idx_dx].real / (L * L)
                                if eps_val == 0:
                                    eps_val = 1e-10
                                M_k[idx_1, idx_2] = mag_sq / (4.0 * eps_val)
                            else:
                                M_k[idx_1, idx_2] = 0.0
            
            eigvals = np.linalg.eigvalsh(M_k)
            eigvals = np.sort(eigvals)
            
            # Take the first 5
            if len(eigvals) >= 5:
                omega4 = np.sqrt(eigvals[3])
                omega5 = np.sqrt(eigvals[4])
                
                if omega4 > max_omega4:
                    max_omega4 = omega4
                if omega5 < min_omega5:
                    min_omega5 = omega5
        
        if min_omega5 == float('inf'):
            return 0.0
            
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = 2.0 * gap / midgap
        
        # Negate for minimization
        return -ratio

    # Optimize
    # Use Nelder-Mead
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
    
    # Get best coefficients
    best_coeffs = res.x
    
    # Reconstruct best pixel map
    c00, c10, c11, c20, c21, c22, c30, c31, c32, c33 = best_coeffs
    
    xs = np.arange(L) + 0.5
    ys = np.arange(L) + 0.5
    XX, YY = np.meshgrid(xs, ys, indexing='xy')
    
    field = np.zeros((L, L))
    for m in range(4):
        for n in range(m + 1):
            if m == 0 and n == 0: coeff = c00
            elif m == 1 and n == 0: coeff = c10
            elif m == 1 and n == 1: coeff = c11
            elif m == 2 and n == 0: coeff = c20
            elif m == 2 and n == 1: coeff = c21
            elif m == 2 and n == 2: coeff = c22
            elif m == 3 and n == 0: coeff = c30
            elif m == 3 and n == 1: coeff = c31
            elif m == 3 and n == 2: coeff = c32
            elif m == 3 and n == 3: coeff = c33
            else: continue
            
            term = np.cos(2 * np.pi * m * XX / L) * np.cos(2 * np.pi * n * YY / L)
            if m != n:
                term += np.cos(2 * np.pi * n * XX / L) * np.cos(2 * np.pi * m * YY / L)
            field += coeff * term
            
    threshold = np.median(field)
    cell_best = (field < threshold).astype(int)
    
    # Convert to nested list
    return [list(row) for row in cell_best]
