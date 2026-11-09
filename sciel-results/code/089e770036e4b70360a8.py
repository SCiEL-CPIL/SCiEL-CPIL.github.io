import numpy as np
import time
import sys

def design(problem):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Precompute reciprocal space vectors for 17x17 grid (289 waves)
    # G vectors are in units of 2*pi/a
    N_pw = 17
    G_max = (N_pw - 1) // 2
    Gx = np.arange(-G_max, G_max + 1)
    Gy = np.arange(-G_max, G_max + 1)
    GX, GY = np.meshgrid(Gx, Gy, indexing='ij')
    G_vectors = np.column_stack((GX.ravel(), GY.ravel()))
    N_waves = len(G_vectors)
    
    # Function to compute Fourier coefficients of epsilon
    def compute_fourier_coeffs(cell):
        # cell is 32x32 binary mask (1 for dielectric)
        # epsilon(x,y) = eps_high if cell[x,y]==1 else eps_low
        eps_grid = np.where(cell == 1, eps_high, eps_low)
        # Compute FFT
        # Note: numpy FFT expects origin at (0,0), but our grid is centered?
        # The problem defines the cell from 0 to size-1.
        # The symmetry is about the center (16,16).
        # Standard FFT assumes periodicity.
        # We need epsilon_G for G in G_vectors.
        # epsilon_G = (1/N^2) sum_{x,y} epsilon(x,y) exp(-i G . r_xy)
        # r_xy = (x+0.5, y+0.5) / size * a ?
        # The problem statement says "unit cell". Let a=1.
        # Coordinates x,y in [0,1).
        # G vectors are integers (in 2pi/a units).
        
        # Efficiently compute specific Fourier components
        # epsilon_G = (1/N^2) * FFT(eps_grid) at index corresponding to G
        # numpy.fft.fft2 computes sum_{x,y} eps(x,y) exp(-2pi i (u x + v y) / N)
        # Our G is in 2pi/a units. x = n/N.
        # exp(-i 2pi G_x n/N) matches FFT with u=G_x.
        # So we can just take the FFT and pick the components.
        
        eps_fft = np.fft.fft2(eps_grid)
        eps_fft /= (size * size)
        
        # Map G vectors to FFT indices
        # FFT index k corresponds to frequency k (if k < N/2) or k-N (if k >= N/2)
        # Our G range is -8 to 8. N=32.
        # Indices: -8 -> 24, -7 -> 25, ..., -1 -> 31, 0 -> 0, 1 -> 1, ..., 8 -> 8.
        
        coeffs = np.zeros(N_waves, dtype=complex)
        for i, (gx, gy) in enumerate(G_vectors):
            ix = int(gx) % size
            iy = int(gy) % size
            coeffs[i] = eps_fft[ix, iy]
            
        return coeffs

    # Function to solve band structure for a given cell
    def solve_bands(cell, k_points):
        eps_coeffs = compute_fourier_coeffs(cell)
        # Precompute k+G vectors
        # k_points is a list of (kx, ky)
        
        # We need to solve M A = w^2 A
        # M_{ij} = (k+G_i) . (k+G_j) * eps_{G_i - G_j}
        
        # For speed, we can precompute the eps matrix for each k?
        # No, eps depends on G_i - G_j.
        
        # Let's iterate over k points
        bands = []
        for kx, ky in k_points:
            # k is in 2pi/a units
            kg = G_vectors + np.array([kx, ky])
            kg_sq = np.sum(kg**2, axis=1)
            
            # Build M
            # M[i, j] = (k+G_i) . (k+G_j) * eps_{G_i - G_j}
            # (k+G_i) . (k+G_j) = kx^2 + ky^2 + kx(G_ix+G_jx) + ky(G_iy+G_jy) + G_ix G_jx + G_iy G_jy
            # This is O(N^2) per k point. N=289. 289^2 ~ 83k operations.
            # 30 k points -> 2.5M operations. Very fast in Python/NumPy.
            
            # Vectorized dot product
            kg_dot = kg @ kg.T
            
            # eps_{G_i - G_j}
            # We need to find the index in eps_coeffs for G_i - G_j
            # G_diff = G_i - G_j
            # We can precompute a map from G vector to index in G_vectors
            # But G_diff might be outside the -8..8 range?
            # G_i, G_j in [-8, 8]. Diff in [-16, 16].
            # Our eps_coeffs only has G in [-8, 8].
            # The convolution sum in the strict PWE method uses all G.
            # However, the "exact-coefficient plane-wave method" with 289 waves
            # usually implies truncating the basis to 289 waves.
            # The matrix element M_{ij} requires eps_{G_i - G_j}.
            # If G_i - G_j is outside the stored range, we assume eps=0?
            # No, the Fourier series of a periodic function has non-zero coeffs everywhere.
            # But we only computed coeffs for G in [-8, 8].
            # The standard Galerkin method with N waves uses:
            # M_{ij} = (k+G_i) . (k+G_j) * eps_{G_i - G_j}
            # where eps_{G} is the true Fourier coefficient.
            # If we truncate the basis to N waves, we still need the true eps_{G_i - G_j}.
            # Since the cell is 32x32, the Fourier coefficients decay.
            # For a binary grid, eps_{G} is zero for G > 16? No.
            # The Nyquist frequency is 16.
            # So eps_{G} is non-zero for |Gx|, |Gy| <= 16.
            # Our G_diff can be up to 16.
            # So we should compute eps_coeffs for G in [-16, 16].
            
            # Let's re-compute coeffs for a larger grid to be safe.
            # N_pw_diff = 33 (from -16 to 16)
            pass

        # Refine solver with larger Fourier grid
        N_pw_diff = 33
        G_max_diff = (N_pw_diff - 1) // 2
        Gx_diff = np.arange(-G_max_diff, G_max_diff + 1)
        Gy_diff = np.arange(-G_max_diff, G_max_diff + 1)
        GX_diff, GY_diff = np.meshgrid(Gx_diff, Gy_diff, indexing='ij')
        G_diff_vectors = np.column_stack((GX_diff.ravel(), GY_diff.ravel()))
        
        # Map G_diff vector to index in G_diff_vectors
        # Since it's a regular grid, index = (gx + G_max_diff) * N_pw_diff + (gy + G_max_diff)
        
        # Re-compute eps_coeffs for the larger grid
        eps_grid = np.where(cell == 1, eps_high, eps_low)
        eps_fft_large = np.fft.fft2(eps_grid) / (size * size)
        
        def get_eps_coeff(gx, gy):
            ix = int(gx) % size
            iy = int(gy) % size
            return eps_fft_large[ix, iy]

        bands = []
        for kx, ky in k_points:
            kg = G_vectors + np.array([kx, ky])
            kg_dot = kg @ kg.T
            
            M = np.zeros((N_waves, N_waves), dtype=complex)
            for i in range(N_waves):
                gi = G_vectors[i]
                for j in range(N_waves):
                    gj = G_vectors[j]
                    gd = gi - gj
                    eps_gd = get_eps_coeff(gd[0], gd[1])
                    M[i, j] = kg_dot[i, j] * eps_gd
            
            # Solve eigenvalue problem M A = w^2 A
            # M is Hermitian?
            # eps_{-G} = conj(eps_G).
            # M_{ji} = (k+G_j).(k+G_i) eps_{G_j-G_i} = M_{ij} * eps_{G_j-G_i}/eps_{G_i-G_j}
            # Since eps_{-G} = conj(eps_G), M is Hermitian.
            
            w2, _ = np.linalg.eigh(M)
            # Sort eigenvalues
            w2 = np.sort(w2.real)
            # We want bands 4 and 5 (indices 3 and 4)
            # The gap is between max(w_4) and min(w_5)
            # Wait, the task says "gap between the fourth and fifth TM bands".
            # Band 1 is the lowest.
            # So we look at w2[3] and w2[4].
            
            if len(w2) >= 5:
                bands.append((w2[3], w2[4]))
            else:
                bands.append((0, 0))
                
        return bands

    # Define k-points in irreducible Brillouin Zone
    # Square BZ: 0 <= kx <= ky <= 1 (or similar)
    # Standard path: Gamma (0,0) -> M (0.5, 0) -> X (0.5, 0.5) -> Gamma?
    # Irreducible wedge: 0 <= kx <= ky <= 0.5?
    # Let's sample 30 points.
    k_points = []
    # Gamma to M: (0,0) to (0.5, 0)
    # M to X: (0.5, 0) to (0.5, 0.5)
    # X to Gamma: (0.5, 0.5) to (0,0)
    # Symmetry reduces this to a triangle.
    # Let's just sample a grid in the triangle 0 <= kx <= ky <= 0.5
    # Or simply 30 points along the boundary and interior.
    
    # For simplicity and robustness, let's use a set of standard high-symmetry points and some interior points.
    # Gamma (0,0)
    # M (0.5, 0)
    # X (0.5, 0.5)
    # K (0.25, 0.25) ?
    
    # Let's create a list of 30 points.
    # 10 points Gamma-M
    for i in range(10):
        kx = 0.5 * i / 9.0
        ky = 0.0
        k_points.append((kx, ky))
    # 10 points M-X
    for i in range(10):
        kx = 0.5
        ky = 0.5 * i / 9.0
        k_points.append((kx, ky))
    # 10 points X-Gamma
    for i in range(10):
        kx = 0.5 * (1 - i / 9.0)
        ky = 0.5 * (1 - i / 9.0)
        k_points.append((kx, ky))
        
    def evaluate_cell(cell):
        bands = solve_bands(cell, k_points)
        if not bands:
            return -1.0
        max_w4 = max(b[0] for b in bands)
        min_w5 = min(b[1] for b in bands)
        
        if min_w5 <= max_w4:
            # Overlapping
            return -1.0
            
        gap = min_w5 - max_w4
        midgap = (min_w5 + max_w4) / 2.0
        if midgap == 0:
            return -1.0
        ratio = 2 * gap / midgap
        return ratio

    # Generate candidate cells
    def make_cell(shape_type, param):
        cell = np.zeros((size, size), dtype=int)
        cx, cy = size // 2, size // 2
        
        if shape_type == 'square':
            # Central square rod
            half = param / 2.0
            x0 = int(cx - half)
            x1 = int(cx + half)
            y0 = int(cy - half)
            y1 = int(cy + half)
            cell[x0:x1, y0:y1] = 1
            
        elif shape_type == 'diamond':
            # Central rotated square
            # |x-cx| + |y-cy| < param
            x = np.arange(size)
            y = np.arange(size)
            X, Y = np.meshgrid(x, y, indexing='ij')
            dist = np.abs(X - cx) + np.abs(Y - cy)
            cell[dist < param] = 1
            
        elif shape_type == 'two_squares':
            # Center square + Corner square
            half = param / 2.0
            # Center
            x0 = int(cx - half)
            x1 = int(cx + half)
            y0 = int(cy - half)
            y1 = int(cy + half)
            cell[x0:x1, y0:y1] = 1
            
            # Corner (0,0) and (size-1, size-1) etc to maintain symmetry?
            # A rod at the corner is shared by 4 cells.
            # So we place it at (0,0), (size,0), (0,size), (size,size).
            # In a 32x32 grid, corner is (0,0).
            # To make it symmetric, we place a square centered at (0,0) which wraps around.
            # Simpler: Place a square at (cx, cy) and another at (cx, 0) ? No.
            # Let's try a rod at (size/4, size/4) and its symmetric counterparts.
            # This is 4 rods per cell? No, 4 points are equivalent.
            # If we put a rod at (8,8), we need rods at (8,24), (24,8), (24,24).
            # This is a 2x2 superlattice of rods.
            
            rx, ry = 8, 8
            half = param / 2.0
            for px in [rx, size-rx]:
                for py in [ry, size-ry]:
                    x0 = int(px - half)
                    x1 = int(px + half)
                    y0 = int(py - half)
                    y1 = int(py + half)
                    # Handle wrapping
                    # Since it's binary, we can just assign
                    # But we must ensure symmetry.
                    # The 4 points are symmetric.
                    cell[x0:x1, y0:y1] = 1
                    # Also need to handle wrap-around if x0<0
                    if x0 < 0:
                        cell[0:x0+size, y0:y1] = 1
                    if x1 > size:
                        cell[x1-size:size, y0:y1] = 1
                    if y0 < 0:
                        cell[x0:x1, 0:y0+size] = 1
                    if y1 > size:
                        cell[x0:x1, y1-size:size] = 1
                    # Corners
                    if x0 < 0 and y0 < 0:
                        cell[0:x0+size, 0:y0+size] = 1
                    if x1 > size and y1 > size:
                        cell[x1-size:size, y1-size:size] = 1
                    # ... this is getting messy.
                    # Let's use a simpler "two rods" topology:
                    # One rod at center, one rod at the corner (0,0).
                    # Corner rod is 4x smaller in the cell representation?
                    # No, if we put a rod at (0,0), it appears at all 4 corners.
                    # Let's just try 'square' and 'diamond' first.
                    pass
        
        # Ensure symmetry
        # The generation should be symmetric, but let's enforce it
        # cell = np.maximum(cell, cell[::-1, :])
        # cell = np.maximum(cell, cell[:, ::-1])
        # cell = np.maximum(cell, cell.T)
        # Actually, just return the generated cell.
        
        return cell

    # Optimization Loop
    best_score = -1.0
    best_cell = None
    
    shapes = ['square', 'diamond']
    params = np.linspace(4, 24, 50) # Size in pixels
    
    for shape in shapes:
        for p in params:
            if time.time() > deadline:
                break
            cell = make_cell(shape, p)
            # Convert to list of lists
            cell_list = cell.tolist()
            score = evaluate_cell(cell_list)
            if score > best_score:
                best_score = score
                best_cell = cell_list
                
        if time.time() > deadline:
            break
            
    # If no good cell found, return seed
    if best_cell is None:
        cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
        return cell
        
    return best_cell

