import numpy as np
import time

def design(problem):
    """
    Optimizes a 2x2 array of rods (circles or squares) to maximize the 4-5 TM band gap.
    """
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Time management
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # 1. Setup G-vectors (289 plane waves)
    # Indices -16 to 15
    g_vals = np.arange(-16, 16)
    gx, gy = np.meshgrid(g_vals, g_vals, indexing='ij')
    g_mags = gx**2 + gy**2
    # Flatten and sort
    flat_gx = gx.flatten()
    flat_gy = gy.flatten()
    flat_mags = g_mags.flatten()
    sorted_indices = np.argsort(flat_mags)
    # Take 289
    n_pw = 289
    basis_idx = sorted_indices[:n_pw]
    Gx = flat_gx[basis_idx]
    Gy = flat_gy[basis_idx]
    
    # 2. Setup K-points (30 points in irreducible BZ)
    # Irreducible BZ for square lattice C4v: Triangle Gamma(0,0) - X(1,0) - M(1,1)
    # in units of pi/a.
    k_points = []
    # Sample 6x5 grid in the triangle 0 <= ky <= kx <= 1
    for i in range(6):
        kx = i / 5.0
        for j in range(5):
            # ky goes from 0 to kx
            # j=0 -> ky=0, j=4 -> ky=kx
            ky = kx * (j / 4.0)
            k_points.append((kx * np.pi, ky * np.pi))
            
    # 3. Candidate Generators
    def make_circle_cell(radius):
        cell = np.zeros((size, size), dtype=np.int8)
        # 2x2 array of circles
        # Centers at (7.5, 7.5), (7.5, 23.5), (23.5, 7.5), (23.5, 23.5)
        centers = [(7.5, 7.5), (7.5, 23.5), (23.5, 7.5), (23.5, 23.5)]
        ys, xs = np.mgrid[0:size, 0:size]
        for cx, cy in centers:
            dist2 = (xs - cx)**2 + (ys - cy)**2
            cell[dist2 < radius**2] = 1
        return cell

    def make_square_cell(side):
        cell = np.zeros((size, size), dtype=np.int8)
        # 2x2 array of squares
        # Centers at (7.5, 7.5) etc.
        # Side S, so range is center +/- S/2
        half = side / 2.0
        centers = [(7.5, 7.5), (7.5, 23.5), (23.5, 7.5), (23.5, 23.5)]
        ys, xs = np.mgrid[0:size, 0:size]
        for cx, cy in centers:
            dx = np.abs(xs - cx)
            dy = np.abs(ys - cy)
            cell[(dx < half) & (dy < half)] = 1
        return cell

    best_score = -np.inf
    best_cell = None
    
    # Pre-compute eta FFT for a 64x64 grid to cover differences
    # We will re-compute for each candidate, but we can optimize if needed.
    # 1000 candidates * (FFT + 30 eigen) is fine.
    
    candidates = []
    # Circle radii
    for r in np.arange(3.0, 7.5, 0.1):
        candidates.append(make_circle_cell(r))
    # Square sides
    for s in np.arange(6.0, 13.0, 0.5):
        candidates.append(make_square_cell(s))
        
    # Evaluate candidates
    for cell in candidates:
        if time.time() > deadline:
            break
            
        # Compute eta(G)
        # Zero pad to 64x64
        eta_pixel = np.ones((size, size), dtype=np.float64)
        eta_pixel[cell == 1] = 1.0 / eps_high
        # eta_pixel[cell == 0] = 1.0 / eps_low (which is 1.0)
        
        eta_padded = np.zeros((64, 64), dtype=np.complex128)
        eta_padded[:size, :size] = eta_pixel
        eta_fft = np.fft.fft2(eta_padded)
        eta_fft = np.fft.fftshift(eta_fft)
        # eta_fft[kx, ky] corresponds to G = (kx-32, ky-32)
        
        # Function to get eta(Gx, Gy)
        def get_eta(gx, gy):
            # Map to 0..63 index
            ix = (gx + 32) % 64
            iy = (gy + 32) % 64
            return eta_fft[iy, ix] # Note: numpy meshgrid indexing 'ij' means first index is x?
            # np.fft.fft2 output: F[u, v] where u is row (y), v is col (x) if input is y,x?
            # My eta_padded is [y, x].
            # fft2 output is [u, v]. u corresponds to y frequency, v to x frequency.
            # So get_eta(gx, gy) should be eta_fft[gy_idx, gx_idx].
            
        # Re-define get_eta correctly
        def get_eta_correct(gx, gy):
            ix = (gx + 32) % 64
            iy = (gy + 32) % 64
            return eta_fft[iy, ix]

        min_w5 = np.inf
        max_w4 = -np.inf
        
        # Solve for each k-point
        for kx, ky in k_points:
            # Construct Matrix A (289x289)
            # A_ij = eta(Gi - Gj) * ((kx+Gix)(kx+Gjx) + (ky+Giy)(ky+Gjy))
            
            # Pre-compute k+G vectors
            kGx = kx + Gx
            kGy = ky + Gy
            
            # We need eta for all pairs (Gi - Gj)
            # This is a double loop or vectorized?
            # 289x289 = 83k elements.
            # Vectorizing eta lookup:
            # Create meshgrid of Gx, Gy
            Gx_grid, Gy_grid = np.meshgrid(Gx, Gy, indexing='ij')
            dGx = Gx_grid - Gx_grid.T
            dGy = Gy_grid - Gy_grid.T
            
            # Flatten dGx, dGy for lookup
            # But lookup is element-wise.
            # eta_matrix = get_eta_correct(dGx, dGy)
            
            # Vectorized lookup
            ix = (dGx + 32) % 64
            iy = (dGy + 32) % 64
            eta_matrix = eta_fft[iy, ix]
            
            # Dot products
            # (k+Gx)_i * (k+Gx)_j + (k+Gy)_i * (k+Gy)_j
            # kGx is (N,). kGx[:, None] * kGx[None, :]
            dot_x = kGx[:, None] * kGx[None, :]
            dot_y = kGy[:, None] * kGy[None, :]
            dot_total = dot_x + dot_y
            
            A = eta_matrix * dot_total
            
            # Solve eigenvalues
            # A is real symmetric?
            # eta_matrix is real (since eta_pixel is real and symmetric? No, eta_fft is complex in general).
            # However, for real symmetric eta_pixel, eta(-G) = conj(eta(G)).
            # The matrix A should be Hermitian.
            # If we use real part, it might be wrong if phases are significant.
            # But for TM modes and real permittivity, the eigenvalues are real.
            # np.linalg.eigvalsh works for Hermitian.
            # Is A Hermitian?
            # A_ji = eta(Gj-Gi) * dot(Gj, Gi)
            # A_ij = eta(Gi-Gj) * dot(Gi, Gj)
            # eta(Gj-Gi) = conj(eta(Gi-Gj))
            # dot is symmetric.
            # So A_ji = conj(A_ij). Yes, Hermitian.
            
            try:
                eigs = np.linalg.eigvalsh(A)
            except:
                continue
                
            # Get 4th and 5th bands (indices 3 and 4)
            # Bands are 1-indexed in problem description.
            # eigs[0] is band 1.
            w4 = np.sqrt(eigs[3])
            w5 = np.sqrt(eigs[4])
            
            if w5 < min_w5:
                min_w5 = w5
            if w4 > max_w4:
                max_w4 = w4
                
        if min_w5 == np.inf or max_w4 == -np.inf:
            continue
            
        gap = min_w5 - max_w4
        if gap <= 0:
            score = -1.0
        else:
            score = 2 * gap / (min_w5 + max_w4)
            
        if score > best_score:
            best_score = score
            best_cell = cell.tolist()
            
    if best_cell is None:
        # Fallback to seed
        cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
        return cell
        
    return best_cell
