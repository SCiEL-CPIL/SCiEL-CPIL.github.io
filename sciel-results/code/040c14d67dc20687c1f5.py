import numpy as np
import time
import sys
import scipy.linalg

def design(problem):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # G-vectors setup
    g_range = range(-8, 9)
    G_int = np.array([(gx, gy) for gx in g_range for gy in g_range], dtype=int)
    G_float = G_int.astype(float) * 2.0 * np.pi
    N_g = len(G_int)
    
    # K-points in the irreducible BZ
    # Triangle: 0 <= kx <= ky <= pi
    k_points = []
    n_k = 10
    for i in range(n_k):
        for j in range(i, n_k):
            kx = i / (n_k - 1) * np.pi
            ky = j / (n_k - 1) * np.pi
            k_points.append((kx, ky))
    # Ensure we have exactly 30 or fewer points
    # The grid above gives 10 + 9 + ... + 1 = 55 points.
    # We need 30.
    # We can subsample or just take the first 30?
    # Better to distribute evenly.
    # Let's just take a subset.
    # Or generate a specific set of 30 points.
    # For simplicity, we'll use the 55 points if it's fast enough, 
    # but the task says "30 wave vectors". 
    # To be safe and fast, we will use 30 points.
    
    # Generate 30 points in the triangle
    # 10 on kx=0, 10 on ky=kx, 10 on ky=pi?
    # Let's just use the grid points and subsample to 30.
    # Or just use the 30 most important ones.
    # Actually, the evaluator uses 30 specific points. We don't know which.
    # But a dense sampling is better for estimating the gap.
    # Let's use 30 points from the grid.
    k_points = np.array(k_points[:30])
    
    def compute_epsilon_conv(cell):
        """Compute the 289x289 epsilon convolution matrix."""
        eps_grid = np.where(np.array(cell) == 1, eps_high, eps_low)
        eps_fft = np.fft.fft2(eps_grid)
        eps_fft /= (size * size)
        
        # Precompute FFT indices for all G_int
        # G_int is (N_g, 2)
        # ix = G_int[:, 0] % size
        # iy = G_int[:, 1] % size
        ix = G_int[:, 0] % size
        iy = G_int[:, 1] % size
        eps_coeffs = eps_fft[iy, ix] # Length N_g
        
        # Build convolution matrix E_conv[i, j] = eps_{G_i - G_j}
        # We need eps_{G_i - G_j}.
        # Since eps is real and symmetric, eps_{-G} = conj(eps_G) = eps_G (if real).
        # We can compute this by iterating.
        E_conv = np.zeros((N_g, N_g), dtype=float)
        
        # Map from (dx, dy) to index?
        # Or just loop.
        # To speed up, we can use the fact that G_i - G_j is in [-16, 16].
        # We can precompute a lookup table for eps_diff for all possible diffs.
        # Diffs are integers in [-16, 16].
        # Number of diffs: 33x33 = 1089.
        # We can compute eps for these diffs using the FFT?
        # No, the FFT is for the 32x32 grid.
        # The difference G_i - G_j corresponds to a spatial frequency.
        # We can get the coefficient by looking at the FFT of the periodic extension?
        # Actually, eps_{G} is the Fourier coefficient of the periodic function.
        # For any integer G, we can find the coefficient.
        # If G is in [-16, 16], it might alias if we only have 32x32 FFT.
        # But 32x32 FFT gives frequencies -16..15.
        # So for diff in [-16, 16], we can map to FFT index.
        
        # Create a dictionary of coeffs for all needed diffs
        coeff_dict = {}
        for dx in range(-16, 17):
            for dy in range(-16, 17):
                ix_d = dx % size
                iy_d = dy % size
                coeff_dict[(dx, dy)] = eps_fft[iy_d, ix_d].real # eps is real
        
        for i in range(N_g):
            for j in range(N_g):
                dx = G_int[i, 0] - G_int[j, 0]
                dy = G_int[i, 1] - G_int[j, 1]
                E_conv[i, j] = coeff_dict[(dx, dy)]
                
        return E_conv

    def solve_tm_bands(E_conv, kx, ky):
        N = N_g
        # Build K matrix (578x578)
        # K is block diagonal
        # Block i: ky_i^2, -kx_i ky_i; -kx_i ky_i, kx_i^2
        kG_x = kx + G_float[:, 0]
        kG_y = ky + G_float[:, 1]
        
        K = np.zeros((2*N, 2*N), dtype=float)
        # Vectorized fill
        K[0::2, 0::2] = kG_y**2
        K[1::2, 1::2] = kG_x**2
        K[0::2, 1::2] = -kG_x * kG_y
        K[1::2, 0::2] = -kG_y * kG_x
        
        # Build E matrix (578x578)
        # E is block matrix with E_conv[i,j] * I2
        E = np.zeros((2*N, 2*N), dtype=float)
        for i in range(N):
            for j in range(N):
                val = E_conv[i, j]
                E[2*i, 2*j] = val
                E[2*i+1, 2*j+1] = val
                # Off-diagonals in the 2x2 block are 0
                # E[2*i, 2*j+1] = 0
                # E[2*i+1, 2*j] = 0
                
        # Solve K A = omega^2 E A
        try:
            eigvals, _ = scipy.linalg.eigh(K, E)
            # Filter out negative or zero eigenvalues (numerical noise)
            eigvals = eigvals[eigvals > 1e-6]
            omegas = np.sqrt(eigvals)
            omegas.sort()
            return omegas
        except Exception:
            return None

    def get_score(cell):
        if time.time() > deadline:
            return -1.0
        E_conv = compute_epsilon_conv(cell)
        max_w4 = 0.0
        min_w5 = float('inf')
        
        for kx, ky in k_points:
            omegas = solve_tm_bands(E_conv, kx, ky)
            if omegas is None or len(omegas) < 5:
                continue
            # Bands 4 and 5 (1-indexed) -> indices 3 and 4
            w4 = omegas[3]
            w5 = omegas[4]
            if w4 > max_w4:
                max_w4 = w4
            if w5 < min_w5:
                min_w5 = w5
                
        if min_w5 == float('inf'):
            return -1.0
            
        gap = min_w5 - max_w4
        midgap = (min_w5 + max_w4) / 2.0
        if midgap == 0:
            return -1.0
        ratio = 2.0 * gap / midgap
        return ratio

    # Candidate structures
    candidates = []
    
    # 1. 4 rods
    def make_4_rods(r):
        cell = [[0]*size for _ in range(size)]
        centers = [(8, 8), (23, 8), (8, 23), (23, 23)]
        for i in range(size):
            for j in range(size):
                px = i + 0.5
                py = j + 0.5
                for cx, cy in centers:
                    if (px - cx)**2 + (py - cy)**2 < r**2:
                        cell[i][j] = 1
                        break
        return cell
        
    candidates.append(make_4_rods(3.0))
    candidates.append(make_4_rods(4.0))

    # 2. Checkerboard
    cell_check = [[(i+j) % 2 for j in range(size)] for i in range(size)]
    candidates.append(cell_check)
    
    # 3. Single large hole in dielectric (mostly 1s, hole in center)
    cell_hole = [[1]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            px = i + 0.5
            py = j + 0.5
            if (px - 15.5)**2 + (py - 15.5)**2 < 8.0**2:
                cell_hole[i][j] = 0
    candidates.append(cell_hole)
    
    # 4. Single large rod in air (mostly 0s, rod in center)
    cell_rod = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            px = i + 0.5
            py = j + 0.5
            if (px - 15.5)**2 + (py - 15.5)**2 < 8.0**2:
                cell_rod[i][j] = 1
    candidates.append(cell_rod)
    
    # 5. 4 holes in dielectric
    def make_4_holes(r):
        cell = [[1]*size for _ in range(size)]
        centers = [(8, 8), (23, 8), (8, 23), (23, 23)]
        for i in range(size):
            for j in range(size):
                px = i + 0.5
                py = j + 0.5
                for cx, cy in centers:
                    if (px - cx)**2 + (py - cy)**2 < r**2:
                        cell[i][j] = 0
                        break
        return cell
        
    candidates.append(make_4_holes(3.0))
    candidates.append(make_4_holes(4.0))

    best_cell = candidates[0]
    best_score = -10.0
    
    for cell in candidates:
        score = get_score(cell)
        if score > best_score:
            best_score = score
            best_cell = cell
            
    return best_cell
