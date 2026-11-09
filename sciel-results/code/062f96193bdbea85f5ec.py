import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Time management
    start_time = time.time()
    # Evaluator limit is 230s. We use 210s as a safety margin for return overhead.
    time_limit = 210.0
    
    # --- Solver Setup (Fast Vectorized) ---
    # 289 plane waves implies 17x17 grid. Gx, Gy in [-8, 8].
    N_G = 17
    half_G = N_G // 2
    g_range = np.arange(-half_G, half_G + 1)
    Gx, Gy = np.meshgrid(g_range, g_range, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    N_pw = len(G_vectors)
    
    # Precompute diff indices for H matrix construction
    # H[i, j] depends on eps_inv_coeffs[G_i - G_j]
    dx = G_vectors[:, 0][:, None] - G_vectors[:, 0][None, :]
    dy = G_vectors[:, 1][:, None] - G_vectors[:, 1][None, :]
    
    # Valid differences are those within the FFT grid range [-8, 8]
    # If a difference is outside, the coefficient is assumed 0 (truncation)
    valid_diffs = (np.abs(dx) <= half_G) & (np.abs(dy) <= half_G)
    
    # Map valid diffs to linear index in the 17x17 coefficient array
    # Index = (dx + 8) * 17 + (dy + 8)
    diff_indices = np.where(valid_diffs, (dx + half_G) * N_G + (dy + half_G), 0)
    
    # K points in irreducible BZ
    # We use a set of points covering the boundary and interior to approximate the evaluator's 30 points.
    # Irreducible BZ for square lattice: 0 <= ky <= kx, 0 <= kx <= 0.5 (if using reciprocal lattice units where BZ is [-0.5, 0.5])
    # Wait, standard square lattice BZ is [-pi/a, pi/a]. In fractional coordinates of reciprocal lattice vectors, it's [-0.5, 0.5].
    # The task says "30 wave vectors covering the boundary and interior".
    # Let's generate a dense set of points in the 1/8 wedge: 0 <= ky <= kx <= 0.5
    k_points = []
    # Boundary points
    for i in range(10):
        t = i / 9.0 * 0.5
        k_points.append([t, 0.0])       # Gamma to X
        k_points.append([0.5, t])       # M to X (if X is 0.5,0 and M is 0.5,0.5)
        k_points.append([t, t])         # Gamma to M
    # Interior points
    # Add some random-looking interior points in the wedge
    for i in range(1, 4):
        for j in range(1, 4):
            kx = i * 0.1
            ky = j * 0.05
            if ky <= kx and kx <= 0.5:
                k_points.append([kx, ky])
    
    k_points = np.array(k_points[:30]) # Limit to ~30 points for speed
    
    def get_fourier_eps_inv(cell_arr):
        """
        Computes Fourier coefficients of 1/eps.
        cell_arr: 32x32 int array (0 or 1).
        Returns 1D array of length N_pw (289) with coeffs for G in [-8,8]x[-8,8].
        """
        eps = np.where(cell_arr == 1, eps_high, eps_low)
        eps_inv = 1.0 / eps
        # FFT of the 32x32 array
        # coeffs[i, j] corresponds to Gx = i - 16, Gy = j - 16 (for 0..31 range)
        # We need coeffs for Gx, Gy in -8..8.
        # In FFT output (0..31), index 0 is G=0, 1 is G=1, ..., 15 is G=15, 16 is G=-16 (or 16?), 
        # Standard numpy fft: index k corresponds to frequency k/N.
        # For N=32, index 0 -> 0, 1 -> 1/32, ..., 16 -> 16/32=0.5 (Nyquist, or -0.5), 17 -> -15/32, etc.
        # Actually, fftfreq(32) gives [0, 1, ..., 15, -16, -15, ..., -1].
        # So index 16 is -16? No, fftfreq returns [-16, -15, ..., -1, 0, 1, ..., 15] if shifted?
        # Let's just use the mapping:
        # G = -8 .. 8.
        # In 0..31 grid:
        # 0 -> 0
        # 1 -> 1
        # ...
        # 8 -> 8
        # 16 -> -16 (alias for 16? No, in 32, 16 is the Nyquist point, usually treated as -16 or 16. 
        #      For periodic functions, e^{i 2pi 16 x / 32} = e^{i pi x} = e^{i 2pi (-16) x / 32}. So 16 and -16 are same.)
        # -8 corresponds to 32-8=24.
        # So we need indices: 
        # -8 -> 24, -7 -> 25, ..., -1 -> 31, 0 -> 0, 1 -> 1, ..., 8 -> 8.
        
        coeffs_2d = np.fft.fft2(eps_inv) / (size * size)
        
        # Construct indices for Gx, Gy in -8..8
        # Gx_idx = Gx + 16. If Gx is -8, idx=8. If Gx is 8, idx=24? 
        # Wait. 
        # fftfreq(32, d=1/32) -> [0, 1, ..., 15, -16, -15, ..., -1]
        # So:
        # G=0 -> idx 0
        # G=1 -> idx 1
        # ...
        # G=8 -> idx 8
        # G=-1 -> idx 31
        # G=-8 -> idx 24
        
        Gx_vals = G_vectors[:, 0]
        Gy_vals = G_vectors[:, 1]
        
        Gx_idx = Gx_vals + size # If Gx is -8, -8+32=24. If Gx is 8, 8+32=40? No, 40%32=8.
        # Correct mapping:
        # If G >= 0: idx = G
        # If G < 0: idx = G + size
        Gx_idx = np.where(Gx_vals >= 0, Gx_vals, Gx_vals + size)
        Gy_idx = np.where(Gy_vals >= 0, Gy_vals, Gy_vals + size)
        
        return coeffs_2d[Gx_idx, Gy_idx]

    def evaluate(cell_arr):
        """
        Computes the gap-midgap ratio for bands 4 and 5.
        """
        eps_inv_coeffs = get_fourier_eps_inv(cell_arr)
        
        min_omega_5 = float('inf')
        max_omega_4 = float('-inf')
        
        for k in k_points:
            k_vec = np.array(k)
            # k is in fractional units? The task says "wave vectors". 
            # Usually in PW methods for lattice constant 1, k is in [0, 1] or [0, 2pi].
            # The problem says "irreducible Brillouin zone". 
            # Standard square lattice BZ in fractional coords of reciprocal lattice is [-0.5, 0.5].
            # My k_points are in [0, 0.5]. This is correct for the 1/8 wedge.
            
            k_plus_G = k_vec[None, :] + G_vectors
            k_plus_G_sq = np.sum(k_plus_G**2, axis=1)
            
            # H matrix: H[i, j] = eps_inv_coeffs[G_i - G_j] * (k + G_j)^2
            # We precomputed diff_indices[i, j] which is the index in eps_inv_coeffs for G_i - G_j.
            # If valid_diffs[i, j] is false, the term is 0.
            
            # Vectorized construction:
            # H = coeffs[diff_indices] * k_plus_G_sq[None, :]
            # But we must zero out invalid diffs.
            H = eps_inv_coeffs[diff_indices] * k_plus_G_sq[None, :]
            H[~valid_diffs] = 0.0
            
            # Solve eigenvalue problem
            # H is not Hermitian, but eigenvalues are real for this specific structure (diagonal dominant-ish or symmetric properties in TM?).
            # Actually, for TM, the matrix is A_{mn} = delta_{mn} (k+G_m)^2 / <1/eps>_{mn}.
            # This is a generalized eigenvalue problem or standard if transformed.
            # The standard PW formulation for TM is:
            # sum_n (1/eps)_{mn} (k+G_m) x (k+G_n) . E_n = omega^2 E_m
            # This is a standard eigenvalue problem for the matrix H.
            
            evals = np.linalg.eigvals(H)
            evals = np.sort(np.real(evals))
            
            # Bands 4 and 5 (indices 3 and 4)
            if len(evals) < 5:
                continue
                
            w4 = evals[3]
            w5 = evals[4]
            
            max_omega_4 = max(max_omega_4, w4)
            min_omega_5 = min(min_omega_5, w5)
            
        if min_omega_5 == float('inf') or max_omega_4 == float('-inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap

    # --- Design Search ---
    
    # Strategy: 4-Circle Cluster
    # Symmetry group D4. 
    # To maintain symmetry, we can define the cell by a fundamental domain (1/8 of the cell) 
    # or by placing symmetric objects.
    # A 4-circle cluster centered at (0,0), (0.5,0), (0,0.5), (0.5,0.5) in fractional coords?
    # Or centered at (0.25, 0.25), (0.75, 0.25), etc?
    # The seed was a single rod at center.
    # A common high-gap structure is a square lattice of rods, or a "superlattice" of rods.
    # Let's try 4 rods in a 2x2 supercell arrangement within the 32x32 cell.
    # Centers at: (8,8), (24,8), (8,24), (24,24).
    # This preserves D4 symmetry (inversion, rotation, reflection).
    
    def generate_cell_4circles(radii, centers):
        """
        radii: list of 4 radii (or one radius for all).
        centers: list of 4 (x, y) centers in pixel coords.
        """
        cell = np.zeros((size, size), dtype=int)
        # Precompute x, y grids
        x = (np.arange(size) + 0.5) / size - 0.5
        y = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        for i, (cx, cy) in enumerate(centers):
            # Convert center to pixel coords? 
            # x, y are in [-0.5, 0.5]. 
            # Centers provided in fractional coords [-0.5, 0.5].
            dist_sq = (X - cx)**2 + (Y - cy)**2
            r = radii[i]
            cell[dist_sq < r**2] = 1
        return cell

    # Initial guess: 4 circles of radius 0.15 at (0.25, 0.25), (-0.25, 0.25), (0.25, -0.25), (-0.25, -0.25)
    # In pixel coords (0-31), 0.25 is 8.5? No, 0.25 of 32 is 8.
    # Let's use fractional coords for centers.
    centers_init = [
        (0.25, 0.25),
        (-0.25, 0.25),
        (0.25, -0.25),
        (-0.25, -0.25)
    ]
    radius_init = 0.15
    
    cell = generate_cell_4circles([radius_init]*4, centers_init)
    best_score = evaluate(cell)
    best_cell = cell.copy()
    
    # Local Search: Perturb radii and centers slightly, or flip pixels.
    # Flipping pixels is more general. Let's stick to pixel flipping for robustness,
    # but initialize with the 4-circle structure.
    
    def get_orbit(i, j):
        N = size
        points = [
            (i, j),
            (j, N-1-i),
            (N-1-i, N-1-j),
            (N-1-j, i),
            (i, N-1-j),
            (N-1-i, j),
            (j, i),
            (N-1-j, N-1-i)
        ]
        return list(set(points))

    no_improve_count = 0
    max_no_improve = 1000 # Increased to allow deeper search
    
    while time.time() - start_time < time_limit:
        # Pick random pixel
        i = np.random.randint(size)
        j = np.random.randint(size)
        
        orbit = get_orbit(i, j)
        
        # Flip
        for (r, c) in orbit:
            cell[r, c] = 1 - cell[r, c]
            
        score = evaluate(cell)
        
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
            no_improve_count = 0
        else:
            # Revert
            for (r, c) in orbit:
                cell[r, c] = 1 - cell[r, c]
            no_improve_count += 1
            
        # Restart if stuck
        if no_improve_count > max_no_improve:
            # Reset to best and apply random noise
            cell = best_cell.copy()
            for _ in range(10):
                ri = np.random.randint(size)
                rj = np.random.randint(size)
                orbit = get_orbit(ri, rj)
                for (r, c) in orbit:
                    cell[r, c] = 1 - cell[r, c]
            no_improve_count = 0

    # Final check
    if time.time() - start_time > time_limit - 5:
        return best_cell.tolist()
        
    return best_cell.tolist()
