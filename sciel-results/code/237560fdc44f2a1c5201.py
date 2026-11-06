"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a symmetric 32x32 unit cell to maximize the TM band gap between bands 4 and 5.
    
    Strategy:
    1. Use a parametric representation of the unit cell to enforce symmetry and reduce search space.
    2. Implement an efficient PWE solver with correct integer indexing.
    3. Perform a local search over the parametric space.
    """
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 110.0

    # --- PWE Solver Setup ---
    # Number of plane waves per dimension
    # 289 = 17^2, so -8 to 8
    N_G = 17
    G_range = range(-N_G // 2, N_G // 2 + 1)
    
    # Precompute G vectors as integers
    G_vectors = []
    for n in G_range:
        for m in G_range:
            G_vectors.append((n, m))
    G = np.array(G_vectors, dtype=int)
    num_G = len(G)
    
    # Precompute difference indices for the PWE matrix
    # We need indices into the FFT of 1/eps for dG = G_b - G_a
    # G is (num_G, 2)
    # dG_n[b, a] = G[b, 0] - G[a, 0]
    # dG_m[b, a] = G[b, 1] - G[a, 1]
    
    G_col = G[:, np.newaxis, :]  # (num_G, 1, 2)
    G_row = G[np.newaxis, :, :]  # (1, num_G, 2)
    dG = G_col - G_row  # (num_G, num_G, 2)
    
    # Convert dG to indices in the FFT array (size x size)
    # FFT indices are modulo size
    idx_n = dG[:, :, 0] % size
    idx_m = dG[:, :, 1] % size
    
    # Precompute a list of (idx_m, idx_n) pairs? No, we need the values from the FFT.
    # We will store the indices to be used for advanced indexing later.
    # To speed up, we can precompute the flat indices or just keep the 2D arrays.
    # Advanced indexing with 2D arrays of shape (289, 289) is fast enough.

    # K-points for evaluation
    # Irreducible BZ: 0 <= kx <= ky <= 0.5
    k_points = []
    # High symmetry points
    k_points.append((0.0, 0.0))
    k_points.append((0.5, 0.0))
    k_points.append((0.5, 0.5))
    # Grid points
    for i in range(10):
        for j in range(10):
            kx = i * 0.5 / 9.0
            ky = j * 0.5 / 9.0
            if kx <= ky + 1e-9 and ky <= 0.5 + 1e-9 and kx >= 0:
                k_points.append((kx, ky))

    def compute_inv_eps_fft(cell):
        """Compute the 2D FFT of 1/eps. Returns the 32x32 complex array."""
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        fft_inv_eps = np.fft.fft2(inv_eps)
        return fft_inv_eps / (size * size)

    def solve_bands(cell):
        """Solve for the first 5 TM bands. Returns gap ratio."""
        inv_eps_fft = compute_inv_eps_fft(cell)
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            k = np.array([kx, ky])
            # k_plus_G = k + G
            # Shape: (num_G, 2)
            k_plus_G = k + G.astype(float)
            
            # Dot product matrix M_ab = (k+Ga) . (k+Gb)
            # M = k_plus_G @ k_plus_G.T
            M = k_plus_G @ k_plus_G.T
            
            # Get the 1/eps coefficients for the differences
            # D_ab = inv_eps_fft[idx_m[b,a], idx_n[b,a]]
            D = inv_eps_fft[idx_m, idx_n]
            
            # H = M * D
            H = M * D
            
            # Symmetrize
            H = (H + H.conj().T) / 2.0
            
            eigenvalues = np.linalg.eigvalsh(H)
            
            if len(eigenvalues) >= 5:
                omega_4 = math.sqrt(eigenvalues[3])
                omega_5 = math.sqrt(eigenvalues[4])
                if omega_4 > max_omega_4:
                    max_omega_4 = omega_4
                if omega_5 < min_omega_5:
                    min_omega_5 = omega_5
        
        if min_omega_5 == float('inf'):
            return -1.0
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        ratio = gap / midgap
        return ratio

    # --- Parametric Design Space ---
    # We define a few topological templates and optimize their parameters.
    # Template 1: Air holes in dielectric background (common for TM gaps)
    # Holes at center, corners, and edge centers.
    
    def generate_cell(params):
        """
        params: list of (center_x, center_y, radius) for air holes (0) in dielectric (1).
        Or we can define dielectric rods. Let's stick to air holes in dielectric.
        The cell is initialized to 1 (dielectric).
        Holes are set to 0.
        Symmetry is enforced by placing holes in symmetric sets.
        """
        cell = np.ones((size, size), dtype=int)
        
        # Helper to punch a circular hole
        def punch_hole(cx, cy, r):
            # cx, cy are in pixel coordinates (0..31)
            # r is radius in pixels
            y, x = np.ogrid[:size, :size]
            dist_sq = (x - cx)**2 + (y - cy)**2
            cell[dist_sq <= r**2] = 0
        
        # Define symmetric sets of holes
        # 1. Center hole
        if params[0] > 0:
            punch_hole(15.5, 15.5, params[0])
            
        # 2. Corner holes (4-fold symmetry)
        # Corners are at (0,0), (31,0), (0,31), (31,31)
        # But usually holes are inside. Let's place them at (2,2), (2,29), (29,2), (29,29)
        if params[1] > 0:
            offset = 2.0
            punch_hole(offset, offset, params[1])
            punch_hole(offset, size-1-offset, params[1])
            punch_hole(size-1-offset, offset, params[1])
            punch_hole(size-1-offset, size-1-offset, params[1])
            
        # 3. Edge center holes (4-fold symmetry)
        # Edges: (15.5, 0), (15.5, 31), (0, 15.5), (31, 15.5)
        # Let's place them slightly inside: (15.5, 2), (15.5, 29), (2, 15.5), (29, 15.5)
        if params[2] > 0:
            offset = 2.0
            punch_hole(15.5, offset, params[2])
            punch_hole(15.5, size-1-offset, params[2])
            punch_hole(offset, 15.5, params[2])
            punch_hole(size-1-offset, 15.5, params[2])
            
        # 4. Diagonal holes (optional, to break degeneracy)
        # Place at (8,8), (8,23), (23,8), (23,23)
        if params[3] > 0:
            offset = 8.0
            punch_hole(offset, offset, params[3])
            punch_hole(offset, size-1-offset, params[3])
            punch_hole(size-1-offset, offset, params[3])
            punch_hole(size-1-offset, size-1-offset, params[3])
            
        return cell

    # Optimization: Local search on parameters
    # Parameters: [r_center, r_corner, r_edge, r_diag]
    # Ranges: [0, 5] for each
    
    best_score = -1.0
    best_params = [2.0, 2.0, 2.0, 0.0] # Initial guess
    best_cell = generate_cell(best_params)
    
    # Evaluate initial
    best_score = solve_bands(best_cell)
    
    # Search loop
    # Use a simple hill climbing with random restarts if stuck
    steps_without_improvement = 0
    max_steps = 1000 # Limit to ensure we don't run out of time
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        # Pick a random parameter to perturb
        p_idx = np.random.randint(4)
        perturbation = np.random.uniform(-1.0, 1.0)
        
        new_params = best_params.copy()
        new_params[p_idx] = max(0.0, min(5.0, new_params[p_idx] + perturbation))
        
        new_cell = generate_cell(new_params)
        new_score = solve_bands(new_cell)
        
        if new_score > best_score:
            best_score = new_score
            best_params = new_params
            best_cell = new_cell
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1
            if steps_without_improvement > 20:
                # Random restart
                best_params = [np.random.uniform(0, 4) for _ in range(4)]
                best_cell = generate_cell(best_params)
                best_score = solve_bands(best_cell)
                steps_without_improvement = 0

    return best_cell.tolist()
# EVOLVE-BLOCK-END
