import time
import math
import numpy as np

def design(problem):
    start_time = time.time()
    deadline = start_time + 220  # 10s margin before 230s limit
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # High-fidelity solver parameters
    # 17x17 = 289 plane waves, matching the evaluator's resolution
    N_pw = 17
    half_N = N_pw // 2
    
    # Precompute plane wave vectors
    # nx, ny from -half_N to half_N
    nx_range = np.arange(-half_N, half_N + 1)
    ny_range = np.arange(-half_N, half_N + 1)
    nx_grid, ny_grid = np.meshgrid(nx_range, ny_range, indexing='ij')
    pw_list = np.column_stack((nx_grid.ravel(), ny_grid.ravel()))
    N_pw_actual = len(pw_list)
    
    # K-points in the irreducible Brillouin zone
    # The evaluator uses 30 wave vectors. We should use a dense set to capture the gap accurately.
    # Gamma-M-X-Gamma path.
    # Gamma: (0,0), M: (0.5, 0), X: (0.5, 0.5)
    # Let's sample 10 points on Gamma-M, 10 on M-X, 10 on X-Gamma.
    k_points = []
    for i in range(10):
        t = i / 9.0
        # Gamma to M: (0,0) to (0.5, 0)
        k_points.append((0.5 * t, 0.0))
    for i in range(1, 10):
        t = i / 9.0
        # M to X: (0.5, 0) to (0.5, 0.5)
        k_points.append((0.5, 0.5 * t))
    for i in range(1, 10):
        t = i / 9.0
        # X to Gamma: (0.5, 0.5) to (0,0)
        k_points.append((0.5 * (1 - t), 0.5 * (1 - t)))
        
    # Remove duplicates if any (none expected with this construction)
    k_points = np.array(k_points)
    
    def compute_eps_inv_fourier(cell):
        """
        Compute Fourier coefficients of 1/epsilon for a given 32x32 binary cell.
        Returns a 2D array of coefficients indexed by [ny+half_N, nx+half_N].
        """
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        # FFT
        fft_inv_eps = np.fft.fft2(inv_eps)
        fft_inv_eps /= (size * size)
        fft_inv_eps = np.fft.fftshift(fft_inv_eps)
        return fft_inv_eps

    def solve_bands(fft_inv_eps, kx, ky, num_bands=5):
        """
        Solve the eigenvalue problem for TM bands at a specific k-point.
        Returns the first num_bands eigenvalues (omega^2).
        """
        # k_vectors = pw_list + (kx, ky)
        # We need to construct the matrix H.
        # H_ij = (1/eps)_{k_i - k_j} * (k_i - k_j)^2
        
        # k_i - k_j = (nx_i - nx_j, ny_i - ny_j)
        # Since kx, ky are the same for all i, j, the difference is just the reciprocal lattice vector difference.
        # Let G_i = pw_list[i], G_j = pw_list[j].
        # diff = G_i - G_j.
        
        # We can vectorize the construction of H.
        # G is N x 2.
        # G_diff[i, j] = G[i] - G[j].
        # This is N x N x 2.
        
        # To save memory and time, we can exploit the fact that the coefficients depend only on the difference.
        # However, for N=289, N^2 = 83521. This is manageable.
        
        G_diff = pw_list[:, np.newaxis, :] - pw_list[np.newaxis, :, :]
        # G_diff is (N, N, 2)
        
        # Flatten the last dimension to get indices
        qx = G_diff[:, :, 0].astype(int)
        qy = G_diff[:, :, 1].astype(int)
        
        # We need to look up fft_inv_eps at [qy + half_N, qx + half_N]
        # Check bounds
        # qx, qy are in range [-N_pw, N_pw]
        # fft_inv_eps is size 32x32, shifted.
        # Indices: 0..31. Center is 16.
        # Range of qx: -17..17.
        # Index range: 16-17 = -1 to 16+17 = 33.
        # Wait, size is 32. Indices 0..31.
        # If qx = 17, idx = 17 + 16 = 33 -> Out of bounds.
        # The FFT of a 32x32 grid has frequencies -16..15 (or similar depending on shift).
        # np.fft.fftshift moves the zero frequency to the center.
        # For N=32, frequencies are -16, ..., 15.
        # So valid qx are -16..15.
        # Our G_diff can be up to 17.
        # Coefficients for |q| > 15 are effectively 0 (or wrap around, but physically they are small/zero for smooth eps).
        # We will mask out-of-bounds to 0.
        
        idx_x = qx + half_N
        idx_y = qy + half_N
        
        # Create a mask for valid indices
        valid_mask = (idx_x >= 0) & (idx_x < size) & (idx_y >= 0) & (idx_y < size)
        
        # Initialize coeff array
        coeff = np.zeros((N_pw_actual, N_pw_actual), dtype=complex)
        
        # Fill valid parts
        # We can't directly index with 2D arrays of different shapes easily without advanced indexing
        # coeff[valid_mask] = fft_inv_eps[idx_y[valid_mask], idx_x[valid_mask]]
        # But fft_inv_eps is 2D.
        
        # Let's do it safely
        coeff[valid_mask] = fft_inv_eps[idx_y[valid_mask], idx_x[valid_mask]]
        
        # Compute k_diff_sq
        k_diff_sq = np.sum(G_diff**2, axis=2)
        
        # H = coeff * k_diff_sq
        H = coeff * k_diff_sq
        
        # H is Hermitian (since coeff is Hermitian and k_diff_sq is symmetric real)
        # Solve eigenvalue problem
        try:
            vals, vecs = np.linalg.eigh(H)
        except np.linalg.LinAlgError:
            return None, None
            
        # Filter out small eigenvalues
        vals = vals[vals > 1e-8]
        
        if len(vals) < num_bands:
            return None, None
            
        return vals[:num_bands], vecs[:, :num_bands]

    def evaluate_gap(cell):
        """
        Evaluate the gap-midgap ratio for a given cell.
        """
        fft_inv_eps = compute_eps_inv_fourier(cell)
        
        max_omega_4 = 0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            vals, _ = solve_bands(fft_inv_eps, kx, ky, num_bands=5)
            if vals is None:
                continue
            omegas = np.sqrt(vals)
            max_omega_4 = max(max_omega_4, omegas[3])
            min_omega_5 = min(min_omega_5, omegas[4])
            
        if min_omega_5 == float('inf'):
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap

    def generate_cell(rod_radius, rod_pos_offset):
        """
        Generate a symmetric 32x32 cell with 4 dielectric rods in the corners.
        rod_radius: radius of the rods.
        rod_pos_offset: offset from the ideal corner position (size/4).
        """
        cell = np.zeros((size, size), dtype=int)
        y, x = np.mgrid[0:size, 0:size]
        y = y + 0.5
        x = x + 0.5
        
        # Ideal centers
        c = size / 4.0
        centers = [
            (c - rod_pos_offset, c - rod_pos_offset),
            (c - rod_pos_offset, size - c + rod_pos_offset),
            (size - c + rod_pos_offset, c - rod_pos_offset),
            (size - c + rod_pos_offset, size - c + rod_pos_offset)
        ]
        
        for cx, cy in centers:
            dist = (x - cx)**2 + (y - cy)**2
            cell[dist < rod_radius**2] = 1
            
        # Enforce symmetry
        cell = (cell + cell[::-1, ::-1]) // 2
        cell = (cell + cell.T) // 2
        
        return cell

    def generate_cell_square(square_size, square_pos_offset):
        """
        Generate a symmetric 32x32 cell with 4 square dielectric blocks in the corners.
        """
        cell = np.zeros((size, size), dtype=int)
        
        c = size / 4.0
        # Define square boundaries relative to center
        half_s = square_size / 2.0
        
        # Centers
        centers = [
            (c - square_pos_offset, c - square_pos_offset),
            (c - square_pos_offset, size - c + square_pos_offset),
            (size - c + square_pos_offset, c - square_pos_offset),
            (size - c + square_pos_offset, size - c + square_pos_offset)
        ]
        
        for cx, cy in centers:
            x_min = int(cx - half_s)
            x_max = int(cx + half_s)
            y_min = int(cy - half_s)
            y_max = int(cy + half_s)
            
            # Clip to bounds
            x_min = max(0, x_min)
            x_max = min(size, x_max)
            y_min = max(0, y_min)
            y_max = min(size, y_max)
            
            cell[y_min:y_max, x_min:x_max] = 1
            
        # Enforce symmetry
        cell = (cell + cell[::-1, ::-1]) // 2
        cell = (cell + cell.T) // 2
        
        return cell

    # Search Strategy
    # 1. Coarse Grid Search
    # 2. Local Refinement
    
    best_score = -1.0
    best_cell = None
    
    # Define search space
    # Rod Radius: 2.0 to 8.0
    # Rod Offset: -2.0 to 2.0
    
    # Time check
    if time.time() > deadline - 50:
        # Fallback: just return a reasonable default
        cell = generate_cell(5.5, 0.0)
        return cell.tolist()

    # Coarse Search
    radii = np.arange(3.0, 8.5, 0.5)
    offsets = np.arange(-2.0, 2.5, 1.0)
    
    for r in radii:
        if time.time() > deadline - 30:
            break
        for off in offsets:
            if time.time() > deadline - 30:
                break
            
            cell = generate_cell(r, off)
            score = evaluate_gap(cell)
            
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
                best_params = ('rod', r, off)

    # Try Squares too
    sizes = np.arange(4.0, 12.5, 1.0)
    offsets_sq = np.arange(-2.0, 2.5, 1.0)
    
    for s in sizes:
        if time.time() > deadline - 30:
            break
        for off in offsets_sq:
            if time.time() > deadline - 30:
                break
            
            cell = generate_cell_square(s, off)
            score = evaluate_gap(cell)
            
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
                best_params = ('sq', s, off)

    # Local Refinement around the best coarse solution
    if best_cell is not None and time.time() < deadline - 20:
        param_type, param1, param2 = best_params
        
        # Define refinement ranges
        if param_type == 'rod':
            r_range = np.arange(param1 - 0.5, param1 + 0.5, 0.1)
            off_range = np.arange(param2 - 0.5, param2 + 0.5, 0.1)
            gen_func = generate_cell
        else:
            r_range = np.arange(param1 - 1.0, param1 + 1.0, 0.2)
            off_range = np.arange(param2 - 0.5, param2 + 0.5, 0.1)
            gen_func = generate_cell_square
            
        for p1 in r_range:
            if time.time() > deadline - 10:
                break
            for p2 in off_range:
                if time.time() > deadline - 10:
                    break
                    
                cell = gen_func(p1, p2)
                score = evaluate_gap(cell)
                
                if score > best_score:
                    best_score = score
                    best_cell = cell.copy()

    # If no valid cell was found (shouldn't happen), return a default
    if best_cell is None:
        best_cell = generate_cell(5.5, 0.0)

    return best_cell.tolist()
