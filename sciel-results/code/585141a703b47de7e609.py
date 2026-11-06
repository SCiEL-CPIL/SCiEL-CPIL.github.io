import numpy as np
import time
import random

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell to maximize the TM band gap 
    between the 4th and 5th bands using topological optimization.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    start_time = time.time()
    time_limit = 115  # Leave buffer

    # --- PWE Solver ---
    
    # Use a moderate number of plane waves for the surrogate.
    # 11x11 = 121 PWs is a good balance of speed and accuracy.
    # The evaluator uses 17x17 = 289 PWs.
    n_pw_dim = 11 
    n_pw = n_pw_dim * n_pw_dim
    half_pw = n_pw_dim // 2

    # Precompute G vectors
    G_list = []
    for gx in range(-half_pw, half_pw + 1):
        for gy in range(-half_pw, half_pw + 1):
            G_list.append((gx, gy))
    # Sort by magnitude to ensure we keep the most important ones if we truncated, 
    # but here we keep the full square grid.
    # For PWE, we usually need a square grid of G vectors centered at 0.
    
    def get_fourier_coeffs(cell):
        """
        Computes the Fourier coefficients of 1/epsilon(r).
        cell is a 2D numpy array of 0s and 1s.
        """
        eps_map = np.where(cell == 1, eps_high, eps_low)
        inv_eps_map = 1.0 / eps_map
        fft_coeffs = np.fft.fft2(inv_eps_map)
        fft_coeffs /= (size * size)
        return fft_coeffs

    def fft_index(g, N):
        """Map integer G component to FFT array index."""
        return g % N

    def solve_band_structure(cell, kx, ky, return_eigvecs=False):
        """
        Solves the eigenvalue problem for TM modes at k=(kx, ky).
        Returns sorted omega^2 values, and optionally eigenvectors.
        """
        fft_coeffs = get_fourier_coeffs(cell)
        
        # Build matrix M where M[i,j] = D(G_i - G_j)
        # G_i and G_j are from G_list
        M = np.zeros((n_pw, n_pw), dtype=complex)
        
        # Precompute indices for G vectors to speed up lookup
        # G_list is sorted by magnitude in the original code, but here we need consistent indexing.
        # Let's just use the list order.
        for i in range(n_pw):
            gx_i, gy_i = G_list[i]
            for j in range(n_pw):
                gx_j, gy_j = G_list[j]
                Gx_diff = gx_i - gx_j
                Gy_diff = gy_i - gy_j
                idx_x = fft_index(Gx_diff, size)
                idx_y = fft_index(Gy_diff, size)
                M[i, j] = fft_coeffs[idx_x, idx_y]
        
        # K^2 values
        K_sq = np.zeros(n_pw)
        for i in range(n_pw):
            gx_i, gy_i = G_list[i]
            Kx = kx + gx_i
            Ky = ky + gy_i
            K_sq[i] = Kx**2 + Ky**2
        
        # A = diag(K_sq) - M
        # For TM polarization, the eigenvalue equation is:
        # sum_j D(G_i - G_j) * c_j = (k+G_i)^2 * c_i / c0^2 ? 
        # Actually, the standard PWE eigenvalue problem for TM is:
        # sum_j (k+G_i) x (k+G_j) D(G_i - G_j) c_j = omega^2 c_i
        # Wait, the matrix form is usually:
        # [ (k+G_i)^2 I - D ] c = 0 is for TE?
        # For TM, the operator is 1/eps * (-grad^2).
        # In Fourier space, 1/eps is convolved.
        # The eigenvalue problem is:
        # sum_j D(G_i - G_j) c_j = ( (k+G_i)^2 / omega^2 ) c_i
        # So we solve:
        # omega^2 * sum_j D(G_i - G_j) c_j = (k+G_i)^2 c_i
        # This is a generalized eigenvalue problem: A c = lambda B c
        # Where A_ij = (k+G_i)^2 delta_ij, B_ij = D(G_i - G_j)
        # Or: A c = lambda c where A = diag(K_sq) * inv(B)? No, B is not diagonal.
        # We can solve: inv(B) A c = lambda c.
        # Since B is Hermitian and positive definite, we can use eigh with B as the metric.
        # numpy.linalg.eigh doesn't support generalized directly, but we can transform.
        # Or simply construct the matrix M = B^{-1} A.
        # Given the small size (121x121), we can invert B or use linalg.solve.
        
        # Let's use the generalized eigenvalue formulation via Cholesky or just solve.
        # B is the matrix of 1/eps coefficients.
        # We want eigenvectors c such that B c = (1/omega^2) A c.
        # Let lambda = 1/omega^2.
        # B c = lambda A c.
        # A c = (1/lambda) B c.
        # A^{-1} B c = (1/lambda) c.
        # Since A is diagonal, A^{-1} is easy.
        # Matrix C = A^{-1} B.
        # C is not Hermitian, but similar to a Hermitian matrix if B and A commute? They don't.
        # However, the eigenvalues of A^{-1} B are real and positive.
        # We can use scipy.linalg.eig or np.linalg.eig.
        
        # Alternative: 
        # The standard PWE matrix for TM is:
        # sum_j D(G_i - G_j) c_j = ( (k+G_i)^2 / omega^2 ) c_i
        # This is a generalized symmetric eigenvalue problem if we define the inner product with B.
        # Let's use scipy.linalg.eigh(B, A) -> solves B x = lambda A x.
        # Here B is the D matrix, A is the diag(K_sq) matrix.
        # We want omega^2 = 1/lambda.
        
        try:
            from scipy.linalg import eigh
            # eigh(b, a) solves b x = lambda a x
            eigvals, eigvecs = eigh(M, np.diag(K_sq))
            # eigvals are lambda = 1/omega^2
            # So omega^2 = 1/eigvals
            # Sort by omega (or 1/eigvals)
            # Smallest omega corresponds to largest eigval.
            # We want bands 4 and 5, which are low frequency.
            # So we want the largest eigvals (smallest 1/eigvals).
            
            # Invert to get omega^2
            omega_sq = 1.0 / eigvals
            
            # Sort by omega_sq (ascending)
            idx_sort = np.argsort(omega_sq)
            omega_sq = omega_sq[idx_sort]
            
            if return_eigvecs:
                eigvecs = eigvecs[:, idx_sort]
                return omega_sq, eigvecs
            return omega_sq
        except Exception:
            return [0] * n_pw, np.zeros((n_pw, n_pw), dtype=complex)

    # --- Symmetry handling ---
    
    def fill_symmetric(unique_mask):
        """
        Fills the full 32x32 cell from a 16x16 unique mask.
        """
        cell = np.zeros((size, size), dtype=int)
        for i in range(16):
            for j in range(16):
                val = unique_mask[i][j]
                coords = [
                    (i, j),
                    (31-i, j),
                    (i, 31-j),
                    (31-i, 31-j),
                    (j, i),
                    (31-j, i),
                    (j, 31-i),
                    (31-j, 31-i)
                ]
                for r, c in coords:
                    cell[r][c] = val
        return cell

    # --- Optimization ---
    
    # Define k-points for the surrogate
    # Irreducible BZ of square lattice: 0 <= kx <= ky <= 0.5
    k_points = [
        (0.0, 0.0),    # Gamma
        (0.5, 0.0),    # X
        (0.5, 0.5),    # M
        (0.25, 0.0),
        (0.25, 0.25),
        (0.4, 0.1),
        (0.1, 0.4), # Note: kx > ky here? No, 0.1 < 0.4. But irreducible is kx<=ky.
                    # The evaluator covers the irreducible zone. We should sample it.
                    # Let's use points in 0<=kx<=ky<=0.5
        (0.3, 0.3),
        (0.2, 0.4),
        (0.1, 0.3),
        (0.4, 0.4),
        (0.3, 0.4),
        (0.2, 0.3),
    ]
    
    # Initialize with a structured pattern (circular hole in dielectric)
    # Let's start with a dielectric background and a hole, or air background and dielectric dot.
    # Given eps_high=12, it's better to have a dielectric structure in air or air holes in dielectric.
    # Usually, air holes in dielectric are easier to optimize for large gaps in high contrast.
    # Let's start with a dielectric square with a circular hole.
    unique_mask = np.ones((16, 16), dtype=int)
    center = 7.5
    radius = 4
    for i in range(16):
        for j in range(16):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if dist < radius:
                unique_mask[i][j] = 0  # Air hole
    
    cell = fill_symmetric(unique_mask)
    
    def evaluate_gap(cell):
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        for kx, ky in k_points:
            omega_sq, _ = solve_band_structure(cell, kx, ky, return_eigvecs=False)
            if len(omega_sq) < 5:
                continue
            w4 = np.sqrt(max(0, omega_sq[3]))
            w5 = np.sqrt(max(0, omega_sq[4]))
            max_omega_4 = max(max_omega_4, w4)
            min_omega_5 = min(min_omega_5, w5)
        
        if min_omega_5 == float('inf'):
            return 0.0, max_omega_4, min_omega_5
        gap = min_omega_5 - max_omega_4
        return gap, max_omega_4, min_omega_5

    current_gap, _, _ = evaluate_gap(cell)
    best_gap = current_gap
    best_cell = cell.copy()
    current_cell = cell.copy()

    # Topological optimization loop
    # We will iterate over unique pixels, try flipping them, and accept if gap improves.
    # To speed up, we can estimate the gradient using field intensities.
    
    # Compute field intensities for the current cell to get a gradient map
    # This is expensive, so we do it every few iterations or just use random search with gradient hints.
    
    def compute_gradient(cell):
        """
        Computes a gradient map for each pixel indicating whether to add or remove dielectric.
        """
        grad = np.zeros((size, size))
        # Sample a few k-points for gradient estimation
        grad_k_points = k_points[:5] # Use fewer for speed
        
        for kx, ky in grad_k_points:
            omega_sq, eigvecs = solve_band_structure(cell, kx, ky, return_eigvecs=True)
            if len(omega_sq) < 5:
                continue
            w4 = np.sqrt(omega_sq[3])
            w5 = np.sqrt(omega_sq[4])
            if w4 == 0 or w5 == 0:
                continue
            
            vec4 = eigvecs[:, 3]
            vec5 = eigvecs[:, 4]
            
            # Compute field intensities at pixels
            # We need to evaluate the field at the pixel centers.
            # r = (px+0.5)/size, (py+0.5)/size
            for px in range(size):
                for py in range(size):
                    rx = (px + 0.5) / size
                    ry = (py + 0.5) / size
                    
                    f4 = 0.0
                    f5 = 0.0
                    for i in range(n_pw):
                        gx_i, gy_i = G_list[i]
                        Kx = kx + gx_i
                        Ky = ky + gy_i
                        phase = 2 * np.pi * (Kx * rx + Ky * ry)
                        f4 += vec4[i] * np.exp(1j * phase)
                        f5 += vec5[i] * np.exp(1j * phase)
                    
                    # Gradient contribution to gap:
                    # dGap/dEps ~ |f5|^2/w5 - |f4|^2/w4
                    # If positive, increasing Eps increases Gap.
                    # If negative, decreasing Eps increases Gap.
                    grad[px, py] += (np.abs(f5)**2 / w5 - np.abs(f4)**2 / w4)
        
        return grad

    # Optimization Loop
    for iteration in range(500):
        if time.time() - start_time > time_limit:
            break
            
        # Compute gradient every 10 iterations
        if iteration % 10 == 0:
            grad = compute_gradient(current_cell)
        
        # Find the best pixel to flip
        # We prioritize pixels where the gradient suggests a large change in gap.
        # If grad > 0, we want to increase Eps (set to 1).
        # If grad < 0, we want to decrease Eps (set to 0).
        
        best_move = None
        best_improvement = -1e9
        
        # Sample a subset of unique pixels to speed up search
        # 16x16 = 256 pixels. We can check all of them.
        for i in range(16):
            for j in range(16):
                # Get the average gradient for this symmetric pixel
                # The pixel (i,j) in unique mask corresponds to multiple pixels in full cell.
                # We use the gradient at the representative pixel (i,j) or average them.
                # Since the field is symmetric, the gradient should be symmetric.
                g_val = grad[i, j]
                
                current_val = unique_mask[i, j]
                
                # Determine desired value
                if g_val > 0.01:
                    desired_val = 1
                elif g_val < -0.01:
                    desired_val = 0
                else:
                    continue # No strong signal
                
                if desired_val == current_val:
                    continue # Already in desired state
                
                # Evaluate the move
                unique_mask[i, j] = desired_val
                new_cell = fill_symmetric(unique_mask)
                new_gap, _, _ = evaluate_gap(new_cell)
                
                improvement = new_gap - current_gap
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_move = (i, j, desired_val)
                
                # Revert for next candidate
                unique_mask[i, j] = current_val
                
        if best_move is not None and best_improvement > 0:
            i, j, val = best_move
            unique_mask[i, j] = val
            current_cell = fill_symmetric(unique_mask)
            current_gap = best_improvement + current_gap
            
            if current_gap > best_gap:
                best_gap = current_gap
                best_cell = current_cell.copy()
        else:
            # Random move to escape local minima
            if iteration % 20 == 19:
                ri = random.randint(0, 15)
                rj = random.randint(0, 15)
                old_val = unique_mask[ri, rj]
                unique_mask[ri, rj] = 1 - old_val
                new_cell = fill_symmetric(unique_mask)
                new_gap, _, _ = evaluate_gap(new_cell)
                
                # Simple acceptance: accept if better, or with small prob if worse
                if new_gap > current_gap:
                    current_cell = new_cell
                    current_gap = new_gap
                    if current_gap > best_gap:
                        best_gap = current_gap
                        best_cell = current_cell.copy()
                else:
                    unique_mask[ri, rj] = old_val
                    current_cell = fill_symmetric(unique_mask)

    return best_cell.tolist()
