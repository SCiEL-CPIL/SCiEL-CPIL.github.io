import numpy as np
import time

def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell to maximize
    the TM band gap between bands 4 and 5.
    """
    start_time = time.time()
    deadline = start_time + 200.0
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Precompute G vectors for the plane-wave expansion
    N_pw = 8
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)
    
    # Define k-points in the irreducible Brillouin zone
    # The irreducible BZ for a square lattice is 1/8 of the full BZ
    # We use a grid of k-points in the triangle (0,0)-(0.5,0)-(0.5,0.5)
    k_points = []
    # Along Gamma-M: (t, 0) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 6):
        k_points.append((t, 0))
    # Along M-X: (0.5, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 6):
        k_points.append((0.5, t))
    # Along Gamma-X: (t, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 6):
        k_points.append((t, t))
    # Remove duplicates
    unique_k = []
    seen = set()
    for k in k_points:
        key = (round(k[0], 6), round(k[1], 6))
        if key not in seen:
            seen.add(key)
            unique_k.append(k)
    
    def compute_fourier_coeffs(eps_grid):
        """
        Compute Fourier coefficients of 1/epsilon for the TM mode.
        Returns a dictionary mapping (dx, dy) to the coefficient.
        """
        inv_eps = 1.0 / eps_grid
        fft_inv_eps = np.fft.fft2(inv_eps)
        fft_inv_eps /= (size * size)
        
        coeff_dict = {}
        for dx in range(-2 * N_pw, 2 * N_pw + 1):
            for dy in range(-2 * N_pw, 2 * N_pw + 1):
                ix = dx % size
                iy = dy % size
                coeff_dict[(dx, dy)] = fft_inv_eps[ix, iy]
        return coeff_dict
    
    def evaluate_band_gap(eps_grid):
        """
        Compute the TM band gap between bands 4 and 5.
        Returns the gap-midgap ratio.
        """
        coeff_dict = compute_fourier_coeffs(eps_grid)
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in unique_k:
            k = np.array([kx, ky])
            
            # Build the Hamiltonian matrix
            M = np.zeros((num_pw, num_pw), dtype=complex)
            for i in range(num_pw):
                Gi = G_vectors[i]
                k_plus_Gi = k + Gi
                omega_sq_i = float(np.dot(k_plus_Gi, k_plus_Gi))
                
                for j in range(num_pw):
                    Gj = G_vectors[j]
                    diff = tuple((Gi - Gj).astype(int))
                    coeff = coeff_dict.get(diff, 0)
                    M[i, j] = coeff * omega_sq_i
            
            # Solve eigenvalue problem
            eigvals = np.linalg.eigvals(M)
            real_eigvals = np.sort(np.real(eigvals))
            
            if len(real_eigvals) >= 5 and real_eigvals[3] > 0 and real_eigvals[4] > 0:
                omega_4 = np.sqrt(real_eigvals[3])
                omega_5 = np.sqrt(real_eigvals[4])
                
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap == 0:
            return -1.0
        ratio = 2 * gap / midgap
        return float(ratio)
    
    def create_cell_from_params(params):
        """
        Create a binary 32x32 cell from a list of symmetry-respecting parameters.
        params[i] corresponds to a specific pattern of pixels.
        """
        cell = np.zeros((size, size), dtype=int)
        
        # Parameter 0: center dot (1 pixel at center)
        if params[0] > 0.5:
            cell[size//2, size//2] = 1
        
        # Parameter 1: small ring (4 pixels around center)
        if params[1] > 0.5:
            cell[size//2, size//2 - 1] = 1
            cell[size//2, size//2 + 1] = 1
            cell[size//2 - 1, size//2] = 1
            cell[size//2 + 1, size//2] = 1
        
        # Parameter 2: cross shape
        if params[2] > 0.5:
            for d in range(1, size//2):
                cell[size//2, size//2 - d] = 1
                cell[size//2, size//2 + d] = 1
                cell[size//2 - d, size//2] = 1
                cell[size//2 + d, size//2] = 1
        
        # Parameter 3: large ring
        if params[3] > 0.5:
            r = size // 3
            for i in range(size):
                for j in range(size):
                    dist = np.sqrt((i - size/2 + 0.5)**2 + (j - size/2 + 0.5)**2)
                    if abs(dist - r) < 0.7:
                        cell[i, j] = 1
        
        # Parameter 4: checkerboard-like pattern
        if params[4] > 0.5:
            for i in range(size):
                for j in range(size):
                    if (i + j) % 4 == 0 or (i + j) % 4 == 3:
                        cell[i, j] = 1
        
        # Parameter 5: diagonal stripes
        if params[5] > 0.5:
            for i in range(size):
                for j in range(size):
                    if (i - j) % 8 in [0, 1, 7]:
                        cell[i, j] = 1
        
        # Parameter 6: anti-diagonal stripes
        if params[6] > 0.5:
            for i in range(size):
                for j in range(size):
                    if (i + j) % 8 in [0, 1, 7]:
                        cell[i, j] = 1
        
        # Parameter 7: square ring
        if params[7] > 0.5:
            r = size // 4
            for i in range(size):
                for j in range(size):
                    dist = max(abs(i - size/2 + 0.5), abs(j - size/2 + 0.5))
                    if abs(dist - r) < 0.7:
                        cell[i, j] = 1
        
        # Parameter 8: dense center
        if params[8] > 0.5:
            for i in range(size):
                for j in range(size):
                    dist = np.sqrt((i - size/2 + 0.5)**2 + (j - size/2 + 0.5)**2)
                    if dist < size / 3:
                        cell[i, j] = 1
        
        # Parameter 9: sparse outer
        if params[9] > 0.5:
            for i in range(size):
                for j in range(size):
                    dist = np.sqrt((i - size/2 + 0.5)**2 + (j - size/2 + 0.5)**2)
                    if dist > size / 2.5:
                        cell[i, j] = 1
        
        # Enforce symmetry
        cell = enforce_symmetry(cell)
        
        return cell.tolist()
    
    def enforce_symmetry(cell):
        """
        Enforce C4v symmetry: cell == cell[::-1] row-wise, column-wise, and transpose.
        """
        cell = np.array(cell, dtype=int)
        # Symmetry about center: cell[i,j] == cell[size-1-i, size-1-j]
        cell = np.maximum(cell, cell[::-1, ::-1])
        # Symmetry: cell[i,j] == cell[j,i]
        cell = np.maximum(cell, cell.T)
        # Symmetry: cell[i,j] == cell[size-1-i, j]
        cell = np.maximum(cell, cell[::-1, :])
        # Symmetry: cell[i,j] == cell[i, size-1-j]
        cell = np.maximum(cell, cell[:, ::-1])
        
        # Average to get binary
        cell = (cell > 0).astype(int)
        
        # Final check and fix
        for i in range(size):
            for j in range(size):
                val = cell[i, j]
                # Enforce all symmetries
                symmetric_positions = [
                    (i, j),
                    (size - 1 - i, j),
                    (i, size - 1 - j),
                    (size - 1 - i, size - 1 - j),
                    (j, i),
                    (size - 1 - j, i),
                    (j, size - 1 - i),
                    (size - 1 - j, size - 1 - i),
                ]
                for si, sj in symmetric_positions:
                    cell[si, sj] = val
        
        return cell
    
    def params_to_epsilon(params):
        """Convert binary params to epsilon grid."""
        cell = create_cell_from_params(params)
        cell = np.array(cell)
        eps_grid = np.where(cell == 1, eps_high, eps_low)
        return eps_grid
    
    def evaluate_params(params):
        """Evaluate a set of parameters."""
        eps_grid = params_to_epsilon(params)
        return evaluate_band_gap(eps_grid)
    
    # Initial candidate: try several heuristic patterns
    candidates = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Center dot
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Small ring
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Cross
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Large ring
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Checkerboard
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Diagonal stripes
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Anti-diagonal stripes
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # Square ring
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Dense center
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Sparse outer
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    ]
    
    best_params = None
    best_score = -np.inf
    
    # Evaluate initial candidates
    for params in candidates:
        if time.time() > deadline - 30:
            break
        score = evaluate_params(params)
        if score > best_score:
            best_score = score
            best_params = params[:]
    
    # If no good candidate found, use a default
    if best_params is None:
        best_params = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        best_score = evaluate_params(best_params)
    
    # Local search: mutate parameters
    num_iterations = 200
    for i in range(num_iterations):
        if time.time() > deadline - 20:
            break
        
        # Create a mutated copy
        current_params = best_params[:]
        
        # Flip 1-2 random bits
        num_flips = np.random.randint(1, 3)
        for _ in range(num_flips):
            idx = np.random.randint(len(current_params))
            current_params[idx] = 1 - current_params[idx]
        
        # Evaluate
        score = evaluate_params(current_params)
        
        if score > best_score:
            best_score = score
            best_params = current_params[:]
    
    # Return the best cell
    cell = create_cell_from_params(best_params)
    
    # Final symmetry check
    cell_arr = np.array(cell)
    cell_arr = enforce_symmetry(cell_arr)
    
    return cell_arr.tolist()
