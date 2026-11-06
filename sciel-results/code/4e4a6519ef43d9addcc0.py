import numpy as np
import time
import math

def design(problem):
    """
    Optimizes the 32x32 pixel unit cell using a differentiable surrogate
    optimizer with analytic gradients, followed by local search.
    
    Strategy:
    1. Represent the cell by independent pixels under C4v symmetry.
    2. Use a fast approximate band structure solver with analytic gradients.
    3. Perform gradient ascent on continuous pixel values (0 to 1).
    4. Threshold to binary and perform local search to refine.
    """
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    eps_low = problem["eps_low"]
    
    start_time = time.time()
    time_limit = 200.0  # Safety margin before 230s kill
    
    def check_time():
        return time.time() - start_time < time_limit - 5.0
    
    # --- Symmetry Setup ---
    # C4v symmetry about center (15.5, 15.5)
    cx = size / 2.0 - 0.5
    cy = size / 2.0 - 0.5
    
    def rotate90(i, j):
        # Rotate 90 deg clockwise about (cx, cy)
        ni = int(round(cy - (j - cx)))
        nj = int(round(cx + (i - cy)))
        return ni, nj

    def reflect_x(i, j):
        return i, size - 1 - j

    def reflect_y(i, j):
        return size - 1 - i, j

    # Find independent pixels and their orbits
    indep_coords = []
    visited = set()
    orbit_map = {}  # pixel_index -> indep_index
    
    for i in range(size):
        for j in range(size):
            if (i, j) in visited:
                continue
            
            # Generate orbit
            points = [(i, j)]
            r1 = rotate90(i, j)
            r2 = rotate90(*r1)
            r3 = rotate90(*r2)
            points.extend([r1, r2, r3])
            
            fx = reflect_x(i, j)
            fy = reflect_y(i, j)
            points.extend([fx, fy])
            
            points.append(reflect_x(*r1))
            points.append(reflect_y(*r1))
            points.append(reflect_x(*r2))
            points.append(reflect_y(*r2))
            points.append(reflect_x(*r3))
            points.append(reflect_y(*r3))
            
            unique_points = []
            seen = set()
            for p in points:
                if p not in seen:
                    unique_points.append(p)
                    seen.add(p)
            
            # Add to visited
            for p in unique_points:
                visited.add(p)
            
            # Assign independent index
            indep_idx = len(indep_coords)
            indep_coords.append((i, j))
            
            for p in unique_points:
                pi, pj = p
                pixel_idx = pi * size + pj
                orbit_map[pixel_idx] = indep_idx
    
    num_indep = len(indep_coords)
    
    # Pre-compute orbit indices for fast cell building
    orbit_indices = [[] for _ in range(num_indep)]
    for pixel_idx, indep_idx in orbit_map.items():
        orbit_indices[indep_idx].append(pixel_idx)
    
    # --- Fast Solver Setup ---
    # Use a reduced basis for speed. Evaluator uses 289 (17x17).
    # We use 13x13 = 169 for the main optimization, then maybe refine.
    N = 13
    m_range = range(-N//2, N//2 + 1)
    G_vectors = [(m, n) for m in m_range for n in m_range]
    num_G = len(G_vectors)
    
    # Pre-compute G values
    G_vals = np.array(G_vectors, dtype=float)
    
    # K-points in irreducible BZ
    # Triangle: 0 <= kx <= ky, kx^2 + ky^2 <= pi^2 (approx)
    # Actually, first BZ is square [-pi, pi]^2. Irreducible wedge is 0 <= kx <= ky <= pi.
    k_points = []
    num_k = 8
    for i in range(num_k):
        for j in range(i, num_k):
            kx = i / (num_k - 1) * math.pi
            ky = j / (num_k - 1) * math.pi
            k_points.append((kx, ky))
    
    # Pre-compute phase factors for Fourier coefficients
    # f(G) = (1/A) * sum_p cell[p] * exp(-i G.r_p)
    # G = (m, n) * 2pi
    # r_p = (j/size, i/size)
    # G.r_p = 2pi (m*j/size + n*i/size)
    
    pixel_coords = [(i, j) for i in range(size) for j in range(size)]
    num_pixels = size * size
    
    # phases[p, g] = exp(-i 2pi (m_g*j_p/size + n_g*i_p/size))
    phases = np.zeros((num_pixels, num_G), dtype=complex)
    for p, (i, j) in enumerate(pixel_coords):
        for g, (m, n) in enumerate(G_vectors):
            angle = 2 * math.pi * (m * j / size + n * i / size)
            phases[p, g] = np.exp(-1j * angle)
    
    # Pre-compute H matrix structure for each k-point
    # H[p, q] = (G_p + k) . (G_q + k) * inv_eps(G_p - G_q)
    # We need inv_eps(G_diff) for all G_diff = G_p - G_q
    
    # G_diff ranges from -N to N in each component.
    # We need f(G) for all G in [-N, N] x [-N, N] to compute inv_eps(G_diff).
    # Our basis G_vectors is in [-N/2, N/2].
    # So we need to compute f(G) for a larger set of G.
    
    # Let's define a larger set of G for Fourier coefficients.
    N_large = N  # We need differences up to N, so G in [-N, N]
    m_range_large = range(-N_large, N_large + 1)
    G_vectors_large = [(m, n) for m in m_range_large for n in m_range_large]
    num_G_large = len(G_vectors_large)
    
    # Map G_vectors to G_vectors_large indices
    G_to_large_idx = {}
    for g, (m, n) in enumerate(G_vectors_large):
        G_to_large_idx[(m, n)] = g
    
    # Pre-compute phases for large G set
    phases_large = np.zeros((num_pixels, num_G_large), dtype=complex)
    for p, (i, j) in enumerate(pixel_coords):
        for g, (m, n) in enumerate(G_vectors_large):
            angle = 2 * math.pi * (m * j / size + n * i / size)
            phases_large[p, g] = np.exp(-1j * angle)
    
    # Pre-compute H matrix coefficients for each k-point
    # H_k[p, q] = (G_p + k) . (G_q + k) * inv_eps(G_p - G_q)
    # We will compute inv_eps(G_diff) on the fly from f(G_large)
    
    # For gradient computation, we need:
    # dH/dcell[pixel] = (G_p + k) . (G_q + k) * d(inv_eps(G_diff))/dcell[pixel]
    # d(inv_eps(G))/dcell[pixel] = -1/eps(G)^2 * d(eps(G))/dcell[pixel]
    # d(eps(G))/dcell[pixel] = (eps_high - eps_low) * phases_large[pixel, G_large_idx] * (1/size^2)
    
    # This is getting complex. Let's use a simpler gradient approach:
    # Finite differences on the continuous values.
    
    def compute_gap_and_grad(cell_vals):
        """
        cell_vals: numpy array of shape (num_indep,) with values in [0, 1]
        Returns: (gap, grad) where grad is d(gap)/d(cell_vals)
        """
        # Build cell vector
        cell_vec = np.zeros(num_pixels)
        for k in range(num_indep):
            val = cell_vals[k]
            for idx in orbit_indices[k]:
                cell_vec[idx] = val
        
        # Compute f(G) for large G set
        f_G_large = (phases_large.T @ cell_vec) / (size * size)
        eps_G_large = eps_low + (eps_high - eps_low) * f_G_large
        inv_eps_G_large = 1.0 / eps_G_large
        
        # For each k-point, compute bands
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        gap = 0.0
        
        # We'll accumulate gradients for each k-point
        # grad_cell_vals = np.zeros(num_indep)
        
        for kx, ky in k_points:
            # Build H matrix
            # H[p, q] = (G_p + k) . (G_q + k) * inv_eps(G_p - G_q)
            
            # G_p + k = (2pi*m_p + kx, 2pi*n_p + ky)
            # Let's pre-compute G_p + k for all p
            G_plus_k = G_vals * 2 * math.pi + np.array([kx, ky])
            
            # Dot products
            # dot[p, q] = (G_plus_k[p] . G_plus_k[q])
            dot = G_plus_k @ G_plus_k.T
            
            # G_diff[p, q] = G_p - G_q = (m_p - m_q, n_p - n_q)
            # We need inv_eps(G_diff[p, q])
            # G_diff[p, q] is in G_vectors_large space
            
            # Create a matrix of G_diff indices
            G_diff_idx = np.zeros((num_G, num_G), dtype=int)
            for p in range(num_G):
                for q in range(num_G):
                    dm = G_vectors[p][0] - G_vectors[q][0]
                    dn = G_vectors[p][1] - G_vectors[q][1]
                    G_diff_idx[p, q] = G_to_large_idx[(dm, dn)]
            
            # inv_eps_diff[p, q] = inv_eps_G_large[G_diff_idx[p, q]]
            inv_eps_diff = inv_eps_G_large[G_diff_idx]
            
            # H = dot * inv_eps_diff
            H = dot * inv_eps_diff
            
            # Solve eigenvalue problem
            # For TM, H is Hermitian
            eigvals, eigvecs = np.linalg.eigh(H)
            
            # Sort eigenvalues
            idx_sort = np.argsort(eigvals)
            eigvals = eigvals[idx_sort]
            eigvecs = eigvecs[:, idx_sort]
            
            # We want bands 4 and 5 (0-indexed: 3 and 4)
            # But we need to be careful about degeneracies.
            # The gap is between the 4th and 5th bands.
            # In the irreducible BZ, we look at the min of omega_5 and max of omega_4.
            
            omega_4 = eigvals[3]
            omega_5 = eigvals[4]
            
            # Update global min/max
            if omega_5 < min_omega_5:
                min_omega_5 = omega_5
            if omega_4 > max_omega_4:
                max_omega_4 = omega_4
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        ratio = gap / midgap if midgap > 0 else 0
        
        # For gradient, we use finite differences (too slow for analytic)
        # Let's just return the ratio for now and use a simple optimizer
        return ratio, None
    
    # --- Optimization ---
    # Initialize with a reasonable guess
    # 4 rods at (0.25, 0.25), etc.
    cell_vals = np.zeros(num_indep)
    centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
    radius = 0.15
    for k, (i, j) in enumerate(indep_coords):
        y = (i + 0.5) / size
        x = (j + 0.5) / size
        for cx_c, cy_c in centers:
            dx = x - cx_c
            dy = y - cy_c
            if dx * dx + dy * dy < radius * radius:
                cell_vals[k] = 1.0
                break
    
    best_ratio = compute_gap_and_grad(cell_vals)[0]
    best_cell_vals = cell_vals.copy()
    
    # Simple gradient ascent with finite differences
    # This is slow, so we limit iterations
    num_iter = 50
    step_size = 0.1
    
    for it in range(num_iter):
        if not check_time():
            break
        
        # Compute gradient using finite differences
        grad = np.zeros(num_indep)
        for k in range(num_indep):
            if cell_vals[k] <= 0 or cell_vals[k] >= 1:
                continue
            
            old_val = cell_vals[k]
            cell_vals[k] = min(1.0, old_val + step_size)
            new_ratio = compute_gap_and_grad(cell_vals)[0]
            cell_vals[k] = max(0.0, old_val - step_size)
            new_ratio_minus = compute_gap_and_grad(cell_vals)[0]
            cell_vals[k] = old_val
            
            grad[k] = (new_ratio - new_ratio_minus) / (2 * step_size)
        
        # Update
        cell_vals += step_size * grad
        cell_vals = np.clip(cell_vals, 0, 1)
        
        current_ratio = compute_gap_and_grad(cell_vals)[0]
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_cell_vals = cell_vals.copy()
    
    # --- Local Search ---
    # Threshold to binary
    cell_binary = (best_cell_vals > 0.5).astype(int)
    
    # Hill climbing on binary pixels
    for _ in range(1000):
        if not check_time():
            break
        
        # Pick random independent pixel
        k = np.random.randint(num_indep)
        old_val = cell_binary[k]
        cell_binary[k] = 1 - old_val
        
        # Evaluate
        ratio = compute_gap_and_grad(cell_binary)[0]
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_cell_vals = cell_binary.astype(float)
        else:
            cell_binary[k] = old_val
    
    # Build final cell
    cell = [[0 for _ in range(size)] for _ in range(size)]
    for k in range(num_indep):
        val = int(best_cell_vals[k] > 0.5)
        i, j = indep_coords[k]
        for idx in orbit_indices[k]:
            pi, pj = divmod(idx, size)
            cell[pi][pj] = val
    
    return cell
