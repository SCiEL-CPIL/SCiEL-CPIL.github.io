import numpy as np
import time
import math

def design(problem):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    size = problem['pixels']
    eps_high = problem['eps_high']
    eps_low = problem['eps_low']
    
    # 1. Define G vectors (289 planes waves)
    # N=289 = 17^2. So G_x, G_y in [-8, 8] * 2*pi?
    # Lattice constant a=1. G = (2*pi/h) * (n_x, n_y).
    # Usually N is odd. 17x17 grid of G vectors.
    # G_x = 2*pi * n_x, G_y = 2*pi * n_y for n in -8..8.
    n_max = 8
    G_vecs = []
    for nx in range(-n_max, n_max+1):
        for ny in range(-n_max, n_max+1):
            G_vecs.append((2*np.pi*nx, 2*np.pi*ny))
    G_vecs = np.array(G_vecs)
    N_pw = len(G_vecs)
    
    # 2. Define K points
    k_points = [
        np.array([0, 0]),
        np.array([np.pi, 0]),
        np.array([np.pi/2, np.pi/2]),
        np.array([np.pi/4, 0]),
        np.array([np.pi/2, np.pi/4])
    ]
    
    # 3. Precompute pixel centers and areas
    pixel_centers = []
    pixel_areas = []
    for i in range(size):
        for j in range(size):
            cx = (i + 0.5) / size
            cy = (j + 0.5) / size
            pixel_centers.append((cx, cy))
            pixel_areas.append(1.0 / (size * size))
    pixel_centers = np.array(pixel_centers)
    
    # 4. Helper to compute eta(G) (Fourier coeff of 1/eps)
    def compute_eta_fast(cell_1d):
        # cell_1d: flat array of 0/1
        # 1 -> dielectric (1/12), 0 -> air (1)
        inv_eps = np.where(cell_1d == 1, 1.0/eps_high, 1.0)
        # eta(G) = sum_p inv_eps_p * Area * exp(-i G . r_p)
        # Vectorized:
        # G_vecs shape (N_pw, 2), pixel_centers shape (N_px, 2)
        # Dot product: (N_pw, N_px)
        dots = np.dot(G_vecs, pixel_centers.T)
        phases = np.exp(-1j * dots)
        # Weighted sum
        weighted = phases * inv_eps[None, :] * pixel_areas[None, :]
        eta = np.sum(weighted, axis=1)
        return eta

    def compute_eta_exact(cell_1d):
        # Exact integration over pixels
        # I(G) = int int exp(-i(Gx x + Gy y)) dx dy
        # = exp(-i(Gx x1 + Gy y1)) * (1 - exp(i Gx dx)) / (i Gx) * (1 - exp(i Gy dy)) / (i Gy)
        # Handle Gx=0 or Gy=0
        eta = np.zeros(N_pw, dtype=complex)
        for idx, (cx, cy) in enumerate(pixel_centers):
            val = 1.0/eps_high if cell_1d[idx] == 1 else 1.0
            if val == 0: continue
            for g_idx, (gx, gy) in enumerate(G_vecs):
                # Integration limits: cx +/- 0.5/size
                x0 = cx - 0.5/size
                x1 = cx + 0.5/size
                y0 = cy - 0.5/size
                y1 = cy + 0.5/size
                
                # Analytic integral
                if abs(gx) < 1e-9:
                    ix = x1 - x0
                else:
                    ix = (np.exp(-1j*gx*x0) - np.exp(-1j*gx*x1)) / (1j*gx)
                if abs(gy) < 1e-9:
                    iy = y1 - y0
                else:
                    iy = (np.exp(-1j*gy*y0) - np.exp(-1j*gy*y1)) / (1j*gy)
                
                eta[g_idx] += val * ix * iy
        # Note: The fast version used Area * exp(-i G rc). 
        # The exact version sums the integrals.
        # The fast version is an approximation.
        return eta

    def calc_gap(eta, use_fast=False):
        min_w5 = np.inf
        max_w4 = -np.inf
        for k in k_points:
            # Build matrix
            # M_GG' = (k+G) dot (k+G') * eta(G-G') / sqrt((k+G)^2 (k+G')^2)
            # k+G: (N_pw, 2)
            KG = G_vecs + k
            KG_sq = np.sum(KG**2, axis=1) # (N_pw,)
            
            # eta_diff: eta(G_i - G_j)
            # G_i - G_j = G_i + (-G_j)
            # We have eta for all G in G_vecs.
            # We need a lookup or matrix.
            # Since G_vecs is a grid, we can map index to G.
            # Or just compute the matrix directly using the fact that eta is defined on the grid.
            # eta_matrix[i,j] = eta(G_i - G_j)
            # G_i - G_j = (2pi(n_i - n_j), 2pi(m_i - m_j))
            
            # Let's create a 2D array for eta for faster lookup
            # n_max=8, size=17
            eta_grid = np.zeros((17, 17), dtype=complex)
            for idx, (gx, gy) in enumerate(G_vecs):
                nx = int(gx / (2*np.pi))
                ny = int(gy / (2*np.pi))
                eta_grid[nx + 8, ny + 8] = eta[idx]
            
            # Build M
            # M[i,j] = KG[i] dot KG[j] * eta(G_i - G_j) / (sqrt(KG_sq[i]) * sqrt(KG_sq[j]))
            # KG dot KG: (N, 1) x (1, N)
            KG_dot = np.outer(KG, KG) # (N, N)
            KG_dot_sum = np.sum(KG_dot, axis=2) # (N, N)
            
            # eta_diff matrix
            # G_i = (2pi n_i, 2pi m_i), G_j = (2pi n_j, 2pi m_j)
            # G_i - G_j = (2pi (n_i-n_j), 2pi (m_i-m_j))
            # We need eta at these differences.
            # The differences range from -16 to 16.
            # Our eta_grid only covers -8 to 8.
            # Wait, G_vecs only has n in -8..8.
            # G_i - G_j can be -16..16.
            # We need eta(G) for G up to 16?
            # The PW method truncates at N. The convolution sum is over G' in the set.
            # The term eta(G_i - G_j) requires the Fourier coefficient at G_i - G_j.
            # If G_i - G_j is outside the calculated set, we need it.
            # We calculated eta for G in -8..8.
            # We need eta for -16..16.
            # So we should calculate eta for a larger range, or accept that high G differences are 0 (approximation).
            # Given N=289 (17x17), the differences go up to 16.
            # Let's calculate eta for n in -16..16 to be safe?
            # That's 33x33 = 1089 coefficients.
            # This increases cost.
            # Alternatively, assume eta(G) = 0 for |G| > 8. This is a common truncation error.
            # Let's stick to the available eta and set to 0 if index out of bounds.
            
            eta_diff = np.zeros((N_pw, N_pw), dtype=complex)
            for i in range(N_pw):
                for j in range(N_pw):
                    diff_nx = int(G_vecs[i][0] / (2*np.pi)) - int(G_vecs[j][0] / (2*np.pi))
                    diff_ny = int(G_vecs[i][1] / (2*np.pi)) - int(G_vecs[j][1] / (2*np.pi))
                    idx_nx = diff_nx + 8
                    idx_ny = diff_ny + 8
                    if 0 <= idx_nx < 17 and 0 <= idx_ny < 17:
                        eta_diff[i, j] = eta_grid[idx_nx, idx_ny]
                    else:
                        eta_diff[i, j] = 0.0
            
            M = KG_dot_sum * eta_diff / np.outer(np.sqrt(KG_sq), np.sqrt(KG_sq))
            M = (M + M.conj().T) / 2 # Ensure Hermitian
            
            eigvals = np.linalg.eigvalsh(M)
            # Sort
            eigvals = np.sort(eigvals.real)
            # We need 4th and 5th.
            # Note: The lowest eigenvalues might be near 0?
            # For TM, the lowest band is usually non-zero.
            # We take index 3 and 4 (0-based).
            w4 = np.sqrt(eigvals[3])
            w5 = np.sqrt(eigvals[4])
            
            if w4 > max_w4: max_w4 = w4
            if w5 < min_w5: min_w5 = w5
            
        gap = 2 * (min_w5 - max_w4) / (min_w5 + max_w4)
        return gap

    # 5. Initial Design
    # Try a few static shapes and pick the best
    def make_cell(shape_type, params):
        cell = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if shape_type == 'rod':
                    r = np.sqrt(x*x + y*y)
                    if r < params[0]: cell[i,j] = 1
                elif shape_type == 'square':
                    if abs(x) < params[0] and abs(y) < params[0]: cell[i,j] = 1
                elif shape_type == 'superellipse':
                    n = params[1]
                    r = abs(x)**n + abs(y)**n
                    if r < params[0]**n: cell[i,j] = 1
                elif shape_type == 'annulus':
                    r = np.sqrt(x*x + y*y)
                    if params[0] < r < params[1]: cell[i,j] = 1
        return cell

    candidates = [
        make_cell('rod', [0.3]),
        make_cell('rod', [0.35]),
        make_cell('square', [0.3]),
        make_cell('superellipse', [0.35, 4]),
        make_cell('superellipse', [0.4, 6]),
        make_cell('annulus', [0.1, 0.35])
    ]
    
    best_cell = None
    best_score = -np.inf
    
    # Evaluate initial candidates with Fast Solver
    for c in candidates:
        c_flat = c.flatten()
        eta = compute_eta_fast(c_flat)
        score = calc_gap(eta)
        if score > best_score:
            best_score = score
            best_cell = c.copy()
            
    # 6. Local Search (Simulated Annealing)
    # Use Fast Solver
    current_cell = best_cell.copy()
    current_score = best_score
    T = 0.05
    cooling = 0.999
    
    # Symmetry: Only modify the 16x16 quadrant
    # Indices: i in 0..15, j in 0..15
    # Map to full cell:
    # (i, j) -> (i, j), (i, 31-j), (31-i, j), (31-i, 31-j)
    # Also transpose symmetry: (i, j) == (j, i)
    # So we only need to consider i <= j?
    # The group is D4.
    # Orbits:
    # Center (15.5, 15.5) - not a pixel.
    # Pixels on axes: (i, 15.5) no, pixels are integer.
    # The cell is 32x32. Center is between 15 and 16.
    # Symmetry axes are x=15.5, y=15.5.
    # So (i, j) is equivalent to (31-i, j), (i, 31-j), (31-i, 31-j).
    # Also (i, j) equivalent to (j, i).
    # Independent pixels:
    # i in 0..15, j in 0..15.
    # But due to transpose, we only need i <= j?
    # If i < j, then (i, j) and (j, i) are distinct in the 16x16 block?
    # No, the block is top-left.
    # (i, j) in top-left.
    # (j, i) is also in top-left if j <= 15 and i <= 15.
    # So we can restrict to i <= j in the 16x16 block.
    
    independent_indices = []
    for i in range(16):
        for j in range(16):
            if i <= j:
                independent_indices.append((i, j))
                
    def apply_symmetry(cell, i, j, val):
        # Set pixel and its symmetric counterparts
        cell[i, j] = val
        cell[31-i, j] = val
        cell[i, 31-j] = val
        cell[31-i, 31-j] = val
        if i != j:
            cell[j, i] = val
            cell[31-j, i] = val
            cell[j, 31-i] = val
            cell[31-j, 31-i] = val
        return cell

    # SA Loop
    while time.time() < deadline:
        T *= cooling
        idx = np.random.choice(len(independent_indices))
        i, j = independent_indices[idx]
        
        new_val = 1 - current_cell[i, j]
        
        # Apply
        old_val = current_cell[i, j]
        apply_symmetry(current_cell, i, j, new_val)
        
        # Evaluate
        c_flat = current_cell.flatten()
        eta = compute_eta_fast(c_flat)
        new_score = calc_gap(eta)
        
        # Accept
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_cell = current_cell.copy()
        else:
            delta = new_score - current_score
            if np.random.rand() < np.exp(delta / T):
                current_score = new_score
            else:
                # Revert
                apply_symmetry(current_cell, i, j, old_val)
                
    return best_cell.tolist()
