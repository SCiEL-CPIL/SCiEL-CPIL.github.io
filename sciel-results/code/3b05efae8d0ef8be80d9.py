"""Design a 32x32 photonic crystal unit cell to maximize TM band 4-5 gap."""
import numpy as np
import time
import math

def design(problem):
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]   # 1
    eps_high = problem["eps_high"] # 12
    
    t_start = time.time()
    DEADLINE = 225.0  # safety margin before 230s kill
    
    def time_left():
        return DEADLINE - (time.time() - t_start)
    
    # Precompute pixel coordinates
    coords = np.array([(i + 0.5) / size - 0.5 for i in range(size)])
    XX, YY = np.meshgrid(coords, coords, indexing='ij')
    
    def enforce_symmetry(cell):
        """Enforce D4 symmetry by averaging over the symmetry group."""
        sym_cells = [
            cell,
            cell[::-1, :],
            cell[:, ::-1],
            cell[::-1, ::-1],
            cell.T,
            cell.T[::-1, :],
            cell.T[:, ::-1],
            cell.T[::-1, ::-1]
        ]
        avg = np.mean(sym_cells, axis=0)
        return (avg >= 0.5).astype(int)
    
    def calculate_gap(cell, n_pw=9):
        """
        Calculate the band 4-5 gap ratio using the plane-wave method.
        Uses a subset of G-vectors for speed.
        """
        eps = eps_low + (eps_high - eps_low) * cell.astype(float)
        inv_eps = 1.0 / eps
        
        N = size
        half = n_pw // 2
        
        # G-vectors
        G_list = []
        for gx in range(-half, half + 1):
            for gy in range(-half, half + 1):
                G_list.append((gx, gy))
        
        nG = len(G_list)
        
        # Precompute inv_eps Fourier coefficients using FFT
        inv_eps_fft = np.fft.fft2(inv_eps)
        
        inv_eps_F = {}
        for gx, gy in G_list:
            fft_gx = gx % N
            fft_gy = gy % N
            inv_eps_F[(gx, gy)] = inv_eps_fft[fft_gx, fft_gy] / (N * N)
        
        # Key k-points in the irreducible BZ (in units where BZ is [0, 0.5]x[0, 0.5])
        # Irreducible BZ: 0 <= ky <= kx <= 0.5
        k_points_list = [
            (0.0, 0.0),      # Gamma
            (0.1, 0.0),
            (0.2, 0.0),
            (0.3, 0.0),
            (0.4, 0.0),
            (0.5, 0.0),      # X
            (0.5, 0.1),
            (0.5, 0.2),
            (0.5, 0.3),
            (0.5, 0.4),
            (0.5, 0.5),      # M
            (0.3, 0.1),
            (0.4, 0.2),
            (0.3, 0.2),
            (0.2, 0.1),
            (0.4, 0.3),
            (0.25, 0.25)     # Interior
        ]
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points_list:
            # Build eigenvalue matrix
            # M_{G,G'} = (G-k)·(G'-k) * inv_eps_F[G-G']
            M = np.zeros((nG, nG), dtype=complex)
            
            for ig, (gx, gy) in enumerate(G_list):
                Gkx = gx - kx
                Gky = gy - ky
                for igp, (gxp, gyp) in enumerate(G_list):
                    Gpkx = gxp - kx
                    Gpky = gyp - ky
                    dGx = gx - gxp
                    dGy = gy - gyp
                    if (dGx, dGy) in inv_eps_F:
                        M[ig, igp] = (Gkx * Gpkx + Gky * Gpky) * inv_eps_F[(dGx, dGy)]
            
            # Solve eigenvalue problem
            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.sort(eigvals.real)
            
            if len(eigvals) >= 5:
                omega_4 = np.sqrt(max(0, eigvals[3]))
                omega_5 = np.sqrt(max(0, eigvals[4]))
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == float('inf'):
            return -1.0
        
        gap_ratio = 2.0 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
        return gap_ratio
    
    def make_cell_from_radii(radii):
        """
        Create a cell from a 4x4 array of radii for 8x8 sub-cells.
        Each sub-cell is an 8x8 block with a circular disk of the given radius.
        radii: 4x4 array of radius values (0 to 0.5 in sub-cell units)
        """
        cell = np.zeros((size, size), dtype=int)
        sub_size = size // 4  # 8 pixels per sub-cell
        
        for i in range(4):
            for j in range(4):
                r = radii[i, j]
                # Create 8x8 sub-cell with circular disk
                for di in range(sub_size):
                    for dj in range(sub_size):
                        # Center of sub-cell
                        cx = (i * sub_size + di + 0.5) / size - 0.5
                        cy = (j * sub_size + dj + 0.5) / size - 0.5
                        # Center of the 4x4 superlattice unit
                        super_cx = (i + 0.5) / 4.0 - 0.5
                        super_cy = (j + 0.5) / 4.0 - 0.5
                        # Distance from super-cell center
                        dx = cx - super_cx
                        dy = cy - super_cy
                        # Radius in pixel units (sub-cell is 8 pixels wide, so max radius is 4)
                        r_pixels = r * sub_size
                        if dx*dx + dy*dy <= r_pixels*r_pixels:
                            cell[i*sub_size + di, j*sub_size + dj] = 1
        
        return enforce_symmetry(cell)
    
    # New structural idea: single centrally located dielectric disk with fractal boundary
    # r(theta) = r0 + sum_{m in {4,8}} a_m * cos(m*theta)
    # This maintains inversion symmetry and D4 symmetry (m=4,8 are multiples of 4)
    
    def make_cell_from_fractal(r0, a4, a8):
        """
        Create a cell from a single centrally located dielectric disk with fractal boundary.
        r(theta) = r0 + a4 * cos(4*theta) + a8 * cos(8*theta)
        r0: base radius (in pixels, 0 to 16)
        a4: amplitude of 4th harmonic (in pixels)
        a8: amplitude of 8th harmonic (in pixels)
        """
        cell = np.zeros((size, size), dtype=int)
        
        # Pixel centers
        x = np.arange(size) + 0.5
        y = np.arange(size) + 0.5
        XX, YY = np.meshgrid(x, y, indexing='ij')
        
        # Center of the cell
        cx = size / 2.0
        cy = size / 2.0
        
        # Compute distance and angle for each pixel
        dx = XX - cx
        dy = YY - cy
        dist = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        
        # Compute fractal radius at each angle
        r_theta = r0 + a4 * np.cos(4 * theta) + a8 * np.cos(8 * theta)
        
        # Fill pixels inside the fractal boundary
        cell[dist <= r_theta] = 1
        
        return enforce_symmetry(cell)
    
    best_cell = None
    best_score = -float('inf')
    
    # Phase 1: Coarse grid search over fractal parameters
    # r0 in [4, 12], a4 in [-3, 3], a8 in [-2, 2]
    r0_vals = np.arange(4, 12.1, 2.0)
    a4_vals = np.arange(-3, 3.1, 1.0)
    a8_vals = np.arange(-2, 2.1, 0.5)
    
    for r0 in r0_vals:
        if time_left() < 10:
            break
        for a4 in a4_vals:
            if time_left() < 10:
                break
            for a8 in a8_vals:
                cell = make_cell_from_fractal(r0, a4, a8)
                score = calculate_gap(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell.copy()
    
    # Phase 2: Local refinement around best fractal parameters
    if best_cell is not None and time_left() > 30:
        # We need to recover the best parameters or do a random local search
        # Since we don't store the best params, let's do a random local search
        # But we need a starting point. Let's assume the best was found in the grid.
        # We can re-evaluate the grid points near the best score? No, we didn't store them.
        # Let's just do a random search with a tighter range around the previous best?
        # Actually, we can store the best params in the loop.
        
        # Let's restart the search with a finer grid around the best found in Phase 1?
        # We didn't store the best params. Let's modify the loop to store them.
        pass
    
    # Let's redo the logic to store best params
    best_params = (8, 0, 0)
    
    # Phase 1: Coarse Grid Search
    for r0 in r0_vals:
        if time_left() < 10:
            break
        for a4 in a4_vals:
            if time_left() < 10:
                break
            for a8 in a8_vals:
                cell = make_cell_from_fractal(r0, a4, a8)
                score = calculate_gap(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell.copy()
                    best_params = (r0, a4, a8)
    
    # Phase 2: Local Refinement
    if time_left() > 30:
        r0_best, a4_best, a8_best = best_params
        # Fine grid around best
        # Step 0.5 for r0, 0.25 for a4, 0.25 for a8
        r0_fine = np.arange(max(4, r0_best-2), min(12, r0_best+2)+0.5, 0.5)
        a4_fine = np.arange(max(-3, a4_best-1), min(3, a4_best+1)+0.25, 0.25)
        a8_fine = np.arange(max(-2, a8_best-1), min(2, a8_best+1)+0.25, 0.25)
        
        for r0 in r0_fine:
            if time_left() < 10:
                break
            for a4 in a4_fine:
                if time_left() < 10:
                    break
                for a8 in a8_fine:
                    cell = make_cell_from_fractal(r0, a4, a8)
                    score = calculate_gap(cell)
                    if score > best_score:
                        best_score = score
                        best_cell = cell.copy()
                        best_params = (r0, a4, a8)
    
    # Phase 3: Random search if time remains
    if time_left() > 20:
        np.random.seed(42)
        for _ in range(int(time_left() / 2)):  # ~2s per eval estimate
            if time_left() < 10:
                break
            r0 = np.random.uniform(4, 12)
            a4 = np.random.uniform(-3, 3)
            a8 = np.random.uniform(-2, 2)
            cell = make_cell_from_fractal(r0, a4, a8)
            score = calculate_gap(cell)
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
                best_params = (r0, a4, a8)
    
    # Phase 4: Try the original 4x4 superlattice approach if time remains
    if time_left() > 30:
        # Initial candidates: various radius patterns
        candidates = []
        
        # Candidate 1: Uniform radius (baseline)
        candidates.append(np.full((4, 4), 0.3))
        
        # Candidate 2: Checkerboard pattern
        checker = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                if (i + j) % 2 == 0:
                    checker[i, j] = 0.4
                else:
                    checker[i, j] = 0.2
        candidates.append(checker)
        
        # Candidate 3: Radial gradient
        radial = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                d = math.sqrt((i - 1.5)**2 + (j - 1.5)**2)
                radial[i, j] = 0.4 - 0.1 * d
        candidates.append(radial)
        
        # Candidate 4: Cross pattern
        cross = np.full((4, 4), 0.2)
        cross[1, :] = 0.4
        cross[2, :] = 0.4
        cross[:, 1] = 0.4
        cross[:, 2] = 0.4
        candidates.append(cross)
        
        # Candidate 5: Diagonal stripes
        diag = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                if (i - j) % 2 == 0:
                    diag[i, j] = 0.4
                else:
                    diag[i, j] = 0.2
        candidates.append(diag)
        
        # Candidate 6: Anti-diagonal stripes
        anti_diag = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                if (i + j) % 2 == 0:
                    anti_diag[i, j] = 0.2
                else:
                    anti_diag[i, j] = 0.4
        candidates.append(anti_diag)
        
        # Candidate 7: Corner emphasis
        corner = np.full((4, 4), 0.2)
        corner[0, 0] = 0.5
        corner[0, 3] = 0.5
        corner[3, 0] = 0.5
        corner[3, 3] = 0.5
        candidates.append(corner)
        
        # Candidate 8: Center emphasis
        center = np.full((4, 4), 0.2)
        center[1, 1] = 0.5
        center[1, 2] = 0.5
        center[2, 1] = 0.5
        center[2, 2] = 0.5
        candidates.append(center)
        
        # Evaluate initial candidates
        for cand in candidates:
            if time_left() < 10:
                break
            cell = make_cell_from_radii(cand)
            score = calculate_gap(cell)
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
        
        # Local optimization using coordinate descent on radii
        if best_cell is not None and time_left() > 30:
            # Find which candidate gave the best score
            best_radii = None
            for cand in candidates:
                cell = make_cell_from_radii(cand)
                if np.array_equal(cell, best_cell):
                    best_radii = cand.copy()
                    break
            
            if best_radii is None:
                # Fallback: use uniform
                best_radii = np.full((4, 4), 0.3)
            
            best_score_local = best_score
            
            # Coordinate descent on each radius
            for round_num in range(5):
                if time_left() < 15:
                    break
                
                improved = False
                for i in range(4):
                    for j in range(4):
                        if time_left() < 10:
                            break
                        
                        current_val = best_radii[i, j]
                        
                        # Try values around current
                        for delta in [-0.05, -0.02, 0.02, 0.05, 0.1]:
                            new_val = current_val + delta
                            if new_val < 0.05 or new_val > 0.5:
                                continue
                            
                            new_radii = best_radii.copy()
                            new_radii[i, j] = new_val
                            
                            cell = make_cell_from_radii(new_radii)
                            score = calculate_gap(cell)
                            
                            if score > best_score_local:
                                best_score_local = score
                                best_radii = new_radii
                                best_cell = cell.copy()
                                improved = True
                                break  # Move to next radius
                
                if not improved:
                    break
    
    # Final fallback: if we didn't find a good cell, use a simple circular rod
    if best_cell is None or best_score < 0:
        # Circular rod, radius 0.2
        r = 0.2
        val = XX**2 + YY**2
        cell = (val <= r**2).astype(int)
        best_cell = enforce_symmetry(cell)
    
    return best_cell.tolist()
