import numpy as np
import time
import math

def design(problem):
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin before 120s limit
    
    size = problem['pixels']  # 32
    eps_low = problem['eps_low']  # 1
    eps_high = problem['eps_high']  # 12
    lower_band = problem['lower_band']  # 4
    
    # Define independent pixels under D4 symmetry
    # For a 32x32 cell (indices 0..31), center is at (15.5, 15.5)
    # Symmetries: 180° rotation, reflection about x-axis, y-axis, and diagonals
    # The fundamental domain for D4 symmetry on a grid is the triangular region
    # in the upper-left corner where i <= j and i <= 31-j, i.e., i <= 15 and i <= j <= 31-i
    indep_indices = []
    for i in range(16):
        for j in range(i, 32 - i):
            indep_indices.append((i, j))
    
    n_indep = len(indep_indices)
    
    def expand_cell(indep_vals):
        """Expand independent pixels to full 32x32 cell using D4 symmetry."""
        cell = np.zeros((size, size), dtype=np.uint8)
        for idx, val in enumerate(indep_vals):
            i, j = indep_indices[idx]
            # Apply all 8 symmetry operations
            sym_positions = [
                (i, j),
                (31-i, 31-j),
                (31-i, j),
                (i, 31-j),
                (j, i),
                (31-j, 31-i),
                (j, 31-i),
                (31-j, i)
            ]
            for (ii, jj) in sym_positions:
                cell[ii, jj] = val
        return cell
    
    def compute_score_fast(cell, n_pw=17, n_kpts=14):
        """Compute approximate band gap score using plane wave method."""
        inv_eps = np.where(cell == 1, 1.0/eps_high, 1.0/eps_low)
        fft_inv = np.fft.fft2(inv_eps)
        
        # Plane wave indices
        half = n_pw // 2
        g_vals = np.arange(-half, half + 1)
        gx, gy = np.meshgrid(g_vals, g_vals, indexing='ij')
        g_vecs = np.stack([gx.flatten(), gy.flatten()], axis=1)
        n_pw_actual = len(g_vecs)
        
        # K-points along the irreducible Brillouin zone boundary
        # Gamma (0,0) -> X (0.5, 0) -> M (0.5, 0.5) -> Gamma
        k_points = []
        # Gamma to X: (t, 0) for t in [0, 0.5]
        n_seg = max(2, n_kpts // 3)
        for t in np.linspace(0, 0.5, n_seg):
            k_points.append((t, 0))
        # X to M: (0.5, t) for t in [0, 0.5]
        for t in np.linspace(0.5, 0.5, 1):
            pass
        for t in np.linspace(0, 0.5, n_seg - 1):
            k_points.append((0.5, t))
        # M to Gamma: (0.5 - t, 0.5 - t) for t in [0, 0.5]
        for t in np.linspace(0, 0.5, n_seg - 1):
            k_points.append((0.5 - t, 0.5 - t))
        
        max_omega_4 = 0.0
        min_omega_5 = np.inf
        
        for k in k_points:
            k_vec = np.array(k)
            matrix = np.zeros((n_pw_actual, n_pw_actual), dtype=np.complex128)
            
            for alpha in range(n_pw_actual):
                ga = g_vecs[alpha]
                for beta in range(n_pw_actual):
                    gb = g_vecs[beta]
                    g_diff = ga - gb
                    idx_x = int(g_diff[0]) % size
                    idx_y = int(g_diff[1]) % size
                    coeff = fft_inv[idx_x, idx_y]
                    
                    k_plus_g = k_vec + gb
                    val = coeff * (k_plus_g[0]**2 + k_plus_g[1]**2)
                    matrix[alpha, beta] = val
            
            eigs = np.linalg.eigvalsh(matrix)
            if len(eigs) > 4:
                w4 = np.sqrt(max(eigs[3], 0))
                w5 = np.sqrt(max(eigs[4], 0))
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
        
        if min_omega_5 == np.inf or max_omega_4 == 0:
            return -1.0
        gap = min_omega_5 - max_omega_4
        if gap < 0:
            return -1.0
        ratio = 2 * gap / (min_omega_5 + max_omega_4)
        return ratio
    
    def generate_ring_cell(ring_assignments):
        """
        Generate a cell from concentric square ring assignments.
        ring_assignments is a list of 0/1 for each ring from center outward.
        For a 32x32 cell, the Chebyshev distance from center defines rings.
        Center is at (15.5, 15.5), so max Chebyshev distance is 15.5, 
        giving rings 0..15 (16 rings).
        """
        center = 15.5
        n_rings = len(ring_assignments)
        cell = np.zeros((size, size), dtype=np.uint8)
        for i in range(size):
            for j in range(size):
                dx = abs(i + 0.5 - center)
                dy = abs(j + 0.5 - center)
                ring_idx = int(max(dx, dy))  # Chebyshev distance
                if ring_idx < n_rings:
                    cell[i, j] = ring_assignments[ring_idx]
        return cell
    
    def generate_cross_cell(arm_width):
        """Generate a cross-shaped dielectric pattern."""
        cell = np.zeros((size, size), dtype=np.uint8)
        center = 15.5
        for i in range(size):
            for j in range(size):
                dx = abs(i + 0.5 - center)
                dy = abs(j + 0.5 - center)
                if dx < arm_width or dy < arm_width:
                    cell[i, j] = 1
        return cell
    
    def generate_four_rods_cell(rod_radius):
        """Generate four dielectric rods at the corners of the fundamental domain."""
        cell = np.zeros((size, size), dtype=np.uint8)
        center = 15.5
        # Rods at positions relative to center: (±8, ±8) approximately
        rod_positions = [(8, 8), (8, -8), (-8, 8), (-8, -8)]
        for i in range(size):
            for j in range(size):
                dx = i + 0.5 - center
                dy = j + 0.5 - center
                for rx, ry in rod_positions:
                    r = math.sqrt((dx - rx)**2 + (dy - ry)**2)
                    if r < rod_radius:
                        cell[i, j] = 1
                        break
        return cell
    
    def generate_checkerboard_cell(period):
        """Generate a checkerboard pattern with given period."""
        cell = np.zeros((size, size), dtype=np.uint8)
        for i in range(size):
            for j in range(size):
                if ((i // period) + (j // period)) % 2 == 0:
                    cell[i, j] = 1
        return cell
    
    # Strategy 1: Explore concentric square rings
    # There are 16 rings (Chebyshev distance 0 to 15)
    # We'll search over binary sequences of length 16
    
    best_cell = None
    best_score = -np.inf
    
    # Evaluate several initial strategies
    strategies = []
    
    # Ring-based strategies: try various patterns
    # Pattern 1: Alternating rings
    for pattern_type in range(6):
        ring_assign = []
        for r in range(16):
            if pattern_type == 0:
                ring_assign.append(1 if r % 2 == 0 else 0)
            elif pattern_type == 1:
                ring_assign.append(1 if r % 2 == 1 else 0)
            elif pattern_type == 2:
                ring_assign.append(1 if r < 4 else 0)  # Central block
            elif pattern_type == 3:
                ring_assign.append(0 if r < 4 else 1)  # Ring with hole
            elif pattern_type == 4:
                ring_assign.append(1 if 2 <= r <= 6 else 0)  # Annulus
            elif pattern_type == 5:
                ring_assign.append(1 if r >= 8 else 0)  # Outer frame
        cell = generate_ring_cell(ring_assign)
        strategies.append(cell)
    
    # Cross shapes
    for w in [3, 4, 5, 6]:
        strategies.append(generate_cross_cell(w))
    
    # Four rods
    for r in [3.0, 4.0, 5.0]:
        strategies.append(generate_four_rods_cell(r))
    
    # Checkerboards
    for p in [4, 6, 8]:
        strategies.append(generate_checkerboard_cell(p))
    
    # Evaluate all strategies
    for cell in strategies:
        score = compute_score_fast(cell)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Hill climbing optimization
    if best_cell is not None:
        current_cell = best_cell.copy()
        current_score = best_score
        
        np.random.seed(42)
        
        # Get current independent values from best_cell
        current_indep = []
        for (i, j) in indep_indices:
            current_indep.append(int(current_cell[i, j]))
        
        while time.time() < deadline:
            # Random pixel flip
            idx = np.random.randint(n_indep)
            old_val = current_indep[idx]
            current_indep[idx] = 1 - old_val
            
            new_cell = expand_cell(current_indep)
            new_score = compute_score_fast(new_cell)
            
            if new_score > current_score:
                current_score = new_score
                current_cell = new_cell
                best_cell = new_cell
                best_score = new_score
            else:
                # Revert
                current_indep[idx] = old_val
            
            # Occasionally try multi-pixel moves
            if np.random.rand() < 0.15 and time.time() < deadline - 30:
                # Try flipping 2-3 random pixels together
                n_flips = np.random.randint(2, 4)
                flip_indices = np.random.choice(n_indep, size=n_flips, replace=False)
                for fi in flip_indices:
                    current_indep[fi] = 1 - current_indep[fi]
                
                new_cell = expand_cell(current_indep)
                new_score = compute_score_fast(new_cell)
                
                if new_score > current_score:
                    current_score = new_score
                    current_cell = new_cell
                    best_cell = new_cell
                    best_score = new_score
                else:
                    # Revert all
                    for fi in flip_indices:
                        current_indep[fi] = 1 - current_indep[fi]
    
    if best_cell is None:
        # Fallback: simple dielectric rod
        center = 15.5
        best_cell = np.zeros((size, size), dtype=np.uint8)
        for i in range(size):
            for j in range(size):
                dx = i + 0.5 - center
                dy = j + 0.5 - center
                r = math.sqrt(dx*dx + dy*dy)
                if r < 5.0:
                    best_cell[i, j] = 1
    
    return best_cell.tolist()
