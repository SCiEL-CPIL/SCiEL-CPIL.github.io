# EVOLVE-BLOCK-START
import time
import numpy as np

def _compute_bands(cell, size=32, num_pw=7, num_k=12, band_max=6):
    """
    Compute the first 'band_max' TM bands for a given cell using plane-wave method.
    Returns (max_omega_4, min_omega_5) in units of a/(2*pi*c).
    """
    # Dielectric function
    eps = np.array(cell, dtype=float)
    eps[eps == 0] = 1.0
    eps[eps == 1] = 12.0
    
    # Inverse dielectric function
    inv_eps = 1.0 / eps
    
    # Plane wave grid
    g_vals = np.arange(-num_pw, num_pw + 1)
    Gx, Gy = np.meshgrid(g_vals, g_vals, indexing='ij')
    G_flat = np.stack([Gx.ravel(), Gy.ravel()], axis=1)
    num_G = len(G_flat)
    
    # Fourier coefficients of inv_eps using FFT
    inv_eps_fft = np.fft.fft2(inv_eps)
    inv_eps_fft = inv_eps_fft / (size * size)
    
    # Precompute S matrix: S[i,j] = inv_eps_{G_i - G_j}
    S = np.zeros((num_G, num_G), dtype=complex)
    for i in range(num_G):
        for j in range(num_G):
            G_diff = G_flat[i] - G_flat[j]
            m_diff = G_diff[0] % size
            n_diff = G_diff[1] % size
            S[i, j] = inv_eps_fft[m_diff, n_diff]
    
    def get_band_structure(kx, ky):
        k_vec = np.array([kx, ky])
        k_plus_G = G_flat + k_vec
        k_plus_G_sq = np.sum(k_plus_G**2, axis=1)
        k_plus_G_sq = np.maximum(k_plus_G_sq, 1e-12)
        sqrt_k_plus_G = np.sqrt(k_plus_G_sq)
        
        B = (sqrt_k_plus_G[:, None] * S) * sqrt_k_plus_G[None, :]
        vals, _ = np.linalg.eigh(B)
        return vals[:band_max]
    
    # Sample k-points in irreducible BZ (C4v symmetry)
    # Irreducible zone: triangle with vertices (0,0), (0.5,0), (0.5,0.5)
    # Condition: 0 <= ky <= kx <= 0.5
    
    max_omega_4 = 0.0
    min_omega_5 = float('inf')
    
    # Use a grid of k-points in the irreducible triangle
    k_points = set()
    # High symmetry points
    k_points.add((0.0, 0.0))
    k_points.add((0.5, 0.0))
    k_points.add((0.5, 0.5))
    
    # Grid points
    for i in range(num_k + 1):
        for j in range(num_k + 1):
            kx = 0.5 * i / num_k
            ky = 0.5 * j / num_k
            if 0 <= ky <= kx <= 0.5 + 1e-9:
                k_points.add((round(kx, 10), round(ky, 10)))
    
    for kx, ky in k_points:
        vals = get_band_structure(kx, ky)
        omegas = np.sqrt(vals)
        if len(omegas) >= 5:
            max_omega_4 = max(max_omega_4, omegas[3])
            min_omega_5 = min(min_omega_5, omegas[4])
    
    if min_omega_5 == float('inf'):
        return 0.0, 0.0
        
    return max_omega_4, min_omega_5

def _score_cell(cell):
    """
    Calculate the gap-midgap ratio.
    """
    max_4, min_5 = _compute_bands(cell, num_pw=7, num_k=12, band_max=6)
    if min_5 == 0 or max_4 == 0:
        return 0.0
    gap = min_5 - max_4
    midgap = min_5 + max_4
    if midgap == 0:
        return 0.0
    return 2.0 * gap / midgap

def _make_cell_from_rings(size, ring_bits):
    """Create cell from concentric square rings."""
    cell = [[0] * size for _ in range(size)]
    for i in range(size):
        dx = abs(i - (size - 1) / 2.0)
        for j in range(size):
            dy = abs(j - (size - 1) / 2.0)
            r = int(max(dx, dy))
            if r < len(ring_bits):
                cell[i][j] = ring_bits[r]
    return cell

def _make_cross_cell(size, thickness=3):
    """Create a cross pattern."""
    cell = [[0] * size for _ in range(size)]
    center = (size - 1) / 2.0
    for i in range(size):
        for j in range(size):
            dx = abs(i - center)
            dy = abs(j - center)
            if dx < thickness or dy < thickness:
                cell[i][j] = 1
    return cell

def _make_four_rods_cell(size, rod_radius=3):
    """Create four dielectric rods at the corners of the cell."""
    cell = [[0] * size for _ in range(size)]
    positions = [(4, 4), (4, size-5), (size-5, 4), (size-5, size-5)]
    for cx, cy in positions:
        for i in range(size):
            for j in range(size):
                if (i - cx)**2 + (j - cy)**2 < rod_radius**2:
                    cell[i][j] = 1
    return cell

def _make_air_holes_cell(size, hole_radius=3):
    """Create air holes in a dielectric background."""
    cell = [[1] * size for _ in range(size)]
    # Air holes at center of each "sub-cell" - for a single unit cell, 
    # we place holes at positions that maintain C4v symmetry
    positions = [(4, 4), (4, size-5), (size-5, 4), (size-5, size-5),
                 (size//2, 4), (size//2, size-5), (4, size//2), (size-5, size//2)]
    for cx, cy in positions:
        for i in range(size):
            for j in range(size):
                if (i - cx)**2 + (j - cy)**2 < hole_radius**2:
                    cell[i][j] = 0
    return cell

def _make_checkerboard_cell(size, block=4):
    """Create a checkerboard pattern."""
    cell = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if (i // block + j // block) % 2 == 0:
                cell[i][j] = 1
    return cell

def _make_concentric_circles_cell(size, radii=None):
    """Create concentric circular rings."""
    cell = [[0] * size for _ in range(size)]
    center = (size - 1) / 2.0
    if radii is None:
        radii = [3, 6, 9, 12, 14]
    for i in range(size):
        for j in range(size):
            r = np.sqrt((i - center)**2 + (j - center)**2)
            # Determine which ring
            for idx, rad in enumerate(radii):
                if r < rad:
                    # Alternate: even rings are dielectric, odd are air
                    if idx % 2 == 0:
                        cell[i][j] = 1
                    break
    return cell

def _compute_orbits(size):
    """Compute all symmetry orbits under C4v group."""
    orbits = []
    visited = set()
    for i in range(size):
        for j in range(size):
            if (i, j) in visited:
                continue
            # Generate all symmetry equivalents
            pts = set()
            # Identity
            pts.add((i, j))
            # 90 deg rotation
            pts.add((j, size-1-i))
            # 180 deg rotation
            pts.add((size-1-i, size-1-j))
            # 270 deg rotation
            pts.add((size-1-j, i))
            # Reflection about vertical axis
            pts.add((size-1-i, j))
            # Reflection about horizontal axis
            pts.add((i, size-1-j))
            # Reflection about main diagonal
            pts.add((j, i))
            # Reflection about anti-diagonal
            pts.add((size-1-j, size-1-i))
            
            orbit = list(pts)
            for p in orbit:
                visited.add(p)
            orbits.append(orbit)
    return orbits

def design(problem):
    size = problem["pixels"]
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin before 200s limit
    
    # Generate diverse candidates
    candidates = []
    
    # 1. Concentric rings with various patterns
    for pattern in [
        [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
        [1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0],
        [1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1],
        [1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0],
        [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
        [1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1],
    ]:
        candidates.append(_make_cell_from_rings(size, pattern))
    
    # 2. Cross patterns
    for thickness in [2, 3, 4, 5]:
        candidates.append(_make_cross_cell(size, thickness))
    
    # 3. Four rods
    for r in [2, 3, 4, 5]:
        candidates.append(_make_four_rods_cell(size, r))
    
    # 4. Air holes in dielectric
    for r in [2, 3, 4]:
        candidates.append(_make_air_holes_cell(size, r))
    
    # 5. Checkerboard
    for block in [3, 4, 5, 6, 8]:
        candidates.append(_make_checkerboard_cell(size, block))
    
    # 6. Concentric circles
    for radii in [
        [3, 6, 9, 12, 14],
        [4, 8, 12, 15],
        [2, 5, 8, 11, 14],
    ]:
        candidates.append(_make_concentric_circles_cell(size, radii))
    
    # 7. Single circle (seed)
    seed_cell = [[0] * size for _ in range(size)]
    center = (size - 1) / 2.0
    for i in range(size):
        for j in range(size):
            if (i - center)**2 + (j - center)**2 < 4.0:
                seed_cell[i][j] = 1
    candidates.append(seed_cell)
    
    # Evaluate all candidates
    best_cell = None
    best_score = -float('inf')
    
    # Precompute orbits for local search
    orbits = _compute_orbits(size)
    # Sort orbits by size (smaller orbits first - they have less impact per flip)
    orbits.sort(key=len)
    
    for cell in candidates:
        if time.time() > deadline - 30:
            break
        score = _score_cell(cell)
        if score > best_score:
            best_score = score
            best_cell = [row[:] for row in cell]
    
    if best_cell is None:
        best_cell = seed_cell
        best_score = _score_cell(seed_cell)
    
    # Local search: hill climbing with symmetric orbit flips
    current_cell = [row[:] for row in best_cell]
    current_score = best_score
    
    # Try multiple passes
    max_passes = 10
    for pass_num in range(max_passes):
        if time.time() > deadline - 10:
            break
        
        improved = False
        # Try orbits in random order for diversity
        np.random.seed(42 + pass_num)
        shuffled_orbits = orbits[:]
        np.random.shuffle(shuffled_orbits)
        
        for orbit in shuffled_orbits:
            if time.time() > deadline - 10:
                break
            
            # Flip orbit
            for (r, c) in orbit:
                current_cell[r][c] = 1 - current_cell[r][c]
                
            new_score = _score_cell(current_cell)
            
            if new_score > current_score + 1e-10:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_cell = [row[:] for row in current_cell]
                improved = True
            else:
                # Revert
                for (r, c) in orbit:
                    current_cell[r][c] = 1 - current_cell[r][c]
        
        if not improved:
            break
    
    return best_cell

# EVOLVE-BLOCK-END
