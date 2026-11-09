import time
import math
import numpy as np

def design(problem):
    """Design a 32x32 photonic crystal unit cell maximizing the 4-5 TM band gap."""
    start_time = time.time()
    deadline = start_time + 200  # 200s safety margin within 230s limit
    
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    # --- Fast band solver ---
    # Use a moderate number of plane waves for speed during search
    # The evaluator uses 289 (17x17), but we can use fewer for fast scoring
    # and refine at the end with more.
    
    def get_g_vectors(nmax):
        """Generate G-vectors for a square grid of size (2*nmax+1)^2."""
        G = []
        for mx in range(-nmax, nmax + 1):
            for my in range(-nmax, nmax + 1):
                G.append((mx, my))
        return np.array(G, dtype=float)
    
    # K-points in the irreducible Brillouin zone (30 points as specified)
    # The irreducible BZ is the triangle (0,0)-(pi,0)-(pi,pi)
    # We'll use a representative set
    k_points = []
    # Along Gamma-X: (0,0) to (pi,0)
    for t in np.linspace(0, 1, 7):
        k_points.append((t * np.pi, 0))
    # Along X-M: (pi,0) to (pi,pi)
    for t in np.linspace(0, 1, 7):
        k_points.append((np.pi, t * np.pi))
    # Along M-Gamma: (pi,pi) to (0,0)
    for t in np.linspace(0, 1, 7):
        k_points.append((t * np.pi, t * np.pi))
    # Interior points
    k_points.append((np.pi/2, np.pi/4))
    k_points.append((3*np.pi/4, np.pi/4))
    k_points.append((np.pi/2, np.pi/2))
    k_points.append((np.pi/4, np.pi/4))
    k_points.append((3*np.pi/4, np.pi/2))
    
    def compute_fourier_fast(cell, nmax):
        """Compute Fourier coefficients of 1/eps using FFT."""
        eps = np.where(np.array(cell) == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        
        # Use FFT to compute Fourier coefficients
        # F[mx, my] = (1/N^2) * sum_{i,j} inv_eps[i,j] * exp(-i*2*pi*(mx*i/N + my*j/N))
        # This is the DFT, which FFT computes efficiently
        F_full = np.fft.fft2(inv_eps) / (size * size)
        
        # Extract the needed range
        # F_full has indices 0..size-1, corresponding to frequencies 0..size-1
        # We need frequencies from -nmax to nmax
        F = np.zeros((2*nmax+1, 2*nmax+1), dtype=complex)
        for mx in range(-nmax, nmax+1):
            for my in range(-nmax, nmax+1):
                # Map to FFT index
                mx_idx = mx % size
                my_idx = my % size
                F[mx + nmax, my + nmax] = F_full[mx_idx, my_idx]
        
        return F
    
    def solve_bands_fast(cell, nmax):
        """Fast band solver for TM polarization."""
        F = compute_fourier_fast(cell, nmax)
        G = get_g_vectors(nmax)
        NG = len(G)
        
        max_omega4 = 0
        min_omega5 = np.inf
        
        for kx, ky in k_points:
            # Build the eigenvalue matrix
            # A[i,j] = (G_i - k)^2 * F[G_i - G_j]
            # Note: F is indexed by [dGx + nmax, dGy + nmax]
            
            # Precompute (G_i - k)^2 for each i
            Gi_k2 = np.array([(G[i,0] - kx)**2 + (G[i,1] - ky)**2 for i in range(NG)])
            
            A = np.zeros((NG, NG), dtype=complex)
            for i in range(NG):
                for j in range(NG):
                    dGx = int(G[i,0] - G[j,0])
                    dGy = int(G[i,1] - G[j,1])
                    if -nmax <= dGx <= nmax and -nmax <= dGy <= nmax:
                        Fval = F[dGx + nmax, dGy + nmax]
                    else:
                        Fval = 0
                    A[i, j] = Gi_k2[i] * Fval
            
            # Solve eigenvalue problem
            try:
                eigvals = np.linalg.eigvalsh(A)
                eigvals = np.sort(np.real(eigvals))
                # Filter out negative/zero
                eigvals = eigvals[eigvals > 1e-10]
                if len(eigvals) >= 5:
                    omega4 = np.sqrt(eigvals[3])
                    omega5 = np.sqrt(eigvals[4])
                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
            except:
                pass
        
        if min_omega5 == np.inf or max_omega4 <= 0:
            return -1.0
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2
        if midgap <= 0:
            return -1.0
        return gap / midgap
    
    # --- Symmetric cell generation from 8x8 fundamental domain ---
    def get_fundamental(cell):
        """Get the 8x8 top-left quadrant (fundamental domain for C4v symmetry)."""
        return [cell[i][:8] for i in range(8)]
    
    def set_fundamental(fund):
        """Set the full 32x32 cell from the 8x8 fundamental domain with C4v symmetry."""
        cell = [[0]*size for _ in range(size)]
        for i in range(8):
            for j in range(8):
                val = fund[i][j]
                # The 8x8 quadrant maps to 4 quadrants of 16x16
                # Top-left: (i, j)
                # Top-right: (i, size-1-j)
                # Bottom-left: (size-1-i, j)
                # Bottom-right: (size-1-i, size-1-j)
                # Also need transpose symmetry: (j, i), etc.
                # For C4v, the fundamental domain is the 8x8 corner
                # Actually, for 32x32 with symmetry about center, the fundamental
                # domain is 16x16 with additional mirror symmetries.
                # Let me reconsider: the cell is invariant under:
                # 1. 180-degree rotation: cell[i][j] = cell[31-i][31-j]
                # 2. Transpose: cell[i][j] = cell[j][i]
                # 3. Horizontal flip: cell[i][j] = cell[i][31-j]
                # 4. Vertical flip: cell[i][j] = cell[31-i][j]
                # The fundamental domain is the 8x8 top-left corner? No.
                # With all these symmetries, the fundamental domain is actually
                # the 8x8 corner because 32/4 = 8.
                # Wait: 32x32, symmetry about center means the cell is determined
                # by the 16x16 top-left quadrant. But with transpose symmetry,
                # it's determined by the 16x16 upper triangle. With horizontal
                # flip, it's the 16x8 left half. With vertical flip, it's the
                # 8x8 top-left corner.
                
                # So the 8x8 fundamental domain determines the whole cell.
                # Mapping: for pixel (i,j) in 0..31:
                # i' = min(i, 31-i)  # fold vertically
                # j' = min(j, 31-j)  # fold horizontally
                # Then if i' > j', swap (transpose symmetry)
                # So the fundamental domain is indexed by (i', j') where i' <= j',
                # but since we're folding, i', j' are in 0..15, and i' <= j'.
                # That's not 8x8.
                
                # Let me think again. The symmetry group is the square group D4.
                # The orbit of a point (i,j) under D4 has size 8 (generically).
                # 32*32 / 8 = 128, so the fundamental domain has 128 pixels.
                # That's a 16x8 rectangle or similar.
                
                # Actually, for a 32x32 grid with D4 symmetry about the center,
                # the fundamental domain is the set of points in the triangle
                # (0,0)-(16,0)-(16,16) but that's not quite right either.
                
                # Let me just use the 16x16 top-left quadrant and enforce
                # the symmetries explicitly.
                pass
        
        # Simpler approach: use 16x16 quadrant and enforce symmetries
        cell = [[0]*size for _ in range(size)]
        for i in range(16):
            for j in range(16):
                val = fund[i][j] if i < 8 and j < 8 else 0  # This is wrong
                pass
        
        # Let me restart with a cleaner approach
        cell = [[0]*size for _ in range(size)]
        # The 8x8 fundamental domain is the top-left 8x8
        # For any pixel (i,j), we can map it to the fundamental domain
        # using the D4 symmetries
        for i in range(size):
            for j in range(size):
                # Fold to fundamental domain
                # First, fold about the center
                ii = i if i < 16 else 31 - i
                jj = j if j < 16 else 31 - j
                # Now ii, jj in 0..15
                # Apply transpose symmetry
                if ii > jj:
                    ii, jj = jj, ii
                # Now ii <= jj, both in 0..15
                # But we need to map to 8x8
                # Hmm, this doesn't give 8x8 directly.
                
                # Alternative: the fundamental domain for D4 on a 32x32 grid
                # is indeed 128 pixels. Let me just store the 16x8 upper-left
                # triangle (i <= j, i in 0..15, j in i..15)
                pass
        
        # I'll use a different approach: store the full 16x16 quadrant
        # and enforce symmetries when generating the full cell
        pass
    
    # Let me use a simpler representation: 16x16 quadrant with transpose symmetry
    def get_quadrant(cell):
        """Get the 16x16 top-left quadrant."""
        return [cell[i][:16] for i in range(16)]
    
    def set_quadrant(quad):
        """Generate the full 32x32 cell from the 16x16 top-left quadrant."""
        cell = [[0]*size for _ in range(size)]
        for i in range(16):
            for j in range(16):
                val = quad[i][j]
                # Top-left
                cell[i][j] = val
                # Top-right (horizontal flip)
                cell[i][31-j] = val
                # Bottom-left (vertical flip)
                cell[31-i][j] = val
                # Bottom-right (180 rotation)
                cell[31-i][31-j] = val
                # Transpose symmetry
                cell[j][i] = val
                cell[j][31-i] = val
                cell[31-j][i] = val
                cell[31-j][31-i] = val
        return cell
    
    # --- Initial candidates ---
    def make_square_rod(radius_frac):
        """Square dielectric rod in the center."""
        cell = [[0]*size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                x = abs((i + 0.5) / size - 0.5)
                y = abs((j + 0.5) / size - 0.5)
                if x < radius_frac and y < radius_frac:
                    cell[i][j] = 1
        return cell
    
    def make_air_hole(hole_frac):
        """Dielectric background with square air hole in the center."""
        cell = [[1]*size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                x = abs((i + 0.5) / size - 0.5)
                y = abs((j + 0.5) / size - 0.5)
                if x < hole_frac and y < hole_frac:
                    cell[i][j] = 0
        return cell
    
    def make_cross(arm_width_frac):
        """Cross-shaped dielectric element."""
        cell = [[0]*size for _ in range(size)]
        half = arm_width_frac / 2
        for i in range(size):
            for j in range(size):
                x = abs((i + 0.5) / size - 0.5)
                y = abs((j + 0.5) / size - 0.5)
                if x < half or y < half:
                    cell[i][j] = 1
        return cell
    
    def make_ring(inner_frac, outer_frac):
        """Square ring dielectric."""
        cell = [[0]*size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                x = abs((i + 0.5) / size - 0.5)
                y = abs((j + 0.5) / size - 0.5)
                r_max = max(x, y)
                if inner_frac <= r_max <= outer_frac:
                    cell[i][j] = 1
        return cell
    
    # Try several candidates with fast solver
    candidates = []
    candidates.append(make_square_rod(0.25))
    candidates.append(make_square_rod(0.3))
    candidates.append(make_square_rod(0.35))
    candidates.append(make_air_hole(0.2))
    candidates.append(make_air_hole(0.25))
    candidates.append(make_air_hole(0.3))
    candidates.append(make_cross(0.2))
    candidates.append(make_cross(0.25))
    candidates.append(make_cross(0.3))
    candidates.append(make_ring(0.15, 0.35))
    candidates.append(make_ring(0.2, 0.4))
    candidates.append(make_ring(0.25, 0.45))
    
    best_score = -1e10
    best_quad = None
    
    # Use nmax=6 (13x13=169 plane waves) for fast search
    nmax_fast = 6
    
    for cell in candidates:
        if time.time() > deadline:
            break
        score = solve_bands_fast(cell, nmax_fast)
        if score > best_score:
            best_score = score
            best_quad = get_quadrant(cell)
    
    # --- Local search: flip pixels in the 16x16 quadrant ---
    # The quadrant has 256 pixels, but with transpose symmetry, only ~136 are independent
    # For simplicity, we'll flip pixels in the 16x16 quadrant and enforce symmetry
    
    current_quad = [row[:] for row in best_quad] if best_quad else get_quadrant(candidates[0])
    current_score = best_score
    best_score_sa = best_score
    best_quad_sa = [row[:] for row in best_quad] if best_quad else get_quadrant(candidates[0])
    
    temp = 0.15
    cooling = 0.997
    min_temp = 0.005
    
    # Precompute which pixels in the quadrant affect the symmetry
    # We'll just flip random pixels in the 16x16 quadrant
    
    while time.time() < deadline - 10 and temp > min_temp:
        # Pick a random pixel in the 16x16 quadrant
        i = np.random.randint(0, 16)
        j = np.random.randint(0, 16)
        
        # Flip it
        current_quad[i][j] = 1 - current_quad[i][j]
        
        cell = set_quadrant(current_quad)
        new_score = solve_bands_fast(cell, nmax_fast)
        
        if new_score > current_score or np.random.random() < math.exp((new_score - current_score) / temp):
            current_score = new_score
            if new_score > best_score_sa:
                best_score_sa = new_score
                best_quad_sa = [row[:] for row in current_quad]
        else:
            # Revert
            current_quad[i][j] = 1 - current_quad[i][j]
        
        temp *= cooling
    
    # --- Final refinement with more plane waves ---
    if time.time() < deadline - 20:
        # Use nmax=8 (17x17=289 plane waves) for final scoring
        nmax_final = 8
        final_quad = [row[:] for row in best_quad_sa]
        final_cell = set_quadrant(final_quad)
        final_score = solve_bands_fast(final_cell, nmax_final)
        
        # Do a few more local search steps with the more accurate solver
        temp2 = 0.05
        cooling2 = 0.99
        min_temp2 = 0.01
        current_quad2 = [row[:] for row in final_quad]
        current_score2 = final_score
        best_score2 = final_score
        best_quad2 = [row[:] for row in final_quad]
        
        while time.time() < deadline - 5 and temp2 > min_temp2:
            i = np.random.randint(0, 16)
            j = np.random.randint(0, 16)
            
            current_quad2[i][j] = 1 - current_quad2[i][j]
            cell2 = set_quadrant(current_quad2)
            new_score2 = solve_bands_fast(cell2, nmax_final)
            
            if new_score2 > current_score2 or np.random.random() < math.exp((new_score2 - current_score2) / temp2):
                current_score2 = new_score2
                if new_score2 > best_score2:
                    best_score2 = new_score2
                    best_quad2 = [row[:] for row in current_quad2]
            else:
                current_quad2[i][j] = 1 - current_quad2[i][j]
            
            temp2 *= cooling2
        
        final_quad = best_quad2
    
    # Generate final cell
    final_cell = set_quadrant(final_quad)
    
    # Verify symmetry
    for i in range(size):
        for j in range(size):
            if final_cell[i][j] != final_cell[31-i][31-j]:
                final_cell[31-i][31-j] = final_cell[i][j]
            if final_cell[i][j] != final_cell[j][i]:
                final_cell[j][i] = final_cell[i][j]
            if final_cell[i][j] != final_cell[i][31-j]:
                final_cell[i][31-j] = final_cell[i][j]
            if final_cell[i][j] != final_cell[31-i][j]:
                final_cell[31-i][j] = final_cell[i][j]
    
    return final_cell
