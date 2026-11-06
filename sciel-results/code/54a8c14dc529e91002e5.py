"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell maximizing TM band-4/5 gap
    using direct pixel-level optimization with fast PWE surrogate.
    """
    start_time = time.time()
    deadline = start_time + 220  # Safety margin (evaluator kills at 230s)
    
    size = problem["pixels"]
    half = size // 2
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    N = size
    
    # Precompute k-points for the irreducible BZ
    # The evaluator uses 30 wave vectors; we approximate with a good set
    # Irreducible BZ for square lattice: 0 <= kx <= ky, kx^2+ky^2 <= (pi/a)^2
    # In units where the BZ boundary is at pi/a, the irreducible wedge is
    # 0 <= kx <= ky and kx^2 + ky^2 <= 1 (normalized)
    def make_k_points():
        pts = []
        # High symmetry points
        pts.append((0.0, 0.0))       # Gamma
        pts.append((0.5, 0.5))       # M
        pts.append((0.5, 0.0))       # X
        pts.append((0.0, 0.5))       # X' (equivalent to X by symmetry)
        # Points along Gamma-M edge (kx=ky)
        for t in np.linspace(0.05, 0.45, 8):
            pts.append((t, t))
        # Points along Gamma-X edge (ky=0, kx varies)
        for t in np.linspace(0.05, 0.45, 8):
            pts.append((t, 0.0))
        # Interior points
        for kx in np.linspace(0.05, 0.45, 5):
            for ky in np.linspace(kx, 0.45, 5):
                if kx*kx + ky*ky <= 1.0:
                    pts.append((kx, ky))
        # Deduplicate
        unique_pts = []
        for p in pts:
            is_dup = False
            for u in unique_pts:
                if abs(p[0]-u[0]) < 1e-6 and abs(p[1]-u[1]) < 1e-6:
                    is_dup = True
                    break
            if not is_dup:
                unique_pts.append(p)
        return unique_pts
    
    K_POINTS = make_k_points()
    n_kpts = len(K_POINTS)
    
    # Fast PWE surrogate using FFT
    # We use a reduced plane-wave basis for speed
    n_pw = 7  # 7x7 = 49 plane waves (reduced from 15x15 for speed)
    n_modes = 2 * n_pw + 1  # 15
    
    # Precompute the plane-wave index ranges
    p_range = np.arange(-n_pw, n_pw + 1)
    p_grid_x, p_grid_y = np.meshgrid(p_range, p_range, indexing='ij')
    P2 = p_grid_x**2 + p_grid_y**2  # shape (n_modes, n_modes)
    
    # Precompute |k+p|^2 for each k-point
    # K_k[m,n] = (kx + p_m)^2 + (ky + p_n)^2
    K_matrices = []
    for kx, ky in K_POINTS:
        kx_arr = kx + p_range
        ky_arr = ky + p_range
        K = kx_arr[:, None]**2 + ky_arr[None, :]**2
        K_matrices.append(K)
    
    def compute_tm_gap(cell_array):
        """
        Compute TM band gap ratio using FFT-accelerated PWE.
        cell_array: 32x32 numpy array of 0/1
        Returns: gap ratio 2*(min_omega5 - max_omega4)/(min_omega5 + max_omega4)
        """
        # Compute 1/epsilon field
        eps = np.where(cell_array == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        
        # Full 2D FFT of 1/epsilon
        F_full = np.fft.fft2(inv_eps) / (N * N)
        
        # Extract Fourier coefficients f[m,n] for |m|,|n| <= n_pw
        f = np.zeros((n_modes, n_modes), dtype=complex)
        for m in range(-n_pw, n_pw + 1):
            for n in range(-n_pw, n_pw + 1):
                mi = m % N
                ni = n % N
                f[m + n_pw, n + n_pw] = F_full[mi, ni]
        
        # Precompute FFT of f (shared across k-points)
        F_f = np.fft.fft2(f)
        
        min_omega5 = float('inf')
        max_omega4 = 0.0
        
        for idx, (kx, ky) in enumerate(K_POINTS):
            K = K_matrices[idx]
            F_K = np.fft.fft2(K)
            A = np.real(np.fft.ifft2(F_f * F_K))
            A = (A + A.T) / 2.0
            
            try:
                eigvals = np.linalg.eigvalsh(A)
                eigvals = np.sort(eigvals)
                if len(eigvals) >= 5:
                    omega4 = eigvals[3]
                    omega5 = eigvals[4]
                    if omega4 > 1e-10 and omega5 > 1e-10:
                        if omega4 > max_omega4:
                            max_omega4 = omega4
                        if omega5 < min_omega5:
                            min_omega5 = omega5
            except:
                pass
        
        if min_omega5 == float('inf') or max_omega4 < 1e-10:
            return 0.0
        
        gap_ratio = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
        return gap_ratio
    
    def quadrant_to_cell(quadrant):
        """Convert 16x16 quadrant to full 32x32 cell with C4v symmetry."""
        cell = [[0] * size for _ in range(size)]
        for i in range(half):
            for j in range(half):
                val = quadrant[i][j]
                cell[i][j] = val
                cell[i][size - 1 - j] = val
                cell[size - 1 - i][j] = val
                cell[size - 1 - i][size - 1 - j] = val
        return cell
    
    def cell_to_quadrant(cell):
        """Extract the top-left 16x16 quadrant from a 32x32 cell."""
        return [row[:half] for row in cell[:half]]
    
    def create_structures():
        """Create diverse initial structures informed by photonic crystal physics."""
        structures = []
        cx, cy = 7.5, 7.5
        
        # 1. Single central rod (circle) - radius ~4
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(half):
                x, y = i + 0.5, j + 0.5
                if (x-cx)**2 + (y-cy)**2 < 4.0**2:
                    q[i][j] = 1
        structures.append(q)
        
        # 2. Single central rod - radius ~5
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(half):
                x, y = i + 0.5, j + 0.5
                if (x-cx)**2 + (y-cy)**2 < 5.0**2:
                    q[i][j] = 1
        structures.append(q)
        
        # 3. Square rod
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(half):
                x, y = i + 0.5, j + 0.5
                if abs(x-cx) < 4.5 and abs(y-cy) < 4.5:
                    q[i][j] = 1
        structures.append(q)
        
        # 4. Super-ellipse (rounded square)
        q = [[0]*half for _ in range(half)]
        a, n_exp = 5.0, 3.0
        for i in range(half):
            for j in range(half):
                x, y = i + 0.5, j + 0.5
                dx, dy = abs(x-cx)/a, abs(y-cy)/a
                if dx**n_exp + dy**n_exp < 1.0:
                    q[i][j] = 1
        structures.append(q)
        
        # 5. Ring (annulus)
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(half):
                x, y = i + 0.5, j + 0.5
                r = math.sqrt((x-cx)**2 + (y-cy)**2)
                if 2.5 < r < 5.5:
                    q[i][j] = 1
        structures.append(q)
        
        # 6. Cross shape
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(half):
                x, y = i + 0.5, j + 0.5
                if abs(x-cx) < 2.0 or abs(y-cy) < 2.0:
                    q[i][j] = 1
        structures.append(q)
        
        # 7. Four corner rods (small circles at corners)
        q = [[0]*half for _ in range(half)]
        for cx_r, cy_r in [(2.5, 2.5), (2.5, 13.5), (13.5, 2.5), (13.5, 13.5)]:
            for i in range(half):
                for j in range(half):
                    x, y = i + 0.5, j + 0.5
                    if (x-cx_r)**2 + (y-cy_r)**2 < 2.0**2:
                        q[i][j] = 1
        structures.append(q)
        
        # 8. Central rod + corner dots
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(half):
                x, y = i + 0.5, j + 0.5
                if (x-cx)**2 + (y-cy)**2 < 3.5**2:
                    q[i][j] = 1
        for cx_r, cy_r in [(1.5, 1.5), (1.5, 14.5), (14.5, 1.5), (14.5, 14.5)]:
            for i in range(half):
                for j in range(half):
                    x, y = i + 0.5, j + 0.5
                    if (x-cx_r)**2 + (y-cy_r)**2 < 1.5**2:
                        q[i][j] = 1
        structures.append(q)
        
        # 9. Diamond shape
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(half):
                x, y = i + 0.5, j + 0.5
                if abs(x-cx) + abs(y-cy) < 5.0:
                    q[i][j] = 1
        structures.append(q)
        
        # 10. Octagon approximation
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(half):
                x, y = i + 0.5, j + 0.5
                dx, dy = abs(x-cx), abs(y-cy)
                if dx < 3.5 and dy < 3.5:
                    q[i][j] = 1
                elif dx < 5.0 and dy < 3.5 and dx + dy < 7.0:
                    q[i][j] = 1
                elif dx < 3.5 and dy < 5.0 and dx + dy < 7.0:
                    q[i][j] = 1
        structures.append(q)
        
        # 11. Random structures
        rng = random.Random(42)
        for _ in range(5):
            q = [[0]*half for _ in range(half)]
            for i in range(half):
                for j in range(half):
                    if rng.random() < 0.35:
                        q[i][j] = 1
            structures.append(q)
        
        return structures
    
    # Coordinate descent with simulated annealing
    def optimize_quadrant(quadrant, time_limit):
        """
        Optimize a quadrant using coordinate descent with simulated annealing.
        Flips individual pixels to improve the gap.
        """
        q = [row[:] for row in quadrant]
        cell = quadrant_to_cell(q)
        cell_arr = np.array(cell, dtype=np.float64)
        current_score = compute_tm_gap(cell_arr)
        best_score = current_score
        best_q = [row[:] for row in q]
        
        # Precompute which pixels to try (all 256 in the quadrant)
        pixel_indices = [(i, j) for i in range(half) for j in range(half)]
        
        T = 0.05  # Initial temperature
        T_min = 0.001
        cooling = 0.995
        
        while time.time() < time_limit:
            # Shuffle pixels for this iteration
            random.shuffle(pixel_indices)
            
            improved_this_round = False
            for (i, j) in pixel_indices:
                if time.time() > time_limit:
                    break
                
                # Flip pixel
                old_val = q[i][j]
                q[i][j] = 1 - old_val
                cell = quadrant_to_cell(q)
                cell_arr = np.array(cell, dtype=np.float64)
                new_score = compute_tm_gap(cell_arr)
                
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_q = [row[:] for row in q]
                    improved_this_round = True
                else:
                    # Revert
                    q[i][j] = old_val
            
            # Cool down
            T = max(T_min, T * cooling)
            
            if T <= T_min and not improved_this_round:
                break
        
        return best_q, best_score
    
    # Main optimization loop
    random.seed(123)
    
    # Create initial structures
    structures = create_structures()
    
    # Evaluate all initial structures
    best_score = -float('inf')
    best_quadrant = None
    
    for q in structures:
        if time.time() > deadline - 30:
            break
        cell = quadrant_to_cell(q)
        cell_arr = np.array(cell, dtype=np.float64)
        score = compute_tm_gap(cell_arr)
        if score > best_score:
            best_score = score
            best_quadrant = [row[:] for row in q]
    
    if best_quadrant is None:
        # Fallback: return a simple structure
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(half):
                x, y = i + 0.5, j + 0.5
                if (x-7.5)**2 + (y-7.5)**2 < 4.5**2:
                    q[i][j] = 1
        best_quadrant = q
        best_score = 0.0
    
    # Optimize from the best initial structure
    if time.time() < deadline - 30:
        opt_time = deadline - 30
        best_quadrant, best_score = optimize_quadrant(best_quadrant, opt_time)
    
    # If time remains, try optimizing from other initial structures
    if time.time() < deadline - 20:
        for q in structures:
            if time.time() > deadline - 20:
                break
            # Skip if already optimized
            if q == best_quadrant:
                continue
            cell = quadrant_to_cell(q)
            cell_arr = np.array(cell, dtype=np.float64)
            score = compute_tm_gap(cell_arr)
            if score < best_score - 0.1:
                continue
            # This structure is competitive; optimize it
            opt_time = deadline - 20
            opt_q, opt_score = optimize_quadrant(q, opt_time)
            if opt_score > best_score:
                best_score = opt_score
                best_quadrant = opt_q
    
    # Final local refinement: try all single-pixel flips
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(half):
                if time.time() > deadline - 5:
                    break
                for j in range(half):
                    if time.time() > deadline - 5:
                        break
                    q = [row[:] for row in best_quadrant]
                    q[i][j] = 1 - q[i][j]
                    cell = quadrant_to_cell(q)
                    cell_arr = np.array(cell, dtype=np.float64)
                    score = compute_tm_gap(cell_arr)
                    if score > best_score:
                        best_score = score
                        best_quadrant = q
                        improved = True
    
    # Convert to final cell
    best_cell = quadrant_to_cell(best_quadrant)
    
    # Verify symmetry
    for i in range(size):
        for j in range(size):
            if best_cell[i][j] != best_cell[size-1-i][j]:
                raise ValueError("Row symmetry failed")
            if best_cell[i][j] != best_cell[i][size-1-j]:
                raise ValueError("Col symmetry failed")
            if best_cell[i][j] != best_cell[j][i]:
                raise ValueError("Transpose symmetry failed")
    
    return best_cell
# EVOLVE-BLOCK-END
