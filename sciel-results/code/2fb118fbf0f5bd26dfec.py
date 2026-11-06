"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def design(problem):
    """Photonic crystal unit cell design for large TM 4-5 band gap.
    
    Uses a fast plane-wave TM eigenvalue solver to directly evaluate
    candidate cells, combined with diverse parametric families and
    local pixel mutations to maximize the gap-midgap ratio.
    """
    t_start = time.time()
    DEADLINE = 215.0  # safety margin before 230s kill
    
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    eps_low = problem["eps_low"]
    
    # --- Plane-wave TM solver ---
    # For TM polarization, the eigenvalue problem is:
    # (k + G)^2 * eps^{-1}(G' - G) * c_{G'} = omega^2 * c_G
    # where we use the exact Fourier coefficients of 1/eps.
    
    def compute_eps_inv_fft(cell):
        """Compute Fourier coefficients of 1/eps for TM mode."""
        f = cell.astype(np.float64)  # 1 where dielectric, 0 where air
        eps_inv = np.where(f > 0.5, 1.0 / eps_high, 1.0 / eps_low)
        # Compute 2D FFT and normalize
        F = np.fft.fft2(eps_inv)
        F /= (size * size)
        return F
    
    def setup_plane_waves(npw):
        """Generate plane wave vectors G within radius npw."""
        # npw = number of G-vectors in each dimension (total ~ (2*npw+1)^2)
        # Use a circular cutoff for better accuracy
        G_list = []
        for gx in range(-npw, npw + 1):
            for gy in range(-npw, npw + 1):
                if gx*gx + gy*gy <= npw*npw:
                    G_list.append((gx, gy))
        G_arr = np.array(G_list, dtype=np.float64)
        return G_arr
    
    def tm_eigenvalues(G_arr, eps_inv_F, kx, ky):
        """Solve TM eigenvalue problem at wavevector (kx, ky).
        
        Returns sorted eigenvalues (omega^2) for all bands.
        """
        n = len(G_arr)
        # Build the matrix: M[G, G'] = (k+G)^2 * eps_inv_F[G'-G]
        # eps_inv_F[m, n] corresponds to Fourier component (m, n)
        # We need eps_inv_F at indices corresponding to (G'_x - G_x, G'_y - G_y)
        
        # Precompute (k+G)^2
        kxG = kx + G_arr[:, 0]
        kyG = ky + G_arr[:, 1]
        kG_sq = kxG**2 + kyG**2
        
        # Build matrix
        M = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                dgx = G_arr[j, 0] - G_arr[i, 0]
                dgy = G_arr[j, 1] - G_arr[i, 1]
                # eps_inv_F is indexed with fft2 convention
                # F[m, n] = sum_{x,y} f(x,y) exp(-2pi i (mx + ny)/N)
                # We need F[dgx, dgy] but in fft2, negative indices wrap
                mx = int(dgx) % size
                my = int(dgy) % size
                M[i, j] = kG_sq[i] * eps_inv_F[mx, my].real
        
        # Solve generalized eigenvalue problem (but M is already the matrix)
        # Actually for TM: sum_G' eps_inv(G-G') (k+G')^2 c_{G'} = omega^2 c_G
        # So the matrix is: A[G, G'] = eps_inv(G - G') * (k + G')^2
        # Let me redo:
        M = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                dgx = G_arr[i, 0] - G_arr[j, 0]
                dgy = G_arr[i, 1] - G_arr[j, 1]
                mx = int(dgx) % size
                my = int(dgy) % size
                M[i, j] = eps_inv_F[mx, my].real * kG_sq[j]
        
        # Solve eigenvalue problem
        eigvals = np.linalg.eigvalsh(M)
        return eigvals
    
    def compute_band_ratio(cell, G_arr, n_bands=6, n_kpts=12):
        """Compute the gap-midgap ratio for TM bands 4-5.
        
        Evaluates over several k-points in the irreducible BZ.
        Returns the ratio 2*(min omega_5 - max omega_4) / (min omega_5 + max omega_4).
        """
        eps_inv_F = compute_eps_inv_fft(cell)
        
        # Irreducible BZ for square lattice: triangle from (0,0) to (0.5,0) to (0.5,0.5)
        # Sample points along the boundary and interior
        k_points = []
        # Along X: (0,0) to (0.5, 0)
        for t in np.linspace(0, 0.5, 4):
            k_points.append((t, 0.0))
        # Along M: (0.5, 0) to (0.5, 0.5)
        for t in np.linspace(0, 0.5, 4):
            k_points.append((0.5, t))
        # Along X-M: (0,0) to (0.5, 0.5)
        for t in np.linspace(0, 0.5, 4):
            k_points.append((t, t))
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in k_points:
            eigvals = tm_eigenvalues(G_arr, eps_inv_F, kx, ky)
            # eigvals are omega^2, take sqrt
            omegas = np.sqrt(np.maximum(eigvals, 0))
            if len(omegas) >= n_bands:
                omega_4 = omegas[3]  # 0-indexed: band 4 is index 3
                omega_5 = omegas[4]  # band 5 is index 4
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0
        
        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap <= 0:
            return -1.0
        return 2.0 * gap / midgap
    
    # --- Cell generation functions ---
    
    def apply_symmetry(cell):
        """Enforce full square symmetry: row-reverse, col-reverse, transpose."""
        cell = cell.copy()
        # Row-reverse symmetry: cell[i][j] == cell[N-1-i][j]
        for i in range(size // 2):
            for j in range(size):
                avg = (cell[i, j] + cell[size-1-i, j]) / 2.0
                cell[i, j] = cell[size-1-i, j] = 1 if avg >= 0.5 else 0
        # Col-reverse symmetry: cell[i][j] == cell[i][N-1-j]
        for i in range(size):
            for j in range(size // 2):
                avg = (cell[i, j] + cell[i, size-1-j]) / 2.0
                cell[i, j] = cell[i, size-1-j] = 1 if avg >= 0.5 else 0
        # Transpose symmetry: cell[i][j] == cell[j][i]
        for i in range(size):
            for j in range(i + 1, size):
                avg = (cell[i, j] + cell[j, i]) / 2.0
                cell[i, j] = cell[j, i] = 1 if avg >= 0.5 else 0
        return cell
    
    def make_four_rod_cell(radius, offset, n_rings=1):
        """Four dielectric rods at (±offset, ±offset)."""
        cell = np.zeros((size, size), dtype=np.int8)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                for dx, dy in [(offset, offset), (offset, -offset), (-offset, offset), (-offset, -offset)]:
                    if (x - dx)**2 + (y - dy)**2 < radius**2:
                        cell[i, j] = 1
                        break
        return apply_symmetry(cell)
    
    def make_centered_ring_cell(r_inner, r_outer):
        """Central ring of dielectric."""
        cell = np.zeros((size, size), dtype=np.int8)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                r = np.sqrt(x*x + y*y)
                if r_inner < r < r_outer:
                    cell[i, j] = 1
        return apply_symmetry(cell)
    
    def make_cross_cell(arm_width, arm_length):
        """Cross pattern: horizontal and vertical bars through center."""
        cell = np.zeros((size, size), dtype=np.int8)
        hw = arm_width / 2.0
        hl = arm_length / 2.0
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                # Vertical bar
                if abs(x) < hw and abs(y) < hl:
                    cell[i, j] = 1
                # Horizontal bar
                elif abs(y) < hw and abs(x) < hl:
                    cell[i, j] = 1
        return apply_symmetry(cell)
    
    def make_staggered_cell(r1, r2, off1, off2):
        """Two sets of rods at different positions."""
        cell = np.zeros((size, size), dtype=np.int8)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                # Set 1: four rods
                for dx, dy in [(off1, off1), (off1, -off1), (-off1, off1), (-off1, -off1)]:
                    if (x - dx)**2 + (y - dy)**2 < r1**2:
                        cell[i, j] = 1
                        break
                # Set 2: four rods at different offset
                for dx, dy in [(off2, off2), (off2, -off2), (-off2, off2), (-off2, -off2)]:
                    if (x - dx)**2 + (y - dy)**2 < r2**2:
                        cell[i, j] = 1
                        break
        return apply_symmetry(cell)
    
    def mutate_cell(cell, n_mutations=4):
        """Randomly flip some pixels while maintaining symmetry."""
        cell = cell.copy()
        # Symmetry orbit: for a pixel (i,j), the orbit under D4 is
        # {(i,j), (i,N-1-j), (N-1-i,j), (N-1-i,N-1-j), (j,i), (j,N-1-i), (N-1-i,j), (N-1-i,N-1-i)}
        # Simplified: pick random (i,j) in top-left quadrant and flip its orbit
        for _ in range(n_mutations):
            i = random.randint(0, size // 2 - 1)
            j = random.randint(0, size // 2 - 1)
            # Flip the full orbit
            orbit = [(i, j), (i, size-1-j), (size-1-i, j), (size-1-i, size-1-j),
                     (j, i), (j, size-1-i), (size-1-i, j), (size-1-i, size-1-i)]
            new_val = 1 - cell[i, j]
            for (oi, oj) in orbit:
                cell[oi, oj] = new_val
        return cell
    
    # --- Main search ---
    
    # Use a moderate number of plane waves for speed
    # npw=3 gives 29 G-vectors, npw=4 gives 49, npw=5 gives 81
    G_arr = setup_plane_waves(3)
    
    best_score = -np.inf
    best_cell = None
    
    def evaluate_cell(cell):
        nonlocal best_score, best_cell
        if time.time() > DEADLINE:
            return
        score = compute_band_ratio(cell, G_arr)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
        return score
    
    # Phase 1: Evaluate diverse parametric families
    print(f"Phase 1: Evaluating parametric families... (G-vectors: {len(G_arr)})")
    
    # Four-rod family
    for r in [0.08, 0.10, 0.12, 0.14, 0.16, 0.18]:
        if time.time() > DEADLINE:
            break
        for d in [0.08, 0.12, 0.16, 0.20, 0.24]:
            if time.time() > DEADLINE:
                break
            cell = make_four_rod_cell(r, d)
            evaluate_cell(cell)
    
    # Centered ring family
    for ri in [0.05, 0.08, 0.10, 0.12]:
        if time.time() > DEADLINE:
            break
        for ro in [0.18, 0.22, 0.26, 0.30]:
            if time.time() > DEADLINE:
                break
            cell = make_centered_ring_cell(ri, ro)
            evaluate_cell(cell)
    
    # Cross family
    for aw in [0.06, 0.08, 0.10, 0.12]:
        if time.time() > DEADLINE:
            break
        for al in [0.30, 0.35, 0.40, 0.45]:
            if time.time() > DEADLINE:
                break
            cell = make_cross_cell(aw, al)
            evaluate_cell(cell)
    
    # Staggered family
    for r1 in [0.06, 0.08, 0.10]:
        if time.time() > DEADLINE:
            break
        for r2 in [0.06, 0.08, 0.10]:
            if time.time() > DEADLINE:
                break
            for o1 in [0.10, 0.15, 0.20]:
                if time.time() > DEADLINE:
                    break
                for o2 in [0.10, 0.15, 0.20]:
                    if time.time() > DEADLINE:
                        break
                    if abs(o1 - o2) < 0.01:
                        continue
                    cell = make_staggered_cell(r1, r2, o1, o2)
                    evaluate_cell(cell)
    
    # Phase 2: Local search around best
    if best_cell is not None and time.time() < DEADLINE - 30:
        print(f"Phase 2: Local search around best (score={best_score:.4f})...")
        current = best_cell.copy()
        current_score = best_score
        
        for iteration in range(200):
            if time.time() > DEADLINE:
                break
            mutated = mutate_cell(current, n_mutations=random.randint(1, 3))
            score = evaluate_cell(mutated)
            if score is not None and score > current_score:
                current = mutated
                current_score = score
    
    # Phase 3: Try more diverse random starts if time permits
    if time.time() < DEADLINE - 60:
        print(f"Phase 3: Random diverse starts...")
        for _ in range(30):
            if time.time() > DEADLINE:
                break
            # Random symmetric cell
            cell = np.zeros((size, size), dtype=np.int8)
            for i in range(size // 2):
                for j in range(size // 2):
                    if random.random() < 0.3:
                        cell[i, j] = 1
            cell = apply_symmetry(cell)
            evaluate_cell(cell)
            
            # Local refinement
            if best_cell is not None:
                for _ in range(20):
                    if time.time() > DEADLINE:
                        break
                    mutated = mutate_cell(best_cell, n_mutations=1)
                    evaluate_cell(mutated)
    
    # Fallback
    if best_cell is None:
        best_cell = make_four_rod_cell(0.12, 0.16)
    
    # Final symmetry enforcement
    best_cell = apply_symmetry(best_cell)
    
    # Convert to nested list of ints
    result = [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]
    
    return result
# EVOLVE-BLOCK-END
