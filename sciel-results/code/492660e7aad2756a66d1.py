"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
import random

def design(problem):
    """
    Design a 32x32 photonic crystal unit cell with D4 symmetry to maximize
    the gap between TM bands 4 and 5.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 200  # Safety margin for 230s kill limit

    # Precompute D4 symmetry orbits
    # For a pixel (i, j), its orbit under D4 (symmetric about center)
    # Size is even (32). Center is between (15,15), (15,16), (16,15), (16,16).
    # Symmetries:
    # 1. Identity: (i, j)
    # 2. Rot 90: (j, size-1-i)
    # 3. Rot 180: (size-1-i, size-1-j)
    # 4. Rot 270: (size-1-j, i)
    # 5. Refl X: (size-1-i, j)
    # 6. Refl Y: (i, size-1-j)
    # 7. Refl Diag: (j, i)
    # 8. Refl AntiDiag: (size-1-j, size-1-i)
    
    orbits = {}
    orbit_list = []
    for i in range(size):
        for j in range(size):
            if (i, j) in orbits:
                continue
            orbit = set()
            orbit.add((i, j))
            orbit.add((j, size-1-i))
            orbit.add((size-1-i, size-1-j))
            orbit.add((size-1-j, i))
            orbit.add((size-1-i, j))
            orbit.add((i, size-1-j))
            orbit.add((j, i))
            orbit.add((size-1-j, size-1-i))
            
            orbit_list.append(list(orbit))
            for p in orbit:
                orbits[p] = len(orbit_list) - 1

    def enforce_d4_symmetry(cell):
        """
        Enforce D4 symmetry by averaging the values in each orbit.
        Input: 32x32 list of 0/1 or floats.
        Output: 32x32 list of 0/1.
        """
        # Convert to numpy for speed
        arr = np.array(cell, dtype=float)
        final_arr = np.zeros((size, size), dtype=int)
        
        for orbit in orbit_list:
            # Get values
            vals = [arr[p[0], p[1]] for p in orbit]
            avg = np.mean(vals)
            val = 1 if avg >= 0.5 else 0
            for p in orbit:
                final_arr[p[0], p[1]] = val
                
        return final_arr.tolist()

    def compute_fourier_coefficients(cell, max_order):
        """
        Compute Fourier coefficients of 1/epsilon.
        cell: 32x32 list of 0/1.
        max_order: max G vector index (e.g., 5 means -5..5).
        Returns dict {(m, n): coeff}
        """
        eps = np.array([[eps_high if cell[i][j] == 1 else eps_low for j in range(size)] for i in range(size)])
        inv_eps = 1.0 / eps
        # FFT of inv_eps
        fft_2d = np.fft.fft2(inv_eps)
        coeffs = {}
        for m in range(-max_order, max_order + 1):
            for n in range(-max_order, max_order + 1):
                m_idx = m % size
                n_idx = n % size
                coeffs[(m, n)] = fft_2d[m_idx, n_idx] / (size * size)
        return coeffs

    def approximate_gap_ratio(cell, max_order=6, g_range=6, k_points_count=10):
        """
        Approximate the gap ratio between bands 4 and 5.
        Uses plane-wave expansion.
        """
        coeffs = compute_fourier_coefficients(cell, max_order)
        
        # G vectors
        g_vectors = [(m, n) for m in range(-g_range, g_range + 1) for n in range(-g_range, g_range + 1)]
        N = len(g_vectors)
        
        # K-points in irreducible BZ: Gamma (0,0) -> X (0.5, 0) -> M (0.5, 0.5)
        # We sample points along these segments.
        k_points = []
        for t in np.linspace(0, 1, k_points_count):
            k_points.append((t * 0.5, 0.0)) # Gamma to X
        for t in np.linspace(0, 1, k_points_count):
            k_points.append((0.5, t * 0.5)) # X to M
            
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        # Precompute k_sq for each G? No, k depends on K.
        # H[i, j] = |K + G_i|^2 * eps_inv(G_i - G_j)
        
        for kx, ky in k_points:
            # Build H matrix
            H = np.zeros((N, N), dtype=complex)
            for i, Gp in enumerate(g_vectors):
                kx_total = kx + Gp[0]
                ky_total = ky + Gp[1]
                k_sq = kx_total**2 + ky_total**2
                for j, G in enumerate(g_vectors):
                    m_diff = Gp[0] - G[0]
                    n_diff = Gp[1] - G[1]
                    eps_inv_G = coeffs.get((m_diff, n_diff), 0.0)
                    H[i, j] = k_sq * eps_inv_G
            
            # Solve eigenvalue problem
            # H is Hermitian
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals.real)
            
            if len(eigvals) >= 5:
                # TM bands. We want bands 4 and 5 (0-indexed 3 and 4).
                # Note: In photonic crystals, there are usually multiple bands per k-point.
                # The lowest band is 0.
                # We need the 4th and 5th bands.
                # However, due to symmetry, bands might be degenerate.
                # The evaluator sorts all bands.
                # So we just take index 3 and 4.
                omega_4 = eigvals[3]
                omega_5 = eigvals[4]
                
                if omega_5 < min_omega_5:
                    min_omega_5 = omega_5
                if omega_4 > max_omega_4:
                    max_omega_4 = omega_4
        
        if min_omega_5 == float('inf') or max_omega_4 <= 0:
            return -1.0
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap <= 0:
            return -1.0
        return gap / midgap

    def generate_single_rod(radius_frac):
        """Central circular rod."""
        cell = [[0] * size for _ in range(size)]
        r_px = radius_frac * size
        cx = size / 2.0
        cy = size / 2.0
        for i in range(size):
            for j in range(size):
                x = (j + 0.5) - cx
                y = (i + 0.5) - cy
                if x * x + y * y < r_px * r_px:
                    cell[i][j] = 1
        return cell

    def generate_ring(inner_frac, outer_frac):
        """Circular ring (dielectric annulus)."""
        cell = [[0] * size for _ in range(size)]
        r_in = inner_frac * size
        r_out = outer_frac * size
        cx = size / 2.0
        cy = size / 2.0
        for i in range(size):
            for j in range(size):
                x = (j + 0.5) - cx
                y = (i + 0.5) - cy
                r2 = x * x + y * y
                if r_in * r_in <= r2 < r_out * r_out:
                    cell[i][j] = 1
        return cell

    def generate_four_rods(radius_frac):
        """Four circular rods at corners (D4 symmetric)."""
        cell = [[0] * size for _ in range(size)]
        r_px = radius_frac * size
        # Rod centers at (size/4, size/4) etc.
        centers = [(size/4.0, size/4.0), (size*3/4.0, size/4.0),
                   (size/4.0, size*3/4.0), (size*3/4.0, size*3/4.0)]
        for i in range(size):
            for j in range(size):
                for cx, cy in centers:
                    x = (j + 0.5) - cx
                    y = (i + 0.5) - cy
                    if x * x + y * y < r_px * r_px:
                        cell[i][j] = 1
                        break
        return cell

    def generate_cross(arm_width_frac, arm_length_frac):
        """Cross shape (plus sign)."""
        cell = [[0] * size for _ in range(size)]
        w = max(1, int(arm_width_frac * size))
        l = int(arm_length_frac * size)
        cx = size // 2
        cy = size // 2
        half_w = w // 2
        for i in range(size):
            for j in range(size):
                dx = abs(j - cx)
                dy = abs(i - cy)
                if (dx <= half_w and dy <= l) or (dy <= half_w and dx <= l):
                    cell[i][j] = 1
        return cell

    def generate_checkerboard_block(block_size):
        """Block pattern with D4 symmetry."""
        cell = [[0] * size for _ in range(size)]
        bs = block_size
        for i in range(size):
            for j in range(size):
                bi = i // bs
                bj = j // bs
                if (bi + bj) % 2 == 0:
                    cell[i][j] = 1
        return cell

    def generate_concentric_rings(radii_fracs):
        """Multiple concentric rings."""
        cell = [[0] * size for _ in range(size)]
        cx = size / 2.0
        cy = size / 2.0
        for i in range(size):
            for j in range(size):
                x = (j + 0.5) - cx
                y = (i + 0.5) - cy
                r2 = x * x + y * y
                for k in range(len(radii_fracs) - 1):
                    r_in = radii_fracs[k] * size
                    r_out = radii_fracs[k + 1] * size
                    if r_in * r_in <= r2 < r_out * r_out:
                        cell[i][j] = 1
                        break
        return cell

    # 1. Generate Initial Candidates
    candidates = []
    
    # Single rods
    for r in [0.20, 0.25, 0.30, 0.35, 0.40]:
        candidates.append(generate_single_rod(r))
    
    # Rings
    for inner, outer in [(0.20, 0.40), (0.30, 0.50), (0.10, 0.30), (0.40, 0.60)]:
        candidates.append(generate_ring(inner, outer))
        
    # Four rods
    for r in [0.15, 0.20, 0.25, 0.30]:
        candidates.append(generate_four_rods(r))
        
    # Crosses
    for w, l in [(0.10, 0.40), (0.20, 0.40), (0.30, 0.40), (0.15, 0.30)]:
        candidates.append(generate_cross(w, l))
        
    # Checkerboards
    for bs in [4, 8, 16]:
        candidates.append(generate_checkerboard_block(bs))
        
    # Concentric rings
    for radii in [(0.0, 0.30), (0.20, 0.40, 0.60), (0.10, 0.30, 0.50)]:
        candidates.append(generate_concentric_rings(radii))

    # 2. Evaluate Candidates
    best_score = -1.0
    best_cell = candidates[0]
    
    for cell in candidates:
        if time.time() > deadline - 20:
            break
        cell_sym = enforce_d4_symmetry(cell)
        score = approximate_gap_ratio(cell_sym, max_order=5, g_range=5, k_points_count=5)
        if score > best_score:
            best_score = score
            best_cell = cell_sym

    # 3. Local Search (Hill Climbing with Random Restarts)
    # We will perform local search on the best candidate found so far.
    # To improve, we can also do local search on other top candidates if time permits.
    
    def local_search(cell, max_iter=500):
        """
        Local search by flipping orbits.
        """
        current_cell = cell
        current_score = approximate_gap_ratio(current_cell, max_order=5, g_range=5, k_points_count=5)
        best_local_cell = current_cell
        best_local_score = current_score
        
        # Temperature for Simulated Annealing
        T = 0.05
        T_min = 0.001
        
        for _ in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Pick a random orbit
            orbit_idx = random.randint(0, len(orbit_list) - 1)
            orbit = orbit_list[orbit_idx]
            
            # Flip all pixels in the orbit
            new_cell = [row[:] for row in current_cell]
            # Determine current value (should be uniform in orbit)
            val = current_cell[orbit[0][0]][orbit[0][1]]
            new_val = 1 - val
            for p in orbit:
                new_cell[p[0]][p[1]] = new_val
            
            new_score = approximate_gap_ratio(new_cell, max_order=5, g_range=5, k_points_count=5)
            
            # Acceptance criterion
            if new_score > current_score or (random.random() < math.exp((new_score - current_score) / T)):
                current_cell = new_cell
                current_score = new_score
                
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_cell = [row[:] for row in current_cell]
            
            # Cool down
            T *= 0.99
            if T < T_min:
                T = T_min
                
        return best_local_cell, best_local_score

    # Run local search on the best candidate
    if time.time() < deadline - 30:
        best_cell, best_score = local_search(best_cell, max_iter=1000)

    # If time remains, try local search on other good candidates
    # Re-evaluate candidates with higher precision to find other good seeds?
    # Or just run local search on the 2nd and 3rd best candidates.
    
    # Let's re-evaluate top candidates with slightly better surrogate to pick seeds
    # But for now, let's just do a few more random restarts local searches
    # starting from slightly perturbed best_cell.
    
    if time.time() < deadline - 20:
        for restart in range(3):
            if time.time() > deadline - 20:
                break
            # Perturb best_cell
            perturbed = [row[:] for row in best_cell]
            # Flip a few random orbits
            for _ in range(5):
                orbit_idx = random.randint(0, len(orbit_list) - 1)
                orbit = orbit_list[orbit_idx]
                val = perturbed[orbit[0][0]][orbit[0][1]]
                new_val = 1 - val
                for p in orbit:
                    perturbed[p[0]][p[1]] = new_val
            
            res_cell, res_score = local_search(perturbed, max_iter=500)
            if res_score > best_score:
                best_score = res_score
                best_cell = res_cell

    final_cell = enforce_d4_symmetry(best_cell)
    return final_cell

# EVOLVE-BLOCK-END
