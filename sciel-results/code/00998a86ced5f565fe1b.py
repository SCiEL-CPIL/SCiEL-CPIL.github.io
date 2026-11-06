"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def design(problem):
    """
    Design a 32x32 photonic crystal unit cell with D4 symmetry to maximize
    the gap between TM bands 4 and 5.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 200s limit

    def enforce_d4_symmetry(cell):
        """Enforce D4 symmetry: row/col reversal and transpose invariance."""
        final_cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                vals = [
                    cell[i][j],
                    cell[size-1-i][j],
                    cell[i][size-1-j],
                    cell[size-1-i][size-1-j],
                    cell[j][i],
                    cell[size-1-j][size-1-i],
                    cell[j][size-1-i],
                    cell[size-1-j][i],
                ]
                avg = sum(vals) / len(vals)
                final_cell[i][j] = 1 if avg >= 0.5 else 0
        return final_cell

    def compute_fourier_coefficients(cell, max_order):
        """Compute Fourier coefficients of 1/epsilon using FFT for speed."""
        eps = np.array([[eps_high if cell[i][j] == 1 else eps_low for j in range(size)] for i in range(size)])
        inv_eps = 1.0 / eps
        # Use 2D FFT to get all coefficients at once
        fft_2d = np.fft.fft2(inv_eps)
        coeffs = {}
        for m in range(-max_order, max_order + 1):
            for n in range(-max_order, max_order + 1):
                # Map negative indices to positive using FFT convention
                m_idx = m % size
                n_idx = n % size
                coeffs[(m, n)] = fft_2d[m_idx, n_idx] / (size * size)
        return coeffs

    def approximate_gap_ratio(cell, max_order=5, g_range=3):
        """
        Approximate the gap ratio between bands 4 and 5 using plane-wave expansion.
        """
        coeffs = compute_fourier_coefficients(cell, max_order)
        
        # G vectors in the basis
        g_vectors = [(m, n) for m in range(-g_range, g_range + 1) for n in range(-g_range, g_range + 1)]
        N = len(g_vectors)
        
        # K-points in irreducible BZ: (0,0) to (0.5,0) to (0.5,0.5)
        k_points = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.25, 0.25), (0.5, 0.5)]
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            H = np.zeros((N, N), dtype=complex)
            for i, Gp in enumerate(g_vectors):
                kx_total = kx + Gp[0]
                ky_total = ky + Gp[1]
                k_sq = kx_total**2 + ky_total**2
                for j, G in enumerate(g_vectors):
                    m_diff = Gp[0] - G[0]
                    n_diff = Gp[1] - G[1]
                    if (m_diff, n_diff) in coeffs:
                        eps_inv_G = coeffs[(m_diff, n_diff)]
                    else:
                        eps_inv_G = 0.0
                    H[i, j] = k_sq * eps_inv_G
            
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals.real)
            
            if len(eigvals) >= 5:
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

    def generate_square_pillar(half_width_frac):
        """Central square pillar - different Fourier spectrum than circle."""
        cell = [[0] * size for _ in range(size)]
        hw = half_width_frac * size
        cx = size / 2.0
        cy = size / 2.0
        for i in range(size):
            for j in range(size):
                x = abs((j + 0.5) - cx)
                y = abs((i + 0.5) - cy)
                if x < hw and y < hw:
                    cell[i][j] = 1
        return cell

    def generate_four_square_pillars(half_width_frac):
        """Four square pillars in a 2x2 arrangement."""
        cell = [[0] * size for _ in range(size)]
        hw = half_width_frac * size
        offset = size / 4.0
        cx_center = size / 2.0
        cy_center = size / 2.0
        centers = [
            (cx_center - offset, cy_center - offset),
            (cx_center + offset, cy_center - offset),
            (cx_center - offset, cy_center + offset),
            (cx_center + offset, cy_center + offset),
        ]
        for i in range(size):
            for j in range(size):
                x = (j + 0.5)
                y = (i + 0.5)
                for cx, cy in centers:
                    dx = abs(x - cx)
                    dy = abs(y - cy)
                    if dx < hw and dy < hw:
                        cell[i][j] = 1
                        break
        return cell

    def generate_polynomial_mask(coeffs):
        """
        Generate a D4-symmetric mask from a separable polynomial.
        coeffs is a list of 15 values for x^(2i)*y^(2j) with i,j in 0..4.
        Only even powers guarantee symmetry under x->-x and y->-y.
        """
        cell = [[0] * size for _ in range(size)]
        cx = size / 2.0
        cy = size / 2.0
        for i in range(size):
            for j in range(size):
                # Normalize coordinates to [-1, 1]
                x = ((j + 0.5) - cx) / (size / 2.0)
                y = ((i + 0.5) - cy) / (size / 2.0)
                
                # Evaluate polynomial P(x,y) = sum c_ij * x^(2i) * y^(2j)
                val = 0.0
                idx = 0
                for ii in range(5):
                    for jj in range(5):
                        val += coeffs[idx] * (x ** (2 * ii)) * (y ** (2 * jj))
                        idx += 1
                
                cell[i][j] = 1 if val > 0 else 0
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
        # Rod centers at (size/4, size/4) and symmetric positions
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
        w = int(arm_width_frac * size)
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
                # Map to block coordinates
                bi = i // bs
                bj = j // bs
                # D4 symmetric pattern: fill blocks where (bi+bj) is even
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

    # Generate candidate designs
    candidates = []
    
    # Single rods with various radii
    for r in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.33, 0.35]:
        candidates.append(generate_single_rod(r))
    
    # Square pillars - different Fourier spectrum
    for hw in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]:
        candidates.append(generate_square_pillar(hw))
    
    # Four square pillars in 2x2 arrangement
    for hw in [0.08, 0.10, 0.12, 0.15, 0.18]:
        candidates.append(generate_four_square_pillars(hw))
    
    # Rings
    for inner, outer in [(0.15, 0.30), (0.20, 0.35), (0.25, 0.40), (0.10, 0.25), (0.20, 0.30)]:
        candidates.append(generate_ring(inner, outer))
    
    # Four rods
    for r in [0.10, 0.12, 0.15, 0.18, 0.20]:
        candidates.append(generate_four_rods(r))
    
    # Crosses
    for w, l in [(0.15, 0.40), (0.20, 0.35), (0.25, 0.30), (0.15, 0.35)]:
        candidates.append(generate_cross(w, l))
    
    # Checkerboard blocks
    for bs in [4, 6, 8]:
        candidates.append(generate_checkerboard_block(bs))
    
    # Concentric rings
    for radii in [(0.0, 0.20), (0.0, 0.25, 0.40), (0.15, 0.30, 0.45)]:
        candidates.append(generate_concentric_rings(radii))
    
    # Polynomial-based masks with random initialization
    np.random.seed(42)
    for _ in range(20):
        # Initialize polynomial coefficients
        coeffs = np.random.randn(25) * 0.5
        # Ensure some structure by making constant term significant
        coeffs[0] += 1.0
        candidates.append(generate_polynomial_mask(coeffs.tolist()))
    
    # Evaluate all candidates
    best_score = -1.0
    best_cell = candidates[0]
    
    for cell in candidates:
        if time.time() - start_time > deadline:
            break
        cell_sym = enforce_d4_symmetry(cell)
        score = approximate_gap_ratio(cell_sym)
        if score > best_score:
            best_score = score
            best_cell = cell_sym
    
    # If time remains, try local perturbations around best design
    if time.time() - start_time < deadline - 30:
        base = best_cell
        # Try flipping individual pixels in D4-symmetric groups
        for _ in range(50):
            if time.time() - start_time > deadline:
                break
            # Pick a random position and flip its D4 orbit
            flip_i = np.random.randint(0, size)
            flip_j = np.random.randint(0, size)
            
            cell_trial = [row[:] for row in base]
            # Get all symmetric positions
            positions = set()
            positions.add((flip_i, flip_j))
            positions.add((size-1-flip_i, flip_j))
            positions.add((flip_i, size-1-flip_j))
            positions.add((size-1-flip_i, size-1-flip_j))
            positions.add((flip_j, flip_i))
            positions.add((size-1-flip_j, size-1-flip_i))
            positions.add((flip_j, size-1-flip_i))
            positions.add((size-1-flip_j, flip_i))
            
            for (pi, pj) in positions:
                cell_trial[pi][pj] = 1 - base[pi][pj]
            
            cell_sym = enforce_d4_symmetry(cell_trial)
            # Quick check: only evaluate if it's different from base
            if cell_sym != base:
                score = approximate_gap_ratio(cell_sym)
                if score > best_score:
                    best_score = score
                    best_cell = cell_sym
                    base = best_cell
    
    # Polynomial-based local search if time remains
    if time.time() - start_time < deadline - 20:
        # Start from a random polynomial
        np.random.seed(123)
        best_poly_score = -1.0
        best_poly_coeffs = None
        
        for _ in range(30):
            if time.time() - start_time > deadline:
                break
            coeffs = np.random.randn(25) * 0.3
            coeffs[0] += 0.8
            cell = generate_polynomial_mask(coeffs.tolist())
            cell_sym = enforce_d4_symmetry(cell)
            score = approximate_gap_ratio(cell_sym)
            if score > best_poly_score:
                best_poly_score = score
                best_poly_coeffs = coeffs.copy()
                best_poly_cell = cell_sym
        
        # Local refinement of best polynomial
        if best_poly_coeffs is not None:
            for _ in range(20):
                if time.time() - start_time > deadline:
                    break
                # Perturb one coefficient
                idx = np.random.randint(0, 25)
                delta = np.random.randn() * 0.1
                trial_coeffs = best_poly_coeffs.copy()
                trial_coeffs[idx] += delta
                cell = generate_polynomial_mask(trial_coeffs.tolist())
                cell_sym = enforce_d4_symmetry(cell)
                score = approximate_gap_ratio(cell_sym)
                if score > best_poly_score:
                    best_poly_score = score
                    best_poly_coeffs = trial_coeffs
                    best_poly_cell = cell_sym
            
            if best_poly_score > best_score:
                best_score = best_poly_score
                best_cell = best_poly_cell
    
    final_cell = enforce_d4_symmetry(best_cell)
    return final_cell

# EVOLVE-BLOCK-END
