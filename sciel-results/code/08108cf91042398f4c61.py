"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def design(problem):
    """
    Search over symmetric photonic crystal designs using a fast plane-wave proxy
    to maximize the TM band-4/5 gap ratio.
    """
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    eps_low = problem["eps_low"]
    
    deadline = time.time() + 100  # Leave margin before 230s kill
    
    def enforce_d4_symmetry(cell):
        """Enforce D4 symmetry: cell == cell[::-1] row-wise, col-wise, and transpose."""
        N = len(cell)
        # Average with all 8 D4 transformations to enforce symmetry
        result = np.zeros((N, N), dtype=float)
        transforms = []
        
        # Identity
        transforms.append(np.array(cell))
        # Transpose
        transforms.append(np.array(cell).T)
        # Reverse rows
        transforms.append(np.array(cell)[::-1, :])
        # Reverse cols
        transforms.append(np.array(cell)[:, ::-1])
        # Both reversed
        transforms.append(np.array(cell)[::-1, ::-1])
        # Transpose of reversed rows
        transforms.append(np.array(cell)[::-1, :].T)
        # Transpose of reversed cols
        transforms.append(np.array(cell)[:, ::-1].T)
        # Transpose of both reversed
        transforms.append(np.array(cell)[::-1, ::-1].T)
        
        for t in transforms:
            result += t
        
        result /= 8.0
        return (result > 0.5).astype(int)
    
    def compute_fourier_coefficients(cell, N_basis=17):
        """Compute Fourier coefficients of epsilon for plane-wave expansion."""
        N = len(cell)
        eps = np.array(cell, dtype=float)
        # epsilon = eps_low + (eps_high - eps_low) * cell
        eps_vals = eps_low + (eps_high - eps_low) * eps
        
        # Compute 2D FFT to get Fourier coefficients
        # For a 32x32 grid, we want coefficients up to order N_basis
        fft2d = np.fft.fft2(eps_vals)
        fft2d = np.fft.fftshift(fft2d)
        
        # Get the relevant Fourier coefficients
        coeffs = {}
        for mx in range(-N_basis, N_basis + 1):
            for my in range(-N_basis, N_basis + 1):
                # Map to FFT index
                ix = (mx + N // 2) % N
                iy = (my + N // 2) % N
                coeffs[(mx, my)] = fft2d[iy, ix] / (N * N)
        
        return coeffs
    
    def plane_wave_gap(coeffs, n_pw=13, n_k=10):
        """
        Fast plane-wave calculation of the band-4/5 gap ratio.
        Uses a reduced basis for speed.
        """
        # Generate plane-wave vectors
        pw_vecs = []
        for mx in range(-n_pw, n_pw + 1):
            for my in range(-n_pw, n_pw + 1):
                if mx * mx + my * my <= (n_pw + 0.5) ** 2:
                    pw_vecs.append((mx, my))
        
        n_pw_actual = len(pw_vecs)
        
        # Generate k-points in the irreducible BZ (1/8 of full BZ for square lattice)
        k_points = []
        for i in range(n_k):
            for j in range(n_k - i):
                kx = i / (2 * n_k)
                ky = j / (2 * n_k)
                # Irreducible BZ: 0 <= kx <= ky, kx + ky <= 0.5
                if kx <= ky and kx + ky <= 0.5 + 1e-10:
                    k_points.append((kx, ky))
        
        max_omega4 = 0.0
        min_omega5 = float('inf')
        
        for kx, ky in k_points:
            # Build the TM Hamiltonian matrix
            # H[p, q] = (G_p + k) . (G_q + k) / epsilon(G_p - G_q)
            H = np.zeros((n_pw_actual, n_pw_actual), dtype=float)
            
            for p in range(n_pw_actual):
                Gpx, Gpy = pw_vecs[p]
                kpx = Gpx + kx
                kpy = Gpy + ky
                for q in range(n_pw_actual):
                    Gqx, Gqy = pw_vecs[q]
                    kqx = Gqx + kx
                    kqy = Gqy + ky
                    Gdiff = (Gpx - Gqx, Gpy - Gqy)
                    eps_coeff = coeffs.get(Gdiff, 1.0)
                    if abs(eps_coeff) < 1e-10:
                        eps_coeff = 1.0
                    H[p, q] = (kpx * kqx + kpy * kqy) / eps_coeff
            
            # Solve eigenvalue problem
            try:
                eigenvalues = np.linalg.eigvalsh(H)
                eigenvalues = np.sort(eigenvalues)
                
                # Get bands 4 and 5 (0-indexed: 3 and 4)
                if len(eigenvalues) >= 5:
                    omega4 = math.sqrt(eigenvalues[3])
                    omega5 = math.sqrt(eigenvalues[4])
                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
            except:
                continue
        
        if min_omega5 <= max_omega4:
            return -1.0  # Overlapping bands
        
        gap_ratio = 2 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
        return gap_ratio
    
    def generate_rod_config(centers, radii):
        """Generate a cell from rod centers and radii."""
        cell = np.zeros((size, size), dtype=int)
        for (cx, cy), r in zip(centers, radii):
            for i in range(size):
                for j in range(size):
                    dx = (i + 0.5) - cx
                    dy = (j + 0.5) - cy
                    if dx * dx + dy * dy < r * r:
                        cell[i, j] = 1
        return cell
    
    def generate_ring_config(cx, cy, r_outer, r_inner):
        """Generate a ring (annulus) configuration with D4 symmetry."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                # Check all 8 symmetric positions
                for (dx, dy) in [(0, 0), (0, size), (size, 0), (size, size),
                                  (0, 0), (0, size), (size, 0), (size, size)]:
                    px = (i + 0.5)
                    py = (j + 0.5)
                    # Map to centered coordinates
                    cx_c = cx - size / 2.0
                    cy_c = cy - size / 2.0
                    dx_c = px - size / 2.0 - cx_c
                    dy_c = py - size / 2.0 - cy_c
                    dist_sq = dx_c * dx_c + dy_c * dy_c
                    if r_inner * r_inner <= dist_sq < r_outer * r_outer:
                        cell[i, j] = 1
        return cell
    
    def generate_cross_config(arm_width, arm_length):
        """Generate a cross pattern with D4 symmetry."""
        cell = np.zeros((size, size), dtype=int)
        center = size / 2.0
        for i in range(size):
            for j in range(size):
                dx = abs(i + 0.5 - center)
                dy = abs(j + 0.5 - center)
                # Vertical arm
                if dx < arm_width / 2.0 and dy < arm_length / 2.0:
                    cell[i, j] = 1
                # Horizontal arm
                elif dy < arm_width / 2.0 and dx < arm_length / 2.0:
                    cell[i, j] = 1
        return cell
    
    def generate_multi_rod_config(n_rods, radius, spacing):
        """Generate multiple rods arranged symmetrically."""
        cell = np.zeros((size, size), dtype=int)
        center = size / 2.0
        
        if n_rods == 4:
            # 2x2 grid of rods
            offsets = [(-spacing/2, -spacing/2), (-spacing/2, spacing/2),
                       (spacing/2, -spacing/2), (spacing/2, spacing/2)]
        elif n_rods == 8:
            # 3x3 grid minus center
            offsets = []
            for dx in [-spacing, 0, spacing]:
                for dy in [-spacing, 0, spacing]:
                    if dx != 0 or dy != 0:
                        offsets.append((dx, dy))
        else:
            offsets = [(0, 0)]
        
        for ox, oy in offsets:
            cx = center + ox
            cy = center + oy
            for i in range(size):
                for j in range(size):
                    dx = (i + 0.5) - cx
                    dy = (j + 0.5) - cy
                    if dx * dx + dy * dy < radius * radius:
                        cell[i, j] = 1
        return cell
    
    # Candidate designs to evaluate
    candidates = []
    
    # 1. Single central rod with varying radius
    for r in [3.0, 4.0, 5.0, 6.0, 7.0]:
        cell = generate_rod_config([(16.5, 16.5)], [r])
        candidates.append(cell)
    
    # 2. 2x2 rod grid with varying parameters
    for r in [3.0, 4.0, 4.5, 5.0]:
        for spacing in [8.0, 10.0, 12.0]:
            cell = generate_multi_rod_config(4, r, spacing)
            candidates.append(cell)
    
    # 3. 8-rod configuration
    for r in [2.5, 3.0, 3.5]:
        for spacing in [6.0, 8.0, 10.0]:
            cell = generate_multi_rod_config(8, r, spacing)
            candidates.append(cell)
    
    # 4. Ring configurations
    for r_out in [5.0, 6.0, 7.0]:
        for r_in in [2.0, 3.0, 4.0]:
            if r_in < r_out:
                cell = generate_ring_config(16.5, 16.5, r_out, r_in)
                candidates.append(cell)
    
    # 5. Cross configurations
    for width in [3.0, 4.0, 5.0]:
        for length in [8.0, 10.0, 12.0]:
            cell = generate_cross_config(width, length)
            candidates.append(cell)
    
    # 6. The original 4-rod design
    cell = generate_rod_config([(8.5, 8.5), (8.5, 23.5), (23.5, 8.5), (23.5, 23.5)], [4.5, 4.5, 4.5, 4.5])
    candidates.append(cell)
    
    # Evaluate candidates with the proxy
    best_cell = None
    best_score = -float('inf')
    
    for cell in candidates:
        if time.time() > deadline:
            break
        
        cell_sym = enforce_d4_symmetry(cell)
        coeffs = compute_fourier_coefficients(cell_sym)
        score = plane_wave_gap(coeffs)
        
        if score > best_score:
            best_score = score
            best_cell = cell_sym
    
    # Local refinement around the best candidate
    if best_cell is not None and time.time() < deadline - 20:
        best_cell_arr = np.array(best_cell)
        
        # Try small perturbations
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            
            # Randomly flip a few pixels while maintaining symmetry
            perturbed = best_cell_arr.copy()
            for _ in range(3):
                i = np.random.randint(0, size)
                j = np.random.randint(0, size)
                # Flip the pixel and all its D4 symmetric counterparts
                positions = [(i, j), (j, i), (size-1-i, j), (j, size-1-i),
                             (size-1-i, size-1-j), (size-1-j, size-1-i),
                             (i, size-1-j), (size-1-j, i)]
                val = 1 - perturbed[i, j]
                for pi, pj in positions:
                    if 0 <= pi < size and 0 <= pj < size:
                        perturbed[pi, pj] = val
            
            perturbed_sym = enforce_d4_symmetry(perturbed)
            coeffs = compute_fourier_coefficients(perturbed_sym)
            score = plane_wave_gap(coeffs)
            
            if score > best_score:
                best_score = score
                best_cell = perturbed_sym
    
    # Convert to nested list
    if best_cell is None:
        # Fallback to original design
        best_cell = generate_rod_config([(8.5, 8.5), (8.5, 23.5), (23.5, 8.5), (23.5, 23.5)], [4.5, 4.5, 4.5, 4.5])
        best_cell = enforce_d4_symmetry(best_cell)
    
    return best_cell.tolist()
# EVOLVE-BLOCK-END
