"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START

def _make_symmetric_cell(size, quarter):
    """Build a 32x32 cell from a 16x16 top-left quarter, enforcing full square symmetry."""
    n = size // 2  # 16
    cell = []
    for i in range(size):
        ri = i if i < n else (2 * n - 1 - i)
        row = []
        for j in range(size):
            rj = j if j < n else (2 * n - 1 - j)
            row.append(quarter[ri][rj])
        cell.append(row)
    return cell

def _compute_1over_eps_fourier(cell, size, n_pw):
    """Compute Fourier coefficients of 1/epsilon for the pixel map."""
    half = n_pw // 2
    inv_eps_coeff = {}
    for mx in range(-half, half + 1):
        for my in range(-half, half + 1):
            s = 0.0
            for i in range(size):
                x = (i + 0.5) / size
                for j in range(size):
                    y = (j + 0.5) / size
                    eps = 12.0 if cell[i][j] == 1 else 1.0
                    inv_eps = 1.0 / eps
                    angle = 2.0 * math.pi * (mx * x + my * y)
                    s += inv_eps * (math.cos(angle) + 1j * math.sin(angle))
            inv_eps_coeff[(mx, my)] = s / (size * size)
    return inv_eps_coeff

def _band_structure_tm(inv_eps_coeff, n_pw, n_kpoints=30):
    """Compute TM band structure using plane-wave Galerkin method."""
    half = n_pw // 2
    G_vectors = []
    for mx in range(-half, half + 1):
        for my in range(-half, half + 1):
            G_vectors.append((2.0 * math.pi * mx, 2.0 * math.pi * my))
    
    n_pw_total = len(G_vectors)
    
    k_points = []
    for t in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        k_points.append((math.pi * t, 0.0))
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        k_points.append((math.pi, math.pi * t))
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        k_points.append((math.pi * (1 - t), math.pi * (1 - t)))
    
    bands = []
    
    for kx, ky in k_points:
        M = [[0.0] * n_pw_total for _ in range(n_pw_total)]
        for i in range(n_pw_total):
            kx_i = kx + G_vectors[i][0]
            ky_i = ky + G_vectors[i][1]
            for j in range(n_pw_total):
                kx_j = kx + G_vectors[j][0]
                ky_j = ky + G_vectors[j][1]
                G_diff_x = G_vectors[i][0] - G_vectors[j][0]
                G_diff_y = G_vectors[i][1] - G_vectors[j][1]
                mx = int(round(G_diff_x / (2.0 * math.pi)))
                my = int(round(G_diff_y / (2.0 * math.pi)))
                f_val = inv_eps_coeff.get((mx, my), 0.0)
                M[i][j] = (kx_i * kx_j + ky_i * ky_j) * f_val.real
        
        M_np = np.array(M)
        eigenvalues = np.linalg.eigvalsh(M_np)
        omegas = [math.sqrt(max(0.0, ev)) for ev in eigenvalues]
        bands.append((kx, ky, omegas))
    
    return bands

def _compute_gap_ratio(bands, lower_band=4):
    """Compute the gap-midgap ratio between bands 4 and 5 (0-indexed: 3 and 4)."""
    max_omega_lower = 0.0
    min_omega_upper = float('inf')
    
    for kx, ky, omegas in bands:
        if len(omegas) > lower_band:
            max_omega_lower = max(max_omega_lower, omegas[lower_band - 1])
            min_omega_upper = min(min_omega_upper, omegas[lower_band])
    
    if min_omega_upper == float('inf'):
        return -1.0
    
    gap = min_omega_upper - max_omega_lower
    midgap = (min_omega_upper + max_omega_lower) / 2.0
    if midgap == 0:
        return 0.0
    return gap / midgap

def _square_wave(x, freq, duty):
    """
    1D square wave: returns 1 if the fractional part of x*freq is in [0, duty), else 0.
    This creates a periodic pattern with spatial frequency 'freq' and duty cycle 'duty'.
    """
    phase = (x * freq) % 1.0
    if phase < duty:
        return 1.0
    else:
        return 0.0

def _checkerboard_cell(size, fx, duty_x, fy, duty_y):
    """
    Create a cell defined by the product of two 1D square waves.
    cell[i][j] = square_wave(x_i, fx, duty_x) * square_wave(y_j, fy, duty_y)
    
    This naturally enforces the required symmetries if parameters are chosen appropriately,
    but we will explicitly enforce them via the quarter-cell method to be safe.
    Actually, the product of two symmetric square waves (centered at 0.5) is symmetric.
    Let's define x_i = (i + 0.5)/size.
    The square wave is periodic. To ensure symmetry about the center (0.5),
    we should center the square wave.
    
    Let's use a symmetric square wave definition:
    dist = abs((x * freq) % 1.0 - 0.5)
    if dist < duty/2.0: return 1.0 else 0.0
    This is symmetric about x=0.5 (and x=0, 1, etc. if freq is integer).
    """
    n = size // 2
    quarter = [[0] * n for _ in range(n)]
    
    for i in range(n):
        x = (i + 0.5) / size
        # Symmetric square wave in x
        phase_x = (x * fx) % 1.0
        dist_x = abs(phase_x - 0.5)
        val_x = 1 if dist_x < duty_x / 2.0 else 0
        
        for j in range(n):
            y = (j + 0.5) / size
            phase_y = (y * fy) % 1.0
            dist_y = abs(phase_y - 0.5)
            val_y = 1 if dist_y < duty_y / 2.0 else 0
            
            # Product of the two 1D patterns
            if val_x == 1 and val_y == 1:
                quarter[i][j] = 1
            else:
                quarter[i][j] = 0
                
    return quarter

def design(problem):
    """Design a symmetric 32x32-pixel unit cell maximizing the 4-5 TM band gap."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 200.0  # Leave 30s margin for the 230s limit
    
    best_cell = None
    best_score = -float('inf')
    
    def evaluate_cell(cell):
        """Evaluate the gap ratio for a given cell."""
        try:
            n_pw = 17  # 17x17 = 289 plane waves
            inv_eps_coeff = _compute_1over_eps_fourier(cell, size, n_pw)
            bands = _band_structure_tm(inv_eps_coeff, n_pw, n_kpoints=30)
            score = _compute_gap_ratio(bands, lower_band=4)
            return score
        except Exception:
            return -float('inf')
    
    # Search space for separable square-wave gratings
    # Frequencies: integers to maintain periodicity and symmetry
    # Duty cycles: continuous in (0, 1)
    
    # Initial guess: fx=2, fy=2, duty_x=0.5, duty_y=0.5 (standard checkerboard)
    # This has strong (2,2) component.
    # We want (2,0) and (2,2).
    # fx=2 gives strong (2,0) in x.
    # fy=2 gives strong (0,2) in y.
    # Product gives (2,2).
    # If we use fx=2, fy=1, we get (2,1) and (2,-1)? No, product of square waves.
    
    # Let's scan a grid of parameters.
    freqs = [1, 2, 3]
    duties = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    # We can also try asymmetric duties to break degeneracy and enhance specific modes?
    # But the cell must be symmetric. The symmetric square wave definition handles that.
    
    # To find the best combination, we iterate.
    # Since evaluation is expensive (~1-2s), we can afford ~100 evaluations.
    # 3 * 7 * 3 * 7 = 441 combinations. Too many.
    # Let's prioritize:
    # 1. fx=2, fy=2 (checkerboard)
    # 2. fx=2, fy=1
    # 3. fx=1, fy=2
    # 4. fx=3, fy=3
    # etc.
    
    candidates = []
    for fx in freqs:
        for fy in freqs:
            for dx in duties:
                for dy in duties:
                    candidates.append((fx, fy, dx, dy))
                    
    # Sort by "likelihood" of success? 
    # (2,2) is often good for high band gaps.
    # Let's just sample a subset or run in time order.
    # Random sampling might be better to explore.
    random.shuffle(candidates)
    
    # Ensure the most promising ones are tried first?
    # Let's try (2,2) with various duties first.
    prioritized = []
    for dx in duties:
        for dy in duties:
            prioritized.append((2, 2, dx, dy))
    prioritized.extend(candidates)
    
    # Remove duplicates
    seen = set()
    unique_candidates = []
    for c in prioritized:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)
            
    for fx, fy, dx, dy in unique_candidates:
        if time.time() > deadline - 10.0:
            break
            
        quarter = _checkerboard_cell(size, fx, dx, fy, dy)
        cell = _make_symmetric_cell(size, quarter)
        
        # Check if cell is valid (all 0/1, correct size)
        if not all(all(p in (0, 1) for p in row) for row in cell):
            continue
            
        score = evaluate_cell(cell)
        
        if score > best_score:
            best_score = score
            best_cell = cell
            
        # If we find a very good score, we might want to do a local refinement?
        # For now, just keep searching.
        
    if best_cell is None:
        # Fallback: simple checkerboard
        quarter = _checkerboard_cell(size, 2, 0.5, 2, 0.5)
        best_cell = _make_symmetric_cell(size, quarter)
        
    return best_cell

# EVOLVE-BLOCK-END
