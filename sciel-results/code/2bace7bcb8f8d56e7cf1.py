import numpy as np
import time

def _apply_d4_symmetry(quarter, size=32):
    """Apply full D4 symmetry (rotations + reflections about center) to a quarter cell.
    
    The quarter is the top-left 16x16 block. The full cell must satisfy:
    - cell[r,c] == cell[r, size-1-c]  (horizontal mirror)
    - cell[r,c] == cell[size-1-r, c]  (vertical mirror)
    - cell[r,c] == cell[size-1-r, size-1-c] (180 rotation)
    - cell[r,c] == cell[c, r]  (transpose / main diagonal mirror)
    - cell[r,c] == cell[c, size-1-r]  (anti-diagonal mirror)
    - and all combinations
    """
    cell = np.zeros((size, size), dtype=int)
    half = size // 2
    
    for r in range(half):
        for c in range(half):
            val = quarter[r, c]
            # 8 symmetric positions
            cell[r, c] = val
            cell[r, size - 1 - c] = val
            cell[size - 1 - r, c] = val
            cell[size - 1 - r, size - 1 - c] = val
            cell[c, r] = val
            cell[c, size - 1 - r] = val
            cell[size - 1 - r, c] = val
            cell[size - 1 - r, size - 1 - c] = val
    
    return cell

def _compute_fourier_coeffs(cell, n_pw=17):
    """Compute Fourier coefficients of 1/epsilon for the cell.
    
    Returns f_G array of shape (n_pw, n_pw) where f_G[i,j] corresponds to
    G = (G_indices[i]*2*pi, G_indices[j]*2*pi).
    """
    size = cell.shape[0]
    eps_high = 12.0
    eps_low = 1.0
    
    n_half = n_pw // 2
    G_indices = np.arange(-n_half, n_half + 1)
    
    # 1/epsilon at each pixel
    inv_eps = np.where(cell == 1, 1.0 / eps_high, 1.0 / eps_low)
    
    # 2D FFT
    fft_2d = np.fft.fft2(inv_eps)
    
    f_G = np.zeros((n_pw, n_pw), dtype=complex)
    for i, m in enumerate(G_indices):
        for j, n in enumerate(G_indices):
            fft_idx_i = m % size
            fft_idx_j = n % size
            f_G[i, j] = fft_2d[fft_idx_i, fft_idx_j] / (size * size)
    
    return f_G, G_indices

def _compute_gap_fast(cell, n_pw=11):
    """Fast gap computation using fewer k-points and plane waves.
    
    Returns the gap-midgap ratio: 2*(min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
    """
    try:
        f_G, G_indices = _compute_fourier_coeffs(cell, n_pw=n_pw)
    except Exception:
        return -1.0
    
    n = n_pw
    n_half = n // 2
    
    # Key k-points in the irreducible Brillouin zone (in units of 2pi/a)
    # IBZ for square lattice: 0 <= kx <= ky, kx + ky <= 1
    k_points = [
        (0.0, 0.0),      # Gamma
        (0.5, 0.0),      # along Gamma-X
        (1.0, 0.0),      # X
        (0.5, 0.5),      # midpoint X-M
        (1.0, 1.0),      # M
    ]
    
    min_omega_5 = np.inf
    max_omega_4 = -np.inf
    
    # Precompute G list
    G_list = []
    for i in range(n):
        for j in range(n):
            G_list.append((G_indices[i] * 2.0 * np.pi, G_indices[j] * 2.0 * np.pi))
    
    # Precompute G index lookup
    G_lookup = {}
    for i in range(n):
        for j in range(n):
            G_lookup[(G_indices[i], G_indices[j])] = (i, j)
    
    for kx, ky in k_points:
        k_actual = np.array([kx * 2.0 * np.pi, ky * 2.0 * np.pi])
        
        H = np.zeros((n * n, n * n), dtype=complex)
        
        for a in range(n * n):
            Ga = G_list[a]
            k_plus_Ga = k_actual + np.array(Ga)
            kp2 = k_plus_Ga[0] ** 2 + k_plus_Ga[1] ** 2
            
            for b in range(n * n):
                Gb = G_list[b]
                dGx_idx = int(round((Ga[0] - Gb[0]) / (2.0 * np.pi)))
                dGy_idx = int(round((Ga[1] - Gb[1]) / (2.0 * np.pi)))
                
                key = (dGx_idx, dGy_idx)
                if key in G_lookup:
                    ii, jj = G_lookup[key]
                    f_val = f_G[ii, jj]
                else:
                    f_val = 0.0
                
                H[a, b] = kp2 * f_val
        
        # Solve eigenvalue problem
        try:
            eigenvalues = np.linalg.eigvalsh(H)
            eigenvalues = np.sort(np.real(eigenvalues))
        except Exception:
            continue
        
        if len(eigenvalues) >= 5:
            omega_4 = eigenvalues[3]
            omega_5 = eigenvalues[4]
            
            if omega_4 > max_omega_4:
                max_omega_4 = omega_4
            if omega_5 < min_omega_5:
                min_omega_5 = omega_5
    
    if max_omega_4 == -np.inf or min_omega_5 == np.inf:
        return -1.0
    
    if min_omega_5 <= max_omega_4:
        return -1.0
    
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    
    if midgap <= 0:
        return -1.0
    
    return 2.0 * gap / midgap

def _turing_pattern(size, half, steps=40, feed=0.055, kill=0.062, seed=42):
    """Generate a D4-symmetric binary mask from a Gray-Scott Turing pattern.
    
    Evolves the Gray-Scott reaction-diffusion system from a symmetric seed,
    then thresholds to binary and applies D4 symmetry.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize with a symmetric pattern
    u = np.ones((size, size))
    v = np.zeros((size, size))
    
    # Add a symmetric perturbation in the center
    cx, cy = size // 2 - 0.5, size // 2 - 0.5
    for r in range(size):
        for c in range(size):
            dist = np.sqrt((r - cx) ** 2 + (c - cy) ** 2)
            if dist < size * 0.15:
                v[r, c] = 0.1 + 0.05 * rng.random()
    
    # Ensure D4 symmetry of initial condition
    def enforce_d4(arr):
        result = arr.copy()
        for r in range(half):
            for c in range(half):
                val = arr[r, c]
                result[r, c] = val
                result[r, size - 1 - c] = val
                result[size - 1 - r, c] = val
                result[size - 1 - r, size - 1 - c] = val
                result[c, r] = val
                result[c, size - 1 - r] = val
                result[size - 1 - r, c] = val
                result[size - 1 - r, size - 1 - c] = val
        return result
    
    u = enforce_d4(u)
    v = enforce_d4(v)
    
    # Gray-Scott evolution
    Du = 1.0
    Dv = 0.5
    dt = 1.0
    
    for step in range(steps):
        # Laplacian using 5-point stencil
        u_lap = (np.roll(u, 1, 0) + np.roll(u, -1, 0) +
                 np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4 * u)
        v_lap = (np.roll(v, 1, 0) + np.roll(v, -1, 0) +
                 np.roll(v, 1, 1) + np.roll(v, -1, 1) - 4 * v)
        
        reaction = v * v * u
        u_new = u + dt * (Du * u_lap - reaction + feed * (1 - u))
        v_new = v + dt * (Dv * v_lap + reaction - (feed + kill) * v)
        
        u = np.clip(u_new, 0, 1)
        v = np.clip(v_new, 0, 1)
        
        # Enforce symmetry periodically
        if step % 5 == 0:
            u = enforce_d4(u)
            v = enforce_d4(v)
    
    # Threshold to binary: where v > 0.5, mark as dielectric
    binary = (v > 0.5).astype(int)
    binary = enforce_d4(binary)
    
    return binary

def _square_pillars_init(size, half, pillar_size=6, offset=2):
    """Initialize with square pillars at symmetric positions."""
    cell = np.zeros((size, size), dtype=int)
    
    # Place square pillars in the top-left quarter
    # Pillar 1: near center of quarter
    r0, c0 = half - pillar_size - offset, offset
    cell[r0:r0 + pillar_size, c0:c0 + pillar_size] = 1
    
    # Pillar 2: near corner of quarter
    r1, c1 = offset, offset
    cell[r1:r1 + pillar_size, c1:c1 + pillar_size] = 1
    
    # Apply full D4 symmetry
    cell = _apply_d4_symmetry(cell[:half, :half], size)
    
    return cell

def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell.
    Uses Turing pattern initialization + local search to optimize the 4th-5th TM band gap.
    """
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 230s kill
    
    size = problem["pixels"]
    half = size // 2  # 16
    
    # Generate multiple candidate initial designs
    candidates = []
    
    # 1. Square pillar initialization
    for ps in [4, 5, 6, 7]:
        for off in [1, 2, 3]:
            cell = _square_pillars_init(size, half, pillar_size=ps, offset=off)
            candidates.append(cell)
    
    # 2. Turing pattern initializations with different parameters
    turing_params = [
        (40, 0.055, 0.062, 42),
        (50, 0.050, 0.060, 123),
        (60, 0.060, 0.065, 777),
        (35, 0.052, 0.061, 256),
        (45, 0.058, 0.063, 888),
    ]
    for steps, feed, kill, seed in turing_params:
        try:
            cell = _turing_pattern(size, half, steps=steps, feed=feed, kill=kill, seed=seed)
            candidates.append(cell)
        except Exception:
            pass
    
    # Evaluate all candidates and pick the best
    best_score = -np.inf
    best_quarter = None
    
    for cell in candidates:
        if time.time() > deadline - 30:
            break
        try:
            score = _compute_gap_fast(cell)
            if score > best_score:
                best_score = score
                best_quarter = cell[:half, :half].copy()
        except Exception:
            pass
    
    # If no valid candidate, use a default
    if best_quarter is None:
        best_quarter = _square_pillars_init(size, half, pillar_size=5, offset=2)[:half, :half].copy()
        best_score = _compute_gap_fast(_apply_d4_symmetry(best_quarter, size))
    
    # Local search: flip pixels in the quarter
    all_positions = [(r, c) for r in range(half) for c in range(half)]
    
    improved = True
    no_improve_count = 0
    max_no_improve = 3
    
    while improved and time.time() < deadline - 20:
        improved = False
        no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Try a random restart with a new Turing pattern
            if time.time() < deadline - 20:
                seed = int(time.time() * 1000) % 100000
                try:
                    cell_trial = _turing_pattern(size, half, steps=40 + seed % 20,
                                                 feed=0.05 + (seed % 10) * 0.002,
                                                 kill=0.06 + (seed % 7) * 0.001,
                                                 seed=seed)
                    quarter_trial = cell_trial[:half, :half].copy()
                    score_trial = _compute_gap_fast(cell_trial)
                    if score_trial > best_score:
                        best_score = score_trial
                        best_quarter = quarter_trial.copy()
                        improved = True
                        no_improve_count = 0
                except Exception:
                    pass
            break
        
        for r, c in all_positions:
            if time.time() > deadline - 20:
                break
            
            # Try flipping pixel (r, c) in the quarter
            new_quarter = best_quarter.copy()
            new_quarter[r, c] = 1 - new_quarter[r, c]
            
            # Apply symmetry to get full cell
            new_cell = _apply_d4_symmetry(new_quarter, size)
            
            # Evaluate
            score = _compute_gap_fast(new_cell)
            
            if score > best_score:
                best_score = score
                best_quarter = new_quarter.copy()
                improved = True
                no_improve_count = 0
    
    # Final cell
    final_cell = _apply_d4_symmetry(best_quarter, size)
    
    return final_cell.tolist()
