import numpy as np
import time
import math

def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    Strategy:
    1. Represent the C4v-symmetric dielectric mask as a thresholded sum of
       harmonic (Fourier) modes that respect C4v symmetry.
    2. Optimize the mode amplitudes/phases via simulated annealing.
    3. Use a fast plane-wave evaluator to score candidate designs.
    4. Return the best design found within the time budget.
    """
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)

    # Time management
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin for 200s limit

    # --- 1. C4v-symmetric Fourier basis ---
    # Build a set of (h, k) reciprocal vectors that form C4v orbits.
    # A mode (h,k) generates the orbit:
    #   (h,k), (k,h), (-h,k), (k,-h), (-h,-k), (-k,-h), (h,-k), (-k,h)
    # We pick a set of fundamental vectors and store their orbits.
    # For a 32x32 grid, Fourier modes have integer frequencies.
    
    # Fundamental vectors (h,k) with h>=0, k>=0, not both zero.
    # We'll use modes up to frequency ~6 to capture fine texture.
    max_freq = 6
    # Collect unique C4v orbits (canonical representative: h>=0, k>=0, h>=k)
    orbits = []
    seen = set()
    for h in range(0, max_freq + 1):
        for k in range(0, h + 1):  # k <= h to avoid duplicates under swap
            if h == 0 and k == 0:
                continue
            # Canonical orbit: generate all 8 points
            pts = set()
            for (a, b) in [(h,k), (k,h), (-h,k), (k,-h), (-h,-k), (-k,-h), (h,-k), (-k,h)]:
                pts.add((a,b))
            # Check if this orbit was already seen
            key = frozenset(pts)
            if key not in seen:
                seen.add(key)
                orbits.append(list(pts))
    
    # Number of independent modes
    N_modes = len(orbits)
    
    # For each mode orbit, we store the (h,k) pairs.
    # The field at pixel (i,j) is:
    #   f(i,j) = sum_m amp_m * cos(2*pi*(h_m*i + k_m*j)/size + phase_m)
    # But for C4v symmetry, we need to sum over all points in the orbit.
    # Actually, a single cos(2*pi*(h*i+k*j)/size) is NOT C4v symmetric.
    # The C4v-symmetric combination is:
    #   sum over (h',k') in orbit of cos(2*pi*(h'*i + k'*j)/size)
    # This is real and C4v symmetric.
    
    # Precompute the basis functions for each mode orbit
    # basis[m] is a size x size array
    basis = []
    for orbit_pts in orbits:
        b = np.zeros((size, size), dtype=float)
        for (h, k) in orbit_pts:
            # cos(2*pi*(h*i + k*j)/size)
            angles = 2.0 * np.pi * (h * np.arange(size)[:, None] + k * np.arange(size)[None, :]) / size
            b += np.cos(angles)
        basis.append(b)
    
    basis = np.array(basis)  # shape (N_modes, size, size)
    
    # --- 2. Fast Evaluator ---
    # Use 7x7 plane waves (49 modes) for speed
    pw_half = 3
    G_vectors = [(h, k) for h in range(-pw_half, pw_half + 1) 
                 for k in range(-pw_half, pw_half + 1)]
    N_pw = len(G_vectors)
    
    # Precompute G vector arrays
    Gx_arr = np.array([g[0] for g in G_vectors])
    Gy_arr = np.array([g[1] for g in G_vectors])
    
    # Precompute G differences
    G_diffs = np.zeros((N_pw, N_pw, 2), dtype=int)
    for i in range(N_pw):
        for j in range(N_pw):
            G_diffs[i, j, 0] = G_vectors[i][0] - G_vectors[j][0]
            G_diffs[i, j, 1] = G_vectors[i][1] - G_vectors[j][1]
    
    # FFT index lookup
    G_diff_idx_h = G_diffs[:, :, 0] % size
    G_diff_idx_k = G_diffs[:, :, 1] % size
    
    # K-points in irreducible BZ (subset for speed)
    k_points = []
    # Gamma (0,0)
    k_points.append((0.0, 0.0))
    # Along Gamma-M: (0, t)
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        k_points.append((0.0, t))
    # Along M-X: (t, 0.5-t)
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        k_points.append((t, 0.5 - t))
    # X (0.5, 0.5)
    k_points.append((0.5, 0.5))
    
    def field_to_mask(field):
        """Threshold a continuous field to a binary C4v-symmetric mask."""
        # Enforce C4v symmetry by averaging over all 8 symmetric copies
        sym = np.zeros_like(field)
        for i in range(size):
            for j in range(size):
                points = [
                    (i, j),
                    (size - 1 - i, j),
                    (i, size - 1 - j),
                    (size - 1 - i, size - 1 - j),
                    (j, i),
                    (size - 1 - j, j),
                    (j, size - 1 - i),
                    (size - 1 - j, size - 1 - i)
                ]
                sym[i, j] = np.mean([field[pi, pj] for pi, pj in points])
        
        # Threshold at the median to get ~50% dielectric
        threshold = np.median(sym)
        mask = (sym >= threshold).astype(int)
        return mask
    
    def evaluate_gap(cell_grid):
        """
        Computes the approximate band gap between bands 4 and 5.
        cell_grid: 32x32 array of 0/1 (0=air, 1=dielectric)
        """
        # Build epsilon grid
        eps_grid = np.where(cell_grid == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_grid
        
        # FFT for Fourier coefficients
        fft_1eps = np.fft.fft2(inv_eps)
        coeffs = fft_1eps[G_diff_idx_h, G_diff_idx_k] / (size * size)
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            Kx = Gx_arr + kx
            Ky = Gy_arr + ky
            
            # H = (K_i · K_j) * coeffs_{G_i - G_j}
            dots = np.outer(Kx, Kx) + np.outer(Ky, Ky)
            H = dots * coeffs
            H_real = H.real
            
            # Get eigenvalues (we need indices 3 and 4 for bands 4 and 5)
            eigenvalues = np.linalg.eigvalsh(H_real)
            
            if len(eigenvalues) >= 5:
                w4_sq = eigenvalues[3]
                w5_sq = eigenvalues[4]
                
                w4 = np.sqrt(max(0, w4_sq))
                w5 = np.sqrt(max(0, w5_sq))
                
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
        
        if min_omega_5 == float('inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = 2 * gap / midgap
        return ratio
    
    def amps_to_score(amps):
        """Convert amplitude vector to a gap score."""
        field = np.tensordot(amps, basis, axes=(0, 0))
        mask = field_to_mask(field)
        return evaluate_gap(mask)
    
    # --- 3. Initialize amplitudes ---
    # Start with a mix of low and high frequency modes
    np.random.seed(42)
    current_amps = np.zeros(N_modes)
    # Give stronger weight to lower frequency modes initially
    for m in range(N_modes):
        # Frequency of mode m (max |h|,|k| in its orbit)
        max_hk = max(max(abs(h), abs(k)) for (h, k) in orbits[m])
        current_amps[m] = np.random.randn() * (1.0 / (1.0 + max_hk))
    
    # Compute initial score
    current_score = amps_to_score(current_amps)
    best_amps = current_amps.copy()
    best_score = current_score
    
    # --- 4. Optimization: Simulated Annealing on amplitudes ---
    temperature = 0.3
    cooling_rate = 0.998
    max_iterations = 50000
    iteration = 0
    
    # Track time per evaluation to adapt step size
    eval_times = []
    
    while time.time() < deadline and iteration < max_iterations:
        iteration += 1
        
        # Check time budget more frequently
        if time.time() > deadline - 5.0:
            break
        
        # Pick a random mode to perturb
        m = np.random.randint(N_modes)
        
        # Gaussian perturbation
        step_size = 0.1 * temperature
        current_amps[m] += np.random.randn() * step_size
        
        new_score = amps_to_score(current_amps)
        
        # Accept or reject based on Metropolis criterion
        delta = new_score - current_score
        if delta > 0 or np.random.random() < math.exp(delta / max(temperature, 1e-6)):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_amps = current_amps.copy()
        else:
            # Reject
            current_amps[m] -= np.random.randn() * step_size
        
        # Cool down
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
    
    # --- 5. Finalize ---
    final_field = np.tensordot(best_amps, basis, axes=(0, 0))
    final_cell = field_to_mask(final_field)
    
    return final_cell.tolist()
