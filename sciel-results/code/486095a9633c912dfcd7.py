"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def design(problem):
    """
    Search over D4-symmetric binary masks using a wavelet-based representation
    to capture multi-scale structural features. Uses plane-wave proxy for scoring.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 200s limit
    
    def enforce_d4(cell):
        """Enforce D4 symmetry on a numpy array."""
        cell = np.maximum(cell, cell[::-1, :])
        cell = np.maximum(cell, cell[:, ::-1])
        cell = np.maximum(cell, cell.T)
        cell = np.minimum(cell, cell[::-1, :])
        cell = np.minimum(cell, cell[:, ::-1])
        cell = np.minimum(cell, cell.T)
        return cell.astype(int)
    
    def compute_eps_fourier(cell):
        """Compute Fourier coefficients of 1/epsilon for the cell."""
        eps = np.where(cell == 1, eps_high, eps_low).astype(float)
        inv_eps = 1.0 / eps
        N = size
        coeffs = np.fft.fft2(inv_eps) / (N * N)
        return coeffs
    
    def plane_wave_gap(cell, n_pw=17, n_kpts=8):
        """
        Fast plane-wave approximation of the TM band gap ratio.
        Uses a truncated Fourier basis and a few k-points.
        """
        coeffs = compute_eps_fourier(cell)
        
        # Build reciprocal lattice vectors
        max_g = n_pw // 2
        g_vectors = []
        for mx in range(-max_g, max_g + 1):
            for my in range(-max_g, max_g + 1):
                g_sq = mx*mx + my*my
                if g_sq <= max_g * max_g:
                    g_vectors.append((mx, my))
        
        n_g = len(g_vectors)
        
        # k-points in irreducible BZ
        k_points = [
            (0.0, 0.0),
            (math.pi, 0.0),
            (math.pi, math.pi),
            (math.pi/2, 0.0),
            (math.pi/2, math.pi/2),
            (0.75*math.pi, 0.0),
            (0.75*math.pi, 0.5*math.pi),
            (0.5*math.pi, 0.25*math.pi),
        ]
        
        all_eigs = []
        
        for kx, ky in k_points:
            k = np.array([kx, ky])
            
            # Build matrix
            M = np.zeros((n_g, n_g), dtype=complex)
            D = np.zeros((n_g, n_g))
            
            for ig, (mx, my) in enumerate(g_vectors):
                kg = k - np.array([mx, my]) * 2 * math.pi
                D[ig, ig] = kg[0]**2 + kg[1]**2
                for ig2, (mx2, my2) in enumerate(g_vectors):
                    kg2 = k - np.array([mx2, my2]) * 2 * math.pi
                    diff = (mx - mx2, my - my2)
                    idx_m = diff[0] % size
                    idx_n = diff[1] % size
                    c = coeffs[idx_m, idx_n]
                    dot = kg[0]*kg2[0] + kg[1]*kg2[1]
                    M[ig, ig2] = dot * c
            
            # Avoid division by zero
            D_inv = np.zeros_like(D)
            for ig in range(n_g):
                if abs(D[ig, ig]) > 1e-10:
                    D_inv[ig, ig] = 1.0 / D[ig, ig]
            
            A = D_inv @ M
            eigs = np.linalg.eigvals(A)
            eigs_real = np.real(eigs)
            eigs_real = eigs_real[eigs_real > 0]
            eigs_real = np.sort(eigs_real)
            
            if len(eigs_real) >= 5:
                omega4 = math.sqrt(eigs_real[3])
                omega5 = math.sqrt(eigs_real[4])
                all_eigs.append((omega4, omega5))
        
        if len(all_eigs) < 3:
            return -1.0
        
        max_omega4 = max(o4 for o4, o5 in all_eigs)
        min_omega5 = min(o5 for o4, o5 in all_eigs)
        
        if max_omega4 == 0:
            return 0.0
        
        ratio = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
        return ratio
    
    def generate_wavelet_mask(params):
        """
        Generate a D4-symmetric binary mask from wavelet coefficients.
        params: list of (scale, x, y, amplitude) for each wavelet component
        """
        # Create a continuous field from superposition of wavelets
        field = np.zeros((size, size), dtype=float)
        
        for scale, wx, wy, amp in params:
            # Simple wavelet-like function: Gaussian envelope times oscillation
            # This approximates a Daubechies wavelet at given scale and position
            sigma = scale * 2.0
            for i in range(size):
                for j in range(size):
                    dx = (i + 0.5) - wx
                    dy = (j + 0.5) - wy
                    r2 = dx*dx + dy*dy
                    gaussian = math.exp(-r2 / (2 * sigma * sigma))
                    # Oscillatory component
                    phase = 2 * math.pi * (i + j) / (scale * 4)
                    wavelet_val = gaussian * math.cos(phase)
                    field[i, j] += amp * wavelet_val
        
        # Threshold to binary
        threshold = np.mean(field)
        mask = (field > threshold).astype(int)
        
        # Enforce D4 symmetry
        mask = enforce_d4(mask)
        return mask
    
    def random_wavelet_params(n_components):
        """Generate random wavelet parameters."""
        params = []
        for _ in range(n_components):
            scale = np.random.uniform(1.0, 8.0)
            wx = np.random.uniform(0, size/2)
            wy = np.random.uniform(0, size/2)
            amp = np.random.uniform(-1.0, 1.0)
            params.append((scale, wx, wy, amp))
        return params
    
    # Initialize with a simple rod configuration as baseline
    best_score = -1.0
    best_cell = None
    
    # Start with a simple 2x2 rod configuration
    cx, cy, rad = 8.5, 8.5, 4.5
    cell = np.zeros((size, size), dtype=int)
    r_sq = rad * rad
    for i in range(16):
        for j in range(16):
            dx = (i + 0.5) - cx
            dy = (j + 0.5) - cy
            if dx*dx + dy*dy < r_sq:
                cell[i][j] = 1
    cell[16:, :16] = cell[:16, :16][::-1, :]
    cell[:16, 16:] = cell[:16, :16][:, ::-1]
    cell[16:, 16:] = cell[:16, :16][::-1, ::-1]
    
    best_score = plane_wave_gap(cell, n_pw=13, n_kpts=6)
    best_cell = cell.copy()
    
    # Wavelet-based search
    np.random.seed(42)
    n_components = 4
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Generate random wavelet parameters
        params = random_wavelet_params(n_components)
        
        # Generate mask
        cell = generate_wavelet_mask(params)
        
        # Score
        score = plane_wave_gap(cell, n_pw=15, n_kpts=7)
        
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
    
    # Local refinement: perturb wavelet parameters around best
    # We'll do a simple hill-climbing on the pixel mask
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        # Flip a random pixel and its D4 symmetric partners
        i = np.random.randint(0, size)
        j = np.random.randint(0, size)
        
        # Get D4 orbit of (i, j)
        orbit = [(i, j), (size-1-i, j), (i, size-1-j), (size-1-i, size-1-j),
                 (j, i), (size-1-j, i), (j, size-1-i), (size-1-j, size-1-i)]
        orbit = list(set(orbit))
        
        # Try flipping all in orbit
        cell_trial = best_cell.copy()
        for oi, oj in orbit:
            cell_trial[oi, oj] = 1 - cell_trial[oi, oj]
        
        score = plane_wave_gap(cell_trial, n_pw=15, n_kpts=7)
        
        if score > best_score:
            best_score = score
            best_cell = cell_trial.copy()
    
    # Convert to nested list
    cell_list = [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]
    
    return cell_list
# EVOLVE-BLOCK-END
