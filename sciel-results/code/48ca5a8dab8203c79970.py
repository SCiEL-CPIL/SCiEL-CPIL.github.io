"""Design a 32x32 photonic crystal unit cell to maximize TM band 4-5 gap."""
import numpy as np
import time
import math

def design(problem):
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]   # 1
    eps_high = problem["eps_high"] # 12
    
    t_start = time.time()
    DEADLINE = 215.0  # safety margin before 230s kill
    
    def time_left():
        return DEADLINE - (time.time() - t_start)
    
    def enforce_symmetry(cell):
        """Enforce D4 symmetry by averaging over the symmetry group."""
        sym_cells = [
            cell,
            cell[::-1, :],
            cell[:, ::-1],
            cell[::-1, ::-1],
            cell.T,
            cell.T[::-1, :],
            cell.T[:, ::-1],
            cell.T[::-1, ::-1]
        ]
        avg = np.mean(sym_cells, axis=0)
        return (avg >= 0.5).astype(int)
    
    def calculate_gap(cell, n_pw=17):
        """
        Calculate the band 4-5 gap ratio using the plane-wave method.
        Uses a moderate number of G-vectors for accuracy.
        """
        eps = eps_low + (eps_high - eps_low) * cell.astype(float)
        inv_eps = 1.0 / eps
        
        N = size
        half = n_pw // 2
        
        # G-vectors
        G_list = []
        for gx in range(-half, half + 1):
            for gy in range(-half, half + 1):
                G_list.append((gx, gy))
        
        nG = len(G_list)
        
        # Precompute inv_eps Fourier coefficients using FFT
        inv_eps_fft = np.fft.fft2(inv_eps)
        
        inv_eps_F = {}
        for gx, gy in G_list:
            fft_gx = gx % N
            fft_gy = gy % N
            inv_eps_F[(gx, gy)] = inv_eps_fft[fft_gx, fft_gy] / (N * N)
        
        # Key k-points in the irreducible BZ
        # Irreducible BZ: 0 <= ky <= kx <= 0.5
        k_points_list = [
            (0.0, 0.0),      # Gamma
            (0.1, 0.0),
            (0.2, 0.0),
            (0.3, 0.0),
            (0.4, 0.0),
            (0.5, 0.0),      # X
            (0.5, 0.1),
            (0.5, 0.2),
            (0.5, 0.3),
            (0.5, 0.4),
            (0.5, 0.5),      # M
            (0.3, 0.1),
            (0.4, 0.2),
            (0.3, 0.2),
            (0.2, 0.1),
            (0.4, 0.3),
        ]
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points_list:
            # Build eigenvalue matrix
            M = np.zeros((nG, nG), dtype=complex)
            
            for ig, (gx, gy) in enumerate(G_list):
                Gkx = gx - kx
                Gky = gy - ky
                for igp, (gxp, gyp) in enumerate(G_list):
                    Gpkx = gxp - kx
                    Gpky = gyp - ky
                    dGx = gx - gxp
                    dGy = gy - gyp
                    if (dGx, dGy) in inv_eps_F:
                        M[ig, igp] = (Gkx * Gpkx + Gky * Gpky) * inv_eps_F[(dGx, dGy)]
            
            # Solve eigenvalue problem
            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.sort(eigvals.real)
            
            if len(eigvals) >= 5:
                omega_4 = np.sqrt(max(0, eigvals[3]))
                omega_5 = np.sqrt(max(0, eigvals[4]))
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == float('inf'):
            return -1.0
        
        gap_ratio = 2.0 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
        return gap_ratio
    
    # Strategy: Use two orthogonal sinusoidal gratings with optimized phases and frequencies
    # This creates a deterministic, non-convex pattern that maximizes specific Fourier components
    
    def make_cell_from_gratings(freq_x, freq_y, phase_x, phase_y, threshold=0.5):
        """
        Create a cell from two orthogonal sinusoidal gratings.
        freq_x, freq_y: spatial frequencies (integer multiples for symmetry)
        phase_x, phase_y: phase offsets
        threshold: threshold for binarization
        """
        coords = np.array([(i + 0.5) / size - 0.5 for i in range(size)])
        XX, YY = np.meshgrid(coords, coords, indexing='ij')
        
        # Two orthogonal sinusoidal gratings
        grating_x = np.sin(2 * np.pi * freq_x * XX + phase_x)
        grating_y = np.sin(2 * np.pi * freq_y * YY + phase_y)
        
        # Combine: use product to create checkerboard-like pattern
        combined = grating_x * grating_y
        
        # Binarize
        cell = (combined >= threshold).astype(int)
        
        return enforce_symmetry(cell)
    
    def make_cell_from_cosine(freq_x, freq_y, phase_x, phase_y, threshold=0.5):
        """
        Alternative: use cosine gratings with different combination.
        """
        coords = np.array([(i + 0.5) / size - 0.5 for i in range(size)])
        XX, YY = np.meshgrid(coords, coords, indexing='ij')
        
        grating_x = np.cos(2 * np.pi * freq_x * XX + phase_x)
        grating_y = np.cos(2 * np.pi * freq_y * YY + phase_y)
        
        # Sum for constructive interference at specific points
        combined = 0.5 * (grating_x + grating_y)
        
        cell = (combined >= threshold).astype(int)
        
        return enforce_symmetry(cell)
    
    best_cell = None
    best_score = -float('inf')
    
    # Try different combinations of frequencies and phases
    # For D4 symmetry, we want frequencies that are even multiples
    # and phases that preserve symmetry
    
    freq_options = [1, 2, 3, 4]
    phase_options = [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    
    candidates = []
    
    # Candidate 1: Product of two sine gratings
    for fx in freq_options:
        for fy in freq_options:
            for px in phase_options:
                for py in phase_options:
                    candidates.append(('product', fx, fy, px, py, 0.0))
    
    # Candidate 2: Sum of two cosine gratings
    for fx in freq_options:
        for fy in freq_options:
            for px in phase_options:
                for py in phase_options:
                    candidates.append(('sum', fx, fy, px, py, 0.5))
    
    # Candidate 3: Pure checkerboard (fx=fy=1, phases=0)
    candidates.append(('product', 1, 1, 0.0, 0.0, 0.0))
    
    # Evaluate candidates
    for cand in candidates:
        if time_left() < 5:
            break
        
        if cand[0] == 'product':
            _, fx, fy, px, py, thresh = cand
            cell = make_cell_from_gratings(fx, fy, px, py, thresh)
        else:
            _, fx, fy, px, py, thresh = cand
            cell = make_cell_from_cosine(fx, fy, px, py, thresh)
        
        score = calculate_gap(cell)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
    
    # Local optimization: fine-tune the threshold
    if best_cell is not None and time_left() > 20:
        # Try to find the best threshold for the best pattern
        # We need to track which pattern was best, so let's re-evaluate
        for cand in candidates[:20]:  # Top 20 candidates
            if time_left() < 10:
                break
            
            if cand[0] == 'product':
                _, fx, fy, px, py, _ = cand
                for thresh in [0.0, 0.1, 0.2, -0.1, -0.2]:
                    cell = make_cell_from_gratings(fx, fy, px, py, thresh)
                    score = calculate_gap(cell)
                    if score > best_score:
                        best_score = score
                        best_cell = cell.copy()
            else:
                _, fx, fy, px, py, _ = cand
                for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
                    cell = make_cell_from_cosine(fx, fy, px, py, thresh)
                    score = calculate_gap(cell)
                    if score > best_score:
                        best_score = score
                        best_cell = cell.copy()
    
    # Fallback: simple circular rod
    if best_cell is None or best_score < 0:
        coords = np.array([(i + 0.5) / size - 0.5 for i in range(size)])
        XX, YY = np.meshgrid(coords, coords, indexing='ij')
        r = 0.2
        val = XX**2 + YY**2
        cell = (val <= r**2).astype(int)
        best_cell = enforce_symmetry(cell)
    
    return best_cell.tolist()
