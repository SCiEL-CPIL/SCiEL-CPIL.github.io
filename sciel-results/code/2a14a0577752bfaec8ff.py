"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def design(problem):
    """Design a photonic crystal unit cell using moire patterns to maximize TM 4-5 band gap."""
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    
    start_time = time.time()
    time_limit = 195.0  # Safety margin before 230s kill

    def make_moire_cell(f1, f2, phase1, phase2, threshold):
        """
        Create a symmetric cell using interference of two counter-rotating square waves.
        f1, f2: spatial frequencies
        phase1, phase2: phase shifts
        threshold: threshold for binarization (0.0 to 1.0)
        """
        # Pixel coordinates centered at 0
        x = (np.arange(size) + 0.5) / size - 0.5
        y = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(x, y, indexing='xy')
        
        # Define two square waves with different orientations/frequencies
        # Square wave: sign(sin(...))
        # To ensure symmetry, we use combinations that respect D4 symmetry
        # A common symmetric pattern is |sin(a*x + b*y)| + |sin(a*y + b*x)| or similar
        
        # Let's use a "checkerboard-like" interference
        # Wave 1: oriented along (1, 1)
        # Wave 2: oriented along (1, -1)
        
        # We want to enhance (2,0) and (2,2) coefficients.
        # Frequency 2 implies period 0.5.
        # Let's try a superposition of frequencies 1 and 2.
        
        # Base patterns
        # P1: frequency f1 along x, modulated by y
        # P2: frequency f2 along y, modulated by x
        
        # Simplest symmetric moire:
        # I = |cos(2*pi*f1*(x+y))| + |cos(2*pi*f2*(x-y))|
        
        # To allow more complex shapes, we can add a radial term or mix frequencies
        # Let's stick to the directive: interference of two counter-rotating square waves.
        
        # Square wave function
        def sq(w):
            return np.sign(np.sin(w))
            
        # Counter-rotating implies + and - directions
        # Let's define angles
        # Angle 1: 45 deg (x+y)
        # Angle 2: -45 deg (x-y)
        
        # Frequencies f1, f2
        # Phase shifts phase1, phase2
        
        w1 = 2 * np.pi * f1 * (X + Y) + phase1
        w2 = 2 * np.pi * f2 * (X - Y) + phase2
        
        # Interference pattern intensity
        # Sum of absolute values creates a grid
        intensity = np.abs(np.sin(w1)) + np.abs(np.sin(w2))
        
        # Normalize intensity to [0, 1] roughly (max is 2)
        intensity = intensity / 2.0
        
        # Binarize
        # threshold determines the "thickness" of the dielectric lines
        # If threshold is low, most of the cell is dielectric (1)
        # If threshold is high, less is dielectric
        # We want 1 = dielectric.
        # So cell = 1 where intensity > threshold
        
        cell = (intensity > threshold).astype(np.uint8)
        
        # Enforce symmetry explicitly
        # 1. Invert rows
        cell = (cell + cell[::-1, :]) // 2
        # 2. Invert cols
        cell = (cell + cell[:, ::-1]) // 2
        # 3. Transpose
        cell = (cell + cell.T) // 2
        # 4. Rotate 180 (redundant if 1 and 2 are done, but safe)
        cell = (cell + cell[::-1, ::-1]) // 2
        
        return cell.tolist()

    def fast_score(cell_list):
        """
        Fast approximation of the band gap score.
        Uses a reduced plane-wave basis for speed.
        """
        cell = np.array(cell_list, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell
        
        # Compute Fourier coefficients of epsilon
        N = 9
        G_vals = np.fft.fftfreq(N, d=1.0/N)
        
        eps_f = np.zeros((N, N), dtype=complex)
        
        # Precompute pixel coordinates
        px = (np.arange(size) + 0.5) / size - 0.5
        py = (np.arange(size) + 0.5) / size - 0.5
        PX, PY = np.meshgrid(px, py, indexing='xy')
        
        for i in range(N):
            for j in range(N):
                kx = 2 * np.pi * G_vals[i]
                ky = 2 * np.pi * G_vals[j]
                phase = np.exp(-1j * (kx * PX + ky * PY))
                eps_f[i, j] = np.sum(eps * phase) / (size * size)
                
        # Check a few critical k-points
        k_points = [
            (0.0, 0.0),    # Gamma
            (0.5, 0.5),    # M
            (0.5, 0.0),    # X
            (0.25, 0.25),  # Midpoint Gamma-M
            (0.5, 0.25),   # Midpoint M-X
            (0.25, 0.0),   # Midpoint Gamma-X
        ]
        
        max_omega4 = 0.0
        min_omega5 = np.inf
        
        indices = np.arange(-N//2, N//2 + 1)
        
        for kx, ky in k_points:
            N_small = 5
            idx_small = np.arange(-N_small//2, N_small//2 + 1)
            G_small = []
            for i in idx_small:
                for j in idx_small:
                    G_small.append((i, j))
            
            M = np.zeros((len(G_small), len(G_small)), dtype=complex)
            D = np.zeros(len(G_small))
            
            for i, (gx_i, gy_i) in enumerate(G_small):
                kgx = gx_i + kx
                kgy = gy_i + ky
                kg2 = kgx**2 + kgy**2
                if kg2 < 1e-10:
                    kg2 = 1e-10
                D[i] = kg2
                
                for j, (gx_j, gy_j) in enumerate(G_small):
                    dgx = gx_i - gx_j
                    dgy = gy_i - gy_j
                    ix = (int(dgx) + 4) % 9
                    iy = (int(dgy) + 4) % 9
                    M[i, j] = eps_f[ix, iy] * D[j]
            
            D_inv_sqrt = 1.0 / np.sqrt(D)
            A = M * D_inv_sqrt[:, None] * D_inv_sqrt[None, :]
            A = (A + A.conj().T) / 2.0
            
            try:
                evals = np.linalg.eigvalsh(A)
                evals = np.real(evals)
                evals = np.sort(evals)
                if len(evals) >= 5:
                    omega4 = np.sqrt(evals[3])
                    omega5 = np.sqrt(evals[4])
                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
            except:
                pass
        
        if min_omega5 == np.inf or max_omega4 == 0:
            return -1.0
        
        gap = 2 * (min_omega5 - max_omega4)
        midgap = min_omega5 + max_omega4
        if midgap == 0:
            return -1.0
        return gap / midgap

    # Search Strategy
    best_cell = None
    best_score = -np.inf
    
    # Coarse Grid Search over Moire Parameters
    # f1, f2: frequencies. Since period is 1, integer frequencies create standing waves aligned with pixels.
    # f=1 -> period 1 (1 unit per cell)
    # f=2 -> period 0.5 (2 units per cell)
    # We want to enhance (2,0) and (2,2).
    
    # Let's search f1, f2 in [1.0, 2.0, 3.0]
    # Phases: 0, pi/2, pi
    # Threshold: 0.4 to 0.6
    
    freqs = [1.0, 1.5, 2.0, 2.5, 3.0]
    phases = [0.0, np.pi/2, np.pi, 3*np.pi/2]
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    for f1 in freqs:
        if time.time() - start_time > time_limit - 30:
            break
        for f2 in freqs:
            if time.time() - start_time > time_limit - 30:
                break
            for p1 in phases:
                if time.time() - start_time > time_limit - 30:
                    break
                for p2 in phases:
                    if time.time() - start_time > time_limit - 30:
                        break
                    for th in thresholds:
                        if time.time() - start_time > time_limit - 30:
                            break
                        cell = make_moire_cell(f1, f2, p1, p2, th)
                        score = fast_score(cell)
                        if score > best_score:
                            best_score = score
                            best_cell = cell
                            best_params = (f1, f2, p1, p2, th)
                            # print(f"New best: {score:.4f} at {best_params}")

    if best_cell is None:
        # Fallback: Simple 4-rod
        best_cell = make_moire_cell(2.0, 2.0, 0.0, 0.0, 0.5)
        best_params = (2.0, 2.0, 0.0, 0.0, 0.5)

    # Fine-tuning with Pixel Flips
    # Take the best cell and try flipping individual pixels to improve score
    cell_arr = np.array(best_cell, dtype=np.uint8)
    
    # Symmetry group of the cell:
    # 1. Identity
    # 2. Rotate 90
    # 3. Rotate 180
    # 4. Rotate 270
    # 5. Reflect X
    # 6. Reflect Y
    # 7. Reflect X+Y (main diag)
    # 8. Reflect X-Y (anti-diag)
    
    # We will iterate through pixels in the fundamental domain (1/8 of the cell)
    # and flip them if it improves the score.
    
    # Define fundamental domain mask
    # The irreducible wedge is the triangle 0 <= y <= x <= 0.5
    # In pixel indices (0..31), center is 15.5
    # x_idx, y_idx
    # Condition: y_idx <= x_idx and x_idx <= 15 (left half)
    # Actually, let's just generate the orbit of each pixel and check if flipping the whole orbit improves score.
    
    def get_orbit(r, c):
        """Get all symmetric equivalents of pixel (r, c)"""
        # r, c are 0-indexed
        # Center is (15.5, 15.5)
        # Transformations:
        # 1. (r, c)
        # 2. (c, r)
        # 3. (31-r, c)
        # 4. (31-c, r)
        # 5. (r, 31-c)
        # 6. (c, 31-r)
        # 7. (31-r, 31-c)
        # 8. (31-c, 31-r)
        
        pts = set()
        pts.add((r, c))
        pts.add((c, r))
        pts.add((31-r, c))
        pts.add((31-c, r))
        pts.add((r, 31-c))
        pts.add((c, 31-r))
        pts.add((31-r, 31-c))
        pts.add((31-c, 31-r))
        return list(pts)

    # Iterate through a subset of pixels in the fundamental domain
    # To save time, we'll check pixels in the upper-left triangle
    # r < 16, c < 16, c <= r
    
    improvements = 0
    max_improvements = 50
    current_score = best_score
    
    # We will do a few passes of local search
    for pass_num in range(3):
        if time.time() - start_time > time_limit - 10:
            break
        improved = False
        # Iterate through fundamental domain
        for r in range(16):
            if time.time() - start_time > time_limit - 10:
                break
            for c in range(r + 1): # c <= r
                if time.time() - start_time > time_limit - 10:
                    break
                
                orbit = get_orbit(r, c)
                
                # Check if all pixels in orbit have the same value
                # If not, flipping might break symmetry if we don't flip all
                # Since we flip all in orbit, symmetry is preserved.
                
                # Try flipping
                # Save original values
                orig_vals = [cell_arr[rr, cc] for rr, cc in orbit]
                
                # Flip
                for rr, cc in orbit:
                    cell_arr[rr, cc] = 1 - cell_arr[rr, cc]
                    
                new_cell = cell_arr.tolist()
                new_score = fast_score(new_cell)
                
                # Restore
                for (rr, cc), val in zip(orbit, orig_vals):
                    cell_arr[rr, cc] = val
                    
                if new_score > current_score + 1e-6:
                    # Accept
                    for rr, cc in orbit:
                        cell_arr[rr, cc] = 1 - cell_arr[rr, cc]
                    current_score = new_score
                    improved = True
                    improvements += 1
                    if improvements >= max_improvements:
                        break
            if improvements >= max_improvements:
                break
        if not improved:
            break
            
    best_cell = cell_arr.tolist()
    best_score = current_score

    return best_cell
# EVOLVE-BLOCK-END
