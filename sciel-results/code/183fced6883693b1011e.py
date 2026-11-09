"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell for a 2D square-lattice photonic crystal
    that maximizes the gap between TM bands 4 and 5.
    
    Approach: Parametric Geometric Design with Heuristic Selection.
    We generate several families of symmetric geometric patterns (Cross, Diamond, Circle, Holey).
    We parameterize each family and evaluate them using a Fourier-based heuristic
    that approximates the band gap strength by looking at the scattering coefficients
    at the M point (1,1) relative to other couplings.
    """
    size = problem["pixels"]
    
    # Helper to compute a heuristic score based on Fourier coefficients
    def heuristic_score(cell_np):
        # Compute 2D FFT of the dielectric map (normalized to 0..1)
        # We are interested in the coefficient at (1,1) in reciprocal space.
        # The FFT of a 32x32 array gives frequencies 0..31.
        # Frequency 1 corresponds to k = 2*pi/32.
        # The M point in the BZ is (pi, pi) which corresponds to k = 2*pi/1 in lattice units?
        # No, the lattice constant is 1 pixel? No, the unit cell is 32 pixels.
        # The reciprocal lattice vectors are G = (2*pi/a) * (h, k).
        # If the pixel grid represents the unit cell, the fundamental frequency is 1/size in normalized coords?
        # Let's just look at the FFT coefficients.
        # The M point corresponds to the highest frequency in the discrete grid?
        # For a 32x32 grid, the M point (pi, pi) corresponds to the Nyquist frequency (16, 16) in FFT indices?
        # Actually, for a square lattice, the M point is (pi, pi).
        # The plane waves are exp(i k x).
        # The Fourier series of epsilon(x,y) = sum G_n exp(i n G x).
        # G_11 is the coefficient for the vector (1,1) in reciprocal lattice units.
        # In the discrete FFT of the pixel map, the index corresponding to G_11 is 1 (or 31).
        # Let's extract G_11, G_10, G_01, G_20, G_02.
        
        fft_coeffs = np.fft.fft2(cell_np)
        
        # Indices for (h,k) = (1,1), (1,0), (0,1), (2,0), (0,2)
        # Note: FFT output is ordered 0, 1, ..., N-1.
        # G_11 is at [1, 1].
        # G_10 is at [1, 0].
        # G_01 is at [0, 1].
        # G_20 is at [2, 0].
        # G_02 is at [0, 2].
        
        G_11 = np.abs(fft_coeffs[1, 1])
        G_10 = np.abs(fft_coeffs[1, 0])
        G_01 = np.abs(fft_coeffs[0, 1])
        G_20 = np.abs(fft_coeffs[2, 0])
        G_02 = np.abs(fft_coeffs[0, 2])
        
        # The gap at M is driven by G_11.
        # The gap at X is driven by G_10 and G_01.
        # For higher bands, we want strong M-point coupling.
        # We want G_11 to be large.
        # We also want to avoid too much X-point coupling if it shifts bands unfavorably, 
        # but typically maximizing G_11 is the primary driver for the M-gap.
        
        # Heuristic: Maximize G_11
        # Also penalize if the pattern is too "blocky" (low spatial frequency content)
        # Let's just use G_11.
        
        # To ensure we pick a valid binary pattern, we also check filling fraction.
        # Ideal filling is around 0.5.
        fill = np.mean(cell_np)
        fill_penalty = (fill - 0.5) ** 2
        
        score = G_11 - 10 * fill_penalty
        return score

    candidates = []

    # 1. Cross Pattern (Dielectric Cross)
    # Width parameter w
    for w in range(4, 14, 2):
        cell = np.zeros((size, size), dtype=int)
        # Horizontal bar
        cell[16 - w//2 : 16 + w//2, :] = 1
        # Vertical bar
        cell[:, 16 - w//2 : 16 + w//2] = 1
        # Ensure symmetry (it is inherently symmetric)
        candidates.append(cell)

    # 2. Diamond Pattern (Dielectric Diamond)
    # Radius parameter r
    for r in range(5, 15, 2):
        cell = np.zeros((size, size), dtype=int)
        x = np.arange(size)
        y = np.arange(size)
        X, Y = np.meshgrid(x, y)
        # Center at 15.5
        cx, cy = 15.5, 15.5
        # Diamond equation: |x - cx| + |y - cy| <= r
        dist = np.abs(X - cx) + np.abs(Y - cy)
        cell[dist <= r] = 1
        candidates.append(cell)

    # 3. Circle Pattern (Dielectric Circle)
    for r in range(5, 15, 2):
        cell = np.zeros((size, size), dtype=int)
        x = np.arange(size)
        y = np.arange(size)
        X, Y = np.meshgrid(x, y)
        cx, cy = 15.5, 15.5
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        cell[dist <= r] = 1
        candidates.append(cell)

    # 4. Holey Square (Dielectric Background with Air Holes)
    # This is 1 - (Air Holes)
    # Holes at corners and edge centers?
    # Let's try holes at the 4 corners and 4 edge centers.
    for r_hole in range(2, 6, 1):
        cell = np.ones((size, size), dtype=int)
        # Corner holes
        # Center of corner pixels is (0,0), (31,0), etc?
        # Symmetry center is 15.5.
        # Let's place holes at (15.5, 15.5) +/- offsets.
        # Standard holey lattice: holes at (0,0), (0.5, 0), (0, 0.5) in unit cell coords?
        # In 32x32, let's put holes at the corners of the unit cell (0,0), (31,0), (0,31), (31,31)
        # and edge midpoints (15,0), (0,15), (31,15), (15,31)?
        
        # Let's try a simpler holey pattern: Holes at (8,8), (23,8), (8,23), (23,23)
        # and (15.5, 8), (15.5, 23), (8, 15.5), (23, 15.5)?
        
        # Let's use a generic hole placement:
        # Holes at (x0, y0) and symmetric equivalents.
        # Try (6,6), (25,6), (6,25), (25,25)
        for hx in [6, 25]:
            for hy in [6, 25]:
                # Create mask for hole
                Y, X = np.ogrid[:size, :size]
                mask = (X - hx)**2 + (Y - hy)**2 <= r_hole**2
                cell[mask] = 0
        
        # Also try holes at edge centers
        # (15, 6), (15, 25), (6, 15), (25, 15)
        for hx in [15, 16]: # Symmetric around 15.5
            for hy in [6, 25]:
                Y, X = np.ogrid[:size, :size]
                mask = (X - hx)**2 + (Y - hy)**2 <= r_hole**2
                cell[mask] = 0
        for hx in [6, 25]:
            for hy in [15, 16]:
                Y, X = np.ogrid[:size, :size]
                mask = (X - hx)**2 + (Y - hy)**2 <= r_hole**2
                cell[mask] = 0
                
        candidates.append(cell)

    # 5. Checkerboard variant (High frequency)
    # This might be too high frequency, but let's include one.
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if (i + j) % 2 == 0:
                cell[i, j] = 1
    candidates.append(cell)

    # Evaluate candidates
    best_score = -np.inf
    best_cell = None
    
    for cell in candidates:
        # Ensure strict symmetry
        # The geometric constructions above should be symmetric, but let's verify/enforce.
        # cell == cell[::-1] row-wise and column-wise and cell == its transpose
        # Our constructions are centered at 15.5, so they are symmetric.
        
        score = heuristic_score(cell)
        if score > best_score:
            best_score = score
            best_cell = cell

    # If no candidate is good, fallback to a default (e.g. 50% checkerboard)
    if best_cell is None:
        best_cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                if (i + j) % 2 == 0:
                    best_cell[i, j] = 1

    return best_cell.tolist()
# EVOLVE-BLOCK-END
