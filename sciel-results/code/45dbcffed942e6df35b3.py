import time
import math
import random
import numpy as np

def _make_symmetric(cell):
    """
    Enforce C4v symmetry about the cell center.
    The cell is 32x32. Center is at (15.5, 15.5) if 0-indexed.
    Symmetry:
    1. 180 deg rotation: cell[i][j] == cell[31-i][31-j]
    2. Reflection across vertical axis: cell[i][j] == cell[i][31-j]
    3. Reflection across horizontal axis: cell[i][j] == cell[31-i][j]
    4. Transpose: cell[i][j] == cell[j][i]
    
    Actually, for a 32x32 grid, the center of the cell is between pixels.
    The fundamental domain is 8x8 (top-left quadrant).
    Let's just enforce the symmetries on the full 32x32 array.
    """
    n = len(cell)
    # Use a numpy array for speed
    arr = np.array(cell, dtype=np.int8)
    
    # Symmetries to enforce:
    # 1. arr[i, j] = arr[n-1-i, n-1-j] (180 rot)
    # 2. arr[i, j] = arr[i, n-1-j] (vert ref)
    # 3. arr[i, j] = arr[n-1-i, j] (horiz ref)
    # 4. arr[i, j] = arr[j, i] (transpose)
    
    # It's easier to define the fundamental domain and expand it.
    # The fundamental domain is the top-left 16x16? No, 8x8.
    # Let's check: 32/2 = 16.
    # If we have vertical symmetry, col j == col 31-j.
    # If we have horizontal symmetry, row i == row 31-i.
    # If we have transpose symmetry, row i col j == row j col i.
    
    # The unique pixels are in the top-left 8x8 quadrant?
    # Let's verify.
    # If i < 16 and j < 16, is it unique?
    # If i < 16, j < 16.
    # 180 rot: 31-i > 15, 31-j > 15.
    # Vert ref: 31-j > 15.
    # Horiz ref: 31-i > 15.
    # Transpose: j, i. If i, j < 16, then j, i < 16.
    # So the top-left 16x16 quadrant is mirrored to the rest.
    # Within the top-left 16x16, is there more symmetry?
    # Transpose maps (i, j) to (j, i).
    # So the unique domain is the triangle i <= j in the top-left 16x16?
    # Or just the top-left 8x8?
    # If i < 8, j < 8.
    # Transpose: (j, i) is also in 8x8.
    # So we can just define the top-left 8x8 and expand.
    
    # Let's define the expansion from an 8x8 base.
    base = arr[0:8, 0:8]
    
    # Expand to 16x16 (top-left quadrant of the full cell)
    # The 16x16 quadrant is symmetric under transpose?
    # Yes, the full cell is symmetric under transpose.
    # So the top-left 16x16 must be symmetric under transpose.
    # Let's construct the 16x16 from the 8x8.
    # Wait, the 8x8 is the top-left corner of the 16x16.
    # Is the 16x16 determined by the 8x8?
    # The 16x16 has vertical and horizontal symmetry relative to the full cell?
    # No, the full cell has vertical symmetry at x=15.5.
    # So the left 16 columns are mirrored to the right 16 columns.
    # So the left 16x16 block is independent of the right 16x16 block.
    # Within the left 16x16 block, we have horizontal symmetry (top 8 rows mirrored to bottom 8 rows).
    # And we have transpose symmetry.
    
    # Let's just enforce the symmetries directly on the 32x32 array.
    # It's safer and less error-prone.
    
    # 1. Transpose
    arr = (arr + arr.T) // 2
    
    # 2. Vertical Reflection (left-right)
    # arr[:, j] == arr[:, n-1-j]
    # We can average them.
    arr = (arr[:, :] + arr[:, ::-1]) // 2
    
    # 3. Horizontal Reflection (top-bottom)
    # arr[i, :] == arr[n-1-i, :]
    arr = (arr[:, :] + arr[::-1, :]) // 2
    
    # 4. 180 Rotation
    # arr[i, j] == arr[n-1-i, n-1-j]
    # This is implied by the combination of the above?
    # Let's check.
    # If arr = arr.T, arr = arr[:, ::-1], arr = arr[::-1, :]
    # Then arr[i, j] = arr[j, i] = arr[j, n-1-i] = arr[n-1-j, n-1-i]
    # Yes, 180 rotation is implied.
    
    # Round to 0 or 1
    arr = (arr > 0.5).astype(np.int8)
    
    return arr.tolist()

def _compute_gap_fast(cell, n_pw=15):
    """
    Fast plane-wave calculation to estimate the gap.
    Uses a smaller number of plane waves for speed.
    """
    n = len(cell)
    eps = np.array(cell, dtype=np.float64)
    eps = 1 + 11 * eps  # eps_low=1, eps_high=12
    
    # Discretize the dielectric function
    # We need the Fourier coefficients of 1/eps for TM mode.
    # For a 32x32 pixel grid, we can compute the DFT.
    # However, the "exact" method uses the integral of 1/eps over the pixel.
    # For a binary pixel map, the integral is just the area of the pixel * (1/eps).
    # Since the pixels are square and uniform, the Fourier coefficient G is:
    # G = (1/N^2) * sum_{i,j} (1/eps[i,j]) * exp(-i G . r_ij)
    # where r_ij is the center of the pixel (i+0.5, j+0.5).
    
    inv_eps = 1.0 / eps
    
    # We want to compute the matrix A for the eigenvalue problem.
    # A_{G, G'} = delta_{G, G'} * |G|^2 + (1/|G-G'|^2) * V_{G-G'}
    # Wait, the standard form is:
    # sum_{G'} |G-G'|^2 V_{G-G'} E_{G'} = omega^2 E_G
    # For TM, V is 1/eps.
    # Let's use the standard formulation:
    # A_{G, G'} = (G-G')^2 * (1/eps)_{G-G'}
    # No, that's not right.
    # The wave equation for TM is:
    # ∇^2 (1/eps) ∇^2 H = k^2 H
    # In Fourier space:
    # sum_{G'} (k+G)^2 (1/eps)_{G-G'} H_{G'} = k^2 H_G
    # (k+G)^2 H_G + sum_{G' != G} (k+G)^2 (1/eps)_{G-G'} H_{G'} = k^2 H_G
    # This is an eigenvalue problem for k^2.
    
    # Let's build the matrix.
    # We need a set of G vectors.
    # We'll use a square grid of G vectors from -n_pw to n_pw.
    Gs = []
    for gx in range(-n_pw, n_pw + 1):
        for gy in range(-n_pw, n_pw + 1):
            Gs.append((gx, gy))
            
    num_G = len(Gs)
    
    # Precompute 1/eps Fourier coefficients
    # V_G = (1/N^2) * sum_{i,j} (1/eps[i,j]) * exp(-i 2pi (Gx*i + Gy*j) / N)
    # Note: r_ij = (i+0.5, j+0.5).
    # exp(-i 2pi (Gx*(i+0.5) + Gy*(j+0.5)) / N) = exp(-i pi (Gx+Gy)) * exp(-i 2pi (Gx*i + Gy*j) / N)
    # The phase factor is (-1)^(Gx+Gy).
    
    V = np.zeros(num_G, dtype=np.complex128)
    for idx, (gx, gy) in enumerate(Gs):
        phase = np.exp(-1j * np.pi * (gx + gy))
        # Use FFT for speed?
        # The sum is a 2D DFT of inv_eps.
        # np.fft.fft2(inv_eps) gives sum_{i,j} f[i,j] exp(-i 2pi (kx*i + ky*j)/N)
        # So V_G = phase * fft2(inv_eps)[gx mod N, gy mod N] / N^2
        
        # Map gx, gy to 0..N-1
        kx = gx % n
        ky = gy % n
        V[idx] = phase * np.fft.fft2(inv_eps)[kx, ky] / (n * n)
        
    # Build the matrix A
    # A_{G, G'} = (k+G)^2 * V_{G-G'}
    # Wait, the equation is:
    # sum_{G'} (k+G)^2 V_{G-G'} H_{G'} = k^2 H_G
    # This is a generalized eigenvalue problem? No, it's a standard one if we rearrange.
    # Actually, the standard plane wave method matrix is:
    # M_{G, G'} = (k+G)^2 V_{G-G'}
    # And we solve M H = k^2 H.
    # But V is 1/eps.
    # Let's check the units.
    # If eps is constant, V_0 = 1/eps, V_G!=0 = 0.
    # Then M_{G, G'} = (k+G)^2 (1/eps) delta_{G, G'}.
    # M H = (k+G)^2 (1/eps) H_G = k^2 H_G.
    # (k+G)^2 / eps = k^2.
    # This is correct for a uniform medium.
    
    # We need to evaluate this for several k points in the BZ.
    # The irreducible BZ of a square lattice is the triangle (0,0)-(pi,0)-(pi,pi).
    # We'll sample a few points.
    # (0,0), (pi/2, 0), (pi/2, pi/2), (pi, 0), (pi, pi).
    
    k_points = [
        (0.0, 0.0),
        (math.pi/2, 0.0),
        (math.pi/2, math.pi/2),
        (math.pi, 0.0),
        (math.pi, math.pi)
    ]
    
    min_omega_5 = float('inf')
    max_omega_4 = 0.0
    
    for kx, ky in k_points:
        k_vec = np.array([kx, ky])
        
        # Build matrix M
        M = np.zeros((num_G, num_G), dtype=np.complex128)
        
        for i, (gx, gy) in enumerate(Gs):
            G = np.array([gx, gy])
            k_plus_G = k_vec + G
            k_plus_G_sq = np.dot(k_plus_G, k_plus_G)
            
            for j, (gx2, gy2) in enumerate(Gs):
                G2 = np.array([gx2, gy2])
                diff_G = G - G2
                # Find index of diff_G in Gs
                # Since Gs is a regular grid, we can compute the index directly.
                # diff_G = (gx-gx2, gy-gy2)
                # The index in the list of Gs from -n_pw to n_pw.
                # If diff_G is within the range, we can find it.
                # Otherwise, V is 0 (since we only computed up to n_pw).
                # Actually, V_{G-G'} is periodic in the reciprocal lattice.
                # But for a finite supercell, it's not periodic.
                # However, for a 32x32 cell, the Fourier coefficients decay.
                # We can assume V is 0 if |G-G'| > n_pw.
                
                # Let's find the index.
                # The Gs list is sorted by gx, then gy.
                # We can create a dictionary for faster lookup.
                
                pass # We'll optimize this below.
                
        # Optimization: Precompute indices
        # Let's just do the double loop with a dictionary.
        
        pass
        
    # This is getting complicated. Let's use a simpler, faster approximation.
    # We can just compute the eigenvalues of the matrix for a single k-point, say X=(pi,0).
    # The gap is often largest at X or M.
    
    # Let's build the matrix for k=(pi, 0).
    kx, ky = math.pi, 0.0
    k_vec = np.array([kx, ky])
    
    M = np.zeros((num_G, num_G), dtype=np.complex128)
    
    # Map G to index
    G_to_idx = {}
    for idx, (gx, gy) in enumerate(Gs):
        G_to_idx[(gx, gy)] = idx
        
    for i, (gx, gy) in enumerate(Gs):
        G = np.array([gx, gy])
        k_plus_G = k_vec + G
        k_plus_G_sq = np.dot(k_plus_G, k_plus_G)
        
        for j, (gx2, gy2) in enumerate(Gs):
            G2 = np.array([gx2, gy2])
            diff_G = (gx - gx2, gy - gy2)
            
            if diff_G in G_to_idx:
                idx_diff = G_to_idx[diff_G]
                M[i, j] = k_plus_G_sq * V[idx_diff]
            # else M[i, j] remains 0
            
    # Solve eigenvalue problem
    # M H = k^2 H
    # k^2 are the eigenvalues.
    # omega = k (since c=1).
    
    eigvals = np.linalg.eigvalsh(M)
    eigvals = np.sort(eigvals.real)
    
    # The bands are the square roots of the eigenvalues.
    # We want the 4th and 5th bands.
    # Note: The zero mode might be present? No, k^2 > 0.
    # The lowest eigenvalue corresponds to the lowest band.
    
    # We need to be careful about the number of bands.
    # With n_pw=15, we have (2*15+1)^2 = 961 plane waves.
    # So we have 961 bands.
    # We want bands 4 and 5 (indices 3 and 4).
    
    omega_4 = np.sqrt(eigvals[3])
    omega_5 = np.sqrt(eigvals[4])
    
    gap = omega_5 - omega_4
    midgap = (omega_5 + omega_4) / 2
    
    if midgap > 0:
        return 2 * gap / midgap
    else:
        return 0.0

def design(problem):
    start_time = time.time()
    deadline = start_time + 220  # 10s margin
    
    size = problem["pixels"]
    
    # Initial design: 4 rods
    cell = [[0 for _ in range(size)] for _ in range(size)]
    centers = [
        (size/4, size/4),
        (size/4, 3*size/4),
        (3*size/4, size/4),
        (3*size/4, 3*size/4)
    ]
    r = 5.5
    for i in range(size):
        for j in range(size):
            x = i + 0.5
            y = j + 0.5
            for cx, cy in centers:
                if (x - cx)**2 + (y - cy)**2 < r**2:
                    cell[i][j] = 1
                    break
                    
    current_score = _compute_gap_fast(cell)
    best_cell = [row[:] for row in cell]
    best_score = current_score
    
    # Hill climbing
    # We will flip pixels in the fundamental domain (top-left 8x8)
    # and enforce symmetry.
    
    # Precompute the list of unique pixel indices
    unique_indices = []
    for i in range(8):
        for j in range(8):
            unique_indices.append((i, j))
            
    random.shuffle(unique_indices)
    
    while time.time() < deadline - 5:
        # Pick a random unique pixel to flip
        idx = random.choice(unique_indices)
        i, j = idx
        
        # Flip the pixel in the current cell
        cell[i][j] = 1 - cell[i][j]
        
        # Enforce symmetry
        cell = _make_symmetric(cell)
        
        # Evaluate
        new_score = _compute_gap_fast(cell)
        
        if new_score > best_score:
            best_score = new_score
            best_cell = [row[:] for row in cell]
        else:
            # Revert the flip
            cell[i][j] = 1 - cell[i][j]
            cell = _make_symmetric(cell)
            
    return best_cell
