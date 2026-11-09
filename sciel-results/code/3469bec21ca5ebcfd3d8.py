import numpy as np
import time

# Precompute G vectors for the 289-plane-wave method (N=8)
N_PW = 8
G_range = np.arange(-N_PW, N_PW + 1)
Gx, Gy = np.meshgrid(G_range, G_range, indexing='ij')
G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
Gx_flat = G_vectors[:, 0]
Gy_flat = G_vectors[:, 1]
Diff_x = Gx_flat[:, None] - Gx_flat[None, :]
Diff_y = Gy_flat[:, None] - Gy_flat[None, :]
idx_x = (Diff_x + 16) % 32
idx_y = (Diff_y + 16) % 32

# Precompute k-points (30 points in irreducible BZ)
def generate_k_points(num_points=30):
    points = []
    # Gamma-X: kx in [0, pi], ky = 0
    for i in range(10):
        kx = (i + 0.5) * (np.pi / 10)
        points.append((kx, 0))
    # X-M: kx = pi, ky in [0, pi]
    for i in range(10):
        ky = (i + 0.5) * (np.pi / 10)
        points.append((np.pi, ky))
    # M-Gamma: kx = ky
    for i in range(10):
        t = (i + 0.5) * (np.pi / 10)
        points.append((t, t))
    return np.array(points)

K_POINTS = generate_k_points()
KG_SQ = []
for k in K_POINTS:
    kx, ky = k
    kg = k[None, :] + G_vectors * (2 * np.pi)
    kg_sq = np.sum(kg**2, axis=1)
    KG_SQ.append(kg_sq)
KG_SQ = np.array(KG_SQ)

def evaluate_cell(cell, verbose=False):
    """
    Evaluate the gap-midgap ratio for a 32x32 cell.
    cell: list of lists of 0/1.
    """
    size = len(cell)
    # 1 = dielectric (eps=12), 0 = air (eps=1)
    # mu = 1/eps
    eps = np.array([[12 if c == 1 else 1 for c in row] for row in cell], dtype=float)
    mu = 1.0 / eps
    
    # Compute Fourier coefficients of mu
    mu_fft = np.fft.fft2(mu)
    mu_fft_shifted = np.fft.fftshift(mu_fft)
    mu_diff = mu_fft_shifted[idx_x, idx_y] / (size * size)
    
    min_omega_5 = float('inf')
    max_omega_4 = float('-inf')
    
    for ki in range(len(K_POINTS)):
        kg_sq = KG_SQ[ki]
        B_inv_sqrt = 1.0 / np.sqrt(kg_sq)
        S = mu_diff * B_inv_sqrt[:, None] * B_inv_sqrt[None, :]
        
        # Only compute 5 smallest eigenvalues for speed
        evals = np.linalg.eigvalsh(S)
        w4 = np.sqrt(evals[3])
        w5 = np.sqrt(evals[4])
        
        max_omega_4 = max(max_omega_4, w4)
        min_omega_5 = min(min_omega_5, w5)
        
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    if midgap == 0:
        return 0.0
    return gap / midgap

def generate_cell_from_params(angle_deg, size_sq, center_size, cx, cy):
    """
    Generate a 32x32 cell defined by the union of four rotated squares (diamonds)
    centered at quarter-points, plus a central square background.
    
    Parameters:
    - angle_deg: rotation angle of the squares (0 to 45)
    - size_sq: side length of the rotated squares
    - center_size: side length of the central square
    - cx, cy: coordinates of the quarter-points (relative to center, in pixels)
    """
    size = 32
    half = size / 2.0
    cell = np.zeros((size, size), dtype=int)
    
    # Grid of pixel centers
    # Pixel (i, j) center is at (i - half + 0.5, j - half + 0.5)
    # Let's use 0-based indexing for the array, but coordinates relative to center (15.5, 15.5)
    # Coordinate system: x in [-16, 16), y in [-16, 16)
    # Pixel (i, j) corresponds to x = i - 15.5, y = j - 15.5
    x_coords = np.arange(size) - 15.5
    y_coords = np.arange(size) - 15.5
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    # Define the four centers
    centers = [
        (cx, cy),
        (-cx, cy),
        (-cx, -cy),
        (cx, -cy)
    ]
    
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Mask for central square
    # Centered at (0,0), side length center_size
    if center_size > 0:
        half_cs = center_size / 2.0
        cell[(np.abs(X) < half_cs) & (np.abs(Y) < half_cs)] = 1
    
    # Mask for rotated squares
    for c_x, c_y in centers:
        # Translate to center of the square
        X_t = X - c_x
        Y_t = Y - c_y
        
        # Rotate coordinates by -angle to align with square axes
        X_r = X_t * cos_a + Y_t * sin_a
        Y_r = -X_t * sin_a + Y_t * cos_a
        
        # Check if inside square of side length size_sq
        half_s = size_sq / 2.0
        mask = (np.abs(X_r) < half_s) & (np.abs(Y_r) < half_s)
        cell[mask] = 1
        
    return cell.tolist()

def design(problem):
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    best_cell = None
    best_score = -float('inf')
    
    # Strategy: Grid search over parameters
    # Parameters:
    # angle: 0 to 45 degrees
    # size_sq: side length of the 4 squares
    # center_size: side length of central square
    # cx, cy: position of the 4 squares (quarter points)
    
    # Coarse search
    # The "quarter points" in a 32x32 cell are typically around (8, 8), (8, -8), etc.
    # Let's vary cx, cy in range [4, 12]
    
    angles = np.arange(0, 46, 5) # 0, 5, ..., 45
    sizes = np.arange(4, 16, 2) # 4, 6, ..., 14
    center_sizes = [0, 4, 8]
    positions = np.arange(4, 13, 2) # 4, 6, ..., 12
    
    # To save time, we can fix cx=cy initially and vary them later
    # Or just iterate all. 
    # 10 angles * 6 sizes * 3 centers * 5 pos * 5 pos = 4500 evaluations.
    # Each eval ~10-20ms? 4500 * 0.02 = 90s. Might be tight.
    # Let's reduce.
    
    angles = np.arange(0, 46, 10) # 0, 10, 20, 30, 40
    sizes = np.arange(6, 15, 2) # 6, 8, 10, 12, 14
    center_sizes = [0, 6]
    positions = np.arange(5, 12, 2) # 5, 7, 9, 11
    
    # Total: 5 * 5 * 2 * 4 * 4 = 800 evaluations.
    # 800 * 0.02 = 16s. Safe.
    
    for a in angles:
        if time.time() > deadline:
            break
        for s in sizes:
            if time.time() > deadline:
                break
            for cs in center_sizes:
                if time.time() > deadline:
                    break
                for cx in positions:
                    if time.time() > deadline:
                        break
                    for cy in positions:
                        # Due to symmetry, cx and cy are effectively the same for the shape definition
                        # but let's just iterate.
                        cell = generate_cell_from_params(a, s, cs, cx, cy)
                        score = evaluate_cell(cell)
                        if score > best_score:
                            best_score = score
                            best_cell = [row[:] for row in cell]
        
    # If we have time, do a finer search around the best parameters?
    # Or just return the best.
    
    # Ensure best_cell is not None
    if best_cell is None:
        # Fallback: simple cross
        size = 32
        best_cell = [[0]*size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                if abs(i - 15.5) < 2.5 or abs(j - 15.5) < 2.5:
                    best_cell[i][j] = 1
    
    return best_cell
