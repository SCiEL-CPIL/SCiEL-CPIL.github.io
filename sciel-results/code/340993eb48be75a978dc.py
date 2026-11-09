import numpy as np
import time

# Precompute G vectors
# 289 plane waves => (2*N+1)^2 = 289 => 2N+1=17 => N=8
N_PW = 8
G_range = np.arange(-N_PW, N_PW + 1)
Gx, Gy = np.meshgrid(G_range, G_range, indexing='ij')
G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
Gx_flat = G_vectors[:, 0]
Gy_flat = G_vectors[:, 1]

# Precompute difference indices for mu(G_i - G_j)
Diff_x = Gx_flat[:, None] - Gx_flat[None, :]
Diff_y = Gy_flat[:, None] - Gy_flat[None, :]
idx_x = (Diff_x + 16) % 32
idx_y = (Diff_y + 16) % 32

def generate_k_points(num_points=30):
    """Generate 30 k-points in the irreducible BZ of a square lattice."""
    points = []
    # Gamma-X: kx in [0, pi], ky = 0
    for i in range(10):
        kx = (i + 0.5) * (np.pi / 10)
        points.append((kx, 0))
        
    # X-M: kx = pi, ky in [0, pi]
    for i in range(10):
        ky = (i + 0.5) * (np.pi / 10)
        points.append((np.pi, ky))
        
    # M-Gamma: kx = ky, k in [0, pi]
    for i in range(10):
        t = (i + 0.5) * (np.pi / 10)
        points.append((t, t))
        
    return np.array(points)

K_POINTS = generate_k_points()

def evaluate_cell(cell):
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
    
    # Extract full difference range -16..16 for mu(G_i - G_j)
    # mu_diff[i, j] = mu(G_i - G_j)
    # G_i - G_j is in -16..16.
    # mu_fft_shifted is 32x32, centered at 16.
    # Index for diff d is (d + 16) % 32.
    
    mu_diff = mu_fft_shifted[idx_x, idx_y] / (size * size)
    
    min_omega_5 = float('inf')
    max_omega_4 = float('-inf')
    
    for k in K_POINTS:
        kx, ky = k
        # Precompute (k + G)^2
        kg = k[None, :] + G_vectors * (2 * np.pi)
        kg_sq = np.sum(kg**2, axis=1)
        
        # A = mu_diff (symmetric)
        # B_diag = kg_sq
        # Solve A E = lambda B E
        # Symmetrize: S = B^{-1/2} A B^{-1/2}
        B_inv_sqrt = 1.0 / np.sqrt(kg_sq)
        S = mu_diff * B_inv_sqrt[:, None] * B_inv_sqrt[None, :]
        
        # Compute only the 5 smallest eigenvalues for speed
        # np.linalg.eigvalsh computes all, which is 289x289.
        # We can use scipy.linalg.eigvalsh with subset_by_index.
        from scipy.linalg import eigvalsh
        evals = eigvalsh(S, subset_by_index=[0, 4])
        
        # evals are omega^2.
        w4 = np.sqrt(evals[3])
        w5 = np.sqrt(evals[4])
        
        max_omega_4 = max(max_omega_4, w4)
        min_omega_5 = min(min_omega_5, w5)
        
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    if midgap == 0:
        return 0.0
    return gap / midgap

def flip_pixel(cell, i, j):
    """
    Flip pixel at i, j in the 8x8 quadrant, and update the full cell.
    i, j in 0..7.
    """
    size = 32
    # Get current value
    val = cell[i][j]
    new_val = 1 - val
    
    # Positions to update
    idxs = [
        (i, j), (i, size-1-j), (size-1-i, j), (size-1-i, size-1-j),
        (j, i), (j, size-1-i), (size-1-j, i), (size-1-j, size-1-i)
    ]
    seen = set()
    for r, c in idxs:
        if (r, c) not in seen:
            seen.add((r, c))
            cell[r][c] = new_val
            
    return cell

def design(problem):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Initial Candidates
    candidates = []
    size = 32
    
    # 1. Rod (small circle)
    cell_rod = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.2**2:
                cell_rod[i][j] = 1
    candidates.append(cell_rod)
    
    # 2. Ring (annulus)
    cell_ring = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            if 0.1 < r < 0.3:
                cell_ring[i][j] = 1
    candidates.append(cell_ring)
    
    # 3. Checkerboard
    cell_cb = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if (i + j) % 2 == 0:
                cell_cb[i][j] = 1
    candidates.append(cell_cb)
    
    # 4. Inverted Checkerboard
    cell_icb = [[1 - cell_cb[i][j] for j in range(size)] for i in range(size)]
    candidates.append(cell_icb)
    
    # 5. Cross-hole (dielectric cross with air hole)
    cell_cross = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if (abs(i - 15.5) < 2.5 or abs(j - 15.5) < 2.5):
                cell_cross[i][j] = 1
    # Punch a hole in the center
    for i in range(13, 19):
        for j in range(13, 19):
            if (i-15.5)**2 + (j-15.5)**2 < 1.5**2:
                cell_cross[i][j] = 0
    candidates.append(cell_cross)
    
    best_cell = None
    best_score = -float('inf')
    
    for c in candidates:
        s = evaluate_cell(c)
        if s > best_score:
            best_score = s
            best_cell = [row[:] for row in c]
            
    # Local Search with Multi-Flip Moves
    current_cell = [row[:] for row in best_cell]
    current_score = best_score
    
    while time.time() < deadline:
        # Pick random number of flips (1 or 2)
        n_flips = np.random.randint(1, 3)
        
        new_cell = [row[:] for row in current_cell]
        
        for _ in range(n_flips):
            i = np.random.randint(0, 8)
            j = np.random.randint(0, 8)
            flip_pixel(new_cell, i, j)
            
        new_score = evaluate_cell(new_cell)
        
        if new_score > current_score:
            current_cell = new_cell
            current_score = new_score
            best_cell = [row[:] for row in current_cell]
            best_score = current_score
            
    return best_cell
