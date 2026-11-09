import numpy as np
import time

# Precompute G vectors and difference indices
# Reduced N_PW to 4 (9x9=81 plane waves) for speed. 
# The evaluator uses 289, but 81 is sufficient for search guidance.
N_PW = 4
G_range = np.arange(-N_PW, N_PW + 1)
Gx, Gy = np.meshgrid(G_range, G_range, indexing='ij')
G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
Gx_flat = G_vectors[:, 0]
Gy_flat = G_vectors[:, 1]

# Precompute difference indices for mu lookup
# mu_coeffs will be indexed by (diff_x + 16) % 32, (diff_y + 16) % 32
Diff_x = Gx_flat[:, None] - Gx_flat[None, :]
Diff_y = Gy_flat[:, None] - Gy_flat[None, :]
idx_x = (Diff_x + 16) % 32
idx_y = (Diff_y + 16) % 32

def generate_k_points():
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
KX = K_POINTS[:, 0]
KY = K_POINTS[:, 1]

# Precompute kg_sq for all k-points to avoid redundant calculation in evaluate_cell
# kg_sq[k_idx, pw_idx] = (kx + 2*pi*Gx)^2 + (ky + 2*pi*Gy)^2
KG_SQ = np.zeros((len(K_POINTS), len(Gx_flat)))
for i, (kx, ky) in enumerate(K_POINTS):
    kgx = kx + 2 * np.pi * Gx_flat
    kgy = ky + 2 * np.pi * Gy_flat
    KG_SQ[i] = kgx**2 + kgy**2

def evaluate_cell_fast(cell):
    """
    Evaluate the gap-midgap ratio for a 32x32 cell.
    cell: numpy array 32x32 of 0/1.
    """
    size = 32
    # 1 = dielectric (eps=12), 0 = air (eps=1)
    eps = np.where(cell == 1, 12.0, 1.0)
    mu = 1.0 / eps
    
    # Compute Fourier coefficients of mu
    mu_fft = np.fft.fft2(mu)
    mu_fft_shifted = np.fft.fftshift(mu_fft) / (size * size)
    
    min_omega_5 = float('inf')
    max_omega_4 = float('-inf')
    
    # Precompute mu_diff for all k-points? No, mu_diff is independent of k.
    # mu_diff is just the matrix of Fourier coefficients for the differences.
    # It is the same for all k-points.
    mu_diff = mu_fft_shifted[idx_x, idx_y]
    
    for i in range(len(K_POINTS)):
        kg_sq = KG_SQ[i]
        
        # Symmetrize: S = B^{-1/2} A B^{-1/2}
        # B is diagonal with entries kg_sq
        # A is mu_diff
        B_inv_sqrt = 1.0 / np.sqrt(kg_sq)
        S = mu_diff * B_inv_sqrt[:, None] * B_inv_sqrt[None, :]
        
        # Solve for 5 smallest eigenvalues using dense eigensolver (NumPy)
        # S is symmetric, so use eigvalsh
        evals = np.linalg.eigvalsh(S)
        
        # Ensure we have at least 5 eigenvalues
        if len(evals) < 5:
            continue
            
        w4 = np.sqrt(evals[3])
        w5 = np.sqrt(evals[4])
        
        max_omega_4 = max(max_omega_4, w4)
        min_omega_5 = min(min_omega_5, w5)
        
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    if midgap == 0:
        return 0.0
    return gap / midgap

def flip_pixel_np(cell_np, i, j):
    """
    Flip pixel at i, j in the 8x8 quadrant, and update the full cell.
    i, j in 0..7.
    """
    size = 32
    val = cell_np[i][j]
    new_val = 1 - val
    
    idxs = [
        (i, j), (i, size-1-j), (size-1-i, j), (size-1-i, size-1-j),
        (j, i), (j, size-1-i), (size-1-j, i), (size-1-j, size-1-i)
    ]
    seen = set()
    for r, c in idxs:
        if (r, c) not in seen:
            seen.add((r, c))
            cell_np[r][c] = new_val
            
    return cell_np

def design(problem):
    start_time = time.time()
    deadline = start_time + 220  # 10s safety margin before 230s limit
    
    size = 32
    candidates = []
    
    # 1. Small Rod (radius 0.2)
    cell_rod = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.2**2:
                cell_rod[i][j] = 1
    candidates.append(cell_rod)
    
    # 2. Medium Rod (radius 0.3)
    cell_rod2 = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.3**2:
                cell_rod2[i][j] = 1
    candidates.append(cell_rod2)
    
    # 3. Checkerboard
    cell_cb = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if (i + j) % 2 == 0:
                cell_cb[i][j] = 1
    candidates.append(cell_cb)
    
    # 4. Inverted Checkerboard
    cell_icb = 1 - cell_cb
    candidates.append(cell_icb)
    
    # 5. Cross (bars)
    cell_cross = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if abs(i - 15.5) < 2.5 or abs(j - 15.5) < 2.5:
                cell_cross[i][j] = 1
    candidates.append(cell_cross)
    
    # 6. Ring (annulus)
    cell_ring = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            if 0.15 < r < 0.35:
                cell_ring[i][j] = 1
    candidates.append(cell_ring)
    
    # 7. Large Cross
    cell_cross2 = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if abs(i - 15.5) < 4.5 or abs(j - 15.5) < 4.5:
                cell_cross2[i][j] = 1
    candidates.append(cell_cross2)
    
    # 8. Cross-hole
    cell_cross_hole = np.ones((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            if r < 0.1:
                cell_cross_hole[i][j] = 0
            elif abs(x) > 0.15 and abs(y) > 0.15:
                cell_cross_hole[i][j] = 0
    candidates.append(cell_cross_hole)
    
    # 9. Square ring
    cell_sq_ring = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = max(abs(x), abs(y))
            if 0.12 < r < 0.38:
                cell_sq_ring[i][j] = 1
    candidates.append(cell_sq_ring)
    
    best_cell = None
    best_score = -float('inf')
    
    for c in candidates:
        s = evaluate_cell_fast(c)
        if s > best_score:
            best_score = s
            best_cell = c.copy()
            
    # Local Search with Simulated Annealing
    current_cell = best_cell.copy()
    current_score = best_score
    
    def multi_flip(cell_np, n_flips):
        """Flip n_flips random quadrant pixels."""
        new_cell = cell_np.copy()
        for _ in range(n_flips):
            i = np.random.randint(0, 8)
            j = np.random.randint(0, 8)
            flip_pixel_np(new_cell, i, j)
        return new_cell

    T = 0.05
    T_min = 0.001
    cooling_rate = 0.995
    no_improve_count = 0
    
    while time.time() < deadline:
        # Choose between single flip and multi-flip
        if np.random.random() < 0.7:
            # Single flip
            i = np.random.randint(0, 8)
            j = np.random.randint(0, 8)
            new_cell = current_cell.copy()
            flip_pixel_np(new_cell, i, j)
        else:
            # Multi-flip (2-3 flips) to escape local minima
            n_flips = np.random.randint(2, 4)
            new_cell = multi_flip(current_cell, n_flips)
        
        new_score = evaluate_cell_fast(new_cell)
        
        # Simulated Annealing acceptance
        if new_score > current_score:
            current_cell = new_cell
            current_score = new_score
            no_improve_count = 0
            if new_score > best_score:
                best_score = new_score
                best_cell = new_cell.copy()
        else:
            delta = new_score - current_score
            if T > T_min:
                if np.random.random() < np.exp(delta / T):
                    current_cell = new_cell
                    current_score = new_score
                    no_improve_count += 1
            else:
                no_improve_count += 1
            T *= cooling_rate
        
        # Restart if stuck too long
        if no_improve_count > 200:
            # Random restart from best
            current_cell = best_cell.copy()
            current_score = best_score
            T = 0.05
            no_improve_count = 0
            
    return [list(row) for row in best_cell]
