import numpy as np
import time

# Precompute G vectors
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

# Precompute k-points and (k+G)^2 for each k-point
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

# Precompute inverse sqrt of KG_SQ for each k-point
KG_INV_SQRT = [1.0 / np.sqrt(kg_sq) for kg_sq in KG_SQ]

def evaluate_cell(cell, verbose=False):
    """
    Evaluate the gap-midgap ratio for a 32x32 cell.
    cell: list of lists of 0/1.
    """
    size = len(cell)
    # 1 = dielectric (eps=12), 0 = air (eps=1)
    # mu = 1/eps
    eps_arr = np.array(cell, dtype=float)
    eps_arr[eps_arr == 1] = 12.0
    eps_arr[eps_arr == 0] = 1.0
    mu = 1.0 / eps_arr
    
    # Compute Fourier coefficients of mu
    mu_fft = np.fft.fft2(mu)
    mu_fft_shifted = np.fft.fftshift(mu_fft)
    mu_diff = mu_fft_shifted[idx_x, idx_y] / (size * size)
    
    min_omega_5 = float('inf')
    max_omega_4 = float('-inf')
    
    for ki in range(len(K_POINTS)):
        B_inv_sqrt = KG_INV_SQRT[ki]
        S = mu_diff * B_inv_sqrt[:, None] * B_inv_sqrt[None, :]
        
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

def symmetrize(cell):
    """
    Enforce square symmetry: 
    1. Horizontal flip (row wise reverse)
    2. Vertical flip (column wise reverse)
    3. Transpose
    Actually, the problem says:
    cell == cell[::-1] row-wise and column-wise and cell == its transpose.
    This implies 4-fold symmetry.
    The top-left 8x8 quadrant defines the whole cell.
    """
    size = len(cell)
    half = size // 2
    # Extract top-left quadrant
    # i in 0..7, j in 0..7
    q = [row[:half] for row in cell[:half]]
    
    new_cell = [[0]*size for _ in range(size)]
    
    for i in range(half):
        for j in range(half):
            val = q[i][j]
            # (i, j)
            new_cell[i][j] = val
            # (i, size-1-j)
            new_cell[i][size-1-j] = val
            # (size-1-i, j)
            new_cell[size-1-i][j] = val
            # (size-1-i, size-1-j)
            new_cell[size-1-i][size-1-j] = val
            
            # Transpose symmetry means (j, i) must be val.
            # If we fill all 4 corners, is transpose satisfied?
            # If new_cell[i][j] = val, then new_cell[j][i] should be val.
            # My filling logic:
            # If i != j:
            # (i, j) = val
            # (j, i) will be set when we iterate i'=j, j'=i?
            # Wait, I only iterate i, j in 0..7.
            # If I set (i, j), I don't explicitly set (j, i) unless j,i is in the quadrant?
            # Yes, if i,j < 8, then (j,i) is also in the quadrant.
            # But I must ensure consistency.
            # If I set q[i][j] and q[j][i] independently, they might differ.
            # The problem requires the *final* cell to be symmetric.
            # So I should take the average or just ensure the input q is symmetric?
            # Better: Just define the top-left 8x8, and construct the rest.
            # But the top-left 8x8 itself doesn't need to be symmetric internally, 
            # only the full 32x32 needs to be.
            # However, if new_cell is transpose of itself, then new_cell[i][j] == new_cell[j][i].
            # My construction:
            # new_cell[i][j] = q[i][j]
            # new_cell[j][i] = q[j][i]
            # For new_cell to be transpose symmetric, we need q[i][j] == q[j][i].
            # So the top-left quadrant MUST be symmetric.
            
    # To be safe, let's enforce symmetry on the quadrant too.
    for i in range(half):
        for j in range(i, half):
            # Average or pick one?
            # We will use a generator that produces symmetric quads.
            pass

    return new_cell

def generate_random_cell():
    size = 32
    half = 16 # No, quadrant is 8x8
    q = [[0]*8 for _ in range(8)]
    for i in range(8):
        for j in range(8):
            q[i][j] = 1 if np.random.random() > 0.5 else 0
    # Enforce symmetry in quadrant for transpose
    for i in range(8):
        for j in range(i, 8):
            if q[i][j] != q[j][i]:
                # Make them same
                val = 1 if np.random.random() > 0.5 else 0
                q[i][j] = val
                q[j][i] = val
                
    cell = [[0]*size for _ in range(size)]
    for i in range(8):
        for j in range(8):
            val = q[i][j]
            cell[i][j] = val
            cell[i][size-1-j] = val
            cell[size-1-i][j] = val
            cell[size-1-i][size-1-j] = val
    return cell

def flip_pixel(cell, i, j):
    """
    Flip pixel at i, j in the 8x8 quadrant, and update the full cell.
    i, j in 0..7.
    """
    size = 32
    half = 16 # Not used directly
    # Get current value
    val = cell[i][j]
    new_val = 1 - val
    
    # Positions to update
    # (i, j), (i, 31-j), (31-i, j), (31-i, 31-j)
    # And their transposes:
    # (j, i), (j, 31-i), (31-j, i), (31-j, 31-i)
    
    # If i==j, these sets overlap.
    # Let's just collect unique indices.
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

def quadrant_to_cell(q):
    """
    Convert an 8x8 symmetric quadrant to a 32x32 cell.
    """
    size = 32
    cell = [[0]*size for _ in range(size)]
    for i in range(8):
        for j in range(8):
            val = q[i][j]
            cell[i][j] = val
            cell[i][size-1-j] = val
            cell[size-1-i][j] = val
            cell[size-1-i][size-1-j] = val
    return cell

def cell_to_quadrant(cell):
    """
    Extract the 8x8 quadrant from a 32x32 cell.
    """
    return [row[:8] for row in cell[:8]]

def compute_sensitivity_map(cell, basis_cells):
    """
    Compute a sensitivity map for the gap-midgap ratio using finite differences
    on a set of basis cells. Returns an 8x8 array where each entry indicates
    how much flipping that pixel (and its symmetric partners) affects the score.
    
    cell: 32x32 current cell
    basis_cells: list of 32x32 cells to use as perturbations
    """
    size = 32
    current_score = evaluate_cell(cell)
    sensitivity = np.zeros((8, 8))
    
    for basis in basis_cells:
        # Create a perturbed cell: flip pixels where basis differs from current
        new_cell = [row[:] for row in cell]
        for i in range(8):
            for j in range(8):
                if basis[i][j] != cell[i][j]:
                    flip_pixel(new_cell, i, j)
        
        new_score = evaluate_cell(new_cell)
        delta = new_score - current_score
        
        # Attribute the change to each flipped pixel
        for i in range(8):
            for j in range(8):
                if basis[i][j] != cell[i][j]:
                    sensitivity[i][j] += delta
    
    return sensitivity, current_score

def build_basis_cells():
    """
    Build a diverse set of basis cells for sensitivity computation.
    These should cover different structural features.
    """
    size = 32
    bases = []
    
    # Corner-heavy
    b1 = [[0]*size for _ in range(size)]
    for i in range(8):
        for j in range(8):
            if i < 3 and j < 3:
                b1[i][j] = 1
                b1[i][size-1-j] = 1
                b1[size-1-i][j] = 1
                b1[size-1-i][size-1-j] = 1
    bases.append(b1)
    
    # Center-heavy
    b2 = [[0]*size for _ in range(size)]
    for i in range(8):
        for j in range(8):
            if i >= 5 and j >= 5:
                b2[i][j] = 1
                b2[i][size-1-j] = 1
                b2[size-1-i][j] = 1
                b2[size-1-i][size-1-j] = 1
    bases.append(b2)
    
    # Diagonal
    b3 = [[0]*size for _ in range(size)]
    for i in range(8):
        for j in range(8):
            if abs(i - j) <= 1:
                b3[i][j] = 1
                b3[i][size-1-j] = 1
                b3[size-1-i][j] = 1
                b3[size-1-i][size-1-j] = 1
    bases.append(b3)
    
    # Anti-diagonal
    b4 = [[0]*size for _ in range(size)]
    for i in range(8):
        for j in range(8):
            if abs(i - (7-j)) <= 1:
                b4[i][j] = 1
                b4[i][size-1-j] = 1
                b4[size-1-i][j] = 1
                b4[size-1-i][size-1-j] = 1
    bases.append(b4)
    
    # Ring-like
    b5 = [[0]*size for _ in range(size)]
    for i in range(8):
        for j in range(8):
            x = (i + 0.5) / 8 - 0.5
            y = (j + 0.5) / 8 - 0.5
            r = np.sqrt(x*x + y*y)
            if 0.2 < r < 0.4:
                b5[i][j] = 1
                b5[i][size-1-j] = 1
                b5[size-1-i][j] = 1
                b5[size-1-i][size-1-j] = 1
    bases.append(b5)
    
    # Cross-like
    b6 = [[0]*size for _ in range(size)]
    for i in range(8):
        for j in range(8):
            if i in [3,4] or j in [3,4]:
                b6[i][j] = 1
                b6[i][size-1-j] = 1
                b6[size-1-i][j] = 1
                b6[size-1-i][size-1-j] = 1
    bases.append(b6)
    
    # Checkerboard
    b7 = [[0]*size for _ in range(size)]
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                b7[i][j] = 1
                b7[i][size-1-j] = 1
                b7[size-1-i][j] = 1
                b7[size-1-i][size-1-j] = 1
    bases.append(b7)
    
    # Random sparse
    b8 = [[0]*size for _ in range(size)]
    np.random.seed(42)
    for i in range(8):
        for j in range(8):
            if np.random.random() > 0.7:
                b8[i][j] = 1
                b8[i][size-1-j] = 1
                b8[size-1-i][j] = 1
                b8[size-1-i][size-1-j] = 1
    # Enforce transpose symmetry
    for i in range(8):
        for j in range(i, 8):
            if b8[i][j] != b8[j][i]:
                val = 1 if np.random.random() > 0.5 else 0
                b8[i][j] = val
                b8[j][i] = val
                b8[i][size-1-j] = val
                b8[size-1-i][j] = val
                b8[size-1-i][size-1-j] = val
                b8[j][size-1-i] = val
                b8[size-1-j][i] = val
                b8[size-1-j][size-1-i] = val
    bases.append(b8)
    
    return bases

def design(problem):
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    size = 32
    candidates = []
    
    # Rod
    cell_rod = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.2**2:
                cell_rod[i][j] = 1
    candidates.append(cell_rod)
    
    # Checkerboard
    cell_cb = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if (i + j) % 2 == 0:
                cell_cb[i][j] = 1
    candidates.append(cell_cb)
    
    # Inverted Checkerboard
    cell_icb = [[1 - cell_cb[i][j] for j in range(size)] for i in range(size)]
    candidates.append(cell_icb)
    
    # Cross (Horizontal and Vertical bars)
    cell_cross = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if (abs(i - 15.5) < 2.5 or abs(j - 15.5) < 2.5):
                cell_cross[i][j] = 1
    candidates.append(cell_cross)
    
    # Ring (annulus) - dielectric ring with air center
    cell_ring = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            if 0.15 < r < 0.35:
                cell_ring[i][j] = 1
    candidates.append(cell_ring)
    
    # Thick ring
    cell_ring2 = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            if 0.12 < r < 0.42:
                cell_ring2[i][j] = 1
    candidates.append(cell_ring2)
    
    # Air ring (inverted ring)
    cell_air_ring = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            if 0.15 < r < 0.35:
                cell_air_ring[i][j] = 0
            else:
                cell_air_ring[i][j] = 1
    candidates.append(cell_air_ring)
    
    # Diamond shape
    cell_diamond = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) + abs(y) < 0.25:
                cell_diamond[i][j] = 1
    candidates.append(cell_diamond)
    
    # Hexagonal-like pattern
    cell_hex = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            # Approximate hexagonal symmetry
            if max(abs(x), abs(y), abs(x+y)/1.5, abs(x-y)/1.5) < 0.22:
                cell_hex[i][j] = 1
    candidates.append(cell_hex)
    
    # Cross with hole - cross shape with air center
    cell_cross_hole = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            in_cross = (abs(x) < 0.12 or abs(y) < 0.12)
            in_hole = r < 0.08
            if in_cross and not in_hole:
                cell_cross_hole[i][j] = 1
    candidates.append(cell_cross_hole)
    
    # Square ring
    cell_sq_ring = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            max_coord = max(abs(x), abs(y))
            if 0.15 < max_coord < 0.35:
                cell_sq_ring[i][j] = 1
    candidates.append(cell_sq_ring)
    
    best_cell = None
    best_score = -float('inf')
    
    for c in candidates:
        s = evaluate_cell(c)
        if s > best_score:
            best_score = s
            best_cell = [row[:] for row in c]
            
    # Build basis cells for sensitivity-guided search
    basis_cells = build_basis_cells()
    
    # Gradient-guided local search using sensitivity map
    current_cell = [row[:] for row in best_cell]
    current_score = best_score
    stagnation = 0
    sensitivity = None
    
    while time.time() < deadline:
        # Recompute sensitivity periodically or when stagnation increases
        if sensitivity is None or stagnation % 5 == 0:
            sensitivity, current_score = compute_sensitivity_map(current_cell, basis_cells)
        
        # Find the most beneficial single-pixel flip based on sensitivity
        # For each pixel, estimate the effect of flipping it
        best_flip = None
        best_estimated_gain = 0
        
        for i in range(8):
            for j in range(8):
                # Estimate gain from flipping pixel (i,j)
                # If current pixel is 0, flipping to 1 adds sensitivity[i][j]
                # If current pixel is 1, flipping to 0 subtracts sensitivity[i][j]
                if current_cell[i][j] == 0:
                    estimated_gain = sensitivity[i][j]
                else:
                    estimated_gain = -sensitivity[i][j]
                
                if estimated_gain > best_estimated_gain:
                    best_estimated_gain = estimated_gain
                    best_flip = (i, j)
        
        # Perform the best flip if it looks beneficial
        if best_flip is not None and best_estimated_gain > 0.001:
            i, j = best_flip
            new_cell = [row[:] for row in current_cell]
            flip_pixel(new_cell, i, j)
            new_score = evaluate_cell(new_cell)
            
            if new_score > current_score:
                current_cell = new_cell
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_cell = [row[:] for row in current_cell]
                stagnation = 0
                sensitivity = None  # Force recomputation
            else:
                stagnation += 1
        else:
            # No beneficial flip found; try random exploration
            if np.random.random() < 0.3:
                # Multi-flip random move
                n_flips = min(2 + stagnation // 10, 5)
                new_cell = [row[:] for row in current_cell]
                for _ in range(n_flips):
                    i = np.random.randint(0, 8)
                    j = np.random.randint(0, 8)
                    flip_pixel(new_cell, i, j)
                new_score = evaluate_cell(new_cell)
                if new_score > current_score:
                    current_cell = new_cell
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_cell = [row[:] for row in current_cell]
                    stagnation = 0
                    sensitivity = None
                else:
                    stagnation += 1
            else:
                # Single random flip
                i = np.random.randint(0, 8)
                j = np.random.randint(0, 8)
                new_cell = [row[:] for row in current_cell]
                flip_pixel(new_cell, i, j)
                new_score = evaluate_cell(new_cell)
                if new_score > current_score:
                    current_cell = new_cell
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_cell = [row[:] for row in current_cell]
                    stagnation = 0
                    sensitivity = None
                else:
                    stagnation += 1
        
        # Periodic restart from best if long stagnation
        if stagnation > 30 and time.time() < deadline - 10:
            current_cell = [row[:] for row in best_cell]
            current_score = best_score
            sensitivity = None
            stagnation = 0
    
    return best_cell
