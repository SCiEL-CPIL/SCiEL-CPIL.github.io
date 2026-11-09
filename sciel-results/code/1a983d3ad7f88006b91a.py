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

def superellipse_cell(n1, n2, r1, r2, invert=False):
    """
    Generate a cell from intersection of two superellipses.
    n1, n2: exponents for the two superellipses
    r1, r2: radii
    invert: if True, dielectric background with air holes
    """
    size = 32
    cell = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            
            # Superellipse 1: |x|^(2/n1) + |y|^(2/n1) <= r1^(2/n1)
            se1 = (abs(x)**(2.0/n1) + abs(y)**(2.0/n1)) <= (r1**(2.0/n1))
            # Superellipse 2: |x|^(2/n2) + |y|^(2/n2) <= r2^(2/n2)
            se2 = (abs(x)**(2.0/n2) + abs(y)**(2.0/n2)) <= (r2**(2.0/n2))
            
            if invert:
                # Dielectric background, air holes where either superellipse is true
                cell[i][j] = 1 if (not se1 and not se2) else 0
            else:
                # Air background, dielectric where either superellipse is true
                cell[i][j] = 1 if (se1 or se2) else 0
    return cell

def rose_curve_cell(k, a, b, invert=False):
    """
    Generate a cell from a rose curve.
    k: number of petals
    a, b: scaling parameters
    invert: if True, dielectric background with air holes
    """
    size = 32
    cell = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            theta = np.arctan2(y, x)
            
            # Rose curve: r = a * cos(k*theta)
            rose_r = a * abs(np.cos(k * theta))
            
            # Also add a circular component
            circ_r = b
            
            # Pixel is dielectric if inside rose curve OR circle
            inside_rose = r <= rose_r
            inside_circ = r <= circ_r
            
            if invert:
                cell[i][j] = 1 if (not inside_rose and not inside_circ) else 0
            else:
                cell[i][j] = 1 if (inside_rose or inside_circ) else 0
    return cell

def four_leaf_cell(r_inner, r_outer, n_petals, invert=False):
    """
    Generate a cell with four-fold symmetric petals.
    r_inner, r_outer: inner and outer radii
    n_petals: number of petals (should be multiple of 4 for C4v)
    invert: if True, dielectric background with air holes
    """
    size = 32
    cell = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            theta = np.arctan2(y, x)
            
            # Petal shape using cosine modulation
            petal_mod = 0.5 + 0.5 * np.cos(n_petals * theta)
            effective_r = r_inner + (r_outer - r_inner) * petal_mod
            
            inside = r <= effective_r
            
            if invert:
                cell[i][j] = 1 if not inside else 0
            else:
                cell[i][j] = 1 if inside else 0
    return cell

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
    
    # === NEW: Algebraic curve-based candidates ===
    
    # Superellipse intersections (various parameters)
    # Normal (air background, dielectric features)
    for n1 in [2, 4, 6, 8]:
        for n2 in [2, 4, 6, 8]:
            if n1 == n2:
                continue
            for r1 in [0.15, 0.2, 0.25, 0.3]:
                for r2 in [0.15, 0.2, 0.25, 0.3]:
                    if abs(r1 - r2) < 0.05:
                        continue
                    cell = superellipse_cell(n1, n2, r1, r2, invert=False)
                    candidates.append(cell)
    
    # Inverted superellipse intersections (dielectric background, air holes)
    for n1 in [2, 4, 6]:
        for n2 in [2, 4, 6]:
            if n1 == n2:
                continue
            for r1 in [0.15, 0.2, 0.25, 0.3]:
                for r2 in [0.15, 0.2, 0.25, 0.3]:
                    if abs(r1 - r2) < 0.05:
                        continue
                    cell = superellipse_cell(n1, n2, r1, r2, invert=True)
                    candidates.append(cell)
    
    # Rose curves
    for k in [4, 8]:
        for a in [0.15, 0.2, 0.25, 0.3]:
            for b in [0.05, 0.1, 0.15]:
                cell = rose_curve_cell(k, a, b, invert=False)
                candidates.append(cell)
                cell = rose_curve_cell(k, a, b, invert=True)
                candidates.append(cell)
    
    # Four-leaf patterns
    for r_inner in [0.05, 0.1, 0.15]:
        for r_outer in [0.2, 0.25, 0.3, 0.35]:
            if r_outer <= r_inner:
                continue
            for n_petals in [4, 8]:
                cell = four_leaf_cell(r_inner, r_outer, n_petals, invert=False)
                candidates.append(cell)
                cell = four_leaf_cell(r_inner, r_outer, n_petals, invert=True)
                candidates.append(cell)
    
    best_cell = None
    best_score = -float('inf')
    
    for c in candidates:
        s = evaluate_cell(c)
        if s > best_score:
            best_score = s
            best_cell = [row[:] for row in c]
    
    # Local Search with adaptive multi-flip moves
    current_cell = [row[:] for row in best_cell]
    current_score = best_score
    stagnation = 0
    total_iters = 0
    
    while time.time() < deadline:
        total_iters += 1
        # Adaptive multi-flip: increase probability and size with stagnation
        flip_prob = min(0.3 + stagnation * 0.01, 0.7)
        if stagnation > 20 and np.random.random() < flip_prob:
            # Multi-flip move
            n_flips = min(2 + stagnation // 10, 6)
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
                stagnation = 0
            else:
                stagnation += 1
        else:
            # Single flip
            i = np.random.randint(0, 8)
            j = np.random.randint(0, 8)
            
            new_cell = [row[:] for row in current_cell]
            flip_pixel(new_cell, i, j)
            
            new_score = evaluate_cell(new_cell)
            
            if new_score > current_score:
                current_cell = new_cell
                current_score = new_score
                best_cell = [row[:] for row in current_cell]
                best_score = current_score
                stagnation = 0
            else:
                stagnation += 1
                
    return best_cell
