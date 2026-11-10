import numpy as np
import time

# Precompute G vectors - 17x17 = 289 plane waves
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
N_G = len(G_vectors)

# Precompute k-points and (k+G)^2
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
KG_INV_SQRT = [1.0 / np.sqrt(kg_sq) for kg_sq in KG_SQ]

def compute_mu_fourier(cell):
    """Compute Fourier coefficients of 1/eps for a 32x32 cell."""
    size = len(cell)
    eps = np.array([[12.0 if c == 1 else 1.0 for c in row] for row in cell], dtype=float)
    mu = 1.0 / eps
    mu_fft = np.fft.fft2(mu)
    mu_fft_shifted = np.fft.fftshift(mu_fft)
    mu_diff = mu_fft_shifted[idx_x, idx_y] / (size * size)
    return mu_diff

def evaluate_cell(cell):
    """Evaluate the gap-midgap ratio for a 32x32 cell."""
    mu_diff = compute_mu_fourier(cell)
    min_omega_5 = float('inf')
    max_omega_4 = float('-inf')
    for ki in range(len(K_POINTS)):
        B_inv_sqrt = KG_INV_SQRT[ki]
        S = mu_diff * B_inv_sqrt[:, None] * B_inv_sqrt[None, :]
        S = (S + S.T) / 2.0
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

def make_cell_from_quadrant(q):
    """Create a full 32x32 cell from an 8x8 symmetric quadrant."""
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

def flip_pixel(cell, i, j):
    """Flip pixel at (i,j) in the 8x8 quadrant, updating full cell."""
    size = 32
    val = cell[i][j]
    new_val = 1 - val
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

def generate_harmonic_cell(amplitudes, threshold=0.5):
    """
    Generate a C4v-symmetric cell from harmonic amplitudes.
    amplitudes: array of shape (num_harmonics,) for each harmonic term.
    Uses fixed angular frequencies with optimized amplitudes.
    """
    size = 32
    field = np.zeros((size, size))
    
    # Define harmonic terms: (m, n, phase) where term = amp * cos(2*pi*m*x + phase) * cos(2*pi*n*y)
    # C4v symmetric harmonics
    harmonics = [
        (1, 1, 0), (1, 1, np.pi/2),
        (2, 0, 0), (0, 2, 0),
        (2, 2, 0), (2, 2, np.pi/4),
        (3, 1, 0), (1, 3, 0),
        (3, 3, 0), (3, 3, np.pi/6),
        (4, 0, 0), (0, 4, 0),
        (4, 2, 0), (2, 4, 0),
        (4, 4, 0), (4, 4, np.pi/8),
        (5, 1, 0), (1, 5, 0),
        (5, 5, 0), (5, 5, np.pi/10),
    ]
    
    num_harmonics = min(len(amplitudes), len(harmonics))
    
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            val = 0.0
            for h in range(num_harmonics):
                m, n, phase = harmonics[h]
                amp = amplitudes[h]
                term = amp * np.cos(2*np.pi*m*x + phase) * np.cos(2*np.pi*n*y)
                # Add C4v symmetric copies
                val += term
                val += amp * np.cos(2*np.pi*n*x + phase) * np.cos(2*np.pi*m*y)  # swap m,n
                val += amp * np.cos(2*np.pi*m*x - phase) * np.cos(2*np.pi*n*y)  # phase flip
                val += amp * np.cos(2*np.pi*n*x - phase) * np.cos(2*np.pi*m*y)
            field[i, j] = val
    
    # Normalize
    field -= np.mean(field)
    std = np.std(field)
    if std > 1e-10:
        field /= std
    
    # Threshold
    cell_arr = (field > threshold).astype(int)
    
    # Enforce C4v symmetry by averaging over all 8 symmetry operations
    sym_field = np.zeros((size, size))
    sym_field += cell_arr
    sym_field += np.rot90(cell_arr, 1)
    sym_field += np.rot90(cell_arr, 2)
    sym_field += np.rot90(cell_arr, 3)
    sym_field += cell_arr[:, ::-1]
    sym_field += cell_arr[::-1, :]
    sym_field += np.rot90(cell_arr[:, ::-1], 1)
    sym_field += np.rot90(cell_arr[::-1, :], 1)
    sym_field /= 8.0
    
    cell_arr = (sym_field > 0.5).astype(int)
    
    # Extract top-left 8x8 and enforce symmetry
    q = [[cell_arr[i][j] for j in range(8)] for i in range(8)]
    for i in range(8):
        for j in range(i, 8):
            val = (q[i][j] + q[j][i]) // 2
            q[i][j] = val
            q[j][i] = val
    
    return make_cell_from_quadrant(q)

def generate_random_cell(seed):
    """Generate a random C4v-symmetric cell."""
    rng = np.random.RandomState(seed)
    q = [[rng.randint(0, 2) for j in range(8)] for i in range(8)]
    # Enforce symmetry
    for i in range(8):
        for j in range(i, 8):
            val = (q[i][j] + q[j][i]) // 2
            q[i][j] = val
            q[j][i] = val
    return make_cell_from_quadrant(q)

def generate_structured_cell(seed, type_id):
    """Generate a structured C4v-symmetric cell based on type_id."""
    size = 32
    cell = [[0]*size for _ in range(size)]
    rng = np.random.RandomState(seed)
    
    if type_id == 0:  # Dielectric background with air holes
        cell = [[1]*size for _ in range(size)]
        # Add 1-4 air holes
        n_holes = rng.randint(1, 5)
        for _ in range(n_holes):
            cx = rng.randint(0, size)
            cy = rng.randint(0, size)
            r = rng.randint(2, 6)
            for i in range(size):
                for j in range(size):
                    if (i-cx)**2 + (j-cy)**2 < r**2:
                        cell[i][j] = 0
    elif type_id == 1:  # Air background with dielectric rods
        n_rods = rng.randint(1, 5)
        for _ in range(n_rods):
            cx = rng.randint(0, size)
            cy = rng.randint(0, size)
            r = rng.randint(2, 6)
            for i in range(size):
                for j in range(size):
                    if (i-cx)**2 + (j-cy)**2 < r**2:
                        cell[i][j] = 1
    elif type_id == 2:  # Checkerboard-like
        pattern_size = rng.randint(4, 8)
        for i in range(size):
            for j in range(size):
                if (i // pattern_size + j // pattern_size) % 2 == 0:
                    cell[i][j] = 1
    elif type_id == 3:  # Ring pattern
        r_inner = rng.randint(4, 8)
        r_outer = rng.randint(r_inner+2, 14)
        for i in range(size):
            for j in range(size):
                r = np.sqrt((i-15.5)**2 + (j-15.5)**2)
                if r_inner < r < r_outer:
                    cell[i][j] = 1
    elif type_id == 4:  # Cross pattern
        width = rng.randint(2, 6)
        for i in range(size):
            for j in range(size):
                if abs(i-15.5) < width or abs(j-15.5) < width:
                    cell[i][j] = 1
    
    # Enforce C4v symmetry
    cell_arr = np.array(cell)
    sym_field = np.zeros((size, size))
    sym_field += cell_arr
    sym_field += np.rot90(cell_arr, 1)
    sym_field += np.rot90(cell_arr, 2)
    sym_field += np.rot90(cell_arr, 3)
    sym_field += cell_arr[:, ::-1]
    sym_field += cell_arr[::-1, :]
    sym_field += np.rot90(cell_arr[:, ::-1], 1)
    sym_field += np.rot90(cell_arr[::-1, :], 1)
    sym_field /= 8.0
    cell_arr = (sym_field > 0.5).astype(int)
    
    q = [[cell_arr[i][j] for j in range(8)] for i in range(8)]
    for i in range(8):
        for j in range(i, 8):
            val = (q[i][j] + q[j][i]) // 2
            q[i][j] = val
            q[j][i] = val
    
    return make_cell_from_quadrant(q)

def design(problem):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 230s kill
    
    size = 32
    candidates = []
    
    # Generate initial candidates
    # Random cells
    for seed in range(30):
        candidates.append(generate_random_cell(seed))
    
    # Structured cells
    for seed in range(20):
        for type_id in range(5):
            candidates.append(generate_structured_cell(seed, type_id))
    
    # Harmonic-based cells with random amplitudes
    num_harmonics = 20
    for seed in range(30):
        rng = np.random.RandomState(seed)
        amplitudes = rng.randn(num_harmonics) * 0.5
        for thresh in [0.3, 0.5, 0.7]:
            candidates.append(generate_harmonic_cell(amplitudes, thresh))
    
    # Evaluate all candidates
    best_cell = None
    best_score = -float('inf')
    
    for c in candidates:
        s = evaluate_cell(c)
        if s > best_score:
            best_score = s
            best_cell = [row[:] for row in c]
    
    # Simulated Annealing on harmonic amplitudes
    num_harmonics = 20
    rng = np.random.RandomState(12345)
    current_amplitudes = rng.randn(num_harmonics) * 0.5
    current_cell = generate_harmonic_cell(current_amplitudes, 0.5)
    current_score = evaluate_cell(current_cell)
    
    if current_score > best_score:
        best_score = current_score
        best_cell = [row[:] for row in current_cell]
    
    T = 0.3
    T_min = 0.01
    cooling = 0.998
    
    while time.time() < deadline - 30 and T > T_min:
        # Single parameter update
        new_amplitudes = current_amplitudes.copy()
        idx = np.random.randint(0, num_harmonics)
        new_amplitudes[idx] += np.random.randn() * 0.1
        
        new_cell = generate_harmonic_cell(new_amplitudes, 0.5)
        new_score = evaluate_cell(new_cell)
        delta = new_score - current_score
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_amplitudes = new_amplitudes
            current_cell = new_cell
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_cell = [row[:] for row in current_cell]
        
        # Multi-parameter update
        if np.random.random() < 0.2:
            new_amplitudes2 = current_amplitudes.copy()
            n_params = np.random.randint(2, 5)
            for _ in range(n_params):
                idx2 = np.random.randint(0, num_harmonics)
                new_amplitudes2[idx2] += np.random.randn() * 0.1
            
            new_cell2 = generate_harmonic_cell(new_amplitudes2, 0.5)
            new_score2 = evaluate_cell(new_cell2)
            delta2 = new_score2 - current_score
            
            if delta2 > 0 or np.random.random() < np.exp(delta2 / T):
                current_amplitudes = new_amplitudes2
                current_cell = new_cell2
                current_score = new_score2
                if current_score > best_score:
                    best_score = current_score
                    best_cell = [row[:] for row in current_cell]
        
        T *= cooling
    
    # Final local search on pixel flips
    current_cell = [row[:] for row in best_cell]
    current_score = best_score
    T = 0.05
    T_min = 0.005
    cooling = 0.997
    
    while time.time() < deadline and T > T_min:
        i = np.random.randint(0, 8)
        j = np.random.randint(0, 8)
        
        new_cell = [row[:] for row in current_cell]
        flip_pixel(new_cell, i, j)
        
        new_score = evaluate_cell(new_cell)
        delta = new_score - current_score
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_cell = new_cell
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_cell = [row[:] for row in current_cell]
        
        T *= cooling
    
    return best_cell
