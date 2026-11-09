import numpy as np
import time

# Precompute G vectors for 289 plane waves (17x17, -8..8)
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
    """
    size = len(cell)
    half = size // 2
    # Extract top-left quadrant
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
            
    # Enforce transpose symmetry on the quadrant
    for i in range(half):
        for j in range(i, half):
            # Average or pick one?
            # We will use a generator that produces symmetric quads.
            pass

    return new_cell

def generate_chebyshev_cell(coeffs, threshold, size=32):
    """
    Generate a 32x32 cell from Chebyshev coefficients.
    coeffs: list of coefficients for the symmetric basis.
    threshold: value to threshold the field.
    """
    x = -1 + (np.arange(size) + 0.5) / (size / 2)
    y = -1 + (np.arange(size) + 0.5) / (size / 2)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    field = np.zeros((size, size))
    
    T2_x = 2*X**2 - 1
    T2_y = 2*Y**2 - 1
    T4_x = 8*X**4 - 8*X**2 + 1
    T4_y = 8*Y**4 - 8*Y**2 + 1
    T6_x = 32*X**6 - 48*X**4 + 18*X**2 - 1
    T6_y = 32*Y**6 - 48*Y**4 + 18*Y**2 - 1
    
    if len(coeffs) > 0:
        field += coeffs[0]
    if len(coeffs) > 1:
        field += coeffs[1] * (T2_x + T2_y)
    if len(coeffs) > 2:
        field += coeffs[2] * (T4_x + T4_y)
    if len(coeffs) > 3:
        field += coeffs[3] * (T2_x * T2_y)
    if len(coeffs) > 4:
        field += coeffs[4] * (T4_x * T2_y + T2_x * T4_y)
    if len(coeffs) > 5:
        field += coeffs[5] * (T4_x * T4_y)
    if len(coeffs) > 6:
        field += coeffs[6] * (T6_x + T6_y)
    if len(coeffs) > 7:
        field += coeffs[7] * (T6_x * T2_y + T2_x * T6_y)
    if len(coeffs) > 8:
        field += coeffs[8] * (T6_x * T4_y + T4_x * T6_y)
    if len(coeffs) > 9:
        field += coeffs[9] * (T6_x * T6_y)
        
    cell = (field > threshold).astype(int)
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
    
    # Ring
    cell_ring = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            if 0.15 < r < 0.35:
                cell_ring[i][j] = 1
    candidates.append(cell_ring)
    
    best_cell = None
    best_score = -float('inf')
    
    for c in candidates:
        s = evaluate_cell(c)
        if s > best_score:
            best_score = s
            best_cell = [row[:] for row in c]
            
    # Continuous Optimization using Chebyshev coefficients
    N_COEFFS = 10
    current_coeffs = np.random.randn(N_COEFFS) * 0.5
    current_threshold = 0.0
    
    # Initialize from a few random guesses
    for _ in range(5):
        current_coeffs = np.random.randn(N_COEFFS) * 1.0
        current_threshold = np.random.uniform(-1, 1)
        cell = generate_chebyshev_cell(current_coeffs, current_threshold)
        score = evaluate_cell(cell)
        if score > best_score:
            best_score = score
            best_cell = [row[:] for row in cell]
            
    current_coeffs = np.random.randn(N_COEFFS) * 0.5
    current_threshold = 0.0
    current_cell = generate_chebyshev_cell(current_coeffs, current_threshold)
    current_score = evaluate_cell(current_cell)
    
    if current_score > best_score:
        best_score = current_score
        best_cell = [row[:] for row in current_cell]
        
    stagnation = 0
    
    while time.time() < deadline:
        # Perturb coefficients and threshold
        step_size = 0.1 / (1 + stagnation * 0.1)
        
        new_coeffs = current_coeffs + np.random.randn(N_COEFFS) * step_size
        new_threshold = current_threshold + np.random.randn() * 0.05
        
        new_cell = generate_chebyshev_cell(new_coeffs, new_threshold)
        new_score = evaluate_cell(new_cell)
        
        if new_score > current_score:
            current_coeffs = new_coeffs
            current_threshold = new_threshold
            current_cell = new_cell
            current_score = new_score
            stagnation = 0
            
            if new_score > best_score:
                best_score = new_score
                best_cell = [row[:] for row in new_cell]
        else:
            stagnation += 1
            
        # Occasionally restart or jump to explore different regions
        if stagnation > 50:
            current_coeffs = np.random.randn(N_COEFFS) * 1.0
            current_threshold = np.random.uniform(-1, 1)
            current_cell = generate_chebyshev_cell(current_coeffs, current_threshold)
            current_score = evaluate_cell(current_cell)
            if current_score > best_score:
                best_score = current_score
                best_cell = [row[:] for row in current_cell]
            stagnation = 0

    return best_cell
