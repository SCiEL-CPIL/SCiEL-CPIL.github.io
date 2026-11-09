import numpy as np
import time
from scipy.optimize import differential_evolution

# Precompute G vectors for the 289 plane wave method
# N_PW = 8 means G ranges from -8 to 8, total 17 values per dimension -> 17*17 = 289
N_PW = 8
G_range = np.arange(-N_PW, N_PW + 1)
Gx, Gy = np.meshgrid(G_range, G_range, indexing='ij')
G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
Gx_flat = G_vectors[:, 0]
Gy_flat = G_vectors[:, 1]

# Precompute indices for the 32x32 pixel grid Fourier coefficients
# The grid is 32x32. We need mu(G-G') for G, G' in the set of 289 vectors.
# The difference G-G' ranges from -16 to 16.
# We map these differences to the 32x32 FFT indices.
# Note: The FFT of a 32x32 grid gives frequencies -16..15 (if using standard fft) or shifted.
# Let's use the explicit definition of Fourier coefficients for a pixelated dielectric.
# mu(x) = sum_mu_G exp(i G x)
# For a pixelated cell of size 32, the coefficients are determined by the pixel values.
# However, calculating the 289x289 matrix from scratch for every candidate is slow.
# We can precompute the "basis" of the matrix if we parameterize the cell continuously?
# No, the cell is discrete 0/1 pixels.
# But since we are using a parametric model (circles), we can generate the 32x32 pixel map
# and then compute the FFT.
# To speed up, we can precompute the mapping from the 32x32 pixel FFT to the 289x289 matrix.

# Let's precompute the index mapping for the 289x289 matrix.
# The matrix element M_{ij} corresponds to mu(G_i - G_j) * (k+G_j)^2 / sqrt((k+G_i)^2 (k+G_j)^2)
# Actually, the standard PWE eigenvalue equation is:
# sum_j  (1/eps(G_i - G_j)) * (k + G_j)^2 * c_j = (omega/c)^2 * (k + G_i)^2 * c_i
# Let's define the matrix A where A_{ij} = (1/eps(G_i - G_j)) * (k + G_j)^2
# And the diagonal D where D_{ii} = (k + G_i)^2
# Then D^{-1} A c = (omega/c)^2 c
# Or A c = (omega/c)^2 D c
# We solve the generalized eigenvalue problem A c = lambda D c.
# Since D is diagonal, we can transform to a standard eigenvalue problem:
# D^{-1/2} A D^{-1/2} c' = lambda c'
# where c' = D^{1/2} c.

# Precompute the indices for mu(G_i - G_j)
# G_i and G_j are integers in [-8, 8].
# Diff = G_i - G_j is in [-16, 16].
# The FFT of the 32x32 pixel grid gives coefficients for frequencies n in [-16, 15] (0-indexed) 
# or we can use fftshift to get [-16, 15] centered?
# np.fft.fft2 returns frequencies 0..31, which correspond to 0..15, -16..-1.
# Let's compute the full 32x32 FFT and index it correctly.

# For a 32x32 grid, the frequency index n in FFT output (0..31) corresponds to:
# if n <= 15: freq = n
# if n > 15: freq = n - 32
# So freq ranges from 0 to 15, -16 to -1.
# We need freq in range -16 to 16.
# Note: The difference G_i - G_j can be 16.
# In a 32x32 grid, the Nyquist frequency is 16. 
# The FFT coefficient for freq 16 is not explicitly present in the 0..31 range 
# (it's aliased with -16). 
# Actually, for even N, the Nyquist bin is N/2 = 16.
# np.fft.fft2 returns index 16 for frequency -16 (or 16, they are the same in real signals).
# So index 16 corresponds to freq -16 (or 16).
# Let's create a mapping from integer frequency f to FFT index.
def freq_to_idx(f, N=32):
    # f is in [-16, 16]
    # FFT indices are 0..31
    # 0 -> 0
    # 1 -> 1
    # ...
    # 15 -> 15
    # 16 -> 16 (Nyquist, represents both 16 and -16)
    # -1 -> 31
    # ...
    # -16 -> 16
    if f == 16 or f == -16:
        return 16
    elif f > 0:
        return f
    else:
        return N + f

# Precompute the 289x289 index pairs for mu
idx_i, idx_j = np.meshgrid(np.arange(289), np.arange(289), indexing='ij')
# G_vectors is (289, 2). 
# We need the difference G_i - G_j for each pair (i, j) in the 289x289 matrix.
# G_i = G_vectors[i], G_j = G_vectors[j]
# Diff_x = G_vectors[:, 0][:, None] - G_vectors[:, 0][None, :]
# This results in a 289x289 array of integer differences.
Diff_x = Gx_flat[:, None] - Gx_flat[None, :]
Diff_y = Gy_flat[:, None] - Gy_flat[None, :]

# Map these differences to FFT indices
# We need to handle the -16/16 case.
idx_mu_x = np.zeros_like(Diff_x, dtype=int)
idx_mu_y = np.zeros_like(Diff_y, dtype=int)

# Vectorized mapping
# For x:
mask_pos = Diff_x > 0
mask_neg = Diff_x < 0
mask_zero = Diff_x == 0
mask_16 = Diff_x == 16
mask_m16 = Diff_x == -16

idx_mu_x[mask_zero] = 0
idx_mu_x[mask_pos] = Diff_x[mask_pos]
idx_mu_x[mask_neg] = 32 + Diff_x[mask_neg]
idx_mu_x[mask_16] = 16
idx_mu_x[mask_m16] = 16

# Same for y
mask_pos_y = Diff_y > 0
mask_neg_y = Diff_y < 0
mask_zero_y = Diff_y == 0
mask_16_y = Diff_y == 16
mask_m16_y = Diff_y == -16

idx_mu_y[mask_zero_y] = 0
idx_mu_y[mask_pos_y] = Diff_y[mask_pos_y]
idx_mu_y[mask_neg_y] = 32 + Diff_y[mask_neg_y]
idx_mu_y[mask_16_y] = 16
idx_mu_y[mask_m16_y] = 16

# Precompute k-points
# The evaluator uses 30 k-points. We should match this for consistency in scoring.
# Standard irreducible BZ for square lattice:
# Gamma (0,0) to X (pi, 0) to M (pi, pi) to Gamma (0,0)
# Symmetry: kx, ky >= 0, kx >= ky (or similar triangular sector)
# Let's generate 30 points in the sector 0 <= ky <= kx <= pi.
# Or use the standard path: Gamma-X-M-Gamma.
# To be safe and match "30 wave vectors covering the boundary and interior",
# we will sample the triangular sector (0,0), (pi,0), (pi,pi).
# Area of sector is pi^2 / 2.
# We can use a grid in this triangle.

def generate_k_points(num_points=30):
    points = []
    # Use a stratified sampling in the triangle
    # Let u, v be random in [0,1]
    # x = u * pi
    # y = v * (1-u) * pi  -> This covers the triangle?
    # Let's just use a deterministic grid for reproducibility and speed.
    # 6x6 grid = 36 points, trim to 30?
    # Or 5x6 = 30 points.
    
    # Let's use a simple grid in the irreducible wedge:
    # kx from 0 to pi, ky from 0 to kx.
    # We want 30 points.
    # Let's take 10 points along Gamma-X (ky=0, kx in [0, pi])
    # 10 points along X-M (kx=pi, ky in [0, pi])
    # 10 points along M-Gamma (kx=ky, kx in [0, pi])
    # This is a common set.
    
    # Gamma-X
    for i in range(10):
        kx = (i + 0.5) * (np.pi / 10)
        points.append((kx, 0))
        
    # X-M
    for i in range(10):
        ky = (i + 0.5) * (np.pi / 10)
        points.append((np.pi, ky))
        
    # M-Gamma
    for i in range(10):
        t = (i + 0.5) * (np.pi / 10)
        points.append((t, t))
        
    return np.array(points)

K_POINTS = generate_k_points()

# Precompute (k+G)^2 for each k-point and G-vector
# k is in radians per unit length (a=1).
# G = 2*pi * n.
# (k + G)^2 = (kx + 2*pi*nx)^2 + (ky + 2*pi*ny)^2
KG_SQ = np.zeros((len(K_POINTS), 289))
for ki, k in enumerate(K_POINTS):
    kx, ky = k
    kgx = kx + 2 * np.pi * Gx_flat
    kgy = ky + 2 * np.pi * Gy_flat
    KG_SQ[ki] = kgx**2 + kgy**2

# Helper to evaluate a specific cell (32x32 list of 0/1)
def evaluate_cell(cell):
    # cell: 32x32 list of 0/1. 1=Dielectric (eps=12), 0=Air (eps=1)
    # We want to maximize the gap between band 4 and 5.
    # Inverted structure: Dielectric background (1), Air holes (0).
    # So most pixels are 1.
    
    eps = np.array(cell, dtype=float)
    # 1 -> eps 12, 0 -> eps 1
    eps_vals = np.where(eps == 1, 12.0, 1.0)
    mu = 1.0 / eps_vals
    
    # Compute FFT of mu
    mu_fft = np.fft.fft2(mu)
    
    # Extract the 289x289 matrix of mu coefficients
    # M_{ij} = mu(G_i - G_j)
    # We use the precomputed indices idx_mu_x, idx_mu_y
    mu_mat = mu_fft[idx_mu_x, idx_mu_y]
    
    # Now solve the eigenvalue problem for each k-point
    min_omega_5 = float('inf')
    max_omega_4 = float('-inf')
    
    for ki in range(len(K_POINTS)):
        kg_sq = KG_SQ[ki]
        # D = diag(kg_sq)
        # A = mu_mat * kg_sq[None, :]  (Element-wise multiplication by kg_sq of column j)
        # Wait, the matrix is A_{ij} = mu(G_i - G_j) * (k + G_j)^2
        # So A = mu_mat * kg_sq[None, :]
        
        A = mu_mat * kg_sq[None, :]
        
        # We need to solve A c = lambda D c
        # D is diagonal with entries kg_sq.
        # Transform to standard problem:
        # D^{-1/2} A D^{-1/2} c' = lambda c'
        # Let S = D^{-1/2} A D^{-1/2}
        # S_{ij} = A_{ij} / sqrt(D_i D_j) = mu(G_i-G_j) * kg_sq_j / sqrt(kg_sq_i kg_sq_j)
        # S_{ij} = mu(G_i-G_j) * sqrt(kg_sq_j / kg_sq_i)
        
        # Note: kg_sq can be 0 at Gamma point for G=0.
        # At Gamma, k=0. G=0 -> kg_sq=0.
        # This causes division by zero.
        # The standard PWE method handles this by noting that the c_0 component decouples or using limits.
        # Actually, for TM mode, the field is H_z.
        # The equation is:
        # sum_j (1/eps(G_i-G_j)) (k+G_j)^2 c_j = (omega/c)^2 (k+G_i)^2 c_i
        # If (k+G_i)^2 = 0, then LHS must be 0? No, if RHS is 0, then LHS must be 0.
        # This implies a constraint on c_j.
        # However, in the transformed problem, if kg_sq_i = 0, the i-th row/col of S is problematic.
        # Usually, one removes the zero-frequency plane wave or handles it separately.
        # For Gamma point, the lowest bands are often associated with G=0.
        # Let's add a small epsilon to kg_sq to avoid division by zero, 
        # but this might distort the result.
        # Better: If kg_sq_i is very small, we can skip it or handle it.
        # Actually, for the lowest bands, k is small.
        # Let's check if any kg_sq is 0.
        # At Gamma (0,0), G=(0,0) -> kg_sq=0.
        # We can set 1/sqrt(0) to 0? No.
        # Let's use a regularization: kg_sq_safe = kg_sq + 1e-12
        # This is a common hack.
        
        kg_sq_safe = kg_sq + 1e-12
        sqrt_kg = np.sqrt(kg_sq_safe)
        inv_sqrt_kg = 1.0 / sqrt_kg
        
        S = mu_mat * inv_sqrt_kg[:, None] * inv_sqrt_kg[None, :]
        
        # Solve for the 5 smallest eigenvalues
        # S is Hermitian (mu is real and symmetric in G? mu(G) = mu(-G)* for real mu.
        # mu_mat is Hermitian if mu is real.
        evals = np.linalg.eigvalsh(S)
        
        # The eigenvalues are (omega/c)^2.
        # We want bands 4 and 5 (index 3 and 4).
        # Note: The lowest band (index 0) is often the trivial constant mode?
        # No, for TM, the lowest band is non-trivial.
        # Let's take indices 3 and 4.
        
        if len(evals) >= 5:
            w4 = np.sqrt(evals[3])
            w5 = np.sqrt(evals[4])
            
            max_omega_4 = max(max_omega_4, w4)
            min_omega_5 = min(min_omega_5, w5)
            
    if min_omega_5 == float('inf') or max_omega_4 == float('-inf'):
        return -1.0
        
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    
    if midgap == 0:
        return 0.0
        
    return gap / midgap

# Parametric Model:
# We define the cell by a set of circular holes (Air, 0) in a Dielectric (1) background.
# Due to symmetry, we only need to define the holes in the first quadrant (8x8) 
# and mirror them.
# However, Differential Evolution works on continuous parameters.
# We will define up to 4 holes in the first quadrant.
# Each hole is defined by (x, y, r).
# x, y in [0, 0.5] (normalized to cell size 1, so 0..16 pixels).
# r in [0, 0.5].
# Total parameters: 4 holes * 3 params = 12 parameters.

def create_cell_from_params(params):
    # params: array of length 12: [x1, y1, r1, x2, y2, r2, x3, y3, r3, x4, y4, r4]
    # Coordinates are in units of cell size (0 to 1).
    # Pixel grid is 32x32.
    # Center of pixel (i, j) is ((i+0.5)/32, (j+0.5)/32).
    
    cell = np.ones((32, 32), dtype=int) # Start with Dielectric (1)
    
    for k in range(4):
        x = params[k*3]
        y = params[k*3+1]
        r = params[k*3+2]
        
        # Generate holes in the 4 symmetric quadrants
        # The cell is symmetric under x->1-x, y->1-y, and swap x,y?
        # The problem requires:
        # cell == cell[::-1] row-wise (x->1-x)
        # cell == cell[::-1] col-wise (y->1-y)
        # cell == cell.T (x<->y)
        # This implies 4-fold symmetry.
        # We can place a hole at (x, y) and its mirrors:
        # (x, y), (1-x, y), (x, 1-y), (1-x, 1-y)
        # And their transposes:
        # (y, x), (y, 1-x), (1-y, x), (1-y, 1-x)
        
        # We just need to set pixels to 0 if they are inside ANY of these circles.
        # To be efficient, we can precompute the pixel centers.
        
    # Precompute pixel centers
    xs = (np.arange(32) + 0.5) / 32.0
    ys = (np.arange(32) + 0.5) / 32.0
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    
    for k in range(4):
        x = params[k*3]
        y = params[k*3+1]
        r = params[k*3+2]
        
        # Define the 8 symmetric positions
        centers = [
            (x, y), (1-x, y), (x, 1-y), (1-x, 1-y),
            (y, x), (y, 1-x), (1-y, x), (1-y, 1-x)
        ]
        
        for cx, cy in centers:
            dist_sq = (X - cx)**2 + (Y - cy)**2
            mask = dist_sq < r**2
            cell[mask] = 0 # Air hole
            
    return cell.tolist()

def objective(params):
    # params is a numpy array
    cell = create_cell_from_params(params)
    score = evaluate_cell(cell)
    # We want to maximize score, so we minimize negative score
    return -score

def design(problem):
    start_time = time.time()
    deadline = start_time + 115 # Safety margin
    
    # Define bounds for parameters
    # x, y in [0, 1], r in [0, 0.5]
    # We restrict x, y to [0, 0.5] to avoid redundancy? 
    # No, the symmetry handles the rest. 
    # If we place a hole at (0.6, 0.2), its mirror is (0.4, 0.2).
    # So we can restrict x, y to [0, 0.5] to cover the fundamental domain.
    # However, if we restrict to [0, 0.5], we might miss holes centered on the symmetry axes?
    # The symmetry axes are x=0.5, y=0.5, x=y, x=1-y.
    # It's safer to allow x, y in [0, 1] and let the symmetry duplicate them.
    # But this increases the search space.
    # Let's use [0, 0.5] for x and y, and rely on the fact that the "mirrors" will cover the rest.
    # Wait, if I place a hole at x=0.4, y=0.4, the mirrors are (0.6, 0.4), (0.4, 0.6), (0.6, 0.6).
    # This covers the whole cell.
    # So bounds [0, 0.5] are sufficient.
    
    bounds = [
        (0.0, 0.5), (0.0, 0.5), (0.0, 0.4),
        (0.0, 0.5), (0.0, 0.5), (0.0, 0.4),
        (0.0, 0.5), (0.0, 0.5), (0.0, 0.4),
        (0.0, 0.5), (0.0, 0.5), (0.0, 0.4)
    ]
    
    # Initial best guess
    # Try a simple configuration: 4 holes near the corners of the quadrant
    # (0.25, 0.25, 0.1)
    initial_params = np.array([
        0.25, 0.25, 0.1,
        0.25, 0.25, 0.1,
        0.25, 0.25, 0.1,
        0.25, 0.25, 0.1
    ])
    
    best_cell = create_cell_from_params(initial_params)
    best_score = evaluate_cell(best_cell)
    
    # Run Differential Evolution
    # maxiter: number of generations
    # popsize: population size
    # We have 12 parameters. popsize=15 -> 180 individuals per generation.
    # Each evaluation takes ~0.1s? 
    # 30 k-points * 289x289 eigvalsh. 
    # eigvalsh on 289x289 is fast (~1ms). 
    # Total per eval: ~10ms.
    # 180 evals * 10ms = 1.8s per generation.
    # 50 generations = 90s.
    # This fits in the budget.
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=50, 
            popsize=15, 
            tol=1e-4,
            seed=42,
            polish=True # Use L-BFGS-B to polish the result
        )
        
        if result.success or result.fun < -best_score:
            final_params = result.x
            final_cell = create_cell_from_params(final_params)
            final_score = evaluate_cell(final_cell)
            
            if final_score > best_score:
                best_score = final_score
                best_cell = final_cell
                
    except Exception as e:
        print(f"Optimization failed: {e}")
        
    # If time remains, we can do a local search on the pixels
    # But the parametric search should be good.
    
    return best_cell

# The problem requires the function name `design` and signature `design(problem)`.
# The `problem` dict is not used in this parametric approach, but we keep it for compliance.
