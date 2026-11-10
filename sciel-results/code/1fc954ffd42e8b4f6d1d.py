"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START

# Precompute G vectors for the plane-wave basis
# The evaluator uses 289 plane waves (17x17). 
# For the fast evaluator, we use a subset. 
# Let's use 11x11 = 121 plane waves for a good balance of speed and accuracy.
# G vectors are integers (h, k) in [-N, N].
PW_HALF = 5  # 11x11 = 121 PWs
G_range = np.arange(-PW_HALF, PW_HALF + 1)
Gx, Gy = np.meshgrid(G_range, G_range, indexing='ij')
G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
Gx_flat = G_vectors[:, 0].astype(int)
Gy_flat = G_vectors[:, 1].astype(int)
N_PW = len(G_vectors)

# Precompute difference indices for FFT extraction
# We need coeff for G_i - G_j.
# FFT indices are (h % 32, k % 32).
Diff_x = Gx_flat[:, None] - Gx_flat[None, :]
Diff_y = Gy_flat[:, None] - Gy_flat[None, :]
idx_x = (Diff_x + 32) % 32
idx_y = (Diff_y + 32) % 32

# Precompute k-points in the irreducible Brillouin Zone
# The irreducible BZ of a square lattice is the triangle with vertices
# Gamma(0,0), X(0.5, 0), M(0.5, 0.5) in units of reciprocal lattice vectors?
# No, standard BZ is [-0.5, 0.5] x [-0.5, 0.5] if G is integer.
# Irreducible BZ is 0 <= kx <= ky, kx+ky <= 0.5?
# Let's use the standard 1/8 BZ: 0 <= kx <= ky, kx + ky <= 0.5.
# We will sample points in this triangle.

def generate_k_points(num_points=30):
    points = []
    # We can use a simple grid or specific high-symmetry lines.
    # Let's use a uniform distribution in the triangle for robustness.
    # Triangle vertices: (0,0), (0.5, 0), (0.5, 0.5) ? No.
    # Irreducible BZ for square lattice (C4v):
    # 0 <= kx <= ky
    # kx + ky <= 0.5 (X-M line)
    # Wait, the boundary is kx=0 (Gamma-M), ky=kx (Gamma-X? No, Gamma-M is kx=ky), and kx+ky=0.5 (X-M).
    # Let's stick to the lines for better resolution where gaps often open.
    # Gamma to M: (t, t) for t in [0, 0.5]
    # M to X: (0.5 - t, t) for t in [0, 0.5] ? No.
    # M is (0.5, 0.5). X is (0.5, 0) or (0, 0.5).
    # Let's define X as (0.5, 0).
    # Path Gamma(0,0) -> X(0.5, 0) -> M(0.5, 0.5) -> Gamma(0,0).
    # Irreducible part:
    # 1. Gamma to X: (t, 0) for t in [0, 0.5]
    # 2. X to M: (0.5, t) for t in [0, 0.5]
    # 3. M to Gamma: (t, t) for t in [0, 0.5]
    
    # Sample 10 points on each segment.
    for t in np.linspace(0, 0.5, 10):
        points.append((t, 0.0))
    for t in np.linspace(0, 0.5, 10):
        points.append((0.5, t))
    for t in np.linspace(0, 0.5, 10):
        points.append((t, t))
        
    return np.array(points)

K_POINTS = generate_k_points()
N_K = len(K_POINTS)

# Precompute (k+G) vectors for each k-point
# k is in units of reciprocal lattice vector (1/a).
# G is integer.
# (k+G) = (kx + Gx, ky + Gy)
KG = np.zeros((N_K, N_PW, 2))
for i_k, k in enumerate(K_POINTS):
    kx, ky = k
    KG[i_k, :, 0] = kx + Gx_flat
    KG[i_k, :, 1] = ky + Gy_flat

# Precompute dot products (k+G_i) . (k+G_j) for each k-point
# This is a matrix of shape (N_K, N_PW, N_PW)
DOTS = np.zeros((N_K, N_PW, N_PW))
for i_k in range(N_K):
    KG_k = KG[i_k] # (N_PW, 2)
    # Dots[i_k, i, j] = KG_k[i] dot KG_k[j]
    DOTS[i_k] = KG_k @ KG_k.T

def evaluate_gap_fast(eps_grid):
    """
    Fast evaluation of the gap-midgap ratio between bands 4 and 5.
    eps_grid: 32x32 array of epsilon values.
    """
    inv_eps = 1.0 / eps_grid
    fft_1eps = np.fft.fft2(inv_eps)
    
    # Extract coefficients for all G differences
    # coeffs[i, j] = fft_1eps[idx_x[i,j], idx_y[i,j]] / 32^2
    # Note: FFT normalization. numpy.fft.fft2 is unnormalized forward.
    # coeff_hk = sum_{x,y} f(x,y) exp(-i 2pi (h x + k y) / N)
    # So coeff = fft_1eps[h%N, k%N] / N^2 is the standard Fourier series coeff if f is defined on [0,1) with N points?
    # Let's check.
    # f(x) = sum c_k exp(i 2pi k x / N).
    # c_k = 1/N sum f(n) exp(-i 2pi k n / N).
    # So yes, divide by N. Wait, 2D. Divide by N^2.
    coeffs = fft_1eps[idx_x, idx_y] / (32 * 32)
    
    max_omega_4_sq = 0.0
    min_omega_5_sq = float('inf')
    
    for i_k in range(N_K):
        # H = DOTS[i_k] * coeffs
        # DOTS is real, coeffs is complex (but should be real for symmetric eps).
        # Let's take real part for speed and stability, assuming symmetry.
        H = DOTS[i_k] * coeffs.real
        
        # H is symmetric.
        # Solve eigenvalues.
        eigenvalues = np.linalg.eigvalsh(H)
        
        if len(eigenvalues) >= 5:
            w4_sq = eigenvalues[3]
            w5_sq = eigenvalues[4]
            
            if w4_sq > max_omega_4_sq:
                max_omega_4_sq = w4_sq
            if w5_sq < min_omega_5_sq:
                min_omega_5_sq = w5_sq
                
    if min_omega_5_sq == float('inf'):
        return 0.0
        
    # The ratio is 2 (min omega_5 - max omega_4) / (min omega_5 + max omega_4)
    # omega = sqrt(omega_sq).
    # Since sqrt is monotonic, we can compute the ratio using sqrt.
    w4 = np.sqrt(max(0, max_omega_4_sq))
    w5 = np.sqrt(max(0, min_omega_5_sq))
    
    gap = w5 - w4
    midgap = (w5 + w4) / 2.0
    
    if midgap == 0:
        return 0.0
        
    ratio = 2 * gap / midgap
    return ratio

def design(problem):
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin
    
    # Parameterization:
    # We use a superposition of cosine modes to create the pattern.
    # Modes: (1,1), (1,0), (0,1), (2,2), (2,0), (0,2), (2,1), (1,2), etc.
    # To ensure C4v symmetry, we combine modes carefully.
    # Let's use a set of basis functions that are symmetric.
    
    # Coordinates
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Centered coordinates
    x_c = (X - (size - 1) / 2.0) / size
    y_c = (Y - (size - 1) / 2.0) / size
    
    # Basis functions (all symmetric under x->-x, y->-y, x<->y)
    # 1. cos(2pi x_c) * cos(2pi y_c)  [Mode (1,1)]
    # 2. cos(2pi x_c) + cos(2pi y_c)   [Mode (1,0) + (0,1)]
    # 3. cos(4pi x_c) * cos(4pi y_c)   [Mode (2,2)]
    # 4. cos(4pi x_c) + cos(4pi y_c)   [Mode (2,0) + (0,2)]
    # 5. cos(2pi x_c) * cos(4pi y_c) + cos(4pi x_c) * cos(2pi y_c) [Mode (1,2) + (2,1)]
    
    def get_eps_grid(params):
        # params: array of amplitudes
        # A0: DC (mean) - not used in tanh, but shifts threshold effectively.
        # Let's just use the AC modes and add a DC offset.
        # field = sum A_i * mode_i
        # eps = 0.5*(eps_low+eps_high) + 0.5*(eps_high-eps_low)*tanh(field)
        
        field = np.zeros((size, size))
        
        # Mode 1: (1,1)
        if len(params) > 0:
            field += params[0] * np.cos(2 * np.pi * x_c) * np.cos(2 * np.pi * y_c)
        # Mode 2: (1,0)+(0,1)
        if len(params) > 1:
            field += params[1] * (np.cos(2 * np.pi * x_c) + np.cos(2 * np.pi * y_c))
        # Mode 3: (2,2)
        if len(params) > 2:
            field += params[2] * np.cos(4 * np.pi * x_c) * np.cos(4 * np.pi * y_c)
        # Mode 4: (2,0)+(0,2)
        if len(params) > 3:
            field += params[3] * (np.cos(4 * np.pi * x_c) + np.cos(4 * np.pi * y_c))
        # Mode 5: (1,2)+(2,1)
        if len(params) > 4:
            field += params[4] * (np.cos(2 * np.pi * x_c) * np.cos(4 * np.pi * y_c) + 
                                   np.cos(4 * np.pi * x_c) * np.cos(2 * np.pi * y_c))
        # Mode 6: (3,3)
        if len(params) > 5:
            field += params[5] * np.cos(6 * np.pi * x_c) * np.cos(6 * np.pi * y_c)
        # Mode 7: (3,0)+(0,3)
        if len(params) > 6:
            field += params[6] * (np.cos(6 * np.pi * x_c) + np.cos(6 * np.pi * y_c))
            
        # Add DC offset to control filling fraction
        # We can just threshold based on median later, so DC is less critical for the shape,
        # but tanh saturation depends on amplitude.
        
        eps_grid = 0.5 * (eps_low + eps_high) + 0.5 * (eps_high - eps_low) * np.tanh(field)
        return eps_grid

    num_params = 7
    # Initial guess: Strong (1,1) and (1,0) components, typical for high gap structures.
    # Inspiration 2 suggests 4 holes, which corresponds to strong (1,1) and (1,0) terms.
    # Let's start with a guess that creates 4 holes.
    # Holes at (+/-0.25, +/-0.25).
    # cos(2pi*0.25)=0. cos(2pi*(-0.25))=0.
    # So at (0.25, 0.25), Mode 1 is 0. Mode 2 is 0.
    # We want low eps (air) there. So field should be negative.
    # tanh(negative) -> negative -> eps low.
    # So we want field < 0 at (0.25, 0.25).
    # At (0,0), field should be positive (dielectric).
    # Mode 1 at (0,0) is 1. Mode 2 at (0,0) is 2.
    # If we set A0=1, A1=0, field(0,0)=1, field(0.25,0.25)=0.
    # This gives a checkerboard.
    # Let's try A0=1.0, A1=0.5.
    params = np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_score = -10.0
    best_params = params.copy()
    
    # Evaluate initial
    eps_grid = get_eps_grid(params)
    score = evaluate_gap_fast(eps_grid)
    if score > best_score:
        best_score = score
        best_params = params.copy()
    
    # Optimization: Hill Climbing / Random Restart
    step_size = 0.1
    num_iterations = 0
    
    while time.time() < deadline and num_iterations < 5000:
        num_iterations += 1
        
        # Pick random parameter
        idx = np.random.randint(num_params)
        
        # Perturb
        delta = np.random.choice([-1, 1]) * step_size
        new_params = params.copy()
        new_params[idx] += delta
        
        # Clamp parameters to reasonable range
        new_params = np.clip(new_params, -3.0, 3.0)
        
        new_eps_grid = get_eps_grid(new_params)
        new_score = evaluate_gap_fast(new_eps_grid)
        
        if new_score > best_score:
            params = new_params
            best_score = new_score
            best_params = params.copy()
            # Increase step size slightly
            step_size *= 1.05
            if step_size > 0.5:
                step_size = 0.5
        else:
            # Decrease step size
            step_size *= 0.9
            if step_size < 0.01:
                step_size = 0.01
        
        # Random restarts
        if num_iterations % 200 == 0:
            # Try a new random point
            random_params = np.random.uniform(-2.0, 2.0, num_params)
            random_eps = get_eps_grid(random_params)
            random_score = evaluate_gap_fast(random_eps)
            if random_score > best_score:
                params = random_params
                best_score = random_score
                best_params = params.copy()
                step_size = 0.1
            else:
                # Keep current best params as base for next hill climb
                params = best_params.copy()
                step_size = 0.1

    # Finalize with best parameters
    final_eps_grid = get_eps_grid(best_params)
    
    # Threshold to binary
    # We want to preserve the shape. 
    # Using median ensures 50% filling, which is often optimal.
    threshold = np.median(final_eps_grid)
    cell = (final_eps_grid > threshold).astype(int)
    
    # Enforce symmetry
    sym_cell = np.zeros_like(cell)
    for i in range(size):
        for j in range(size):
            points = [
                (i, j),
                (size - 1 - i, j),
                (i, size - 1 - j),
                (size - 1 - i, size - 1 - j),
                (j, i),
                (size - 1 - j, j),
                (j, size - 1 - i),
                (size - 1 - j, size - 1 - i)
            ]
            s = 0
            for pi, pj in points:
                s += cell[pi, pj]
            avg = s / 8.0
            sym_cell[i, j] = 1 if avg >= 0.5 else 0
            
    return sym_cell.tolist()

# EVOLVE-BLOCK-END
