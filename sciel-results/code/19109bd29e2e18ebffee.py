"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def design(problem):
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 210.0

    # --- 1. Parameterization ---
    # We use a level-set representation defined by a low-order polynomial in Chebyshev basis
    # constrained to C4v symmetry.
    # f(x, y) = sum A_mn T_m(u) T_n(v) + T_n(u) T_m(v) for m < n, and A_mm T_m(u) T_m(v) for m=n.
    # u = 2x - 1, v = 2y - 1.
    # Symmetry: f(u, v) is even in u and v, and symmetric f(u, v) = f(v, u).
    
    max_cheb = 8
    cheb_orders = [0, 2, 4, 6, 8]
    param_indices = []
    for i, m in enumerate(cheb_orders):
        for j, n in enumerate(cheb_orders):
            if m <= n:
                param_indices.append((m, n))
    num_params = len(param_indices)

    x_coords = np.array([(i + 0.5) / size for i in range(size)])
    y_coords = np.array([(j + 0.5) / size for j in range(size)])
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    U = 2 * X - 1
    V = 2 * Y - 1

    def cheb_poly(t, k):
        if k == 0:
            return np.ones_like(t)
        if k == 1:
            return t.copy()
        T_prev2 = np.ones_like(t)
        T_prev1 = t.copy()
        for _ in range(2, k + 1):
            T_curr = 2 * t * T_prev1 - T_prev2
            T_prev2 = T_prev1
            T_prev1 = T_curr
        return T_prev1

    basis = []
    for m, n in param_indices:
        Tm_U = cheb_poly(U, m)
        Tn_V = cheb_poly(V, n)
        if m == n:
            basis.append(Tm_U * Tn_V)
        else:
            Tn_U = cheb_poly(U, n)
            Tm_V = cheb_poly(V, m)
            basis.append(Tm_U * Tn_V + Tn_U * Tm_V)
    
    def get_eps_grid(params):
        field = np.zeros((size, size))
        for i in range(num_params):
            field += params[i] * basis[i]
        eps_grid = 0.5 * (eps_low + eps_high) + 0.5 * (eps_high - eps_low) * np.tanh(field)
        return eps_grid

    # --- 2. Fast Evaluator ---
    # Use 7x7 = 49 plane waves.
    pw_size = 7
    G_vectors = []
    for h in range(-3, 4):
        for k in range(-3, 4):
            G_vectors.append((h, k))
    N_pw = len(G_vectors)
    G_arr = np.array(G_vectors)
    Gx_flat = G_arr[:, 0]
    Gy_flat = G_arr[:, 1]
    
    # Precompute difference indices for FFT extraction
    Diff_x = Gx_flat[:, None] - Gx_flat[None, :]
    Diff_y = Gy_flat[:, None] - Gy_flat[None, :]
    idx_x = (Diff_x + 16) % 32
    idx_y = (Diff_y + 16) % 32

    # Precompute k-points and dot products
    # Irreducible BZ: 0 <= kx <= ky, kx + ky <= 0.5 (in units of 2pi/a? No, 1/a)
    # Actually, standard BZ for square lattice is [-0.5, 0.5] x [-0.5, 0.5].
    # Irreducible wedge: 0 <= kx <= ky, kx + ky <= 0.5.
    # Key points: Gamma (0,0), M (0.5, 0), X (0.5, 0.5).
    # Path: Gamma -> M -> X.
    
    k_points = []
    # Gamma to M: (0, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 5):
        k_points.append((0.0, t))
    # M to X: (t, 0.5-t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 5):
        k_points.append((t, 0.5 - t))
    # Remove duplicates if any (M is in both)
    k_points = np.array(k_points)
    
    # Precompute K_i dot K_j for each k-point? No, K depends on k.
    # K_i = G_i + k.
    # We can precompute G_i G_j dot products, but k part varies.
    # (G_i + k) . (G_j + k) = G_i . G_j + G_i . k + G_j . k + k . k
    # Let's just compute it on the fly or vectorize.
    
    def evaluate_gap(eps_grid):
        inv_eps = 1.0 / eps_grid
        fft_1eps = np.fft.fft2(inv_eps)
        
        # Extract Fourier coefficients for all G_i - G_j pairs
        # coeff[i, j] = fft_1eps[idx_x[i, j], idx_y[i, j]] / size^2
        mu_diff = fft_1eps[idx_x, idx_y] / (size * size)
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            # Build H matrix
            # K_i = (Gx_i + kx, Gy_i + ky)
            Kx = Gx_flat + kx
            Ky = Gy_flat + ky
            
            # Dot product matrix: K_i dot K_j
            # K_i dot K_j = Kx_i Kx_j + Ky_i Ky_j
            dot_matrix = np.outer(Kx, Kx) + np.outer(Ky, Ky)
            
            # H = dot_matrix * mu_diff
            H = dot_matrix * mu_diff
            
            # H should be Hermitian. Since mu_diff is conjugate symmetric (mu is real),
            # H is Hermitian.
            # For real symmetric mu, H is real symmetric?
            # mu_diff[i, j] = coeff(G_i - G_j).
            # coeff(-G) = conj(coeff(G)).
            # mu_diff[j, i] = coeff(G_j - G_i) = coeff(-(G_i - G_j)) = conj(coeff(G_i - G_j)) = conj(mu_diff[i, j]).
            # So H is Hermitian.
            # If mu is real, coeff is real?
            # FFT of real function is conjugate symmetric.
            # coeff(h, k) = conj(coeff(-h, -k)).
            # If mu is even (symmetric), coeff is real.
            # Our mu is symmetric (C4v), so coeff should be real (numerically ~real).
            # Let's take real part to be safe and fast.
            H_real = H.real
            
            eigenvalues = np.linalg.eigvalsh(H_real)
            
            if len(eigenvalues) >= 5:
                w4_sq = eigenvalues[3]
                w5_sq = eigenvalues[4]
                
                w4 = np.sqrt(max(0, w4_sq))
                w5 = np.sqrt(max(0, w5_sq))
                
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
        
        if min_omega_5 == float('inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = 2 * gap / midgap
        return ratio

    # --- 3. Optimization Loop ---
    
    # Initial guess: A central rod or 4 holes?
    # Let's try a few starting points and pick the best.
    # Start 1: Central rod (A00 > 0)
    # Start 2: 4 holes (A00 < 0, A22 < 0? No, holes are air, so low eps.
    # Low eps means tanh(f) < 0. So f < 0.
    # 4 holes at (0.25, 0.25) etc.
    # f should be negative there.
    
    starts = [
        np.array([2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), # Rod
        np.array([-2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), # Holes (if A00<0 means air? No, A00<0 -> f<0 -> eps<6.5 -> air. So uniform air? No, uniform dielectric is A00>0. Uniform air is A00<0.
        # Let's try a mix.
        np.array([1.0, -1.0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        np.array([0.5, -1.5, 1.0, -0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ]
    
    best_score = -float('inf')
    best_params = None
    
    for s in starts:
        if len(s) < num_params:
            s = np.pad(s, (0, num_params - len(s)))
        else:
            s = s[:num_params]
            
        score = evaluate_gap(get_eps_grid(s))
        if score > best_score:
            best_score = score
            best_params = s.copy()
            
    params = best_params.copy()
    step_size = 0.2
    
    num_iterations = 0
    max_iterations = 10000
    
    while time.time() < deadline and num_iterations < max_iterations:
        num_iterations += 1
        
        # Pick a random parameter to perturb
        idx = np.random.randint(num_params)
        
        # Try a perturbation
        delta = np.random.choice([-1, 1]) * step_size
        
        new_params = params.copy()
        new_params[idx] += delta
        
        # Compute new score
        new_eps_grid = get_eps_grid(new_params)
        new_score = evaluate_gap(new_eps_grid)
        
        if new_score > best_score:
            params = new_params
            best_score = new_score
            best_params = params.copy()
            # Increase step size slightly
            step_size *= 1.05
            # Cap step size
            if step_size > 1.0:
                step_size = 1.0
        else:
            # Decrease step size
            step_size *= 0.9
            # Floor step size
            if step_size < 0.01:
                step_size = 0.01
        
        # Occasionally, do a larger jump (restarting) to escape local minima
        if num_iterations % 200 == 0:
            params = best_params + np.random.normal(0, 0.5, num_params)
            score = evaluate_gap(get_eps_grid(params))
            if score > best_score:
                best_score = score
                best_params = params.copy()
    
    # --- 4. Finalize Design ---
    
    # Use the best parameters found
    final_params = best_params
    final_eps_grid = get_eps_grid(final_params)
    
    # Threshold to binary 0/1
    # eps_low = 1, eps_high = 12.
    # Midpoint = 6.5.
    threshold = (eps_low + eps_high) / 2.0
    cell = (final_eps_grid > threshold).astype(int)
    
    # Enforce symmetry exactly (just in case of floating point errors in thresholding)
    # The construction ensures symmetry, but let's verify/force it.
    # Symmetry: cell[i][j] == cell[size-1-i][size-1-j] == cell[j][i]
    # Our basis functions are symmetric. The thresholding might break it if values are close to threshold.
    # We can average the 4 symmetric pixels and threshold the average.
    
    sym_cell = np.zeros_like(cell)
    for i in range(size):
        for j in range(size):
            # Symmetric equivalents:
            # (i, j)
            # (size-1-i, j)  -> reflection x
            # (i, size-1-j)  -> reflection y
            # (size-1-i, size-1-j) -> rotation 180
            # Also transpose: (j, i), etc.
            # C4v group has 8 elements.
            # Let's just average the 4 unique ones under the group?
            # Actually, the group is generated by reflection x, reflection y, and transpose.
            # If we enforce reflection x and y, and transpose, we get C4v.
            
            # For now, let's just take the majority vote or average of the 4 main symmetric points:
            # (i,j), (size-1-i, j), (i, size-1-j), (size-1-i, size-1-j)
            # And their transposes.
            # It's safer to just re-compute the average of the 8 symmetric points.
            
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
            # Unique points?
            # For a 32x32 grid, these might not all be unique.
            # Let's just sum them and divide by 8.
            s = 0
            for pi, pj in points:
                s += cell[pi, pj]
            avg = s / 8.0
            sym_cell[i, j] = 1 if avg >= 0.5 else 0
    
    # Convert to list of lists
    return sym_cell.tolist()

# EVOLVE-BLOCK-END
