"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """Design a symmetric 32x32 unit cell maximizing the TM band-4/5 gap-midgap ratio."""
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12)
    
    t0 = time.time()
    deadline = t0 + 200.0  # Safety margin before 230s limit

    N = size  # 32
    half = N // 2  # 16

    # --- Symmetric cell construction from 16x16 quadrant ---
    def build_cell_from_quadrant(quadrant):
        """Build full 32x32 cell from 16x16 quadrant with C4v symmetry."""
        cell = [[0]*N for _ in range(N)]
        for i in range(half):
            for j in range(half):
                val = quadrant[i][j]
                # C4v: 8 symmetric positions
                cell[i][j] = val
                cell[i][N-1-j] = val
                cell[N-1-i][j] = val
                cell[N-1-i][N-1-j] = val
                cell[j][i] = val
                cell[j][N-1-i] = val
                cell[N-1-j][i] = val
                cell[N-1-j][N-1-i] = val
        return cell

    # --- Fast PWE solver with coarse basis ---
    max_g = 4  # 9x9 = 81 plane waves for speed
    g_indices = [(mx, my) for mx in range(-max_g, max_g+1) for my in range(-max_g, max_g+1)]
    num_g = len(g_indices)
    G_vecs = np.array([(2*math.pi*mx, 2*math.pi*my) for mx, my in g_indices])
    
    # Critical k-points in irreducible BZ
    k_points = [
        (0.0, math.pi),           # X
        (0.0, 2*math.pi/3),       
        (0.0, math.pi/2),         # midpoint Gamma-X
        (math.pi/2, math.pi/2),   # M
        (math.pi/3, math.pi/3),
        (math.pi/6, math.pi/2),
        (0.0, math.pi*0.8),
        (0.0, math.pi*0.6),
        (math.pi/4, math.pi/2),
        (math.pi/6, math.pi/3),
    ]

    def compute_eps_inv_fourier(cell):
        """Compute Fourier coefficients of 1/epsilon using FFT."""
        eps_inv = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                eps_inv[i, j] = 1.0 / eps_high if cell[i][j] == 1 else 1.0
        fft_coeff = np.fft.fft2(eps_inv) / (N * N)
        return fft_coeff

    def evaluate_gap(cell, fft_coeff=None):
        """Evaluate the gap-midgap ratio between TM bands 4 and 5."""
        if fft_coeff is None:
            fft_coeff = compute_eps_inv_fourier(cell)
        
        # Build eps_inv difference matrix
        eps_inv_mat = np.zeros((num_g, num_g))
        for p in range(num_g):
            mx_p, my_p = g_indices[p]
            for q in range(num_g):
                mx_q, my_q = g_indices[q]
                dm = (mx_p - mx_q) % N
                dn = (my_p - my_q) % N
                eps_inv_mat[p, q] = fft_coeff[dn, dm]
        
        max_omega4 = 0.0
        min_omega5 = float('inf')
        
        for kx, ky in k_points:
            kG_sq = (kx + G_vecs[:, 0])**2 + (ky + G_vecs[:, 1])**2
            M = eps_inv_mat * kG_sq[None, :]
            
            try:
                eigvals = np.linalg.eigvalsh(M)
            except:
                continue
            
            eigvals = np.sort(eigvals)
            if len(eigvals) < 5:
                continue
            
            omega4 = math.sqrt(max(eigvals[3], 0))
            omega5 = math.sqrt(max(eigvals[4], 0))
            
            max_omega4 = max(max_omega4, omega4)
            min_omega5 = min(min_omega5, omega5)
        
        if min_omega5 == float('inf') or max_omega4 == 0:
            return -1.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        if midgap <= 0:
            return -1.0
        
        return gap / midgap

    # --- Parametric design: concentric C4v-symmetric rings with varying thickness ---
    # Use 4 concentric rings with adjustable inner/outer radii and radial modulation
    def make_parametric_cell(params):
        """
        params: [r1_in, r1_out, r2_in, r2_out, r3_in, r3_out, r4_in, r4_out, 
                 mod_amp, mod_freq, phase1, phase2]
        Creates multi-scale fractal-like dielectric distribution.
        """
        r1_in, r1_out = params[0], params[1]
        r2_in, r2_out = params[2], params[3]
        r3_in, r3_out = params[4], params[5]
        r4_in, r4_out = params[6], params[7]
        mod_amp = params[8]
        mod_freq = params[9]
        phase1 = params[10]
        phase2 = params[11]
        
        quadrant = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(half):
                # Pixel center in normalized coords [0,1]
                x = (j + 0.5) / N
                y = (i + 0.5) / N
                r = math.sqrt(x*x + y*y)
                theta = math.atan2(y, x)
                
                # Radial modulation for fractal-like structure
                mod = 1.0 + mod_amp * math.sin(mod_freq * theta + phase1) * math.cos(2*theta + phase2)
                
                val = 0
                if r1_in * mod <= r <= r1_out * mod:
                    val = 1
                if r2_in * mod <= r <= r2_out * mod:
                    val = 1
                if r3_in * mod <= r <= r3_out * mod:
                    val = 1
                if r4_in * mod <= r <= r4_out * mod:
                    val = 1
                    
                quadrant[i][j] = val
        return build_cell_from_quadrant(quadrant)

    # --- Initial designs: various ring configurations ---
    initial_params_list = [
        [0.15, 0.30, 0.35, 0.45, 0.50, 0.58, 0.62, 0.68, 0.1, 4, 0, 0],
        [0.10, 0.25, 0.30, 0.40, 0.45, 0.52, 0.56, 0.62, 0.15, 6, 0.5, 0.3],
        [0.20, 0.35, 0.40, 0.50, 0.55, 0.62, 0.66, 0.72, 0.2, 8, 1.0, 0.5],
        [0.12, 0.28, 0.32, 0.42, 0.46, 0.54, 0.58, 0.65, 0.05, 3, 0.2, 0.1],
    ]

    # Evaluate initial designs
    best_score = -1.0
    best_params = initial_params_list[0][:]
    
    for params in initial_params_list:
        if time.time() > deadline - 100:
            break
        cell = make_parametric_cell(params)
        score = evaluate_gap(cell)
        if score > best_score:
            best_score = score
            best_params = params[:]

    # --- Nelder-Mead optimization on continuous parameters ---
    n_dim = len(best_params)
    
    # Bounds for parameters
    bounds = [
        (0.05, 0.40),  # r1_in
        (0.15, 0.50),  # r1_out
        (0.25, 0.55),  # r2_in
        (0.35, 0.65),  # r2_out
        (0.45, 0.65),  # r3_in
        (0.55, 0.75),  # r3_out
        (0.60, 0.80),  # r4_in
        (0.70, 0.90),  # r4_out
        (0.0, 0.3),    # mod_amp
        (2, 10),       # mod_freq
        (0, 2*math.pi), # phase1
        (0, 2*math.pi), # phase2
    ]
    
    def objective(params):
        """Objective function: negative gap ratio (to minimize)."""
        # Clamp to bounds
        clamped = []
        for i, p in enumerate(params):
            lo, hi = bounds[i]
            clamped.append(max(lo, min(hi, p)))
        
        cell = make_parametric_cell(clamped)
        score = evaluate_gap(cell)
        return -score
    
    # Nelder-Mead implementation
    def nelder_mead(f, x0, bounds, max_iter=50, timeout=150.0):
        """Simple Nelder-Mead optimizer with bounds and timeout."""
        n = len(x0)
        simplex = [x0[:]]
        simplex_f = [f(x0)]
        
        # Initial simplex
        for i in range(n):
            xi = x0[:]
            step = 0.1 * (bounds[i][1] - bounds[i][0]) if i < len(bounds) else 0.1
            xi[i] += step
            simplex.append(xi)
            simplex_f.append(f(xi))
        
        t_start = time.time()
        
        for iteration in range(max_iter):
            if time.time() - t_start > timeout:
                break
            
            # Sort by function value
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:n], axis=0)
            
            # Reflection
            alpha = 1.0
            x_ref = centroid - alpha * (simplex[-1] - centroid)
            f_ref = f(x_ref)
            
            if f_ref < simplex_f[0]:
                # Expansion
                gamma = 2.0
                x_exp = centroid + gamma * (x_ref - centroid)
                f_exp = f(x_exp)
                simplex[-1] = x_exp if f_exp < f_ref else x_ref
                simplex_f[-1] = min(f_exp, f_ref)
                continue
            elif f_ref < simplex_f[-2]:
                simplex[-1] = x_ref
                simplex_f[-1] = f_ref
                continue
            else:
                # Contraction
                rho = 0.5
                if f_ref < simplex_f[-1]:
                    x_con = centroid + rho * (x_ref - centroid)
                else:
                    x_con = centroid - rho * (centroid - simplex[-1])
                f_con = f(x_con)
                if f_con < min(f_ref, simplex_f[-1]):
                    simplex[-1] = x_con
                    simplex_f[-1] = f_con
                    continue
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex_f[i] = f(simplex[i])
        
        best_idx = np.argmin(simplex_f)
        return simplex[best_idx], -simplex_f[best_idx]
    
    # Run optimization with time budget
    remaining_time = deadline - time.time() - 30  # Reserve 30s for final evaluation
    if remaining_time > 60:
        opt_params, opt_score = nelder_mead(objective, best_params, bounds, 
                                             max_iter=30, timeout=remaining_time)
        if opt_score > best_score:
            best_score = opt_score
            best_params = opt_params
    
    # --- Local pixel search to refine ---
    current_cell = make_parametric_cell(best_params)
    current_score = best_score
    current_quadrant = [[current_cell[i][j] for j in range(half)] for i in range(half)]
    
    positions = [(i, j) for i in range(half) for j in range(half)]
    rng = random.Random(42)
    
    # Quick local search with a few sweeps
    for sweep in range(5):
        if time.time() > deadline - 20:
            break
        
        shuffled_pos = positions[:]
        rng.shuffle(shuffled_pos)
        
        improved = False
        for (i, j) in shuffled_pos:
            if time.time() > deadline - 15:
                break
            
            old_val = current_quadrant[i][j]
            current_quadrant[i][j] = 1 - old_val
            new_cell = build_cell_from_quadrant(current_quadrant)
            new_score = evaluate_gap(new_cell)
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_params = best_params[:]  # Keep parametric version
                improved = True
            else:
                current_quadrant[i][j] = old_val
        
        if not improved:
            break
    
    # Build final cell from parametric best (more robust than pixel-local)
    final_cell = make_parametric_cell(best_params)
    
    # Verify and enforce symmetry
    for i in range(N):
        for j in range(N):
            if final_cell[i][j] != final_cell[N-1-i][j]:
                final_cell[N-1-i][j] = final_cell[i][j]
            if final_cell[i][j] != final_cell[i][N-1-j]:
                final_cell[i][N-1-j] = final_cell[i][j]
            if final_cell[i][j] != final_cell[j][i]:
                final_cell[j][i] = final_cell[i][j]
    
    return final_cell
# EVOLVE-BLOCK-END
