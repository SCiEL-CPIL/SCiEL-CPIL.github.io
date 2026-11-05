import numpy as np
import time
# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell to maximize the TM band gap between bands 4 and 5.
    
    Strategy:
    1. Implement a fast internal PWE solver using the standard Galerkin method.
    2. Use a symmetry-constrained parametric representation with geometric primitives.
    3. Optimize parameters via gradient-free search (Nelder-Mead style).
    4. Maintain D4 symmetry and stay within the time budget.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    
    # Time budget: 230s limit, keep 20s margin for overhead
    start_time = time.time()
    deadline = start_time + 210.0
    
    # Fast internal PWE solver for TM modes
    def compute_tm_bands(cell, n_pw=21, n_kpts=8):
        """
        Compute TM band structure using plane-wave expansion.
        """
        cell = np.array(cell, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell
        inv_eps = 1.0 / eps
        
        fft_coeffs = np.fft.fft2(inv_eps) / (size * size)
        
        def get_coeff(m, n):
            idx_m = m % size
            idx_n = n % size
            return fft_coeffs[idx_m, idx_n]
        
        M = n_pw // 2
        m_vals = list(range(-M, M + 1))
        g_indices = [(m, n) for m in m_vals for n in m_vals]
        n_total = len(g_indices)
        
        # Precompute G vectors
        G_vecs = np.array([(2.0 * np.pi * m, 2.0 * np.pi * n) for m, n in g_indices])
        
        # Build S matrix once
        S = np.zeros((n_total, n_total), dtype=complex)
        for i, (mi, ni) in enumerate(g_indices):
            for j, (mj, nj) in enumerate(g_indices):
                gm = mi - mj
                gn = ni - nj
                S[i, j] = get_coeff(gm, gn)
        
        # Define k-points in the irreducible Brillouin zone
        k_points = []
        for t in np.linspace(0, np.pi, n_kpts//3 + 1):
            k_points.append((t, 0))
        for t in np.linspace(0, np.pi, n_kpts//3 + 1):
            k_points.append((np.pi, t))
        for t in np.linspace(np.pi, 0, n_kpts//3 + 1):
            k_points.append((t, t))
        
        k_points_unique = []
        for kp in k_points:
            if not any(np.allclose(kp, kp2, atol=1e-10) for kp2 in k_points_unique):
                k_points_unique.append(kp)
        
        n_kpts_actual = len(k_points_unique)
        omega = np.zeros((n_kpts_actual, n_total))
        
        for ki, (kx, ky) in enumerate(k_points_unique):
            # Build eigenvalue matrix
            # M_{ij} = S(G_i - G_j) * |G_j + k|^2
            k_vec = np.array([kx, ky])
            G_plus_k = G_vecs + k_vec
            k_mag_sq = np.sum(G_plus_k**2, axis=1)
            
            M = S * k_mag_sq[np.newaxis, :]
            
            try:
                eigvals = np.linalg.eigvalsh(M)
                eigvals = np.sort(eigvals)
                omega[ki, :] = np.sqrt(np.real(eigvals))
            except Exception:
                omega[ki, :] = np.inf
        
        return omega

    def make_parametric_cell(params):
        """
        Create a cell using geometric primitives with D4 symmetry.
        params: (ring_width, cross_width, corner_size)
        """
        ring_width = params[0]
        cross_width = params[1]
        corner_size = params[2]
        
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='ij')
        
        # Concentric square ring
        max_coord = np.maximum(np.abs(X), np.abs(Y))
        min_coord = np.minimum(np.abs(X), np.abs(Y))
        
        # Ring: between r_inner and r_outer
        r_inner = 0.25 - ring_width/2
        r_outer = 0.25 + ring_width/2
        ring_mask = (max_coord >= r_inner) & (max_coord <= r_outer)
        
        # Cross: horizontal and vertical bars through center
        cross_mask = (np.abs(X) < cross_width/2) | (np.abs(Y) < cross_width/2)
        
        # Corner regions
        corner_mask = (np.abs(X) > 0.35) & (np.abs(Y) > 0.35) & (np.abs(X) < 0.35 + corner_size) & (np.abs(Y) < 0.35 + corner_size)
        
        # Combine: union of ring, cross, and corners
        cell[ring_mask | cross_mask | corner_mask] = 1
        
        # Ensure D4 symmetry by averaging with rotated/reflected versions
        cell = np.array(cell, dtype=float)
        cell = (cell + cell[::-1, :] + cell[:, ::-1] + cell[::-1, ::-1]) / 4.0
        cell = (cell + cell.T) / 2.0
        cell = (cell > 0.5).astype(int)
        
        return cell.tolist()

    def compute_score(cell):
        """Compute the gap-midgap ratio for a given cell."""
        try:
            bands = compute_tm_bands(cell, n_pw=15, n_kpts=6)
            omega_4 = bands[:, 3]
            omega_5 = bands[:, 4]
            
            max_omega_4 = np.max(omega_4)
            min_omega_5 = np.min(omega_5)
            
            if min_omega_5 > max_omega_4:
                gap = min_omega_5 - max_omega_4
                ratio = 2.0 * gap / (min_omega_5 + max_omega_4)
            else:
                ratio = -1.0
            return ratio
        except Exception:
            return -1.0

    # Nelder-Mead optimization
    def nelder_mead(f, x0, bounds, max_iter=50, tol=1e-4):
        """Simple Nelder-Mead optimizer with bounds."""
        n = len(x0)
        # Initialize simplex
        simplex = [np.array(x0, dtype=float)]
        for i in range(n):
            xi = np.array(x0, dtype=float)
            step = 0.1 * (bounds[i][1] - bounds[i][0])
            xi[i] += step
            simplex.append(xi)
        
        # Evaluate initial simplex
        fvals = [f(np.clip(x, bounds[:, 0], bounds[:, 1])) for x in simplex]
        
        for it in range(max_iter):
            # Sort by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            if abs(fvals[-1] - fvals[0]) < tol:
                break
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_ref = centroid + (centroid - simplex[-1])
            x_ref = np.clip(x_ref, bounds[:, 0], bounds[:, 1])
            f_ref = f(x_ref)
            
            if fvals[0] <= f_ref < fvals[-1]:
                simplex[-1] = x_ref
                fvals[-1] = f_ref
            elif f_ref < fvals[0]:
                # Expansion
                x_exp = centroid + 2 * (x_ref - centroid)
                x_exp = np.clip(x_exp, bounds[:, 0], bounds[:, 1])
                f_exp = f(x_exp)
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
            else:
                # Contraction
                x_con = centroid + 0.5 * (simplex[-1] - centroid)
                x_con = np.clip(x_con, bounds[:, 0], bounds[:, 1])
                f_con = f(x_con)
                if f_con < fvals[-1]:
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i])
        
        # Return best point
        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx]

    # Bounds for parameters
    bounds = np.array([
        [0.05, 0.20],  # ring_width
        [0.02, 0.15],  # cross_width
        [0.0, 0.15],   # corner_size
    ])
    
    # Initial guess
    x0 = np.array([0.10, 0.08, 0.05])
    
    # Evaluate initial point
    best_cell = make_parametric_cell(x0)
    best_score = compute_score(best_cell)
    
    # Run optimization
    if time.time() < deadline - 30:
        try:
            x_opt, f_opt = nelder_mead(
                lambda x: -compute_score(make_parametric_cell(x)),  # Minimize negative score
                x0,
                bounds,
                max_iter=30,
                tol=1e-3
            )
            if -f_opt > best_score:
                best_score = -f_opt
                best_cell = make_parametric_cell(x_opt)
        except Exception:
            pass
    
    # If optimization didn't help, try a few more random starting points
    if time.time() < deadline - 20:
        np.random.seed(42)
        for _ in range(3):
            if time.time() > deadline - 15:
                break
            x_rand = np.random.uniform(bounds[:, 0], bounds[:, 1])
            cell = make_parametric_cell(x_rand)
            score = compute_score(cell)
            if score > best_score:
                best_score = score
                best_cell = cell
    
    return best_cell

# EVOLVE-BLOCK-END
