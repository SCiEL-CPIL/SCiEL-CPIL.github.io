import numpy as np
import time

def design(problem):
    """
    Optimizes a C4v-symmetric dielectric distribution using a continuous
    Fourier parameterization with smooth surrogates for the band-gap objective.
    Uses a two-phase search: broad random exploration followed by adaptive
    local refinement with gradient-free optimization.
    """
    start_time = time.time()
    deadline = start_time + 215.0  # Safety margin before 230s kill
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Parameterization: C4v symmetric Fourier series centered at (0.5, 0.5)
    # f(x,y) = sum A_{mn} cos(2*pi*m*(x-0.5)) cos(2*pi*n*(y-0.5))
    # Symmetry about x=0.5, y=0.5 is automatic.
    # Symmetry about y=x requires A_{mn} = A_{nm}.
    
    max_harmonic = 4
    params_idx = []
    for m in range(max_harmonic + 1):
        for n in range(m, max_harmonic + 1):
            params_idx.append((m, n))
    num_params = len(params_idx)
    
    # Precompute grid coordinates
    x = np.linspace(0.5/size, 1 - 0.5/size, size)
    X, Y = np.meshgrid(x, x, indexing='ij')
    Xc = X - 0.5
    Yc = Y - 0.5
    
    # Precompute basis functions
    basis_funcs = []
    for (m, n) in params_idx:
        if m == 0 and n == 0:
            basis_funcs.append(np.ones((size, size)))
        else:
            cm = np.cos(2 * np.pi * m * Xc)
            cn = np.cos(2 * np.pi * n * Yc)
            basis_funcs.append(cm * cn)
    basis_funcs = np.array(basis_funcs)  # shape (num_params, size, size)
    
    def get_field(params):
        """Returns the 32x32 epsilon grid from parameters."""
        field = np.tensordot(params, basis_funcs, axes=1)
        eps_grid = 0.5 * (eps_low + eps_high) + 0.5 * (eps_high - eps_low) * np.tanh(field)
        return eps_grid
    
    # Precompute G vectors and k points for the evaluator approximation
    N_pw = 8
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)
    
    # K points: 15 points in irreducible BZ (Gamma-M, M-X, Gamma-X)
    k_points = []
    for t in np.linspace(0, 0.5, 5):
        k_points.append((t, 0))
        k_points.append((0.5, t))
        k_points.append((t, t))
    # Remove duplicates
    seen = set()
    unique_k = []
    for k in k_points:
        key = (round(k[0], 10), round(k[1], 10))
        if key not in seen:
            seen.add(key)
            unique_k.append(k)
    
    # Precompute G-G' differences for each pair (i,j)
    G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]  # (num_pw, num_pw, 2)
    
    def evaluate_cell(eps_grid):
        """
        Computes the band gap using a simplified plane-wave method.
        Uses 289 PWs and 15 k-points.
        """
        inv_eps_grid = 1.0 / eps_grid
        fft_inv_eps = np.fft.fft2(inv_eps_grid)
        fft_inv_eps /= (size * size)
        
        # Precompute coefficient matrix for all G-G' differences
        coeff_matrix = np.zeros((num_pw, num_pw), dtype=complex)
        for i in range(num_pw):
            for j in range(num_pw):
                dx = int(G_diff[i, j, 0])
                dy = int(G_diff[i, j, 1])
                ix = dx % size
                iy = dy % size
                coeff_matrix[i, j] = fft_inv_eps[ix, iy]
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in unique_k:
            k = np.array([kx, ky])
            
            # (k+G)^2 for each G
            k_plus_G = k + G_vectors
            omega_sq = np.sum(k_plus_G ** 2, axis=1)  # (num_pw,)
            
            # M[i,j] = coeff_matrix[i,j] * omega_sq[i]
            M = coeff_matrix * omega_sq[:, np.newaxis]
            
            eigvals = np.linalg.eigvals(M)
            real_eigvals = np.sort(np.real(eigvals))
            
            if len(real_eigvals) >= 5:
                omega_4 = np.sqrt(real_eigvals[3])
                omega_5 = np.sqrt(real_eigvals[4])
                
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap == 0:
            return -1.0
        ratio = 2 * gap / midgap
        return ratio
    
    def smooth_gap_surrogate(eps_grid, beta=50.0):
        """
        Smooth surrogate for the gap-midgap ratio using log-sum-exp style
        soft-min/soft-max to make the objective differentiable-friendly.
        """
        inv_eps_grid = 1.0 / eps_grid
        fft_inv_eps = np.fft.fft2(inv_eps_grid)
        fft_inv_eps /= (size * size)
        
        coeff_matrix = np.zeros((num_pw, num_pw), dtype=complex)
        for i in range(num_pw):
            for j in range(num_pw):
                dx = int(G_diff[i, j, 0])
                dy = int(G_diff[i, j, 1])
                ix = dx % size
                iy = dy % size
                coeff_matrix[i, j] = fft_inv_eps[ix, iy]
        
        # Collect all omega_4 and omega_5 values across k-points
        all_omega_4 = []
        all_omega_5 = []
        
        for kx, ky in unique_k:
            k = np.array([kx, ky])
            k_plus_G = k + G_vectors
            omega_sq = np.sum(k_plus_G ** 2, axis=1)
            
            M = coeff_matrix * omega_sq[:, np.newaxis]
            eigvals = np.linalg.eigvals(M)
            real_eigvals = np.sort(np.real(eigvals))
            
            if len(real_eigvals) >= 5:
                omega_4 = np.sqrt(real_eigvals[3])
                omega_5 = np.sqrt(real_eigvals[4])
                all_omega_4.append(omega_4)
                all_omega_5.append(omega_5)
        
        if len(all_omega_4) == 0:
            return -1.0
        
        all_omega_4 = np.array(all_omega_4)
        all_omega_5 = np.array(all_omega_5)
        
        # Soft max for omega_4, soft min for omega_5
        beta_safe = max(beta, 1.0)
        log_max_omega_4 = np.log(np.sum(np.exp(beta_safe * all_omega_4)) / len(all_omega_4)) / beta_safe
        log_min_omega_5 = -np.log(np.sum(np.exp(-beta_safe * all_omega_5)) / len(all_omega_5)) / beta_safe
        
        gap = log_min_omega_5 - log_max_omega_4
        midgap = 0.5 * (log_min_omega_5 + log_max_omega_4)
        if midgap == 0:
            return -1.0
        ratio = 2 * gap / midgap
        return ratio
    
    # Optimization: Two-phase search
    best_params = None
    best_score = -np.inf
    
    # Initial guess: a dielectric rod-like shape
    init_params = np.zeros(num_params)
    idx_11 = params_idx.index((1, 1))
    init_params[idx_11] = 2.0
    init_params[0] = -0.5
    
    best_params = init_params.copy()
    best_score = evaluate_cell(get_field(best_params))
    
    # Phase 1: Broad random exploration
    num_phase1 = 80
    for attempt in range(num_phase1):
        if time.time() > deadline - 30:
            break
        
        # Generate candidate
        if attempt < 30:
            # Local perturbation of best
            p = best_params + np.random.normal(0, 0.4, num_params)
        elif attempt < 60:
            # Moderate random restart
            p = np.random.uniform(-1.5, 1.5, num_params)
            if (1, 1) in params_idx:
                idx = params_idx.index((1, 1))
                p[idx] = abs(p[idx]) + 0.5
        else:
            # Wide random restart
            p = np.random.uniform(-2.5, 2.5, num_params)
            if (1, 1) in params_idx:
                idx = params_idx.index((1, 1))
                p[idx] = abs(p[idx]) + 1.0
        
        # Use smooth surrogate for quick evaluation, then exact if promising
        score_smooth = smooth_gap_surrogate(get_field(p), beta=30.0)
        
        if score_smooth > best_score - 0.5:  # Promising candidate
            score = evaluate_cell(get_field(p))
            if score > best_score:
                best_score = score
                best_params = p.copy()
    
    # Phase 2: Adaptive local refinement around best
    num_phase2 = 100
    step_size = 0.15
    for attempt in range(num_phase2):
        if time.time() > deadline - 5:
            break
        
        # Adaptive step size: decrease if no improvement recently
        p = best_params + np.random.normal(0, step_size, num_params)
        
        score = evaluate_cell(get_field(p))
        
        if score > best_score:
            best_score = score
            best_params = p.copy()
            # Keep step size or slightly increase
            step_size = min(step_size * 1.1, 0.3)
        else:
            # Decrease step size
            step_size = max(step_size * 0.9, 0.03)
        
        # Every 10 iterations, try a larger jump to escape local minima
        if (attempt + 1) % 10 == 0:
            p_jump = best_params + np.random.normal(0, 0.5, num_params)
            score_jump = evaluate_cell(get_field(p_jump))
            if score_jump > best_score:
                best_score = score_jump
                best_params = p_jump.copy()
    
    # Final polish: very fine local search
    if time.time() < deadline - 3:
        for _ in range(30):
            if time.time() > deadline - 1:
                break
            p = best_params + np.random.normal(0, 0.03, num_params)
            score = evaluate_cell(get_field(p))
            if score > best_score:
                best_score = score
                best_params = p.copy()
    
    if best_params is None:
        # Fallback to seed
        cell = []
        for i in range(size):
            x_c = (i + 0.5) / size - 0.5
            row = []
            for j in range(size):
                y_c = (j + 0.5) / size - 0.5
                if x_c * x_c + y_c * y_c < 0.2 ** 2:
                    row.append(1)
                else:
                    row.append(0)
            cell.append(row)
        return cell
        
    final_grid = get_field(best_params)
    
    # Binarize
    threshold = 0.5 * (eps_low + eps_high)
    binary_cell = []
    for i in range(size):
        row = []
        for j in range(size):
            if final_grid[i, j] > threshold:
                row.append(1)
            else:
                row.append(0)
        binary_cell.append(row)
    
    # Enforce strict symmetry
    sym_cell = np.zeros((size, size), dtype=int)
    counts = np.zeros((size, size), dtype=int)
    
    for i in range(size):
        for j in range(size):
            val = binary_cell[i][j]
            # All 8 symmetric positions
            positions = [
                (i, j),
                (size-1-i, j),
                (i, size-1-j),
                (size-1-i, size-1-j),
                (j, i),
                (size-1-j, i),
                (j, size-1-i),
                (size-1-j, size-1-i)
            ]
            for pi, pj in positions:
                sym_cell[pi, pj] += val
                counts[pi, pj] += 1
    
    # Final symmetric binary cell
    final_binary_cell = []
    for i in range(size):
        row = []
        for j in range(size):
            if counts[i, j] > 0:
                row.append(1 if sym_cell[i, j] > counts[i, j] / 2.0 else 0)
            else:
                row.append(0)
        final_binary_cell.append(row)
        
    return final_binary_cell
