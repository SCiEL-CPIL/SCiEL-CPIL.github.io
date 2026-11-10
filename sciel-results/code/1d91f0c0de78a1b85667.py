import numpy as np
import time

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell with 4 air holes in a dielectric
    background using C4v symmetry. Optimizes hole parameters to maximize the
    TM band gap between bands 4 and 5.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s budget
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # C4v symmetric parameterization of 4 air holes
    # Holes at (±dx, 0) and (0, ±dy) in cell coordinates (0 to 1)
    # dx, dy in [0.1, 0.45], radius in [0.05, 0.25], superellipse exponent n in [1, 8]
    
    def make_cell(params):
        """
        params: [dx, dy, radius, n_exp]
        Creates a 32x32 binary cell with dielectric background and 4 air holes.
        """
        dx, dy, radius, n_exp = params
        cell = np.ones((size, size), dtype=int)
        
        x = np.linspace(0.5/size, 1 - 0.5/size, size)
        X, Y = np.meshgrid(x, x, indexing='ij')
        
        # Center of cell
        cx, cy = 0.5, 0.5
        
        # Four holes: (cx±dx, cy), (cx, cy±dy)
        hole_centers = [
            (cx + dx, cy),
            (cx - dx, cy),
            (cx, cy + dy),
            (cx, cy - dy)
        ]
        
        for hx, hy in hole_centers:
            # Superellipse: |x-hx|^n + |y-hy|^n <= radius^n
            dist = np.abs(X - hx) ** n_exp + np.abs(Y - hy) ** n_exp
            cell[dist <= radius ** n_exp] = 0
        
        # Enforce strict C4v symmetry by averaging with all 8 symmetric counterparts
        sym_cell = np.zeros((size, size), dtype=float)
        counts = np.zeros((size, size), dtype=float)
        
        for i in range(size):
            for j in range(size):
                val = cell[i, j]
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
        
        final_cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                if counts[i, j] > 0:
                    final_cell[i, j] = 1 if sym_cell[i, j] > counts[i, j] / 2.0 else 0
                else:
                    final_cell[i, j] = 0
        
        return final_cell
    
    def evaluate_cell(cell):
        """
        Computes the TM band gap using a plane-wave method.
        Returns the gap-midgap ratio.
        """
        eps_grid = eps_low + (eps_high - eps_low) * cell.astype(float)
        inv_eps_grid = 1.0 / eps_grid
        fft_inv_eps = np.fft.fft2(inv_eps_grid)
        fft_inv_eps /= (size * size)
        
        # Use 17x17 = 289 plane waves
        N_pw = 8
        G_vals = np.arange(-N_pw, N_pw + 1)
        Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
        G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
        num_pw = len(G_vectors)
        
        # Precompute G-G' differences
        G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]
        
        # K points in irreducible BZ
        k_points = []
        for t in np.linspace(0, 0.5, 8):
            k_points.append((t, 0))
            k_points.append((0.5, t))
            k_points.append((t, t))
        seen = set()
        unique_k = []
        for k in k_points:
            key = (round(k[0], 10), round(k[1], 10))
            if key not in seen:
                seen.add(key)
                unique_k.append(k)
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in unique_k:
            k = np.array([kx, ky])
            k_plus_G = k + G_vectors
            omega_sq = np.sum(k_plus_G ** 2, axis=1)
            
            # Build coefficient matrix
            coeff_matrix = np.zeros((num_pw, num_pw), dtype=complex)
            for i in range(num_pw):
                for j in range(num_pw):
                    dx = int(G_diff[i, j, 0])
                    dy = int(G_diff[i, j, 1])
                    ix = dx % size
                    iy = dy % size
                    coeff_matrix[i, j] = fft_inv_eps[ix, iy]
            
            M = coeff_matrix * omega_sq[:, np.newaxis]
            
            eigvals = np.linalg.eigvalsh(M)
            real_eigvals = np.sort(np.real(eigvals))
            
            if len(real_eigvals) >= 5:
                omega_4 = np.sqrt(max(real_eigvals[3], 0))
                omega_5 = np.sqrt(max(real_eigvals[4], 0))
                
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
    
    # Optimization: Nelder-Mead style local search with restarts
    best_params = [0.25, 0.25, 0.15, 3.0]
    best_score = evaluate_cell(make_cell(best_params))
    
    # Local search
    step_sizes = [0.03, 0.03, 0.02, 0.5]
    
    num_iterations = 0
    while time.time() < deadline - 5:
        num_iterations += 1
        
        # Generate candidate
        if num_iterations % 50 == 0:
            # Random restart
            p = [
                np.random.uniform(0.15, 0.4),
                np.random.uniform(0.15, 0.4),
                np.random.uniform(0.08, 0.22),
                np.random.uniform(1.5, 6.0)
            ]
        else:
            # Local perturbation
            p = best_params.copy()
            for i in range(len(p)):
                p[i] += np.random.normal(0, step_sizes[i])
            # Clamp
            p[0] = np.clip(p[0], 0.1, 0.45)
            p[1] = np.clip(p[1], 0.1, 0.45)
            p[2] = np.clip(p[2], 0.05, 0.25)
            p[3] = np.clip(p[3], 1.0, 8.0)
        
        score = evaluate_cell(make_cell(p))
        
        if score > best_score:
            best_score = score
            best_params = p.copy()
            # Reduce step sizes
            step_sizes = [s * 0.9 for s in step_sizes]
        else:
            # Gradually increase step sizes if stuck
            if num_iterations % 20 == 0:
                step_sizes = [min(s * 1.1, 0.05) for s in step_sizes[:3]] + [min(step_sizes[3] * 1.05, 1.0)]
    
    final_cell = make_cell(best_params)
    
    # Convert to nested list
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(final_cell[i, j]))
        result.append(row)
    
    return result
