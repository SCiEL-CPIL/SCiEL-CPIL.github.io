import numpy as np
import time

def design(problem):
    """
    Design a 32x32 photonic crystal unit cell to maximize the TM band gap between bands 4 and 5.
    Uses a symmetry-constrained parametric representation with concentric square rings and
    cross-shaped dielectric bars, optimized via rapid coarse-grained plane-wave evaluation.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Pre-compute symmetry orbits for the square cell
    # Symmetries: (x,y) -> (y, size-1-x), (x,y) -> (x, size-1-y)
    orbits = []
    seen = set()
    for i in range(size):
        for j in range(size):
            if (i, j) in seen:
                continue
            orbit = set()
            x, y = i, j
            for _ in range(4):
                orbit.add((x, y))
                orbit.add((x, size - 1 - y))
                x, y = y, size - 1 - x
            orbits.append(orbit)
            seen.update(orbit)
    
    n_orbits = len(orbits)
    
    def orbit_to_grid(orbit_vals):
        grid = np.zeros((size, size), dtype=int)
        for k, orbit in enumerate(orbits):
            val = int(orbit_vals[k])
            for (i, j) in orbit:
                grid[i, j] = val
        return grid
    
    # Parametric representation: define the cell using a small set of continuous parameters
    # Parameters: widths of concentric square rings and cross bars
    # We use 7 parameters to define the dielectric structure
    # p[0]: outer ring width
    # p[1]: gap after outer ring
    # p[2]: inner ring width
    # p[3]: cross arm width (horizontal)
    # p[4]: cross arm width (vertical) 
    # p[5]: center dot radius
    # p[6]: additional offset parameter
    
    def params_to_grid(params):
        """Convert continuous parameters to a 32x32 grid respecting square symmetry."""
        grid = np.zeros((size, size), dtype=int)
        
        # Extract parameters (clamped to valid ranges)
        w_outer = np.clip(params[0], 0.02, 0.15)  # Outer ring width
        gap1 = np.clip(params[1], 0.0, 0.1)       # Gap after outer ring
        w_inner = np.clip(params[2], 0.02, 0.12)  # Inner ring width
        w_cross_h = np.clip(params[3], 0.02, 0.15)  # Cross horizontal width
        w_cross_v = np.clip(params[4], 0.02, 0.15)  # Cross vertical width
        r_center = np.clip(params[5], 0.0, 0.1)    # Center dot radius
        offset = np.clip(params[6], -0.05, 0.05)   # Offset
        
        # Create coordinate grid
        xs = (np.arange(size) + 0.5) / size - 0.5
        ys = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(xs, ys)
        
        # Outer square ring: |x| or |y| in [0.5 - w_outer, 0.5]
        outer_mask = (np.abs(np.abs(X) - (0.5 - offset)) < w_outer/2) | \
                     (np.abs(np.abs(Y) - (0.5 - offset)) < w_outer/2)
        
        # Inner square ring: |x| or |y| in [0.5 - gap1 - w_outer - w_inner, 0.5 - gap1 - w_outer]
        inner_inner = 0.5 - gap1 - w_outer - w_inner
        inner_outer = 0.5 - gap1 - w_outer
        inner_mask = (np.abs(np.abs(X) - (inner_inner + inner_outer)/2) < w_inner/2) | \
                     (np.abs(np.abs(Y) - (inner_inner + inner_outer)/2) < w_inner/2)
        
        # Cross shape: horizontal bar and vertical bar
        cross_h = (np.abs(Y) < w_cross_h/2) & (np.abs(X) < 0.45)
        cross_v = (np.abs(X) < w_cross_v/2) & (np.abs(Y) < 0.45)
        
        # Center dot
        center_mask = (X**2 + Y**2) < r_center**2
        
        # Combine: dielectric is outer ring + inner ring + cross + center
        dielectric_mask = outer_mask | inner_mask | cross_h | cross_v | center_mask
        
        # Apply to grid
        grid[dielectric_mask] = 1
        
        # Enforce symmetry by averaging over orbits
        orbit_vals = np.zeros(n_orbits)
        for k, orbit in enumerate(orbits):
            val = 0
            for (i, j) in orbit:
                val += grid[i, j]
            orbit_vals[k] = 1 if val > len(orbit) / 2 else 0
        
        return orbit_to_grid(orbit_vals), orbit_vals
    
    # Fast PWE evaluator using reduced plane-wave basis
    N_pw = 9  # -4 to 4
    G_x = np.arange(-N_pw//2, N_pw//2 + 1)
    G_y = np.arange(-N_pw//2, N_pw//2 + 1)
    G_vectors = np.array([(gx, gy) for gx in G_x for gy in G_y])
    N_modes = len(G_vectors)
    
    # K-points in irreducible BZ (1/8 of full BZ)
    k_points = [
        (0.0, 0.0),       # Gamma
        (0.5, 0.5),       # M
        (0.5, 0.0),       # X
        (0.25, 0.25),     # Mid-Gamma-M
        (0.25, 0.0),      # Mid-Gamma-X
        (0.5, 0.25),      # Mid-X-M
    ]
    
    def fast_evaluate(orbit_vals):
        """
        Calculate the gap metric for a given orbit value vector using reduced PWE.
        """
        grid = orbit_to_grid(orbit_vals)
        eps_grid = eps_low + (eps_high - eps_low) * grid
        
        # Compute Fourier coefficients via FFT
        eps_fft = np.fft.fft2(eps_grid) / (size * size)
        
        global_max_4 = 0
        global_min_5 = float('inf')
        
        for kx, ky in k_points:
            # Build PWE matrix
            M = np.zeros((N_modes, N_modes), dtype=float)
            
            for i in range(N_modes):
                gx, gy = G_vectors[i]
                kx_i = kx + gx
                ky_i = ky + gy
                k_sq_i = kx_i**2 + ky_i**2
                
                for j in range(N_modes):
                    gx_j, gy_j = G_vectors[j]
                    kx_j = kx + gx_j
                    ky_j = ky + gy_j
                    
                    dot = kx_i * kx_j + ky_i * ky_j
                    
                    dgx = gx - gx_j
                    dgy = gy - gy_j
                    idx_i = dgx % size
                    idx_j = dgy % size
                    eps_val = eps_fft[idx_i, idx_j].real
                    
                    if i == j:
                        M[i, j] = k_sq_i - dot / eps_val
                    else:
                        M[i, j] = - dot / eps_val
            
            try:
                eigvals = np.linalg.eigvalsh(M)
            except:
                return -1.0
            
            if len(eigvals) < 5:
                return -1.0
            
            omega_4 = np.sqrt(max(eigvals[3], 0))
            omega_5 = np.sqrt(max(eigvals[4], 0))
            
            if omega_4 > global_max_4:
                global_max_4 = omega_4
            if omega_5 < global_min_5:
                global_min_5 = omega_5
        
        if global_min_5 == float('inf'):
            return -1.0
        
        gap = global_min_5 - global_max_4
        midgap = (global_min_5 + global_max_4) / 2.0
        if midgap == 0:
            return 0.0
        ratio = gap / midgap
        return ratio
    
    # Optimization: start with multiple random initializations and local search
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 230s kill
    
    best_score = -float('inf')
    best_orbit_vals = None
    best_grid = None
    
    # Try several random initializations
    n_restarts = 20
    steps_per_restart = 50
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Random initial parameters
        params = np.array([
            np.random.uniform(0.05, 0.12),  # Outer ring width
            np.random.uniform(0.0, 0.08),   # Gap
            np.random.uniform(0.05, 0.10),  # Inner ring width
            np.random.uniform(0.05, 0.12),  # Cross H width
            np.random.uniform(0.05, 0.12),  # Cross V width
            np.random.uniform(0.0, 0.08),   # Center radius
            np.random.uniform(-0.03, 0.03), # Offset
        ])
        
        _, current_orbit_vals = params_to_grid(params)
        current_score = fast_evaluate(current_orbit_vals)
        
        if current_score > best_score:
            best_score = current_score
            best_orbit_vals = current_orbit_vals.copy()
            best_grid = orbit_to_grid(current_orbit_vals)
        
        # Local search: perturb parameters
        for step in range(steps_per_restart):
            if time.time() > deadline - 15:
                break
            
            # Gaussian perturbation
            delta = np.random.normal(0, 0.015, size=7)
            new_params = params + delta
            new_params = np.clip(new_params, 0.01, 0.15)
            new_params[1] = np.clip(new_params[1], 0.0, 0.1)
            new_params[6] = np.clip(new_params[6], -0.05, 0.05)
            
            _, new_orbit_vals = params_to_grid(new_params)
            new_score = fast_evaluate(new_orbit_vals)
            
            if new_score > current_score:
                params = new_params
                current_score = new_score
                current_orbit_vals = new_orbit_vals
                
                if current_score > best_score:
                    best_score = current_score
                    best_orbit_vals = current_orbit_vals.copy()
                    best_grid = orbit_to_grid(current_orbit_vals)
    
    # If no valid result, fallback to a simple rod
    if best_grid is None or best_score <= 0:
        # Simple rod fallback
        xs = (np.arange(size) + 0.5) / size - 0.5
        ys = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(xs, ys)
        r = np.sqrt(X**2 + Y**2)
        grid = (r < 0.2).astype(int)
        
        # Enforce symmetry
        orbit_vals = np.zeros(n_orbits)
        for k, orbit in enumerate(orbits):
            val = 0
            for (i, j) in orbit:
                val += grid[i, j]
            orbit_vals[k] = 1 if val > len(orbit) / 2 else 0
        
        best_grid = orbit_to_grid(orbit_vals)
    
    # Ensure exact 0/1 integers and symmetry
    best_grid = np.round(best_grid).astype(int)
    
    # Final symmetry check and enforcement
    for k, orbit in enumerate(orbits):
        val = best_grid[list(orbit)[0][0], list(orbit)[0][1]]
        for (i, j) in orbit:
            best_grid[i, j] = val
    
    return best_grid.tolist()
