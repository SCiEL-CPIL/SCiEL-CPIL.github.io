import numpy as np
import time
import sys

def design(problem):
    """
    Design a 32x32 photonic crystal unit cell to maximize the TM band gap between bands 4 and 5.
    
    Strategy: Spectral-domain inverse design using a low-resolution continuous representation.
    1. Represent the dielectric distribution as a low-resolution grid (e.g., 8x8) of continuous values in [0, 1].
    2. Upsample this to 32x32 using bilinear interpolation to get the exact pixel map for Fourier coefficients.
    3. Use a reduced plane-wave basis (e.g., 15x15) for fast evaluation.
    4. Optimize the low-resolution grid values using Gradient Descent on the gap metric.
    5. Finalize by binarizing the result and performing a few local bit-flips to refine.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Symmetry handling for final binarization
    # The cell is 32x32. Center is effectively at (15.5, 15.5).
    # Symmetries:
    # 1. Identity: (i, j)
    # 2. Rot 90: (j, size-1-i)
    # 3. Rot 180: (size-1-i, size-1-j)
    # 4. Rot 270: (size-1-j, i)
    # 5. Refl X (vertical axis): (i, size-1-j)
    # 6. Refl Y (horizontal axis): (size-1-i, j)
    # 7. Refl Diag: (j, i)
    # 8. Refl AntiDiag: (size-1-j, size-1-i)
    
    orbits = []
    seen = set()
    for i in range(size):
        for j in range(size):
            if (i, j) in seen:
                continue
            orbit = set()
            x, y = i, j
            # Generate orbit under the symmetry group
            points = []
            points.append((x, y))
            points.append((y, size-1-x))
            points.append((size-1-x, size-1-y))
            points.append((size-1-y, x))
            points.append((x, size-1-y))
            points.append((size-1-x, y))
            points.append((y, x))
            points.append((size-1-y, size-1-x))
            
            for p in points:
                if p not in orbit:
                    orbit.add(p)
            
            orbits.append(orbit)
            seen.update(orbit)
    
    n_orbits = len(orbits)
    
    def orbit_to_grid(orbit_vals):
        grid = np.zeros((size, size), dtype=np.float64)
        for k in range(n_orbits):
            val = orbit_vals[k]
            for (i, j) in orbits[k]:
                grid[i, j] = val
        return grid

    # Fast Evaluator
    # We use a reduced plane wave basis for speed during optimization.
    # N_pw = 15 (from -7 to 7) -> 225 modes. 
    # This is a balance between accuracy and speed.
    N_pw = 15
    G_x = np.arange(-N_pw//2, N_pw//2 + 1)
    G_y = np.arange(-N_pw//2, N_pw//2 + 1)
    G_vectors = np.array([(gx, gy) for gx in G_x for gy in G_y])
    N_modes = len(G_vectors)
    
    # K-points in the irreducible Brillouin zone
    # We sample key points to approximate the gap.
    k_points = [
        (0.0, 0.0),
        (0.5, 0.5),
        (0.5, 0.0),
        (0.25, 0.25),
        (0.25, 0.0),
        (0.5, 0.25),
        (0.125, 0.125),
        (0.375, 0.125),
        (0.125, 0.375),
        (0.375, 0.375)
    ]

    # Precompute G_diff for epsilon matrix construction
    # G_diff[i, j] = G_i - G_j
    G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]
    idx_x = G_diff[:, :, 0] % size
    idx_y = G_diff[:, :, 1] % size

    def fast_evaluate(grid):
        """
        Calculate the gap metric for a given 32x32 grid (float values 0-1).
        """
        # Compute Fourier coefficients epsilon_G
        # epsilon(r) = eps_low + (eps_high - eps_low) * grid(r)
        eps_grid = eps_low + (eps_high - eps_low) * grid
        
        # FFT to get coefficients
        eps_fft = np.fft.fft2(eps_grid) / (size * size)
        
        global_max_4 = 0.0
        global_min_5 = float('inf')
        
        for kx, ky in k_points:
            # Build the PWE matrix
            # M_{G, G'} = (k+G)^2 delta_{G,G'} - (k+G).(k+G') / epsilon_{G-G'}
            
            # Precompute (k+G) vectors and their squares
            kG = G_vectors + np.array([kx, ky])
            kG_sq = np.sum(kG**2, axis=1)
            
            # Dot product matrix
            dot = np.dot(kG, kG.T)
            
            # Epsilon matrix
            eps_mat = eps_fft[idx_x, idx_y].real
            
            # M = diag(kG_sq) - dot / eps_mat
            # Note: division is element-wise
            # Handle potential division by zero? eps_low is 1, so min eps is 1.
            # So eps_mat >= 1. No zero division.
            
            M = np.diag(kG_sq) - dot / eps_mat
            
            # Solve for eigenvalues
            try:
                eigvals = np.linalg.eigvalsh(M)
            except:
                return -1.0
                
            if len(eigvals) < 5:
                return -1.0
                
            omega_4 = np.sqrt(eigvals[3])
            omega_5 = np.sqrt(eigvals[4])
            
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

    # Initial design: A low-resolution grid
    # We use a 4x4 grid to represent the unit cell, upsampling to 32x32.
    # This reduces the number of variables from 1024 to 16.
    res = 4
    init_grid_low = np.ones((res, res), dtype=np.float64) * 0.5
    # Make it symmetric under the square group
    # The 4x4 grid maps to 32x32.
    # Symmetry of the 32x32 grid implies symmetry of the 4x4 grid if we align properly.
    # For now, let's just start with a simple pattern.
    # Try a rod-like pattern: center high, edges low.
    init_grid_low[1:3, 1:3] = 0.8
    init_grid_low[0:2, 0:2] = 0.2
    init_grid_low[2:4, 2:4] = 0.2
    init_grid_low[0:2, 2:4] = 0.2
    init_grid_low[2:4, 0:2] = 0.2

    def upsample(low_grid):
        # Bilinear interpolation from res x res to size x size
        # low_grid is defined on a grid of points.
        # We want to map (i, j) in 0..size-1 to (u, v) in 0..res-1
        # u = i * (res-1) / (size-1)
        # v = j * (res-1) / (size-1)
        u = np.linspace(0, res-1, size)
        v = np.linspace(0, res-1, size)
        U, V = np.meshgrid(u, v, indexing='ij')
        
        # Bilinear interpolation
        U0 = np.floor(U).astype(int)
        U1 = np.minimum(U0 + 1, res - 1)
        V0 = np.floor(V).astype(int)
        V1 = np.minimum(V0 + 1, res - 1)
        
        dU = U - U0
        dV = V - V0
        
        # Values
        a = low_grid[U0, V0]
        b = low_grid[U1, V0]
        c = low_grid[U0, V1]
        d = low_grid[U1, V1]
        
        # Interpolate
        grid = a*(1-dU)*(1-dV) + b*dU*(1-dV) + c*(1-dU)*dV + d*dU*dV
        return grid

    current_low = init_grid_low.copy()
    current_grid = upsample(current_low)
    current_score = fast_evaluate(current_grid)
    best_low = current_low.copy()
    best_score = current_score
    
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 20s margin for safety
    
    # Gradient Descent
    lr = 0.05
    momentum = 0.9
    velocity = np.zeros_like(current_low)
    
    while time.time() < deadline:
        # Calculate gradient using finite differences
        # Since res is small (4x4), we can do this efficiently
        grad = np.zeros_like(current_low)
        h = 0.01
        
        for i in range(res):
            for j in range(res):
                orig = current_low[i, j]
                
                # +h
                current_low[i, j] = orig + h
                grid_plus = upsample(current_low)
                score_plus = fast_evaluate(grid_plus)
                
                # -h
                current_low[i, j] = orig - h
                grid_minus = upsample(current_low)
                score_minus = fast_evaluate(grid_minus)
                
                # Restore
                current_low[i, j] = orig
                
                grad[i, j] = (score_plus - score_minus) / (2 * h)
        
        # Update with momentum
        velocity = momentum * velocity + lr * grad
        current_low += velocity
        
        # Clip to [0, 1]
        current_low = np.clip(current_low, 0.0, 1.0)
        
        # Evaluate new score
        current_grid = upsample(current_low)
        current_score = fast_evaluate(current_grid)
        
        if current_score > best_score:
            best_score = current_score
            best_low = current_low.copy()
            
        # Adaptive learning rate
        if time.time() > start_time + 60:
            lr *= 0.95
        if lr < 0.001:
            break

    # Finalize: Binarize the best low-res grid and refine with bit flips
    best_grid_float = upsample(best_low)
    best_grid_bin = (best_grid_float > 0.5).astype(int)
    
    # Convert to orbit values
    best_vals = np.zeros(n_orbits, dtype=int)
    for k in range(n_orbits):
        # Take the value at the first pixel in the orbit
        i, j = list(best_orbits_list[0][0]) if False else list(orbits[k])[0]
        best_vals[k] = best_grid_bin[i, j]
        
    # Refine with bit flips
    current_vals = best_vals.copy()
    current_score = fast_evaluate(orbit_to_grid(current_vals))
    
    T = 0.01
    cooling_rate = 0.999
    
    while time.time() < deadline:
        idx = np.random.randint(n_orbits)
        new_vals = current_vals.copy()
        new_vals[idx] = 1 - new_vals[idx]
        
        new_score = fast_evaluate(orbit_to_grid(new_vals))
        
        if new_score > current_score:
            current_vals = new_vals
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_vals = new_vals.copy()
        else:
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if np.random.rand() < prob:
                    current_vals = new_vals
                    current_score = new_score
                    
        T *= cooling_rate
        if T < 0.001:
            T = 0.001

    # Return the best grid found
    best_grid = orbit_to_grid(best_vals)
    return best_grid.astype(int).tolist()

# Helper to avoid global state issues if needed, but we compute orbits inside design
# The code above computes orbits inside design, so it's self-contained.

# Wait, I need to make sure the code is fully contained in the design function or module level.
# The current draft has orbits computed inside design.
# I will clean up the code to ensure it's a single valid python script.

