"""Photonic crystal unit cell design for 4-5 TM band gap maximization."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def design(problem):
    """
    Design a 32x32 unit cell to maximize the 4-5 TM band gap.
    Uses a parametric shape with multiple parameters optimized via
    local search using a fast PWE solver as a surrogate.
    """
    start_time = time.time()
    deadline = start_time + 215.0  # Safety margin before 230s kill

    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)

    def make_cell(params):
        """
        Generate a 32x32 cell from parameters.
        params: (r0, d, p, cx)
        r0: base radius
        d: dimple depth
        p: superellipse exponent (higher = more square)
        cx: x-offset of centers (0.25 is symmetric)
        """
        r0, d, p, cx = params
        
        # Pixel centers
        xs = np.arange(size)
        ys = np.arange(size)
        X, Y = np.meshgrid(xs, ys)
        
        px = (X + 0.5) / size - 0.5
        py = (Y + 0.5) / size - 0.5
        
        # Four centers
        centers = [
            (-cx, -cx),
            (cx, -cx),
            (-cx, cx),
            (cx, cx)
        ]
        
        inside = np.zeros((size, size), dtype=bool)
        
        for cx_c, cy_c in centers:
            dx = px - cx_c
            dy = py - cy_c
            
            # Superellipse with dimples
            # Base: (|dx|/r0)^p + (|dy|/r0)^p <= 1
            # Dimple: modulate r0 by (1 - d * cos(4*theta))
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            r_shape = r0 * (1 - d * np.cos(4 * theta))
            
            # Superellipse test
            abs_dx = np.abs(dx)
            abs_dy = np.abs(dy)
            
            # Avoid division by zero
            safe_r = np.where(r_shape > 1e-10, r_shape, 1.0)
            val = (abs_dx / safe_r) ** p + (abs_dy / safe_r) ** p
            
            inside |= (val <= 1.0) & (r < r_shape)
        
        cell = inside.astype(int)
        return cell

    def compute_fourier(cell):
        """
        Compute Fourier coefficients of 1/epsilon.
        """
        eps_grid = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_grid
        
        N = size
        g_max = 8
        fourier = {}
        
        F = np.fft.fft2(inv_eps)
        
        for nx in range(-g_max, g_max + 1):
            for ny in range(-g_max, g_max + 1):
                u = nx % N
                v = ny % N
                fourier[(nx, ny)] = F[u, v] / (N * N)
                
        return fourier

    def compute_band_gap(cell, k_points=None):
        """
        Compute the 4-5 TM band gap ratio.
        """
        fourier = compute_fourier(cell)
        
        g_max = 8
        G = []
        for nx in range(-g_max, g_max + 1):
            for ny in range(-g_max, g_max + 1):
                G.append((nx, ny))
        G = np.array(G)
        num_G = len(G)
        
        # Precompute the Fourier matrix
        F_mat = np.zeros((num_G, num_G), dtype=complex)
        for i in range(num_G):
            for j in range(num_G):
                diff = (G[i][0] - G[j][0], G[i][1] - G[j][1])
                F_mat[i, j] = fourier.get(diff, 0.0)
        
        if k_points is None:
            # Use a set of k-points in the irreducible BZ
            # Irreducible BZ for square lattice: 0 <= ky <= kx <= pi
            kx_vals = np.linspace(0, math.pi, 5)
            ky_vals = np.linspace(0, math.pi, 5)
            k_points = []
            for kx in kx_vals:
                for ky in ky_vals:
                    if ky <= kx:
                        k_points.append((kx, ky))
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            k = np.array([kx, ky])
            
            # Build matrix A
            kG = k + G
            D = kG @ kG.T
            
            A = D * F_mat
            
            try:
                eigvals = np.linalg.eigvalsh(A)
            except np.linalg.LinAlgError:
                continue
            
            if len(eigvals) < 5:
                continue
            
            eigvals = np.sort(eigvals)
            omega_4 = math.sqrt(max(eigvals[3], 0))
            omega_5 = math.sqrt(max(eigvals[4], 0))
            
            max_omega_4 = max(max_omega_4, omega_4)
            min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == float('inf') or max_omega_4 == 0:
            return 0.0
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
        
        return gap / midgap

    # Optimization
    # Start with several initial guesses
    initial_params_list = [
        (0.2, 0.2, 4.0, 0.25),
        (0.18, 0.1, 6.0, 0.25),
        (0.22, 0.3, 8.0, 0.25),
        (0.15, 0.0, 4.0, 0.25),
        (0.25, 0.15, 10.0, 0.25),
    ]
    
    best_params = initial_params_list[0]
    best_score = compute_band_gap(make_cell(best_params))
    
    # Local search with multiple restarts
    num_restarts = len(initial_params_list)
    steps_per_restart = 15
    
    for restart in range(num_restarts):
        if time.time() > deadline - 40:
            break
            
        params = list(initial_params_list[restart])
        current_score = compute_band_gap(make_cell(tuple(params)))
        
        if current_score > best_score:
            best_score = current_score
            best_params = tuple(params)
        
        # Local search
        step_sizes = [0.02, 0.05, 0.5, 0.02]
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 15:
                break
            
            improved = False
            
            # Try perturbations in each dimension
            for dim in range(4):
                for sign in [1, -1]:
                    new_params = params.copy()
                    new_params[dim] += sign * step_sizes[dim]
                    
                    # Clip params to valid ranges
                    new_params[0] = max(0.1, min(0.3, new_params[0]))  # r0
                    new_params[1] = max(0.0, min(0.5, new_params[1]))  # d
                    new_params[2] = max(2.0, min(15.0, new_params[2]))  # p
                    new_params[3] = max(0.2, min(0.3, new_params[3]))  # cx
                    
                    new_params = tuple(new_params)
                    s = compute_band_gap(make_cell(new_params))
                    
                    if s > current_score:
                        params = list(new_params)
                        current_score = s
                        improved = True
            
            if not improved:
                # Shrink step sizes
                step_sizes = [s * 0.7 for s in step_sizes]
                if step_sizes[0] < 0.001:
                    break
        
        if current_score > best_score:
            best_score = current_score
            best_params = tuple(params)

    # Final refinement with smaller steps
    if time.time() < deadline - 20:
        params = list(best_params)
        current_score = best_score
        step_sizes = [0.005, 0.01, 0.1, 0.005]
        
        for step in range(10):
            if time.time() > deadline - 10:
                break
            
            improved = False
            for dim in range(4):
                for sign in [1, -1]:
                    new_params = params.copy()
                    new_params[dim] += sign * step_sizes[dim]
                    
                    new_params[0] = max(0.1, min(0.3, new_params[0]))
                    new_params[1] = max(0.0, min(0.5, new_params[1]))
                    new_params[2] = max(2.0, min(15.0, new_params[2]))
                    new_params[3] = max(0.2, min(0.3, new_params[3]))
                    
                    new_params = tuple(new_params)
                    s = compute_band_gap(make_cell(new_params))
                    
                    if s > current_score:
                        params = list(new_params)
                        current_score = s
                        improved = True
            
            if not improved:
                step_sizes = [s * 0.7 for s in step_sizes]
                if step_sizes[0] < 0.0001:
                    break
        
        if current_score > best_score:
            best_score = current_score
            best_params = tuple(params)

    # Final cell
    final_cell = make_cell(best_params)
    final_cell = final_cell.tolist()
    
    return final_cell

# EVOLVE-BLOCK-END
