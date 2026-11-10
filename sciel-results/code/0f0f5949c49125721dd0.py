# EVOLVE-BLOCK-START
import numpy as np


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.

    We build N*N = 64 "generalized tubes" indexed by (i, j), i, j in {0, ..., N-1}.
    Tube (i, j) has a base position (x, y) and, at height c in [0, 1], occupies the
    axis-aligned square
        [x + c*i/N, x + 1/N + c*i/N] x [y + c*j/N, y + 1/N + c*j/N].

    The task is to choose the 64 base positions so that the VOLUME of the union of
    all tubes is as SMALL as possible.

    This implementation uses a spectral grid-search over Fourier coefficients of
    the tube displacement functions. Due to translational symmetry and the
    separability of the problem in x and y, we can optimize the x-displacements
    and y-displacements independently. For each dimension, we represent the
    displacement function as a sum of sinusoidal components (Fourier series)
    and perform a grid search over the Fourier coefficients to find the
    configuration that minimizes the 1-D union length, which in turn minimizes
    the 2-D union area (and thus the 3-D union volume).

    Returns:
        positions: np.ndarray of shape (64, 2). Row k = tube k in (i, j) raster order
            (i outer, j inner, i.e., numpy meshgrid indexing="ij" then flatten),
            column 0 = base x-coordinate, column 1 = base y-coordinate.
    """
    N = 8
    N2 = N * N  # 64 tubes total

    def optimize_1d_spectral(N, num_harmonics=3, grid_points=15, search_range=0.4):
        """
        Optimize the 1-D base positions for N tubes using spectral grid search.
        
        Each tube i has base position x_i and at height c occupies
        [x_i + c*i/N, x_i + c*i/N + 1/N].
        
        We parameterize x_i as a Fourier series:
        x_i = sum_{m=1}^{M} (a_m * cos(2*pi*m*i/N) + b_m * sin(2*pi*m*i/N))
        
        Then we do a grid search over the Fourier coefficients a_m, b_m.
        """
        i_indices = np.arange(N)
        harmonics = np.arange(1, num_harmonics + 1)
        
        cos_basis = np.zeros((num_harmonics, N))
        sin_basis = np.zeros((num_harmonics, N))
        
        for m in range(1, num_harmonics + 1):
            angle = 2 * np.pi * m * i_indices / N
            cos_basis[m-1] = np.cos(angle)
            sin_basis[m-1] = np.sin(angle)
        
        # Precompute c samples for union length
        c_samples = np.linspace(0, 1, 200)
        
        def compute_union_length(d):
            """
            Compute the 1-D union length for base positions d.
            """
            total_length = 0.0
            for c in c_samples:
                starts = d + c * i_indices / N
                ends = starts + 1.0 / N
                
                order = np.argsort(starts)
                s_sorted = starts[order]
                e_sorted = ends[order]
                
                union_len = 0.0
                cur_start = s_sorted[0]
                cur_end = e_sorted[0]
                
                for k in range(1, N):
                    if s_sorted[k] <= cur_end:
                        cur_end = max(cur_end, e_sorted[k])
                    else:
                        union_len += cur_end - cur_start
                        cur_start = s_sorted[k]
                        cur_end = e_sorted[k]
                union_len += cur_end - cur_start
                total_length += union_len
            
            return total_length / len(c_samples)

        best_length = np.inf
        best_coeffs = np.zeros(2 * num_harmonics)
        
        # Coarse-to-fine grid search
        # Start with 2 harmonics for the coarse search
        M_coarse = 2
        
        # Build search grids for coarse stage
        grid_vals_coarse = np.linspace(-search_range, search_range, grid_points)
        
        # Evaluate all combinations for first M_coarse harmonics
        for a1 in grid_vals_coarse:
            for b1 in grid_vals_coarse:
                for a2 in grid_vals_coarse:
                    for b2 in grid_vals_coarse:
                        d = (a1 * cos_basis[0] + b1 * sin_basis[0] + 
                             a2 * cos_basis[1] + b2 * sin_basis[1])
                        
                        length = compute_union_length(d)
                        if length < best_length:
                            best_length = length
                            best_coeffs = np.array([a1, b1, a2, b2] + [0.0] * (2 * num_harmonics - 4))
        
        # Local refinement using coordinate descent with adaptive step size
        def update_coeff(coeff_idx, new_val):
            coeffs = best_coeffs.copy()
            coeffs[coeff_idx] = new_val
            d = np.zeros(N)
            for m in range(1, num_harmonics + 1):
                a_m = coeffs[2*(m-1)]
                b_m = coeffs[2*(m-1)+1]
                d += a_m * cos_basis[m-1] + b_m * sin_basis[m-1]
            return d

        step = 0.02
        for iteration in range(100):
            improved = False
            for k in range(2 * num_harmonics):
                old_val = best_coeffs[k]
                # Try several deltas
                for delta in [-step, -step/2, step/2, step]:
                    new_val = old_val + delta
                    d = update_coeff(k, new_val)
                    length = compute_union_length(d)
                    if length < best_length - 1e-7:
                        best_length = length
                        best_coeffs[k] = new_val
                        improved = True
                        break
            if not improved:
                # Reduce step size
                step *= 0.5
                if step < 1e-4:
                    break
                continue
        
        d_final = np.zeros(N)
        for m in range(1, num_harmonics + 1):
            a_m = best_coeffs[2*(m-1)]
            b_m = best_coeffs[2*(m-1)+1]
            d_final += a_m * cos_basis[m-1] + b_m * sin_basis[m-1]
            
        return d_final

    # Optimize 1-D positions for x and y (they are the same by symmetry)
    base_x = optimize_1d_spectral(N, num_harmonics=3, grid_points=15, search_range=0.4)
    base_y = base_x.copy()
    
    # Build the 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x_flat = base_x[i_flat]
    base_y_flat = base_y[j_flat]
    
    positions = np.stack([base_x_flat, base_y_flat], axis=1)
    return positions


# EVOLVE-BLOCK-END
