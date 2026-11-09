# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def _compute_volume_fast(positions, n_quad=100):
    """
    Compute union volume using interval merging and Gauss-Legendre quadrature.
    This is much faster than Monte Carlo and more accurate for search.
    """
    N = 8
    x = positions[:, 0]
    y = positions[:, 1]
    i_idx = np.arange(64) // N
    j_idx = np.arange(64) % N
    
    # Gauss-Legendre nodes and weights on [0, 1]
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    c_vals = 0.5 * (nodes + 1)
    w_vals = 0.5 * weights
    
    total_area = 0.0
    for c, w in zip(c_vals, w_vals):
        # For this height c, compute the union area of the 64 rectangles
        # Rect k: [x_k + c*i_k/N, x_k + c*i_k/N + 1/N] x [y_k + c*j_k/N, y_k + c*j_k/N + 1/N]
        
        x_intervals = []
        y_intervals = []
        
        for k in range(64):
            i_k = i_idx[k]
            j_k = j_idx[k]
            x_left = x[k] + c * i_k / N
            x_right = x_left + 1.0 / N
            y_left = y[k] + c * j_k / N
            y_right = y_left + 1.0 / N
            
            x_intervals.append((x_left, x_right))
            y_intervals.append((y_left, y_right))
            
        # Merge intervals in X and Y
        # Note: The union area of rectangles is NOT simply (merged X length) * (merged Y length).
        # We need to compute the area of the union of rectangles directly.
        # Standard sweep-line algorithm for union area of rectangles.
        
        # Events: (x, type, y1, y2)
        # type 0: start, type 1: end
        events = []
        for k in range(64):
            x_left, x_right = x_intervals[k]
            y_left, y_right = y_intervals[k]
            events.append((x_left, 0, y_left, y_right))
            events.append((x_right, 1, y_left, y_right))
            
        events.sort(key=lambda e: (e[0], e[1]))
        
        active_y_intervals = []
        area = 0.0
        last_x = events[0][0]
        
        for event in events:
            curr_x, type_, y1, y2 = event
            # Add area from last_x to curr_x
            if active_y_intervals:
                # Merge active y intervals to find total covered length
                active_y_intervals.sort()
                merged = []
                start, end = active_y_intervals[0]
                for s, e in active_y_intervals[1:]:
                    if s <= end:
                        end = max(end, e)
                    else:
                        merged.append((start, end))
                        start, end = s, e
                merged.append((start, end))
                
                total_len = sum(e - s for s, e in merged)
                area += (curr_x - last_x) * total_len
            
            # Update active intervals
            if type_ == 0:
                active_y_intervals.append((y1, y2))
            else:
                # Remove the interval (y1, y2)
                # This is O(N) but N is small (64)
                active_y_intervals = [iv for iv in active_y_intervals if not (iv[0] == y1 and iv[1] == y2)]
            
            last_x = curr_x
        
        total_area += w * area
        
    return total_area

def _spectral_grid_search(max_time=500):
    """
    Perform a grid search over Fourier coefficients to find a good configuration.
    """
    start_time = time.time()
    N = 8
    num_tubes = 64
    
    # Define spectral representation
    # x_k = mu_x + sum_m (a_m cos(2*pi*m*k/64) + b_m sin(2*pi*m*k/64))
    # y_k = mu_y + sum_m (a_m cos(2*pi*m*k/64) + b_m sin(2*pi*m*k/64))
    
    # We will search over mu_x, mu_y, and the first M coefficients.
    # To keep the search space small, we restrict to M=1 and M=2.
    
    # Grid parameters
    mu_range = np.linspace(-0.5, 0.5, 5)
    coeff_range = np.linspace(-0.2, 0.2, 5)
    
    best_positions = None
    best_volume = float('inf')
    
    # Precompute basis functions
    k = np.arange(num_tubes)
    angles = 2 * np.pi * k / num_tubes
    
    # Search over mu_x, mu_y, a_x1, b_x1, a_y1, b_y1
    # This is 6 dimensions. 5^6 = 15625 evaluations.
    # Each evaluation takes ~0.001s with fast volume? Let's check.
    
    for mu_x in mu_range:
        for mu_y in mu_range:
            for ax1 in coeff_range:
                for bx1 in coeff_range:
                    for ay1 in coeff_range:
                        for by1 in coeff_range:
                            if time.time() - start_time > max_time - 60:
                                return best_positions, best_volume
                            
                            # Construct positions
                            x = mu_x + ax1 * np.cos(angles) + bx1 * np.sin(angles)
                            y = mu_y + ay1 * np.cos(angles) + by1 * np.sin(angles)
                            
                            positions = np.column_stack((x, y))
                            
                            # Estimate volume
                            vol = _compute_volume_fast(positions, n_quad=50)
                            
                            if vol < best_volume:
                                best_volume = vol
                                best_positions = positions.copy()
                                
    # If we have time, do a second pass with finer grid or higher modes
    if best_positions is not None and time.time() - start_time < max_time - 30:
        # Refine around best found
        center_x, center_y = best_positions[0, 0], best_positions[0, 1] # Approximate
        # Actually, just do a local search on the 128 variables from the best spectral solution
        # This combines the global spectral search with local coordinate optimization
        
        # Use a simple local search
        current = best_positions.copy()
        current_V = best_volume
        rng = np.random.default_rng(123)
        
        step_sizes = [0.01, 0.005, 0.001]
        for step in step_sizes:
            for _ in range(100): # Max steps
                if time.time() - start_time > max_time - 10:
                    break
                improved = False
                for k_idx in range(num_tubes):
                    for coord in range(2):
                        for direction in [-1, 1]:
                            trial = current.copy()
                            trial[k_idx, coord] += direction * step * rng.uniform(0.5, 1.5)
                            if trial[k_idx, coord] < -1 or trial[k_idx, coord] > 1:
                                continue
                            trial_V = _compute_volume_fast(trial, n_quad=20)
                            if trial_V < current_V - 1e-6:
                                current = trial
                                current_V = trial_V
                                improved = True
                if not improved:
                    break
                    
        if current_V < best_volume:
            best_volume = current_V
            best_positions = current
    
    return best_positions, best_volume

def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Uses spectral grid search over Fourier coefficients as the primary optimization strategy.
    """
    start_time = time.time()
    max_time = 600 # Safety margin
    
    # Initial guess: Keich binary construction
    N = 8
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = keich[i_flat]
    base_y = keich[j_flat]
    positions_keich = np.stack([base_x, base_y], axis=1)
    
    # Run Spectral Grid Search
    try:
        positions, vol = _spectral_grid_search(max_time=max_time - 20)
        if positions is None:
            positions = positions_keich
    except Exception as e:
        print(f"Spectral search failed: {e}")
        positions = positions_keich
        
    return positions

# EVOLVE-BLOCK-END
