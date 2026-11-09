import numpy as np
import time

def compute_union_area_at_c(c, x_positions, y_positions, N=8):
    """
    Compute the area of the union of rectangles at height c.
    x_positions: array of shape (N,) for x-base positions (one per i)
    y_positions: array of shape (N,) for y-base positions (one per j)
    """
    i = np.arange(N)
    x_left = x_positions + c * i / N
    x_right = x_left + 1.0 / N
    j = np.arange(N)
    y_left = y_positions + c * j / N
    y_right = y_left + 1.0 / N
    
    x_edges = np.unique(np.concatenate([x_left, x_right]))
    y_edges = np.unique(np.concatenate([y_left, y_right]))
    
    area = 0.0
    for px in range(len(x_edges) - 1):
        x0 = x_edges[px]
        x1 = x_edges[px+1]
        if x1 <= x0: continue
        cover_i = np.where((x_left <= x0) & (x_right >= x1))[0]
        if len(cover_i) == 0:
            continue
        for py in range(len(y_edges) - 1):
            y0 = y_edges[py]
            y1 = y_edges[py+1]
            if y1 <= y0: continue
            cover_j = np.where((y_left <= y0) & (y_right >= y1))[0]
            if len(cover_j) == 0:
                continue
            area += (x1 - x0) * (y1 - y0)
            
    return area

def compute_volume(x_positions, y_positions, N=8, num_c=10):
    """
    Estimate the volume by integrating the area over c in [0,1].
    """
    c_vals = np.linspace(0, 1, num_c, endpoint=False) + 0.5/num_c
    areas = [compute_union_area_at_c(c, x_positions, y_positions, N) for c in c_vals]
    return np.mean(areas)

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    start_time = time.time()
    max_time = 590.0 # Safety margin
    
    # Initial Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    x_positions = keich.copy()
    y_positions = keich.copy()
    
    best_vol = compute_volume(x_positions, y_positions)
    best_x = x_positions.copy()
    best_y = y_positions.copy()
    
    step_size = 0.05
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < max_time:
        # Local search step
        improved = False
        for _ in range(10): # Inner loop for multiple moves
            if time.time() - start_time > max_time:
                break
            
            idx = rng.integers(0, N)
            is_x = rng.random() > 0.5
            
            if is_x:
                old_val = x_positions[idx]
                x_positions[idx] += rng.uniform(-step_size, step_size)
            else:
                old_val = y_positions[idx]
                y_positions[idx] += rng.uniform(-step_size, step_size)
                
            new_vol = compute_volume(x_positions, y_positions)
            
            if new_vol < best_vol:
                best_vol = new_vol
                best_x = x_positions.copy()
                best_y = y_positions.copy()
                improved = True
            else:
                if is_x:
                    x_positions[idx] = old_val
                else:
                    y_positions[idx] = old_val
        
        # Adaptive step size
        if improved:
            step_size *= 0.8
        else:
            step_size *= 0.5
            if step_size < 1e-5:
                # Reset to a larger step and randomize positions slightly
                step_size = 0.05
                x_positions += rng.uniform(-0.1, 0.1, N)
                y_positions += rng.uniform(-0.1, 0.1, N)
                # Ensure we don't drift too far, maybe pull back towards best
                x_positions = 0.5 * x_positions + 0.5 * best_x
                y_positions = 0.5 * y_positions + 0.5 * best_y

    x_positions = best_x
    y_positions = best_y
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_positions[i_flat]
    base_y = y_positions[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
