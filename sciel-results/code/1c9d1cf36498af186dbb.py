# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _union_area(x, n=64, M=20000):
    """
    Compute the area of the union of n triangles using the specified quadrature.
    This is the exact metric used by the evaluator.
    """
    m = np.arange(M)
    y = (m + 0.5) / M
    i_idx = np.arange(n)
    a = (i_idx + 1) / n
    w = (1.0 - y) / n
    
    # L_i(y) = x_i + a_i * y
    # Shape: (n, M)
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    # Sort L along axis 0
    L_sorted = np.sort(L, axis=0)
    
    # Consecutive differences
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    
    # Union length = w + sum over j of min(d_j, w)
    min_d_w = np.minimum(d, w[np.newaxis, :])
    
    union_lengths = w[np.newaxis, :] + np.sum(min_d_w, axis=0)
    
    area = np.mean(union_lengths)
    return area

def _keich_init(n):
    """
    Generate initial positions based on the Keich construction.
    """
    k = int(np.log2(n))
    j_arr = np.arange(1, k + 1)
    x0 = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k)
    return x0

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem.
    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    M = 20000
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin (10s buffer)
    
    # Initialize with Keich construction, which is a known good starting point
    best_x = _keich_init(n)
    best_area = _union_area(best_x, n, M)
    
    # Simulated Annealing Parameters
    # We need to estimate a reasonable initial temperature.
    # The area is roughly ~0.5 to 0.8. Small changes in x (0.01) can change area by ~0.001.
    # Let's start with T = 0.01 and cool down.
    T_start = 0.05
    T_end = 1e-6
    alpha = 0.999  # Cooling rate
    
    current_x = best_x.copy()
    current_area = best_area
    
    # Pre-compute constants for speed
    a = (np.arange(n) + 1) / n
    
    # Main SA Loop
    step = 0
    while time.time() < deadline:
        # Cooling
        T = T_start * (alpha ** step)
        if T < T_end:
            # If temperature is too low, we can break or just do a few more steps
            # To be safe, let's break if T is very small and we've had no improvement recently
            # But for simplicity, we just stop when T is below T_end
            # Actually, let's continue for a bit longer to polish, but with smaller steps
            pass
        
        # Generate a move
        # Randomly select a subset of indices to perturb
        # Perturbing all 64 at once with small noise is a good strategy for this dimension
        # Or perturb a random subset? Let's perturb all with Gaussian noise.
        # Scale of noise: start with 0.1, decrease with T?
        # Standard SA: step size is often fixed or scaled with T.
        # Let's scale noise std with sqrt(T) or just T? 
        # Since the landscape is piecewise linear, small steps are better for polishing.
        # Let's use a step size that decreases with temperature.
        noise_scale = 0.05 * T / T_start + 0.001
        
        x_new = current_x + noise_scale * np.random.randn(n)
        
        # Evaluate new point
        area_new = _union_area(x_new, n, M)
        
        # Acceptance criterion
        delta = area_new - current_area
        if delta < 0:
            # Improvement
            current_x = x_new
            current_area = area_new
            if current_area < best_area:
                best_area = current_area
                best_x = current_x.copy()
        else:
            # Deterioration
            # Accept with probability exp(-delta / T)
            if T > 0:
                prob = math.exp(-delta / T)
                if np.random.random() < prob:
                    current_x = x_new
                    current_area = area_new
        
        step += 1
        
        # Periodic check to ensure we don't exceed deadline
        if time.time() > deadline - 5.0:
            break

    # Final polish: try a few random restarts from best_x with very small noise
    # Or just return best_x.
    # Given the time budget, a few final gradient-free local searches (random steps) can help.
    if time.time() < deadline - 10.0:
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            x_try = best_x + 0.01 * np.random.randn(n)
            area_try = _union_area(x_try, n, M)
            if area_try < best_area:
                best_area = area_try
                best_x = x_try.copy()

    # Validation
    if not np.all(np.isfinite(best_x)):
        return _keich_init(n)
    if best_area <= 0:
        return _keich_init(n)
        
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
