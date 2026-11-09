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
    This is a strong baseline for Kakeya needle problems.
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
    Uses Simulated Annealing to explore the discrete combinatorial space of triangle orderings/positions.
    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    M = 20000
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s kill
    
    # Initial candidates
    candidates = [
        _keich_init(n),
        np.linspace(-1.0, 1.0, n),
        np.random.uniform(-1.0, 1.0, n),
    ]
    
    best_x = None
    best_area = np.inf
    
    # Evaluate initial candidates
    for x0 in candidates:
        area = _union_area(x0, n, M)
        if area < best_area:
            best_area = area
            best_x = x0.copy()
    
    current_x = best_x.copy()
    current_area = best_area
    
    # Simulated Annealing Parameters
    # We treat the positions as continuous variables but use large jumps to explore
    # the combinatorial structure (ordering of triangles at different heights).
    T_start = 2.0
    T_end = 0.01
    alpha = 0.995 # Cooling rate
    max_iterations = 20000 # Adjusted for speed
    
    # We will adapt the step size. 
    # The problem is scale-invariant in x (shifting all x_i doesn't change area, only relative positions matter).
    # We can fix x_0 = 0 to reduce dimensionality? No, the evaluator expects 64 floats.
    # But we can work in relative coordinates or just shift the best solution at the end.
    # Actually, let's just optimize the 64 variables.
    
    T = T_start
    iteration = 0
    
    while time.time() < deadline - 30.0 and iteration < max_iterations:
        # Generate a new candidate
        # Use a mix of small local moves and larger global moves
        if np.random.random() < 0.1:
            # Global jump: randomize a subset of points
            mask = np.random.random(n) < 0.2
            step_size = 1.0
        else:
            # Local move
            mask = np.random.random(n) < 0.1 # Perturb 10% of points
            step_size = 0.1 * T / T_start + 0.01
        
        x_new = current_x.copy()
        x_new[mask] += np.random.randn(np.sum(mask)) * step_size
        
        area_new = _union_area(x_new, n, M)
        
        # Acceptance criterion
        if area_new < current_area:
            current_x = x_new
            current_area = area_new
            if current_area < best_area:
                best_area = current_area
                best_x = current_x.copy()
        else:
            # Metropolis criterion
            delta = area_new - current_area
            if delta < 0:
                current_x = x_new
                current_area = area_new
            else:
                # Probability of accepting worse solution
                # Scale delta: area is around 0.5-1.0. Delta is small.
                # Use a scaled delta to make T meaningful
                scale = 0.01
                prob = math.exp(-delta / (T * scale))
                if np.random.random() < prob:
                    current_x = x_new
                    current_area = area_new
        
        T *= alpha
        iteration += 1
        
        # Optional: Periodic restart from best if stuck
        if iteration % 500 == 0 and current_area > best_area * 1.05:
             current_x = best_x.copy()
             current_area = best_area

    # Final validation
    if best_x is None or not np.all(np.isfinite(best_x)):
        return _keich_init(n)
    
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
