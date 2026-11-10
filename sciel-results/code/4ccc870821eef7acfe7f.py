import numpy as np
import time

def _compute_area(x):
    """
    Compute the area of the union of the triangles for given positions x.
    x: np.ndarray of shape (64,)
    """
    n = 64
    num_heights = 20000
    
    # Precomputed constants can be passed or created. 
    # For speed in a loop, it's better to pass them or make them global.
    # But for self-contained function, we create them. 
    # Actually, the function is called many times. 
    # Let's make a closure or pass constants.
    # The prompt says "Keep the function names and signatures of the seed program".
    # The seed program only has kakeya_needle_2d_positions64.
    # I can define helper functions.
    
    a = (np.arange(n) + 1) / n
    y = (np.arange(num_heights) + 0.5) / num_heights
    w = (1.0 - y) / n
    
    # L[i, m] = x[i] + a[i] * y[m]
    # x is (64,), a is (64,), y is (20000,)
    # L = x[:, None] + a[:, None] * y[None, :]
    L = x[:, None] + a[:, None] * y[None, :]
    
    # Sort L along axis 0 (triangle index) for each height m
    # L shape: (64, 20000)
    L_sorted = np.sort(L, axis=0)
    
    # Differences between consecutive sorted L's
    # d shape: (63, 20000)
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    
    # Union length contribution from gaps
    # min(d, w) where w is broadcast to (63, 20000)
    # w shape: (20000,) -> (1, 20000)
    gap_contrib = np.sum(np.minimum(d, w[None, :]), axis=0)
    
    # Total union length at each height
    # U_m = w_m + gap_contrib_m
    U = w + gap_contrib
    
    # Area is mean of U
    return np.mean(U)

def _keich_initialization(n=64):
    """Generate initial positions using Keich construction."""
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    start_time = time.time()
    time_limit = 590.0 # Safety margin
    
    x = _keich_initialization(n)
    best_x = x.copy()
    best_area = _compute_area(best_x)
    
    # Local search
    # Random coordinate ascent
    current_step = 0.01
    no_improve_count = 0
    max_no_improve = 1000
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < time_limit:
        # Pick a random index
        i = rng.integers(0, n)
        
        # Try a few perturbations
        improved = False
        for _ in range(10):
            delta = rng.uniform(-current_step, current_step)
            new_x = best_x.copy()
            new_x[i] += delta
            
            new_area = _compute_area(new_x)
            
            if new_area < best_area:
                best_x = new_x
                best_area = new_area
                improved = True
                no_improve_count = 0
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Reduce step size or restart
                current_step *= 0.9
                no_improve_count = 0
                if current_step < 1e-6:
                    break

    return best_x
