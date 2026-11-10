# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar

def _area_fast(x, n=64, M=20000):
    """Compute union area via quadrature at M heights."""
    a = (np.arange(n, dtype=np.float64) + 1.0) / n
    y = (np.arange(M, dtype=np.float64) + 0.5) / M
    w = (1.0 - y) / n
    # L shape: (M, n)
    L = x[None, :] + a[None, :] * y[:, None]
    Ls = np.sort(L, axis=1)
    gaps = Ls[:, 1:] - Ls[:, :-1]
    union_len = w + np.sum(np.minimum(gaps, w[:, None]), axis=1)
    return float(np.mean(union_len))

def _aligned_positions(n=64):
    """
    Initialize such that all triangle apices converge to x=0.5 at y=1.
    Vertex at y=1 is (x_i + a_i, 1).
    We want x_i + a_i = 0.5 => x_i = 0.5 - a_i.
    """
    a = (np.arange(n, dtype=np.float64) + 1.0) / n
    return 0.5 - a

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    deadline = time.time() + 580.0  # Leave 20s safety margin
    safety = 20.0
    
    # 1. Initialization
    # Start with the aligned construction, which is a strong baseline
    best_x = _aligned_positions(n)
    current_area = _area_fast(best_x, n, 20000)
    
    # 2. Coordinate Descent with Line Search
    # We optimize each coordinate x_i by finding the best shift delta.
    # Moving x_i by delta changes L_i(y) = x_i + a_i*y + delta.
    # The other L_j are fixed. We can compute the area as a function of delta efficiently.
    
    # Precompute constants for speed
    a = (np.arange(n, dtype=np.float64) + 1.0) / n
    # Use a moderate M for optimization to balance accuracy and speed.
    # M=2000 is ~10x faster than M=20000. The landscape is smooth enough that this should suffice for finding the basin.
    M_opt = 2000
    y = (np.arange(M_opt, dtype=np.float64) + 0.5) / M_opt
    w = (1.0 - y) / n
    
    # Precompute L without x for speed? No, x changes.
    # But we can precompute y and w.
    
    # To speed up the inner loop, we can vectorize the area calculation for a specific delta.
    # L_i(delta) = x_i + a_i*y + delta
    # L_j = x_j + a_j*y for j != i
    
    # Let's define a helper to compute area given x and a specific coordinate i and delta.
    def _area_with_delta(x, i, delta, M=M_opt):
        # L is (M, n)
        # We can construct L efficiently.
        # L[:, i] = x[i] + a[i]*y + delta
        # L[:, j] = x[j] + a[j]*y for j != i
        
        # Instead of building full L every time, we can sort the other n-1 values,
        # then insert the i-th value. But insertion sort is O(n) per row, so O(M*n) total,
        # which is similar to np.sort O(M*n log n).
        # Actually, np.sort is highly optimized in C. Let's just use np.sort.
        
        L = x[None, :] + a[None, :] * y[:, None]
        L[:, i] += delta
        
        Ls = np.sort(L, axis=1)
        gaps = Ls[:, 1:] - Ls[:, :-1]
        union_len = w + np.sum(np.minimum(gaps, w[:, None]), axis=1)
        return float(np.mean(union_len))
    
    # Coordinate Descent Loop
    # We iterate through coordinates multiple times until time runs out or no improvement.
    # Each coordinate optimization uses minimize_scalar (Brent's method) to find the best delta.
    # We restrict delta to a reasonable range, e.g., [-1, 1] or based on current position.
    # Since x can be anywhere, but the geometry is scale-invariant in translation?
    # No, the triangles are fixed relative to each other. Shifting all x by C shifts all triangles, area unchanged.
    # So we can fix one coordinate or just let it drift.
    # However, the relative positions matter.
    # A reasonable search range for delta is maybe [-0.5, 0.5] or so.
    
    # To be safe, we can use a wider range initially and narrow down.
    # Or just use a fixed range like [-1.0, 1.0].
    
    no_improvement_count = 0
    max_no_improvement = 5
    
    while time.time() < deadline - safety:
        improved = False
        
        for i in range(n):
            if time.time() > deadline - safety:
                break
                
            # Define objective for this coordinate
            def obj(delta):
                return _area_with_delta(best_x, i, delta)
            
            # Search range. 
            # If we move x_i too far, it might just go to infinity (area doesn't change if it's far away? 
            # No, if it's far away, it adds its own width w to the union.
            # So the area is bounded below by sum of widths if disjoint, or 1 if fully overlapping.
            # The minimum area is achieved by overlapping as much as possible.
            # So x_i should stay close to the cluster.
            # Let's try a range of [-0.5, 0.5] relative to current x_i? 
            # Actually, x_i is absolute.
            # Let's just search in a window around the current best value.
            # But minimize_scalar needs a bracket or bounds.
            # Let's use bounds=(-1.0, 1.0). If the optimum is outside, we might miss it.
            # But given the geometry, the optimal x_i is likely near the other x_j.
            # Since we start with aligned positions, x_i are in [0, 0.5].
            # So [-1, 1] should cover it.
            
            try:
                res = minimize_scalar(obj, bounds=(-1.0, 1.0), method='bounded', options={'xatol': 1e-6})
                if res.success:
                    new_x_i = best_x[i] + res.x
                    new_area = res.fun
                    
                    # We should re-evaluate the full area with the updated x to be sure
                    # because the objective inside might be slightly different due to M_opt vs M_full?
                    # No, we used M_opt for both.
                    # But we should update best_x and check against current_area (which was computed with M=20000? No, current_area is updated with M_opt if we do so).
                    # Let's keep current_area consistent with M_opt during the loop, and do a final M=20000 check at the end.
                    
                    if new_area < current_area - 1e-9:
                        best_x[i] = new_x_i
                        current_area = new_area
                        improved = True
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
            except Exception:
                pass
        
        if not improved:
            break
            
    # 3. Final Polish with Full Resolution
    # Re-evaluate with M=20000
    final_area = _area_fast(best_x, n, 20000)
    
    # Optional: One more pass of coordinate descent with M=5000 or 10000 if time permits
    if time.time() < deadline - safety - 10:
        M_polish = 5000
        y_polish = (np.arange(M_polish, dtype=np.float64) + 0.5) / M_polish
        w_polish = (1.0 - y_polish) / n
        
        def _area_with_delta_polish(x, i, delta):
            L = x[None, :] + a[None, :] * y_polish[:, None]
            L[:, i] += delta
            Ls = np.sort(L, axis=1)
            gaps = Ls[:, 1:] - Ls[:, :-1]
            union_len = w_polish + np.sum(np.minimum(gaps, w_polish[:, None]), axis=1)
            return float(np.mean(union_len))
        
        current_area_polish = final_area
        for _ in range(2): # Two passes
            for i in range(n):
                if time.time() > deadline - safety:
                    break
                def obj_p(delta):
                    return _area_with_delta_polish(best_x, i, delta)
                try:
                    res = minimize_scalar(obj_p, bounds=(-0.5, 0.5), method='bounded', options={'xatol': 1e-7})
                    if res.success:
                        new_x_i = best_x[i] + res.x
                        new_area = res.fun
                        if new_area < current_area_polish - 1e-9:
                            best_x[i] = new_x_i
                            current_area_polish = new_area
                except Exception:
                    pass
            
    # Final verification
    # The function returns the array. The evaluator computes the score.
    # We just return the best x found.
    
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
