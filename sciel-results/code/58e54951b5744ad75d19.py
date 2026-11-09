# EVOLVE-BLOCK-START
import numpy as np
import time

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x = np.array(positions, dtype=np.float64)
    
    a = (np.arange(n) + 1) / n
    M = 20000
    y = (np.arange(M) + 0.5) / M
    w = 1.0 / n
    
    # Precompute a * y matrix? 64x20000 is 1.28M floats. ~10MB. OK.
    # L_i(y) = x_i + a_i * y
    # We can compute L matrix as x[:, None] + a[:, None] * y[None, :]
    
    start_time = time.time()
    deadline = 600.0
    
    def compute_area(x_vec):
        # x_vec is shape (n,)
        # L is shape (n, M)
        L = x_vec[:, None] + a[:, None] * y[None, :]
        # Sort L along axis 0
        # np.sort is O(M * n log n)
        L_sorted = np.sort(L, axis=0)
        # Diffs
        # d_k = L_sorted[k+1] - L_sorted[k]
        d = np.diff(L_sorted, axis=0) # shape (n-1, M)
        # union_len = w + sum(min(d, w))
        # min(d, w) can be computed as np.minimum(d, w)
        # Sum along axis 0
        union_lens = w + np.sum(np.minimum(d, w), axis=0)
        area = np.mean(union_lens)
        return area

    # Initial area
    current_area = compute_area(x)
    
    # Coordinate Descent with Ternary Search
    # We iterate until time is up or no improvement
    tol = 1e-10
    max_iter = 1000
    
    for it in range(max_iter):
        if time.time() > deadline - 30.0:
            break
            
        improved = False
        for i in range(n):
            # Ternary search for x[i]
            # Range: The optimal x[i] is likely close to current x[i].
            # However, to be safe, we can use a wider range initially, or adapt it.
            # The Kakeya set is bounded. x_i should be in [0, 1] or similar.
            # Actually, relative positions matter.
            # Let's use a window around current x[i].
            # If we hit the boundary, we expand.
            
            # A better range: The "Keich" positions are in [0, 1].
            # Let's search in [x[i] - 0.5, x[i] + 0.5] initially?
            # Or just global [0, 1] if we assume normalization?
            # The problem is translation invariant?
            # If we shift all x_i by c, the area doesn't change.
            # So we can fix x_0 = 0? No, the metric depends on absolute positions?
            # "area of the union". Translation doesn't change area.
            # So we can set x_0 = 0 and optimize x_1..x_63?
            # Or just let it float.
            
            # Let's use a local search window.
            # If the current x[i] is a local min, the global min is nearby.
            # Start with a small window, e.g., +/- 1e-3.
            # If the min is at the edge, expand.
            
            # Actually, let's just use a fixed reasonable range for ternary search.
            # The optimal x_i are bounded.
            # Let's try [0, 1] for all.
            # But if we start far, ternary search might be slow? No, it's log.
            # But if the function is flat, it converges fast.
            
            # Let's use a window that adapts.
            # For simplicity and robustness, let's use a window of size 1.0 centered at current x[i].
            # If the optimum is outside, we might miss it.
            # But coordinate descent usually stays in a good basin.
            
            # Let's try a wider initial range: [min(x)-1, max(x)+1]
            # Or just [0, 1] since Keich is in [0,1].
            
            lo = x[i] - 0.5
            hi = x[i] + 0.5
            
            # Ensure we don't go crazy.
            # We will do ternary search on [lo, hi]
            
            best_x = x[i]
            best_area = current_area
            
            # If the current point is already very good, we might not need to search far.
            # But ternary search is fast.
            
            # Perform ternary search
            # We want to find x that minimizes area.
            # Since it's convex, we can also use subgradient bisection.
            # But ternary is simpler to implement correctly without derivatives.
            
            # Let's do 40 iterations of ternary search.
            for _ in range(40):
                m1 = lo + (hi - lo) / 3.0
                m2 = hi - (hi - lo) / 3.0
                
                # Save old
                old_xi = x[i]
                
                x[i] = m1
                area1 = compute_area(x)
                
                x[i] = m2
                area2 = compute_area(x)
                
                # Restore old for next iteration logic? No, we update x[i] at the end.
                # But we need to keep x[i] consistent for the next variable.
                # We are doing this inside the loop for i.
                
                if area1 < area2:
                    hi = m2
                else:
                    lo = m1
                    
                # Check time
                if time.time() > deadline - 30.0:
                    # Restore best found so far or current
                    x[i] = best_x
                    break
                
            # After ternary search, the optimal is in [lo, hi]
            # The center is the best estimate
            x_new = (lo + hi) / 2.0
            
            # Evaluate the new point
            old_xi = x[i]
            x[i] = x_new
            area_new = compute_area(x)
            
            # Also check the neighbors of the ternary search result?
            # Ternary search converges to the minimum of a unimodal function.
            
            if area_new < best_area - 1e-12:
                best_area = area_new
                best_x = x_new
                current_area = area_new
                improved = True
            else:
                # Revert if not improved
                x[i] = old_xi
                # current_area remains best_area (which is the area with old_xi)
                # Wait, if we reverted, current_area should be the area with old_xi.
                # But we might have updated current_area in previous steps.
                # Let's just track the global best.
                
            # If we broke early due to time, x[i] might be messed up.
            # The loop above restores x[i] to best_x if it breaks.
            # But if it doesn't break, x[i] is x_new.
            # We need to ensure x[i] is the best one.
            if time.time() > deadline - 30.0:
                 x[i] = best_x
                 # Recalculate area? No, best_area is stored.
                 # But x vector might have other changes? No, we only changed x[i].
                 # So area is best_area.
                 break

        if not improved:
            break
            
    return x

# EVOLVE-BLOCK-END
