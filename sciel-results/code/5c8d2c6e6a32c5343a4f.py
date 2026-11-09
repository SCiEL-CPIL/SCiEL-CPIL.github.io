import time
import numpy as np

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for safety
    
    # State
    points = []  # List of (x, fx)
    best_x = float('nan')
    best_f_abs = float('inf')
    
    def is_deadline_reached():
        return time.time() > deadline

    def add_point(x, fx):
        nonlocal best_x, best_f_abs
        if not np.isfinite(x) or not np.isfinite(fx):
            return
        # Avoid duplicates
        for px, pfx in points:
            if abs(x - px) < 1e-14:
                return
        points.append((x, fx))
        if abs(fx) < best_f_abs:
            best_f_abs = abs(fx)
            best_x = x
            
    # 1. Initial Point
    x0 = float(x0)
    fx0 = f(x0)
    add_point(x0, fx0)
    if abs(fx0) <= 1e-8:
        return x0
    if is_deadline_reached():
        return x0 if np.isfinite(x0) else 0.0

    # 2. Get a second point
    # Try a small step
    step = 1e-4 * max(1.0, abs(x0))
    x1 = x0 + step
    fx1 = f(x1)
    add_point(x1, fx1)
    if abs(fx1) <= 1e-8:
        return x1
    if is_deadline_reached():
        return best_x

    # 3. Get a third point using Secant
    if fx0 == fx1:
        # Degenerate case, try a different step
        x1 = x0 - step
        fx1 = f(x1)
        add_point(x1, fx1)
        if abs(fx1) <= 1e-8:
            return x1
        if fx0 == fx1:
            # Still degenerate, just return best
            return best_x

    denom = fx1 - fx0
    if abs(denom) < 1e-15:
        x2 = x0 + 10*step
    else:
        x2 = x0 - fx0 * (x1 - x0) / (fx1 - fx0)
    
    fx2 = f(x2)
    add_point(x2, fx2)
    if abs(fx2) <= 1e-8:
        return x2
    if is_deadline_reached():
        return best_x

    # Now we have 3 points. We can fit a quadratic.
    # But let's try to get a 4th point to fit a cubic for better accuracy.
    # Or just start iterating with quadratic/cubic interpolation.
    
    # Strategy:
    # While we have < 4 points, use quadratic interpolation (3 pts) or secant.
    # Once we have 4 points, use cubic interpolation.
    # We will maintain a window of the best 4 points.
    
    max_calls = 100 # Safety limit on calls, though time is the real limit
    
    calls = 4 # We have made 4 calls so far (x0, x1, x2... wait, x2 was the 3rd call? 
              # x0 (1), x1 (2), x2 (3). Yes, 3 calls.
    
    # Let's re-verify call count.
    # 1. f(x0)
    # 2. f(x1)
    # 3. f(x2)
    # So calls = 3.
    calls = 3

    while calls < 60: # Arbitrary high limit, time will cut it off
        if is_deadline_reached():
            break
            
        # Select points for interpolation
        # We want the points that define the best local approximation.
        # Simple heuristic: use the 4 most recent or the 4 with smallest |fx|?
        # Better: use the points that form the smallest bracket or are closest to the current best guess.
        
        # Let's sort points by x to help with bracketing logic if needed, 
        # but for interpolation we just need distinct x's.
        
        if len(points) < 4:
            # Use secant with the two points with smallest |fx|
            # Or just the last two
            pts = points[-2:]
            x_a, fx_a = pts[0]
            x_b, fx_b = pts[1]
            
            if fx_a == fx_b:
                x_new = x_a + 1e-4
            else:
                x_new = x_a - fx_a * (x_b - x_a) / (fx_b - fx_a)
                
            if abs(x_new - x_a) < 1e-14:
                x_new = x_a + 1e-4
            if abs(x_new - x_b) < 1e-14:
                x_new = x_b + 1e-4

        else:
            # Use 4 points. Which 4?
            # Let's pick the 4 points with the smallest |fx| to focus on the root region.
            # However, this might lose bracketing information.
            # Alternative: Use the 4 most recent points.
            # Let's try: 4 most recent points.
            pts = points[-4:]
            
            xs = np.array([p[0] for p in pts])
            fs = np.array([p[1] for p in pts])
            
            # Fit cubic polynomial
            # np.polyfit expects x, y. Returns coefficients highest to lowest.
            try:
                coeffs = np.polyfit(xs, fs, 3)
                # Find roots
                roots = np.roots(coeffs)
                real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
                
                if not real_roots:
                    # Fallback to secant on best two
                    pts_sorted = sorted(pts, key=lambda p: abs(p[1]))
                    x_a, fx_a = pts_sorted[0]
                    x_b, fx_b = pts_sorted[1]
                    if fx_a == fx_b:
                        x_new = x_a + 1e-4
                    else:
                        x_new = x_a - fx_a * (x_b - x_a) / (fx_b - fx_a)
                else:
                    # Choose the real root that is "best"
                    # Heuristic: Closest to the average of the x's used, or closest to current best_x
                    # Let's pick the one that minimizes the distance to the current best_x
                    # Or the one that lies within the range of the x's used (interpolation vs extrapolation)
                    
                    # Prefer roots within the convex hull of the x points used
                    x_min = min(xs)
                    x_max = max(xs)
                    
                    candidates = []
                    for r in real_roots:
                        # Penalize extrapolation
                        if x_min <= r <= x_max:
                            candidates.append((0, r))
                        else:
                            candidates.append((1, r))
                    
                    if candidates:
                        # Sort by penalty, then by distance to best_x
                        candidates.sort(key=lambda item: (item[0], abs(item[1] - best_x)))
                        x_new = candidates[0][1]
                    else:
                        x_new = real_roots[0]
                        
            except Exception:
                # Fallback to secant
                pts_sorted = sorted(pts, key=lambda p: abs(p[1]))
                x_a, fx_a = pts_sorted[0]
                x_b, fx_b = pts_sorted[1]
                if fx_a == fx_b:
                    x_new = x_a + 1e-4
                else:
                    x_new = x_a - fx_a * (x_b - x_a) / (fx_b - fx_a)

        # Ensure x_new is distinct from existing points
        if any(abs(x_new - px) < 1e-12 for px, _ in points):
            x_new = x_new + 1e-12

        fx_new = f(x_new)
        calls += 1
        add_point(x_new, fx_new)
        
        if abs(fx_new) <= 1e-8:
            return x_new

        # Prune points to keep memory and fitting stable
        # Keep the 6 best points? Or just 4?
        # If we keep too many, polyfit might become ill-conditioned.
        # Let's keep the 4 points with the smallest |fx| + the 2 most recent?
        # Actually, for cubic interpolation, we only need 4.
        # Let's keep the 4 points that form the smallest bracket or are closest to the root.
        # Simple strategy: Keep the 4 points with the smallest |fx|.
        # This focuses the interpolation on the root area.
        if len(points) > 6:
            # Sort by |fx|
            points.sort(key=lambda p: abs(p[1]))
            # Keep the best 4, but ensure we don't lose all x-diversity?
            # Maybe keep the best 3 and the most recent 1?
            # Let's just keep the best 4.
            points = points[:4]
            
    return best_x if np.isfinite(best_x) else x0
