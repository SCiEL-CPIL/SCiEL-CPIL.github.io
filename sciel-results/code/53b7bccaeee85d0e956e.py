import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin under 150s wall-clock
    
    # 1. Validation & Relaxation Utilities
    
    def validate(circles, w, h, tol=1e-9):
        """Check if circles are valid: inside rectangle and non-overlapping."""
        if circles.shape != (N, 3):
            return False
        if np.any(circles[:, 2] <= 0):
            return False
        # Boundary check
        if np.any(circles[:, 0] - circles[:, 2] < -tol) or \
           np.any(circles[:, 0] + circles[:, 2] > w + tol) or \
           np.any(circles[:, 1] - circles[:, 2] < -tol) or \
           np.any(circles[:, 1] + circles[:, 2] > h + tol):
            return False
        # Pairwise distance check
        # Vectorized pairwise distance is faster for N=21
        for i in range(N):
            for j in range(i + 1, N):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < circles[i, 2] + circles[j, 2] - tol:
                    return False
        return True

    def relax(circles, w, h, iters=500):
        """
        Relaxation algorithm to resolve overlaps and boundary violations.
        Returns a valid configuration or the input if it fails.
        """
        circles = circles.copy()
        pos = circles[:, :2]
        rad = circles[:, 2]
        
        for _ in range(iters):
            moved = False
            # Boundary clamp
            for i in range(N):
                if pos[i, 0] < rad[i]:
                    pos[i, 0] = rad[i]; moved = True
                elif pos[i, 0] > w - rad[i]:
                    pos[i, 0] = w - rad[i]; moved = True
                if pos[i, 1] < rad[i]:
                    pos[i, 1] = rad[i]; moved = True
                elif pos[i, 1] > h - rad[i]:
                    pos[i, 1] = h - rad[i]; moved = True
            
            # Pairwise push
            for i in range(N):
                for j in range(i + 1, N):
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    dist_sq = dx * dx + dy * dy
                    min_dist = rad[i] + rad[j]
                    
                    if dist_sq < min_dist * min_dist:
                        dist = np.sqrt(dist_sq)
                        if dist < 1e-9:
                            # Random push if coincident
                            angle = np.random.uniform(0, 2 * np.pi)
                            push = (min_dist - dist) / 2.0
                            pos[i, 0] -= np.cos(angle) * push
                            pos[i, 1] -= np.sin(angle) * push
                            pos[j, 0] += np.cos(angle) * push
                            pos[j, 1] += np.sin(angle) * push
                        else:
                            push = (min_dist - dist) / 2.0
                            ux = dx / dist
                            uy = dy / dist
                            pos[i, 0] -= ux * push
                            pos[i, 1] -= uy * push
                            pos[j, 0] += ux * push
                            pos[j, 1] += uy * push
                        moved = True
            
            if not moved:
                break

        circles[:, :2] = pos
        circles[:, 2] = rad
        
        # Final boundary clamp
        circles[:, 0] = np.clip(circles[:, 0], circles[:, 2], w - circles[:, 2])
        circles[:, 1] = np.clip(circles[:, 1], circles[:, 2], h - circles[:, 2])
        
        return circles

    # 2. Initialization Strategies

    def make_hex_grid(w, h):
        """Create a hexagonal grid packing that fits in w x h."""
        # Estimate radius
        # For hex packing, density is ~0.9069
        # Area per circle ~ 2.309 * r^2
        # N * 2.309 * r^2 <= w * h
        r_est = np.sqrt(w * h / (N * 2.309))
        
        circles = []
        y = r_est
        row = 0
        # Adjust r to fit vertically
        # We want to fit as many rows as needed, but we only have N circles.
        # Let's just generate a dense hex grid and take the first N that fit.
        # Or better, place them systematically.
        
        # Try to fit rows of 5 or 4
        # Let's just generate a hex lattice and filter
        r = r_est * 0.95 # Safety margin
        dy = np.sqrt(3) * r
        x_off = 0
        y = r
        while y + r <= h:
            x = r + x_off
            while x + r <= w:
                circles.append([x, y, r])
                x += 2 * r
            x_off = r if x_off == 0 else 0
            y += dy
            if len(circles) >= N:
                break
            if y + r > h:
                break
                
        if len(circles) < N:
            return None
            
        return np.array(circles[:N])

    def make_grid(w, h, cols):
        """Create a grid packing."""
        rows = int(np.ceil(N / cols))
        circles = []
        cw = w / cols
        ch = h / rows
        
        count = 0
        for r_idx in range(rows):
            for c_idx in range(cols):
                if count >= N:
                    break
                cx = (c_idx + 0.5) * cw
                cy = (r_idx + 0.5) * ch
                # Radius limited by cell size and boundary
                rad = 0.5 * min(cw, ch) * 0.95
                circles.append([cx, cy, rad])
                count += 1
        return np.array(circles)

    def random_start(w, h):
        """Random start with small radii."""
        r0 = 0.05
        circles = np.zeros((N, 3))
        circles[:, 0] = np.random.uniform(r0, w - r0, N)
        circles[:, 1] = np.random.uniform(r0, h - r0, N)
        circles[:, 2] = r0
        return circles

    # 3. Optimization

    def optimize_slsqp(circles, w, h, deadline):
        """
        Optimize radii and positions using SLSQP.
        """
        if time.time() > deadline - 10.0:
            return None

        x0 = np.array([v for c in circles for v in c])

        def objective(x):
            if time.time() > deadline:
                return 0.0
            r = x[2::3]
            return -np.sum(r)

        constraints = []

        # Boundary constraints
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*3] - x[i*3+2]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*3+1] - x[i*3+2]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2] - 1e-6})

        # Overlap constraints
        for i in range(N):
            for j in range(i + 1, N):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: np.sqrt((x[i*3]-x[j*3])**2 + (x[i*3+1]-x[j*3+1])**2) - x[i*3+2] - x[j*3+2]
                })

        bounds = []
        for i in range(N):
            bounds.append((1e-6, w))
            bounds.append((1e-6, h))
            bounds.append((1e-6, min(w, h)/2))

        try:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-9}
            )

            if res.success:
                new_circles = np.array([(res.x[i*3], res.x[i*3+1], res.x[i*3+2]) for i in range(N)])
                # Relax to ensure strict validity
                new_circles = relax(new_circles, w, h, iters=100)
                if validate(new_circles, w, h):
                    return new_circles
        except Exception:
            pass

        return None

    def optimize_growth(circles, w, h, deadline):
        """
        Growth-based optimization: grow radii, relax overlaps.
        """
        circles = circles.copy()
        # Ensure initial validity
        circles = relax(circles, w, h, iters=100)
        
        for _ in range(200):
            if time.time() > deadline - 5.0:
                break
            
            # Find min clearance
            min_clearance = np.inf
            # Boundaries
            for i in range(N):
                min_clearance = min(min_clearance, circles[i, 0], w - circles[i, 0])
                min_clearance = min(min_clearance, circles[i, 1], h - circles[i, 1])
            
            # Pairs
            for i in range(N):
                for j in range(i + 1, N):
                    dist = np.sqrt((circles[i, 0] - circles[j, 0])**2 + (circles[i, 1] - circles[j, 1])**2)
                    clearance = dist - (circles[i, 2] + circles[j, 2])
                    min_clearance = min(min_clearance, clearance)
            
            if min_clearance <= 0:
                # Relax more
                circles = relax(circles, w, h, iters=200)
                continue
                
            # Grow
            growth = min(0.002, min_clearance * 0.5)
            circles[:, 2] += growth
            
            # Relax
            circles = relax(circles, w, h, iters=100)

        return circles

    # 4. Main Search Loop

    best_result = None
    best_score = -np.inf
    best_w = 1.0
    best_h = 1.0

    # Aspect ratios to try
    # Perimeter 4 => 2(w+h)=4 => w+h=2
    # w = ar * h => h(ar+1) = 2 => h = 2/(ar+1), w = 2ar/(ar+1)
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 2.2, 0.45]
    
    # Sort by likelihood? Square is usually good, but elongated might be better for 21.
    # Let's prioritize a mix.
    
    for ar in aspect_ratios:
        if time.time() > deadline - 20.0:
            break
            
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        # Also try swapped orientation if ar != 1
        orientations = [(w, h)]
        if abs(w - h) > 1e-6:
            orientations.append((h, w))
            
        for ww, hh in orientations:
            if time.time() > deadline - 10.0:
                break
                
            # Strategy 1: Hex Grid Init + Growth
            init_hex = make_hex_grid(ww, hh)
            if init_hex is not None:
                init_hex = relax(init_hex, ww, hh, iters=200)
                if validate(init_hex, ww, hh):
                    # Try Growth
                    res = optimize_growth(init_hex, ww, hh, deadline)
                    score = np.sum(res[:, 2])
                    if score > best_score:
                        best_score = score
                        best_result = res.copy()
                        best_w = ww
                        best_h = hh
                    
                    # Try SLSQP on the result of Growth (or init if growth failed)
                    if time.time() < deadline - 15.0:
                        res_slsqp = optimize_slsqp(res, ww, hh, deadline)
                        if res_slsqp is not None:
                            score = np.sum(res_slsqp[:, 2])
                            if score > best_score:
                                best_score = score
                                best_result = res_slsqp.copy()
                                best_w = ww
                                best_h = hh
                                
            # Strategy 2: Grid Init + SLSQP
            init_grid = make_grid(ww, hh, 5)
            if init_grid is not None:
                init_grid = relax(init_grid, ww, hh, iters=200)
                if validate(init_grid, ww, hh):
                    if time.time() < deadline - 15.0:
                        res_slsqp = optimize_slsqp(init_grid, ww, hh, deadline)
                        if res_slsqp is not None:
                            score = np.sum(res_slsqp[:, 2])
                            if score > best_score:
                                best_score = score
                                best_result = res_slsqp.copy()
                                best_w = ww
                                best_h = hh

    # If no result, fallback to simple grid
    if best_result is None:
        w, h = 1.0, 1.0
        init = make_grid(w, h, 5)
        best_result = relax(init, w, h, iters=100)
        best_w, best_h = w, h

    # Final Check and Cleanup
    # The evaluator expects the circles to be in a rectangle of perimeter <= 4.
    # Our w, h satisfy 2(w+h) = 4.
    # We must ensure the returned array is valid.
    
    if not validate(best_result, best_w, best_h):
        # Attempt a final relaxation
        best_result = relax(best_result, best_w, best_h, iters=500)
        if not validate(best_result, best_w, best_h):
            # Shrink radii
            best_result[:, 2] *= 0.9
            best_result = relax(best_result, best_w, best_h, iters=500)
            if not validate(best_result, best_w, best_h):
                # Fallback to seed
                side = int(np.ceil(np.sqrt(N)))
                cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                         for i in range(side) for j in range(side)]
                best_result = np.array(cells[:N])

    return best_result
