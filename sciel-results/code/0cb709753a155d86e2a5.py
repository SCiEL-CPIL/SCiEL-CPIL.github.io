import numpy as np
from scipy.optimize import differential_evolution

N = 21

def construct():
    # Variables: [w, x1, y1, r1, x2, y2, r2, ...]
    # w is the width, h = 2 - w
    # Bounds for w: (0.1, 1.9) to ensure h > 0.1
    # Bounds for x, y: (0, 2)
    # Bounds for r: (0.01, 0.5)
    
    bounds = [(0.1, 1.9)]
    for i in range(N):
        bounds.append((0.0, 2.0)) # x
        bounds.append((0.0, 2.0)) # y
        bounds.append((0.01, 0.5)) # r
        
    def objective(x):
        w = x[0]
        h = 2.0 - w
        
        if w <= 0 or h <= 0:
            return 1e6
            
        radii_sum = 0
        penalty = 0
        
        # Extract circles
        circles = []
        for i in range(N):
            xi = x[1 + 3*i]
            yi = x[2 + 3*i]
            ri = x[3 + 3*i]
            circles.append((xi, yi, ri))
            radii_sum += ri
            
        # Penalty for being outside bounds
        for xi, yi, ri in circles:
            if xi < ri:
                penalty += (ri - xi) ** 2
            if xi > w - ri:
                penalty += (xi - (w - ri)) ** 2
            if yi < ri:
                penalty += (ri - yi) ** 2
            if yi > h - ri:
                penalty += (yi - (h - ri)) ** 2
                
        # Penalty for overlaps
        for i in range(N):
            xi, yi, ri = circles[i]
            for j in range(i+1, N):
                xj, yj, rj = circles[j]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                sum_r = ri + rj
                if dist_sq < sum_r**2:
                    # Penalty proportional to the square of the overlap amount
                    penalty += (sum_r**2 - dist_sq) ** 2
                    
        # We want to maximize sum(radii), so minimize -sum(radii) + penalty
        return -radii_sum + penalty * 100.0

    # Run Differential Evolution
    # Reduce maxiter to ensure it fits in time limit
    # population_size default is 15*L, L=64, so ~960.
    # 100 iterations should be reasonable.
    try:
        res = differential_evolution(objective, bounds, maxiter=100, tol=1e-4, seed=42, polish=True)
        
        if res.fun < 1e5: # Check if it found a reasonable solution
            # Extract result
            w = res.x[0]
            h = 2.0 - w
            circles = []
            for i in range(N):
                x = res.x[1 + 3*i]
                y = res.x[2 + 3*i]
                r = res.x[3 + 3*i]
                circles.append([x, y, r])
            
            # Validate and clean up
            final_circles = np.array(circles)
            
            # Check validity
            valid = True
            for i in range(N):
                xi, yi, ri = final_circles[i]
                if xi < ri or xi > w - ri or yi < ri or yi > h - ri:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    xi, yi, ri = final_circles[i]
                    for j in range(i+1, N):
                        xj, yj, rj = final_circles[j]
                        dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                        if dist < ri + rj:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                return final_circles
            else:
                # If invalid, try to fix by shrinking radii
                scale = 0.95
                for i in range(N):
                    final_circles[i][2] *= scale
                return final_circles
        else:
            # Fallback
            pass
    except Exception:
        pass
        
    # Fallback: Grid layout
    # 5x5 grid in a 1x1 square (perimeter 4)
    side = 1.0
    c = 5
    r = 5
    cw = side / c
    ch = side / r
    circles = []
    for i in range(c):
        for j in range(r):
            if len(circles) >= N:
                break
            x = (i + 0.5) * cw
            y = (j + 0.5) * ch
            rad = 0.4 * min(cw, ch)
            circles.append([x, y, rad])
    return np.array(circles)
