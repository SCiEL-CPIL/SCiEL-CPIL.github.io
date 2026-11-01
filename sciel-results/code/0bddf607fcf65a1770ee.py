import numpy as np
from scipy.optimize import minimize
import time

N = 14

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Initial configurations
    # Config 1: 3x5 grid minus center, corners fixed
    # x in {0, 0.5, 1}, y in {0, 0.25, 0.5, 0.75, 1}
    pts_35 = []
    for y in [0, 0.25, 0.5, 0.75, 1]:
        for x in [0, 0.5, 1]:
            pts_35.append([x, y])
    # Remove center (0.5, 0.5)
    pts_35 = [p for p in pts_35 if not (p[0] == 0.5 and p[1] == 0.5)]
    # Fixed corners: (0,0), (1,0), (1,1), (0,1)
    # Indices of corners in pts_35?
    # It's easier to just construct the array with fixed and free parts
    
    def make_initial(config_type):
        if config_type == '35':
            # 3x5 grid minus center
            # Corners: (0,0), (1,0), (1,1), (0,1)
            # Others:
            others = [
                [0.5, 0], [0, 0.25], [0.5, 0.25], [1, 0.25],
                [0, 0.5], [1, 0.5],
                [0, 0.75], [0.5, 0.75], [1, 0.75],
                [0.5, 1]
            ]
            corners = [[0,0], [1,0], [1,1], [0,1]]
            # Shuffle others slightly? No, keep grid.
            return np.array(corners + others, dtype=float)
        elif config_type == '44':
            # 4x4 grid minus 2 interior
            # x,y in {0, 1/3, 2/3, 1}
            pts = []
            for i in range(4):
                for j in range(4):
                    x = i / 3.0
                    y = j / 3.0
                    pts.append([x, y])
            # Remove (1/3, 1/3) and (2/3, 2/3)
            pts = [p for p in pts if not (p[0] == 1/3.0 and p[1] == 1/3.0) and not (p[0] == 2/3.0 and p[1] == 2/3.0)]
            # Ensure we have 14
            # Corners are fixed
            corners = [[0,0], [1,0], [1,1], [0,1]]
            # The rest
            others = [p for p in pts if p not in corners]
            # This list logic is fragile with floats. Let's just use the grid directly.
            return np.array(pts, dtype=float)
        else:
            return np.random.rand(14, 2)

    best_score = -1
    best_points = None
    
    configs = ['35', '44']
    
    for cfg in configs:
        if time.time() > deadline:
            break
            
        x0 = make_initial(cfg)
        # Jitter slightly to break symmetries if needed, but grid is fine for start
        x0 = x0 + np.random.normal(0, 0.01, x0.shape)
        
        # Optimization
        beta = 100.0
        
        def objective_and_grad(x_flat):
            P = x_flat.reshape(14, 2)
            # Calculate all triangle areas
            areas = np.zeros((364,))
            idx = 0
            for i in range(14):
                for j in range(i+1, 14):
                    for k in range(j+1, 14):
                        p1 = P[i]
                        p2 = P[j]
                        p3 = P[k]
                        area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                        areas[idx] = area
                        idx += 1
            
            # Soft min
            # max area is ~0.5, min is small.
            # Use log-sum-exp trick for stability
            max_area = np.max(areas)
            if max_area < 1e-10:
                return 0, np.zeros_like(x_flat)
            
            # We want to maximize min area.
            # Soft min = -1/beta * log(sum(exp(-beta * area)))
            # To avoid underflow, we can just use the areas directly if beta is not huge.
            # exp(-beta * area) can underflow if area is large.
            # But we are minimizing -soft_min, so we maximize soft_min.
            # Let's compute log_sum_exp(-beta * areas)
            shifted = -beta * areas
            max_shifted = np.max(shifted)
            log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
            soft_min_val = -1.0 / beta * log_sum_exp
            
            # Gradient
            # d(soft_min)/dx = -1/beta * d(log_sum_exp)/dx
            # d(log_sum_exp)/dx = sum( exp(-beta*area_i) / sum(exp(-beta*area_j)) * d(-beta*area_i)/dx )
            # = sum( w_i * (-beta * d(area_i)/dx) )
            # where w_i = exp(-beta*area_i) / sum(exp(-beta*area_j))
            
            exp_vals = np.exp(shifted - max_shifted)
            sum_exp = np.sum(exp_vals)
            weights = exp_vals / sum_exp
            
            grad = np.zeros_like(x_flat)
            
            # Compute d(area)/dx for each triangle
            # We can do this in a loop, but it's 364 * 14 * 2 operations.
            # 364 * 28 ~ 10k ops. Fast.
            
            idx = 0
            for i in range(14):
                for j in range(i+1, 14):
                    for k in range(j+1, 14):
                        p1 = P[i]
                        p2 = P[j]
                        p3 = P[k]
                        
                        # Determinant D = (x2-x1)(y3-y1) - (x3-x1)(y2-y1)
                        # Area = 0.5 * |D|
                        # Assume D > 0 for gradient (if D < 0, gradient flips sign, but we use abs in objective? 
                        # No, we used area = 0.5*abs(D).
                        # The gradient of abs(D) is sign(D) * grad(D).
                        # Let's compute D and sign.
                        D = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1])
                        sign = 1.0 if D > 0 else -1.0
                        if D == 0:
                            sign = 1.0 # Arbitrary, measure zero
                        
                        # d(D)/dx1 = -(y3-y1) = y1-y3
                        # d(D)/dy1 = x2-x1
                        # d(D)/dx2 = y3-y1
                        # d(D)/dy2 = -(x3-x1)
                        # d(D)/dx3 = -(y2-y1) = y1-y2
                        # d(D)/dy3 = x2-x1
                        
                        w = weights[idx] * (-beta) * sign * 0.5 # 0.5 from area def, -beta from soft min def?
                        # Wait.
                        # soft_min = -1/beta * log_sum_exp(-beta * areas)
                        # d(soft_min)/dx = -1/beta * [ sum( exp(-beta A_i) / S * (-beta dA_i/dx) ) ]
                        # = sum( w_i * dA_i/dx )  where w_i = exp(-beta A_i)/S
                        # So the factor is just w_i * dA_i/dx.
                        
                        w_area = weights[idx]
                        
                        grad[2*i]   += w_area * sign * 0.5 * (p1[1] - p3[1])
                        grad[2*i+1] += w_area * sign * 0.5 * (p2[0] - p1[0])
                        
                        grad[2*j]   += w_area * sign * 0.5 * (p3[1] - p1[1])
                        grad[2*j+1] += w_area * sign * 0.5 * (-(p3[0] - p1[0]))
                        
                        grad[2*k]   += w_area * sign * 0.5 * (p1[1] - p2[1])
                        grad[2*k+1] += w_area * sign * 0.5 * (p2[0] - p1[0])
                        
                        idx += 1
            
            # We want to MAXIMIZE soft_min.
            # So we minimize -soft_min.
            val = -soft_min_val
            grad = -grad
            
            return val, grad
        
        try:
            res = minimize(objective_and_grad, x0.flatten(), method='L-BFGS-B', 
                           jac=True, options={'maxiter': 50, 'gtol': 1e-5})
            
            if res.success or res.fun < 1e10:
                P_opt = res.x.reshape(14, 2)
                # Calculate actual score
                min_area = np.inf
                for i in range(14):
                    for j in range(i+1, 14):
                        for k in range(j+1, 14):
                            p1 = P_opt[i]
                            p2 = P_opt[j]
                            p3 = P_opt[k]
                            area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                            if area < min_area:
                                min_area = area
                
                # Convex hull area
                # Simple hull for 14 points
                def hull_area(points):
                    n = len(points)
                    if n < 3: return 0
                    # Graham scan
                    def cross(o, a, b):
                        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
                    
                    # Sort by x, then y
                    sorted_pts = sorted(points, key=lambda p: (p[0], p[1]))
                    
                    lower = []
                    for p in sorted_pts:
                        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                            lower.pop()
                        lower.append(p)
                        
                    upper = []
                    for p in reversed(sorted_pts):
                        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                            upper.pop()
                        upper.append(p)
                        
                    hull = lower[:-1] + upper[:-1]
                    
                    area = 0
                    for i in range(len(hull)):
                        x1, y1 = hull[i]
                        x2, y2 = hull[(i+1)%len(hull)]
                        area += x1*y2 - x2*y1
                    return 0.5 * abs(area)
                
                hull_a = hull_area(P_opt)
                if hull_a > 1e-9:
                    score = min_area / hull_a
                else:
                    score = 0
                    
                if score > best_score:
                    best_score = score
                    best_points = P_opt
                    
        except Exception as e:
            pass

    if best_points is None:
        # Fallback to initial 35
        best_points = make_initial('35')
        
    return best_points

# EVOLVE-BLOCK-END
