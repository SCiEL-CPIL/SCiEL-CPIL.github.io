import numpy as np
from scipy.optimize import minimize
import time

def construct():
    n = 8
    # Regular octagon as fallback
    def regular_octagon():
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])
    
    def score(points):
        # points: (n, 2)
        # Scale to diameter 2
        diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        D = np.max(dists)
        if D == 0:
            return 0.0
        scaled_dists = dists * (2.0 / D)
        # Product of distances for i < j
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        prod = np.prod(scaled_dists[mask])
        return prod / (n ** n)

    start_time = time.time()
    timeout = 140  # seconds, keep 10s margin
    
    best_score = score(regular_octagon())
    best_points = regular_octagon()
    
    # Parametric optimization:
    # Fix p0 = (-1, 0), p1 = (1, 0) as the diameter pair.
    # Optimize p2..p7.
    # Constraints: d(pi, pj) <= 2 for all i,j.
    
    def objective(x):
        # x: (12,) representing p2..p7
        pts = np.zeros((n, 2))
        pts[0] = [-1.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = x.reshape(n-2, 2)
        
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        
        # Check validity (distinct points)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        if np.min(dists[mask]) < 1e-6:
            return 1e10
            
        # Log sum of distances
        log_sum = np.sum(np.log(dists[mask]))
        return -log_sum

    # Constraints for SLSQP: g(x) >= 0
    # We want d(pi, pj) <= 2, so 2 - d(pi, pj) >= 0
    def make_constraint(i, j):
        def constraint(x):
            pts = np.zeros((n, 2))
            pts[0] = [-1.0, 0.0]
            pts[1] = [1.0, 0.0]
            pts[2:] = x.reshape(n-2, 2)
            d = np.linalg.norm(pts[i] - pts[j])
            return 2.0 - d
        return constraint

    constraints = []
    # Pairs among 2..7
    for i in range(2, n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
    # Pairs between 0 and 2..7
    for i in range(2, n):
        constraints.append({'type': 'ineq', 'fun': make_constraint(0, i)})
    # Pairs between 1 and 2..7
    for i in range(2, n):
        constraints.append({'type': 'ineq', 'fun': make_constraint(1, i)})
    
    # Bounds for x (coordinates of p2..p7)
    # They must be within distance 2 of p0 and p1.
    # x in [-3, 3], y in [-3, 3] is a safe box.
    bounds = [(-3.0, 3.0)] * (2 * (n-2))
    
    # Initial guesses
    # 1. Regular octagon perturbed to match p0, p1
    # Regular octagon vertices: (cos(k*pi/4), sin(k*pi/4))
    # Diameter is 2. Vertices at 0 and pi are (-1,0) and (1,0).
    # So p0 corresponds to angle pi, p1 to angle 0.
    # Other points are at angles pi/4, pi/2, 3pi/4, 5pi/4, 3pi/2, 7pi/4.
    # Let's map the regular octagon points to our parameterization.
    # p0 = (-1,0), p1 = (1,0).
    # The other 6 points in regular octagon are:
    # (sqrt(2)/2, sqrt(2)/2), (0, 1), (-sqrt(2)/2, sqrt(2)/2)
    # (-sqrt(2)/2, -sqrt(2)/2), (0, -1), (sqrt(2)/2, -sqrt(2)/2)
    # We need to order them in x.
    # x0 = -1, x1 = 1.
    # Remaining x coords: -0.707, 0, 0.707, -0.707, 0, 0.707.
    # Let's just use the regular octagon points, remove (-1,0) and (1,0), and flatten.
    oct_pts = regular_octagon()
    # Identify which indices are close to (-1,0) and (1,0)
    # In np.linspace(0, 2pi, 8, endpoint=False):
    # 0: (1,0)
    # 1: (0.707, 0.707)
    # 2: (0, 1)
    # 3: (-0.707, 0.707)
    # 4: (-1, 0)
    # 5: (-0.707, -0.707)
    # 6: (0, -1)
    # 7: (0.707, -0.707)
    # So p0 should be index 4, p1 index 0.
    # We want to fix p0=(-1,0) and p1=(1,0).
    # The remaining points are indices 1,2,3,5,6,7.
    init_x = np.array([
        oct_pts[1][0], oct_pts[1][1],
        oct_pts[2][0], oct_pts[2][1],
        oct_pts[3][0], oct_pts[3][1],
        oct_pts[5][0], oct_pts[5][1],
        oct_pts[6][0], oct_pts[6][1],
        oct_pts[7][0], oct_pts[7][1]
    ])
    
    # Run SLSQP
    try:
        res = minimize(
            objective, 
            init_x, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        
        if time.time() - start_time < timeout:
            if res.success:
                pts = np.zeros((n, 2))
                pts[0] = [-1.0, 0.0]
                pts[1] = [1.0, 0.0]
                pts[2:] = res.x.reshape(n-2, 2)
                
                # Verify diameter is actually 2 (or close)
                diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dists = np.linalg.norm(diffs, axis=2)
                D = np.max(dists)
                
                if D > 2.0 + 1e-5:
                    # Constraint violation, skip or scale? 
                    # SLSQP should handle it. If not, it's invalid.
                    pass
                else:
                    # Check distinctness
                    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                    if np.min(dists[mask]) > 1e-5:
                        cand_score = score(pts)
                        if cand_score > best_score:
                            best_score = cand_score
                            best_points = pts
                            
    except Exception as e:
        pass

    # Try another initialization: Random points within the lens
    if time.time() - start_time < timeout:
        np.random.seed(42)
        for _ in range(10):
            # Generate random points in the lens intersection
            # Simple rejection sampling in [-2,2]x[-2,2]
            # But we need to ensure they are within dist 2 of p0 and p1.
            # p0=(-1,0), p1=(1,0).
            # Lens is intersection of disk((-1,0),2) and disk((1,0),2).
            # x in [-1, 1] is not required, x can be up to 1.
            # Actually, max x is 1 (from p1). Min x is -1 (from p0).
            # Wait, d(p, p0)<=2 and d(p, p1)<=2.
            # If p=(x,y), (x+1)^2+y^2 <= 4 and (x-1)^2+y^2 <= 4.
            # x^2+2x+1+y^2 <= 4 => x^2+y^2+2x <= 3
            # x^2-2x+1+y^2 <= 4 => x^2+y^2-2x <= 3
            # Adding: 2x^2+2y^2 <= 6 => x^2+y^2 <= 3.
            # So points are within radius sqrt(3) ~ 1.732 from origin.
            
            valid = True
            pts = np.zeros((n, 2))
            pts[0] = [-1.0, 0.0]
            pts[1] = [1.0, 0.0]
            for i in range(2, n):
                while True:
                    p = np.random.uniform(-2, 2, 2)
                    if np.linalg.norm(p - pts[0]) <= 2.0 and np.linalg.norm(p - pts[1]) <= 2.0:
                        pts[i] = p
                        break
                # Also check against previous points 2..i-1
                for j in range(2, i):
                    if np.linalg.norm(pts[i] - pts[j]) < 1e-3:
                        # Too close, re-sample? Just accept for now, SLSQP will move them
                        pass
            
            x0 = pts[2:].flatten()
            
            try:
                res = minimize(
                    objective, 
                    x0, 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-10}
                )
                if time.time() - start_time < timeout:
                    if res.success:
                        pts[2:] = res.x.reshape(n-2, 2)
                        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                        dists = np.linalg.norm(diffs, axis=2)
                        D = np.max(dists)
                        if D <= 2.0 + 1e-5:
                            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                            if np.min(dists[mask]) > 1e-5:
                                cand_score = score(pts)
                                if cand_score > best_score:
                                    best_score = cand_score
                                    best_points = pts
            except Exception:
                pass

    return best_points.tolist()
