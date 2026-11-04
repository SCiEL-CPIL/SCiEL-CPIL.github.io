import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay, Voronoi

N = 26

def construct():
    
    def largest_empty_circle(centers, radii):
        """
        Finds the center and radius of the largest circle that can be added
        to the given set of circles without overlapping, within [0,1]x[0,1].
        Uses Delaunay triangulation to identify candidate voids (circumcenters).
        """
        if len(centers) == 0:
            return (0.5, 0.5, 0.5)
        
        centers = np.array(centers)
        radii = np.array(radii)
        
        best_r = 0
        best_c = None
        
        # 1. Candidate centers from Delaunay circumcenters
        # The largest empty circle in a set of points is centered at a Voronoi vertex,
        # which is the circumcenter of a Delaunay triangle.
        if len(centers) >= 3:
            try:
                tri = Delaunay(centers)
                # Iterate over triangles
                for simplex in tri.simplices:
                    p1, p2, p3 = centers[simplex]
                    
                    # Calculate circumcenter
                    ax, ay = p1
                    bx, by = p2
                    cx, cy = p3
                    
                    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
                    if abs(d) < 1e-10:
                        continue # Collinear points
                    
                    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
                    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
                    
                    if ux < 0 or ux > 1 or uy < 0 or uy > 1:
                        continue
                    
                    r = min(ux, 1-ux, uy, 1-uy)
                    for i in range(len(centers)):
                        dist = np.sqrt((ux-centers[i][0])**2 + (uy-centers[i][1])**2)
                        r = min(r, dist - radii[i])
                    
                    if r > best_r:
                        best_r = r
                        best_c = np.array([ux, uy])
                        
            except Exception:
                pass
        
        # 2. Candidate centers on the boundary of the square
        # We must check boundaries because the largest void might be against the wall.
        # We check midpoints between adjacent circles projected onto walls, and corners.
        # A more robust way: sample the walls densely.
        num_samples = 200
        for side in range(4):
            for t in np.linspace(0, 1, num_samples):
                if side == 0: # y=0
                    x, y = t, 0.0
                elif side == 1: # y=1
                    x, y = t, 1.0
                elif side == 2: # x=0
                    x, y = 0.0, t
                else: # x=1
                    x, y = 1.0, t
                
                r = min(x, 1-x, y, 1-y)
                for i in range(len(centers)):
                    dist = np.sqrt((x-centers[i][0])**2 + (y-centers[i][1])**2)
                    r = min(r, dist - radii[i])
                
                if r > best_r:
                    best_r = r
                    best_c = np.array([x, y])
        
        # 3. Circle-Circle intersection points (for gaps between 2 circles and a wall, or 3 circles)
        # Delaunay handles 3-circle gaps. 2-circle + wall gaps are handled by boundary sampling.
        # 2-circle + 2-wall (corner) is handled by boundary sampling.
        # However, sometimes the optimal center is at the intersection of two circle boundaries
        # if they are close to a wall. The boundary sampling should catch these if dense enough.
        # To be safe, we add intersection points of circle pairs as candidates too.
        n_centers = len(centers)
        for i in range(n_centers):
            for j in range(i+1, n_centers):
                c1 = centers[i]
                c2 = centers[j]
                r1 = radii[i]
                r2 = radii[j]
                
                d = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                
                if d < 1e-6 or d > r1 + r2 or d < abs(r1 - r2):
                    continue
                
                a = (d**2 + r1**2 - r2**2) / (2*d)
                h = np.sqrt(max(0, r1**2 - a**2))
                
                mid_x = c1[0] + a * (c2[0] - c1[0]) / d
                mid_y = c1[1] + a * (c2[1] - c1[1]) / d
                
                off_x = h * (c2[1] - c1[1]) / d
                off_y = h * (c1[0] - c2[0]) / d
                
                p1 = np.array([mid_x + off_x, mid_y + off_y])
                p2 = np.array([mid_x - off_x, mid_y - off_y])
                
                for p in [p1, p2]:
                    if p[0] < 0 or p[0] > 1 or p[1] < 0 or p[1] > 1:
                        continue
                    
                    r = min(p[0], 1-p[0], p[1], 1-p[1])
                    for k in range(n_centers):
                        dist_k = np.sqrt((p[0]-centers[k][0])**2 + (p[1]-centers[k][1])**2)
                        r = min(r, dist_k - radii[k])
                    
                    if r > best_r:
                        best_r = r
                        best_c = p
        
        if best_c is None:
            best_c = np.array([0.5, 0.5])
            best_r = 0.01
            
        return best_c[0], best_c[1], max(best_r, 1e-6)

    def greedy_initialize(n):
        centers = []
        radii = []
        
        # Start with 4 corners to establish boundaries
        for i in range(4):
            x = 0.1 if i % 2 == 0 else 0.9
            y = 0.1 if i < 2 else 0.9
            centers.append([x, y])
            radii.append(0.1)
            
        for _ in range(n - 4):
            c_x, c_y, r = largest_empty_circle(centers, radii)
            centers.append([c_x, c_y])
            radii.append(r)
            
        return np.array(centers), np.array(radii)

    def simulated_annealing(x_init, y_init, r_init, T0=1.0, T_end=1e-5, cooling=0.9995, steps=5000):
        """
        Simulated Annealing optimizer for circle packing.
        Moves a single circle (x, y, r) at a time.
        """
        n = len(x_init)
        x = x_init.copy()
        y = y_init.copy()
        r = r_init.copy()
        
        # Calculate initial energy
        def calc_energy(x, y, r):
            pen = 0.0
            # Boundary penalty
            for i in range(n):
                pen += max(0, r[i] - x[i])**2
                pen += max(0, r[i] - (1 - x[i]))**2
                pen += max(0, r[i] - y[i])**2
                pen += max(0, r[i] - (1 - y[i]))**2
            # Overlap penalty
            for i in range(n):
                for j in range(i+1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        pen += (min_dist - dist)**2
            return -np.sum(r) + 50.0 * pen

        current_energy = calc_energy(x, y, r)
        best_energy = current_energy
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        rng = np.random.default_rng(42)
        
        for step in range(steps):
            T = T0 * (cooling ** step)
            if T < T_end:
                break
            
            # Pick a random circle to move
            i = rng.integers(0, n)
            
            # Propose a move
            # Move center
            dx = rng.normal(0, 0.05)
            dy = rng.normal(0, 0.05)
            # Move radius
            dr = rng.normal(0, 0.02)
            
            new_x = x.copy()
            new_y = y.copy()
            new_r = r.copy()
            
            new_x[i] = np.clip(new_x[i] + dx, 0, 1)
            new_y[i] = np.clip(new_y[i] + dy, 0, 1)
            new_r[i] = np.clip(new_r[i] + dr, 0.01, 0.5)
            
            new_energy = calc_energy(new_x, new_y, new_r)
            
            delta_E = new_energy - current_energy
            
            # Acceptance probability
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                x, y, r = new_x, new_y, new_r
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
                    
        # Final post-processing to ensure validity
        # We shrink radii slightly if they overlap to ensure strict validity
        for _ in range(10):
            changed = False
            for i in range(n):
                r_max_bound = min(best_x[i], 1-best_x[i], best_y[i], 1-best_y[i])
                if best_r[i] > r_max_bound:
                    best_r[i] = r_max_bound * 0.999
                    changed = True
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.sqrt((best_x[i]-best_x[j])**2 + (best_y[i]-best_y[j])**2)
                    min_dist = best_r[i] + best_r[j]
                    if dist < min_dist:
                        # Scale down radii
                        scale = dist / min_dist * 0.999
                        best_r[i] *= scale
                        best_r[j] *= scale
                        changed = True
            if not changed:
                break
                
        return best_x, best_y, best_r

    # 1. Generate initial configuration using Greedy Method
    c_init, r_init = greedy_initialize(N)
    
    # 2. Optimize using Simulated Annealing
    # Run multiple SA runs with different random seeds for diversity
    best_score = -np.inf
    best_x, best_y, best_r = c_init[:, 0], c_init[:, 1], r_init
    
    np.random.seed(42)
    
    for run in range(5):
        # Perturb initial state slightly for each run
        if run == 0:
            x_start = c_init[:, 0]
            y_start = c_init[:, 1]
            r_start = r_init
        else:
            # Larger perturbations for subsequent runs
            x_start = c_init[:, 0] + np.random.normal(0, 0.1, N)
            y_start = c_init[:, 1] + np.random.normal(0, 0.1, N)
            r_start = r_init * np.random.uniform(0.8, 1.0, N)
            x_start = np.clip(x_start, 0.01, 0.99)
            y_start = np.clip(y_start, 0.01, 0.99)
            r_start = np.clip(r_start, 0.01, 0.5)
            
        x_opt, y_opt, r_opt = simulated_annealing(x_start, y_start, r_start, 
                                                  T0=1.0, T_end=1e-5, cooling=0.9995, steps=10000)
        
        score = np.sum(r_opt)
        if score > best_score:
            best_score = score
            best_x, best_y, best_r = x_opt, y_opt, r_opt

    # 3. Final validation and formatting
    result = np.zeros((N, 3))
    result[:, 0] = best_x
    result[:, 1] = best_y
    result[:, 2] = best_r
    
    return result
