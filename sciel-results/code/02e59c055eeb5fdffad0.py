import numpy as np
import time
import scipy.optimize
import itertools

N = 11
# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2])

def point_in_triangle(p, v0, v1, v2):
    # Using barycentric coordinates or cross products
    # Cross product of (v1-v0) and (p-v0)
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    d1 = cross(p, v0, v1)
    d2 = cross(p, v1, v2)
    d3 = cross(p, v2, v0)
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    return not (has_neg and has_pos)

def min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by 3 points."""
    min_area = float('inf')
    n = len(points)
    # Precompute squared distances? No, area needs cross product.
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]))
                if area < min_area:
                    min_area = area
    return min_area

def negative_min_area(points):
    """Objective function for minimization."""
    # Add penalty if points are outside? 
    # We enforce constraints in the move generation, so points should always be inside.
    # But Nelder-Mead might step outside. 
    # We should handle this.
    for p in points:
        if not point_in_triangle(p, V0, V1, V2):
            # Return a large penalty
            return 1e10
    return -min_triangle_area(points)

def random_point_in_triangle():
    # Rejection sampling
    while True:
        u = np.random.random()
        v = np.random.random()
        if u + v <= 1:
            p = V0 + u * (V1 - V0) + v * (V2 - V0)
            return p
        # Alternative: sqrt method
        # u = np.random.random()
        # v = np.random.random()
        # if v > 1 - u:
        #     u = 1 - u
        #     v = 1 - v
        # p = V0 + u * (V1 - V0) + v * (V2 - V0)
        # return p

def initial_lattice_points():
    """Generate an initial set of 11 points based on a lattice."""
    # Try to fit a grid
    # A 4x3 grid of equilateral triangles has 10 points on vertices?
    # Let's just place points in a hexagonal-like pattern.
    # Or simply: divide the triangle into smaller triangles.
    
    # Simple strategy: Place points at centers of sub-triangles or vertices of a grid.
    # Let's try a 3x3 grid of sub-triangles? 
    # Actually, just scattering 11 points randomly is a valid "lattice" if we jitter a coarse grid.
    
    # Let's create a grid of candidates and pick 11.
    # Grid size 5x5 in barycentric?
    # Let's use Cartesian grid clipped to triangle.
    candidates = []
    step = 0.1
    x = 0.0
    while x <= 1.0:
        y = 0.0
        while y <= np.sqrt(3)/2:
            p = np.array([x, y])
            if point_in_triangle(p, V0, V1, V2):
                candidates.append(p)
            y += step
        x += step
        
    # Shuffle and pick 11
    np.random.shuffle(candidates)
    if len(candidates) >= N:
        return np.array(candidates[:N])
    else:
        # Fallback to random
        return np.array([random_point_in_triangle() for _ in range(N)])

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin
    
    # 1. Initialization
    # Try a few different initializations
    best_points = None
    best_score = -float('inf')
    
    # Init 1: Lattice
    points = initial_lattice_points()
    # Add small noise to break symmetry
    noise = np.random.normal(0, 0.01, points.shape)
    points = points + noise
    # Ensure inside
    for i in range(len(points)):
        if not point_in_triangle(points[i], V0, V1, V2):
             points[i] = random_point_in_triangle()
             
    current_score = min_triangle_area(points)
    if current_score > best_score:
        best_score = current_score
        best_points = points.copy()
        
    # 2. Simulated Annealing
    T = 0.05 # Initial temperature
    alpha = 0.999 # Cooling rate
    delta = 0.05 # Step size
    
    current_points = points.copy()
    current_min_area = current_score
    
    while time.time() < deadline - 5.0: # Leave 5s for polishing
        # Pick a random point to move
        i = np.random.randint(N)
        old_p = current_points[i].copy()
        
        # Generate new position
        # Random direction and magnitude
        angle = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(0, delta)
        new_p = old_p + r * np.array([np.cos(angle), np.sin(angle)])
        
        # Check if inside
        if not point_in_triangle(new_p, V0, V1, V2):
            continue # Reject move
            
        # Calculate new min area
        # Optimization: Only recalculate if the moved point is involved in the current min triangle?
        # Too complex. Just recalculate all. 165 triangles is fast.
        current_points[i] = new_p
        new_min_area = min_triangle_area(current_points)
        
        # Acceptance criterion
        if new_min_area > current_min_area:
            current_min_area = new_min_area
            if current_min_area > best_score:
                best_score = current_min_area
                best_points = current_points.copy()
        else:
            # Worse, accept with probability exp((new - old) / T)
            # Note: We are maximizing, so delta = new - old is negative.
            # Prob = exp(delta / T)
            prob = np.exp((new_min_area - current_min_area) / T)
            if np.random.random() < prob:
                current_min_area = new_min_area
            else:
                # Revert
                current_points[i] = old_p
                
        # Cool down
        T *= alpha
        # Increase step size slightly? Or keep constant?
        # delta = max(0.01, delta * 0.999) # Maybe shrink delta?
        
    # 3. Polishing with Nelder-Mead
    if time.time() < deadline - 2.0:
        # Use the best points found so far
        x0 = best_points.flatten()
        
        def objective(x):
            p = x.reshape(-1, 2)
            return negative_min_area(p)
        
        try:
            result = scipy.optimize.minimize(objective, x0, method='Nelder-Mead', 
                                             options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-6})
            if result.fun < negative_min_area(best_points.flatten()):
                polished_points = result.x.reshape(-1, 2)
                # Check validity
                valid = True
                for p in polished_points:
                    if not point_in_triangle(p, V0, V1, V2):
                        valid = False
                        break
                if valid:
                    final_score = min_triangle_area(polished_points)
                    if final_score > best_score:
                        best_points = polished_points
                        best_score = final_score
        except Exception:
            pass # Ignore errors in polishing
            
    return best_points

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

N = 11
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2])

def point_in_triangle(p, v0, v1, v2):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    d1 = cross(p, v0, v1)
    d2 = cross(p, v1, v2)
    d3 = cross(p, v2, v0)
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    return not (has_neg and has_pos)

def min_triangle_area(points):
    min_area = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]))
                if area < min_area:
                    min_area = area
    return min_area

def negative_min_area(points):
    for p in points:
        if not point_in_triangle(p, V0, V1, V2):
            return 1e10
    return -min_triangle_area(points)

def random_point_in_triangle():
    while True:
        u = np.random.random()
        v = np.random.random()
        if u + v <= 1:
            p = V0 + u * (V1 - V0) + v * (V2 - V0)
            return p

def initial_lattice_points():
    candidates = []
    step = 0.1
    x = 0.0
    while x <= 1.0:
        y = 0.0
        while y <= np.sqrt(3)/2:
            p = np.array([x, y])
            if point_in_triangle(p, V0, V1, V2):
                candidates.append(p)
            y += step
        x += step
        
    np.random.shuffle(candidates)
    if len(candidates) >= N:
        return np.array(candidates[:N])
    else:
        return np.array([random_point_in_triangle() for _ in range(N)])

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_points = None
    best_score = -float('inf')
    
    # Initialization
    points = initial_lattice_points()
    noise = np.random.normal(0, 0.01, points.shape)
    points = points + noise
    for i in range(len(points)):
        if not point_in_triangle(points[i], V0, V1, V2):
             points[i] = random_point_in_triangle()
             
    current_score = min_triangle_area(points)
    if current_score > best_score:
        best_score = current_score
        best_points = points.copy()
        
    # Simulated Annealing
    T = 0.05
    alpha = 0.999
    delta = 0.05
    
    current_points = points.copy()
    current_min_area = current_score
    
    while time.time() < deadline - 5.0:
        i = np.random.randint(N)
        old_p = current_points[i].copy()
        
        angle = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(0, delta)
        new_p = old_p + r * np.array([np.cos(angle), np.sin(angle)])
        
        if not point_in_triangle(new_p, V0, V1, V2):
            continue
            
        current_points[i] = new_p
        new_min_area = min_triangle_area(current_points)
        
        if new_min_area > current_min_area:
            current_min_area = new_min_area
            if current_min_area > best_score:
                best_score = current_min_area
                best_points = current_points.copy()
        else:
            prob = np.exp((new_min_area - current_min_area) / T)
            if np.random.random() < prob:
                current_min_area = new_min_area
            else:
                current_points[i] = old_p
                
        T *= alpha
        
    # Polishing
    if time.time() < deadline - 2.0:
        x0 = best_points.flatten()
        
        def objective(x):
            p = x.reshape(-1, 2)
            return negative_min_area(p)
        
        try:
            result = scipy.optimize.minimize(objective, x0, method='Nelder-Mead', 
                                             options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-6})
            if result.fun < negative_min_area(best_points.flatten()):
                polished_points = result.x.reshape(-1, 2)
                valid = True
                for p in polished_points:
                    if not point_in_triangle(p, V0, V1, V2):
                        valid = False
                        break
                if valid:
                    final_score = min_triangle_area(polished_points)
                    if final_score > best_score:
                        best_points = polished_points
                        best_score = final_score
        except Exception:
            pass
            
    return best_points
# EVOLVE-BLOCK-END
