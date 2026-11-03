"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np

# Triangle vertices
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, np.sqrt(3)/2])

def _is_inside_triangle(p, tol=1e-9):
    """Check if point p is inside or on the boundary of the unit equilateral triangle."""
    x, y = p
    if y < -tol:
        return False
    if y > np.sqrt(3) * (1 - x) + tol:
        return False
    if y > np.sqrt(3) * x + tol:
        return False
    return True

def _project_to_triangle(p):
    """Project a point to the nearest point inside the triangle."""
    # If already inside, return as is
    if _is_inside_triangle(p):
        return p
        
    x, y = p
    # Centroid
    cx, cy = 0.5, np.sqrt(3)/6
    
    # If far outside, move towards centroid
    # We can use a simple iterative projection or barycentric coordinates
    # Barycentric coordinates for point (x,y) with respect to V1, V2, V3
    # lambda1 = area(P, V2, V3) / area(V1, V2, V3)
    # lambda2 = area(V1, P, V3) / area(V1, V2, V3)
    # lambda3 = area(V1, V2, P) / area(V1, V2, V3)
    
    # Area of V1, V2, V3 is sqrt(3)/4
    area_total = np.sqrt(3)/4
    
    # Vectorized area calculation for barycentric
    # P = (x,y)
    # lambda1 = det(V2-V1, V3-V1, P-V1) ... 
    # Simpler: use the half-plane distances.
    
    # Distance to edge 1 (y=0): d1 = y
    # Distance to edge 2 (line through V2, V3): y = -sqrt(3)(x-1) => sqrt(3)x + y - sqrt(3) = 0
    # Normalized distance d2 = (sqrt(3)*x + y - sqrt(3)) / 2  (since norm of (sqrt(3), 1) is 2)
    # Distance to edge 3 (line through V1, V3): y = sqrt(3)x => -sqrt(3)x + y = 0
    # Normalized distance d3 = (-sqrt(3)*x + y) / 2
    
    d1 = y
    d2 = (np.sqrt(3)*x + y - np.sqrt(3)) / 2.0
    d3 = (-np.sqrt(3)*x + y) / 2.0
    
    # If any distance is negative, project onto that edge
    # We can use the barycentric coordinates derived from these distances
    # lambda1 = d3 / h1? No, barycentric weights are proportional to areas.
    # Area1 = 0.5 * base * height. Base V2V3 length is 1. Height from P to V2V3 is -d2 (if d2<0).
    # Let's just use the standard barycentric formula.
    
    # lambda1 = ( (y2-y3)x + (x3-x2)y + x2y3 - x3y2 ) / D
    # lambda2 = ( (y3-y1)x + (x1-x3)y + x3y1 - x1y3 ) / D
    # lambda3 = 1 - lambda1 - lambda2
    # D = (y2-y3)(x3-x1) + (x1-x2)(y3-y1)
    
    x1, y1 = V1
    x2, y2 = V2
    x3, y3 = V3
    
    D = (y2 - y3) * (x3 - x1) + (x1 - x2) * (y3 - y1)
    
    l1 = ((y2 - y3) * x + (x3 - x2) * y + x2 * y3 - x3 * y2) / D
    l2 = ((y3 - y1) * x + (x1 - x3) * y + x3 * y1 - x1 * y3) / D
    l3 = 1.0 - l1 - l2
    
    # Clamp to [0, 1] and renormalize? No, simple projection:
    # If l1 < 0, set l1=0, and renormalize l2, l3
    # This is approximate but effective for SA.
    
    # Better: just move towards centroid if outside
    # This is simpler and robust enough for SA
    t = 0.1
    for _ in range(10):
        if _is_inside_triangle(p):
            break
        p = p + t * (np.array([cx, cy]) - p)
        t *= 0.5
    return p

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    # Use vectorized calculation for speed
    # We need all triplets i, j, k
    # 11 points -> 165 triplets. Python loop is fine, but let's try to be efficient.
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * | (p2-p1) x (p3-p1) |
                v1 = p2 - p1
                v2 = p3 - p1
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _score(points):
    """Score is the minimum triangle area."""
    return _min_triangle_area(points)

def _repulsive_force(points, k=1.0):
    """
    Calculate a repulsive force vector for each point.
    Force is proportional to 1/area for each triplet.
    This encourages points to spread out.
    """
    n = len(points)
    forces = np.zeros_like(points)
    
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                
                v1 = p2 - p1
                v2 = p3 - p1
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                
                # Avoid division by zero
                if area < 1e-12:
                    # If area is very small, push them apart strongly in random direction or along edge
                    # Simple heuristic: push i and j apart
                    forces[i] += 100 * (p1 - p2)
                    forces[j] += 100 * (p2 - p1)
                    continue
                
                # Gradient of area with respect to points
                # Area = 0.5 * | (x2-x1)(y3-y1) - (y2-y1)(x3-x1) |
                # Let A = (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
                # dA/dx1 = -(y3-y1) + (y2-y1) = y2 - y3
                # dA/dy1 = (x2-x1) - (x3-x1) = x2 - x3
                # dA/dx2 = y3 - y1
                # dA/dy2 = -(x3 - x1) = x1 - x3
                # dA/dx3 = -(y2 - y1) = y1 - y2
                # dA/dy3 = x2 - x1
                
                # We want to maximize area, so we move in direction of gradient of A (if A>0) or -A (if A<0)
                # Since we use abs, we just want to increase the magnitude.
                # If A > 0, grad is as above. If A < 0, grad is negated.
                # Effectively, we move points to increase the signed area.
                
                A = v1[0]*v2[1] - v1[1]*v2[0]
                sign = 1.0 if A > 0 else -1.0
                
                grad_x1 = sign * (points[j][1] - points[k][1])
                grad_y1 = sign * (points[j][0] - points[k][0])
                grad_x2 = sign * (points[k][1] - points[i][1])
                grad_y2 = sign * (points[i][0] - points[k][0])
                grad_x3 = sign * (points[i][1] - points[j][1])
                grad_y3 = sign * (points[j][0] - points[i][0])
                
                # Force is gradient / area (inverse proportionality to emphasize small areas)
                # We use k / area to weight the force
                weight = k / (area * area) # Inverse square for stronger repulsion at close range
                
                forces[i] += np.array([grad_x1, grad_y1]) * weight
                forces[j] += np.array([grad_x2, grad_y2]) * weight
                forces[k] += np.array([grad_x3, grad_y3]) * weight
                
    return forces

def construct():
    """
    Use Simulated Annealing to maximize the minimum triangle area.
    """
    np.random.seed(42)
    
    # Initial points: Random
    points = []
    while len(points) < N:
        u, v = np.random.random(), np.random.random()
        if u + v < 1.0:
            # Barycentric to Cartesian
            x = u * V1[0] + v * V2[0] + (1-u-v) * V3[0]
            y = u * V1[1] + v * V2[1] + (1-u-v) * V3[1]
            points.append([x, y])
    points = np.array(points)
    
    current_score = _score(points)
    best_points = points.copy()
    best_score = current_score
    
    # SA Parameters
    T0 = 0.01
    Tend = 0.0001
    alpha = 0.95
    T = T0
    step_size = 0.05
    
    while T > Tend:
        # Generate perturbation
        new_points = points.copy()
        # Pick a random subset of points to move
        n_move = np.random.randint(1, 4)
        indices = np.random.choice(N, n_move, replace=False)
        
        for idx in indices:
            # Random displacement
            disp = np.random.normal(0, step_size, 2)
            new_points[idx] += disp
            # Project to triangle
            new_points[idx] = _project_to_triangle(new_points[idx])
            
        new_score = _score(new_points)
        
        # Acceptance criterion
        if new_score > current_score:
            points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            # Metropolis criterion
            delta = current_score - new_score
            if delta < 0:
                accept = True
            else:
                accept = np.random.random() < np.exp(-delta / T)
            
            if accept:
                points = new_points
                current_score = new_score
        
        T *= alpha
        # Optionally decrease step size
        step_size *= 0.99
        
    return best_points
# EVOLVE-BLOCK-END
