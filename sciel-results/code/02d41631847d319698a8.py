"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _min_triangle_area(points):
    """Compute minimum area among all C(11,3)=165 triangles."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _inside_triangle(p, tol=1e-6):
    """Check if point is inside or on boundary of equilateral triangle."""
    # Triangle: (0,0), (1,0), (0.5, sqrt(3)/2)
    h = np.sqrt(3) / 2
    # Barycentric coordinates
    # Point must satisfy: y >= 0, y <= h*x, y <= h*(1-x) for x in [0,1]
    # More robust: check all three edge constraints
    x, y = p[0], p[1]
    if x < -tol or x > 1+tol or y < -tol or y > h+tol:
        return False
    # Edge from (0,0) to (1,0): y >= 0
    if y < -tol:
        return False
    # Edge from (0,0) to (0.5,h): y <= 2*sqrt(3)*x => y <= 2*h*x
    if y > 2*h*x + tol:
        return False
    # Edge from (1,0) to (0.5,h): y <= -2*sqrt(3)*(x-1) => y <= 2*h*(1-x)
    if y > 2*h*(1-x) + tol:
        return False
    return True

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    h = np.sqrt(3) / 2
    
    # Initial configuration: use a grid-like placement
    # Try several initial configurations and pick the best
    best_points = None
    best_min_area = 0.0
    
    # Configuration 1: Regular grid pattern
    for seed in range(5):
        if time.time() > deadline - 10:
            break
        
        if seed == 0:
            # Uniform grid
            xs = np.linspace(0.1, 0.9, 4)
            ys = np.linspace(0.05, h*0.85, 3)
            points = []
            for x in xs:
                for y in ys:
                    if _inside_triangle([x, y]):
                        points.append([x, y])
            if len(points) < N:
                # Add more points
                for i in range(N - len(points)):
                    x = 0.1 + 0.8 * i / (N - len(points))
                    y = h * 0.5 * (1 - abs(2*x - 1))
                    points.append([x, y])
            points = np.array(points[:N])
        elif seed == 1:
            # Hexagonal-like pattern
            angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            radii = np.linspace(0, 0.35, N)
            cx, cy = 0.5, h/3
            points = np.array([[cx + r*np.cos(a), cy + r*np.sin(a)*0.7] 
                             for a, r in zip(angles, radii)])
        elif seed == 2:
            # Concentric rings
            ring1 = 3  # center + ring
            ring2 = 4
            ring3 = 4
            points = [[0.5, h/3]]  # center
            for r, count in [(0.15, ring1), (0.25, ring2), (0.35, ring3)]:
                angles = np.linspace(0, 2*np.pi, count, endpoint=False)
                for a in angles:
                    px = 0.5 + r*np.cos(a)
                    py = h/3 + r*np.sin(a)*0.6
                    if _inside_triangle([px, py]):
                        points.append([px, py])
            points = np.array(points[:N])
        else:
            # Random with rejection
            rng = np.random.RandomState(seed)
            points = []
            while len(points) < N:
                u, v = rng.random(), rng.random()
                if u + v < 0.95:
                    px = u + 0.5*v
                    py = h*v
                    points.append([px, py])
            points = np.array(points)
        
        # Ensure all points are inside
        valid_points = []
        for p in points:
            if _inside_triangle(p):
                valid_points.append(p)
            else:
                # Project to nearest valid point (center)
                valid_points.append([0.5, h/3])
        points = np.array(valid_points[:N])
        
        min_area = _min_triangle_area(points)
        if min_area > best_min_area:
            best_min_area = min_area
            best_points = points.copy()
    
    # Simulated annealing refinement
    if best_points is not None:
        current_points = best_points.copy()
        current_min_area = best_min_area
        best_sa_points = current_points.copy()
        best_sa_min_area = current_min_area
        
        T = 0.01
        T_min = 1e-6
        cooling = 0.9995
        max_iter = 50000
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Pick a random point to move
            idx = np.random.randint(N)
            old_point = current_points[idx].copy()
            
            # Propose a move
            step = T * 0.5
            new_point = old_point + np.random.normal(0, step, 2)
            
            # Check if inside triangle
            if not _inside_triangle(new_point):
                # Try to reflect or skip
                continue
            
            current_points[idx] = new_point
            new_min_area = _min_triangle_area(current_points)
            
            delta = new_min_area - current_min_area
            
            # Accept if better, or with probability exp(delta/T) if worse
            if delta > 0 or np.random.random() < np.exp(delta / T):
                current_min_area = new_min_area
                if new_min_area > best_sa_min_area:
                    best_sa_min_area = new_min_area
                    best_sa_points = current_points.copy()
            else:
                current_points[idx] = old_point
            
            T *= cooling
            if T < T_min:
                break
        
        if best_sa_min_area > best_min_area:
            best_points = best_sa_points
            best_min_area = best_sa_min_area
    
    return best_points
# EVOLVE-BLOCK-END
