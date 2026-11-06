import numpy as np
import time
import random
from itertools import combinations

def _score(points):
    """Calculate the maximum multiplicity of a positive doubled area."""
    points = np.array(points, dtype=np.int64)
    n = len(points)
    if n < 3:
        return 0
    counts = {}
    # Vectorized computation for speed
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            v1 = p2 - p1
            if j + 1 < n:
                rest = points[j+1:]
                terms = v1[0] * rest[:, 1] - v1[1] * rest[:, 0] + (v1[1]*p1[0] - v1[0]*p1[1])
                areas = np.abs(terms)
                for a in areas:
                    if a > 0:
                        counts[a] = counts.get(a, 0) + 1
    if not counts:
        return 0
    return max(counts.values())

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_points = None
    best_score = 0
    
    # Helper to generate points from a lattice definition
    def generate_lattice(n1, n2, u, v, offset=(0,0)):
        pts = []
        for i in range(n1):
            for j in range(n2):
                x = offset[0] + i*u[0] + j*v[0]
                y = offset[1] + i*u[1] + j*v[1]
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
                else:
                    return None
        if len(pts) == 30 and len(set(pts)) == 30:
            return pts
        return None

    # 1. Initialize with a good baseline: 5x6 grid
    # u=(1,0), v=(0,1), n1=6, n2=5
    # Area base = 1. Smallest area factor is 1.
    # Multiplicity of area 1 in 5x6 grid:
    # Triangles with area 0.5 (doubled area 1) are formed by adjacent points.
    # In an m x n grid, number of unit triangles is 2*(m-1)*(n-1) + ... actually just count them.
    # Let's just run the scorer on a few static baselines first.
    
    baselines = []
    # 5x6 Grid
    baselines.append(generate_lattice(6, 5, (1,0), (0,1), (0,0)))
    # 3x10 Grid
    baselines.append(generate_lattice(10, 3, (1,0), (0,1), (0,0)))
    # 2x15 Grid
    baselines.append(generate_lattice(15, 2, (1,0), (0,1), (0,0)))
    # 1x30 is degenerate (area 0), skip.
    
    # Sheared 5x6: u=(1,0), v=(1,1)
    baselines.append(generate_lattice(6, 5, (1,0), (1,1), (-10, -10)))
    # Sheared 3x10: u=(1,0), v=(1,1)
    baselines.append(generate_lattice(10, 3, (1,0), (1,1), (-15, -5)))
    
    for b in baselines:
        if b:
            s = _score(b)
            if s > best_score:
                best_score = s
                best_points = b

    # 2. Simulated Annealing / Random Search over Lattice Parameters
    # We represent a lattice by:
    # n1, n2 (dimensions, n1*n2=30)
    # u, v (basis vectors)
    # We want to maximize the score.
    # The score is dominated by the frequency of the smallest area.
    # Smallest area in a lattice is |det(u,v)| * min_factor.
    # min_factor is usually 1 (if u,v are primitive and adjacent indices exist).
    # So we want to maximize the count of triangles with area |det(u,v)|.
    # This count depends on the geometry (how many unit parallelograms/triangles fit).
    # For a fixed area A_base, a more "compact" or "regular" shape might have more small triangles?
    # Actually, for a grid, the number of unit triangles is proportional to the number of edges.
    # A 1x30 line has 0 area. A 2x15 grid has more "boundary" relative to area?
    
    # Let's search over valid (n1, n2) pairs and random basis vectors.
    dims = [(6,5), (5,6), (10,3), (3,10), (15,2), (2,15)]
    
    # Current best configuration parameters to perturb
    # We'll just do random restarts with local optimization on u,v
    # u, v are integers.
    
    def random_basis():
        # Generate random primitive vectors? Or just random small vectors.
        # To fit 30 points in [-30,30], vectors must be small.
        # Max extent is 30. If n1=15, u can be at most size 2 roughly.
        # Let's pick u, v from a small range, e.g., [-5, 5].
        u = (random.randint(-5, 5), random.randint(-5, 5))
        v = (random.randint(-5, 5), random.randint(-5, 5))
        # Ensure they are not collinear (det != 0)
        if u[0]*v[1] - u[1]*v[0] == 0:
            return None
        return u, v

    def optimize_lattice(n1, n2, u_init, v_init, steps=200):
        """
        Local search on u, v for fixed n1, n2.
        """
        u = u_init
        v = v_init
        current_points = generate_lattice(n1, n2, u, v, (0,0))
        if not current_points:
            # Try to find an offset that fits
            # Calculate bounding box
            xs = [i*u[0] + j*v[0] for i in range(n1) for j in range(n2)]
            ys = [i*u[1] + j*v[1] for i in range(n1) for j in range(n2)]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            # Center it
            off_x = - (min_x + max_x) // 2
            off_y = - (min_y + max_y) // 2
            current_points = generate_lattice(n1, n2, u, v, (off_x, off_y))
            if not current_points:
                return None, -1
        
        current_score = _score(current_points)
        
        for step in range(steps):
            if time.time() > deadline - 5:
                break
            # Perturb u or v
            if random.random() < 0.5:
                du = random.randint(-2, 2)
                dv = random.randint(-2, 2)
                if du == 0 and dv == 0: continue
                new_u = (u[0] + du, u[1] + dv)
                new_v = v
            else:
                du = random.randint(-2, 2)
                dv = random.randint(-2, 2)
                if du == 0 and dv == 0: continue
                new_u = u
                new_v = (v[0] + du, v[1] + dv)
            
            # Check validity
            if new_u[0]*new_v[1] - new_u[1]*new_v[0] == 0:
                continue
            
            new_points = generate_lattice(n1, n2, new_u, new_v, (0,0))
            if not new_points:
                xs = [i*new_u[0] + j*new_v[0] for i in range(n1) for j in range(n2)]
                ys = [i*new_u[1] + j*new_v[1] for i in range(n1) for j in range(n2)]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                off_x = - (min_x + max_x) // 2
                off_y = - (min_y + max_y) // 2
                new_points = generate_lattice(n1, n2, new_u, new_v, (off_x, off_y))
            
            if not new_points:
                continue
            
            new_score = _score(new_points)
            
            # Accept if better, or with some probability if worse (annealing)
            # Since we are limited in time, simple hill climbing is safer.
            if new_score >= current_score:
                current_score = new_score
                current_points = new_points
                u, v = new_u, new_v
                
        return current_points, current_score

    # Perform search for each dimension
    for n1, n2 in dims:
        if time.time() > deadline - 10:
            break
            
        # Try a few random initializations
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            rb = random_basis()
            if not rb:
                continue
            u_init, v_init = rb
            pts, sc = optimize_lattice(n1, n2, u_init, v_init, steps=100)
            if pts and sc > best_score:
                best_score = sc
                best_points = pts

    # Final check
    if best_points is None:
        # Fallback to a simple grid
        best_points = generate_lattice(6, 5, (1,0), (0,1), (0,0))
        
    return best_points
