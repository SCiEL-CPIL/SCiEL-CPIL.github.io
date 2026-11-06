"""Number-theoretic construction: lattice points in a parallelogram to maximize area multiplicity."""
import itertools
from collections import Counter

def construct():
    best_points = None
    best_score = 0
    
    # Candidate basis vectors (primitive) and grid dimensions
    # We want mn = 30.
    # Basis vectors should be small to keep points in [-30, 30].
    
    candidates = [
        # (u, v, m, n)
        ((1, 0), (0, 1), 5, 6),
        ((1, 0), (0, 1), 6, 5),
        ((1, 0), (0, 1), 3, 10),
        ((1, 0), (0, 1), 10, 3),
        ((1, 0), (0, 1), 2, 15),
        ((1, 0), (0, 1), 15, 2),
        
        ((1, 0), (1, 1), 5, 6),
        ((1, 0), (1, 1), 6, 5),
        ((1, 0), (1, 1), 3, 10),
        ((1, 0), (1, 1), 10, 3),
        
        ((1, 1), (1, -1), 5, 6),
        ((1, 1), (1, -1), 6, 5),
        
        ((2, 1), (1, 0), 5, 6), # K=1
        ((1, 2), (0, 1), 5, 6), # K=1
    ]
    
    for u, v, m, n in candidates:
        # Generate points
        # Center the grid roughly at origin to stay in bounds
        # Points: i*u + j*v for i in [0, m), j in [0, n)
        pts = []
        for i in range(m):
            for j in range(n):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                pts.append((x, y))
        
        # Check bounds and distinctness
        # We can shift the whole grid to center it.
        # Calculate bounding box
        if not pts: continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Shift to center
        shift_x = -(min_x + max_x) // 2
        shift_y = -(min_y + max_y) // 2
        
        shifted_pts = [(x + shift_x, y + shift_y) for x, y in pts]
        
        # Check if all in [-30, 30]
        if not all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in shifted_pts):
            continue
            
        # Check distinctness
        if len(set(shifted_pts)) != 30:
            continue
            
        # Calculate areas
        areas = []
        for i in range(30):
            for j in range(i+1, 30):
                for k in range(j+1, 30):
                    x1, y1 = shifted_pts[i]
                    x2, y2 = shifted_pts[j]
                    x3, y3 = shifted_pts[k]
                    det = abs(x1*(y2-y3) - x2*(y1-y3) + x3*(y1-y2))
                    if det > 0:
                        areas.append(det)
                        
        if not areas:
            continue
            
        counter = Counter(areas)
        max_mult = max(counter.values())
        
        if max_mult > best_score:
            best_score = max_mult
            best_points = shifted_pts
            
    # Fallback to original if something went wrong
    if best_points is None:
        points = [(x - 2, y - 2) for x in range(5) for y in range(6)]
        return points
        
    return best_points
