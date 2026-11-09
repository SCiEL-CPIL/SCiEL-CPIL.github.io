import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Helper to normalize a line defined by two points
    def line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        # Normal vector (dy, -dx) or direction vector (dx, dy)
        # Use direction vector normalized by GCD
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return None
        g = __import__('math').gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        # Ensure unique representation
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        # Line is determined by direction and one point? 
        # No, parallel lines are different.
        # Use intercept? Or (dx, dy, c) where ax+by=c
        # A = dy, B = -dx, C = -(dy*x1 - dx*y1)
        a = dy
        b = -dx
        c = -(dy * x1 - dx * y1)
        # Normalize (a,b,c)
        g2 = __import__('math').gcd(abs(a), __import__('math').gcd(abs(b), abs(c)))
        if g2 > 0:
            a //= g2
            b //= g2
            c //= g2
        if a < 0 or (a == 0 and (b < 0 or (b == 0 and c < 0))):
            a, b, c = -a, -b, -c
        return (a, b, c)

    def is_valid_no4(points):
        # Check if any 4 points are collinear
        # Group points by line
        lines = {}
        for p1, p2 in combinations(points, 2):
            k = line_key(p1, p2)
            if k is None: continue
            if k not in lines:
                lines[k] = set()
            lines[k].add(p1)
            lines[k].add(p2)
        
        for k, pts in lines.items():
            if len(pts) >= 4:
                return False
        return True

    def count_triples(points):
        # Count number of 3-collinear triples
        # Also check for 4+ collinear (should be 0 if valid)
        lines = {}
        for p1, p2 in combinations(points, 2):
            k = line_key(p1, p2)
            if k is None: continue
            if k not in lines:
                lines[k] = set()
            lines[k].add(p1)
            lines[k].add(p2)
        
        total = 0
        for k, pts in lines.items():
            n = len(pts)
            if n >= 4:
                return -1 # Invalid
            if n == 3:
                total += 1
        return total

    # Initialize with random points
    def random_points():
        pts = set()
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            pts.add((x, y))
        return list(pts)

    # Find initial valid configuration
    best_pts = None
    best_triples = -1
    
    # Try to find a configuration with many triples
    # Start with a grid-like structure and perturb? 
    # Or just random search.
    
    # Let's try to start with a "good" heuristic: 
    # Take a 4x5 grid and randomize slightly?
    # Grid: x in 0..4, y in 0..3.
    # Map to large range.
    # But grid has 4-collinear.
    # So we must break them.
    
    # Strategy: Random restarts + Local Search
    import math
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Generate random points
        pts = random_points()
        
        # If not valid, try to fix by moving points
        # Just reject for now? No, that's slow.
        # Let's generate, check, and if invalid, move a random point until valid?
        
        valid = is_valid_no4(pts)
        if not valid:
            # Try to fix
            for _ in range(100):
                idx = random.randint(0, 19)
                old = pts[idx]
                new = (random.randint(-100, 100), random.randint(-100, 100))
                if new == old: continue
                pts[idx] = new
                if is_valid_no4(pts):
                    break
            if not is_valid_no4(pts):
                continue
                
        triples = count_triples(pts)
        if triples > best_triples:
            best_triples = triples
            best_pts = pts[:]
            
        # Local Search
        current_pts = best_pts[:]
        current_triples = best_triples
        no_improve = 0
        temp = 1.0
        
        while time.time() < deadline and no_improve < 100:
            idx = random.randint(0, 19)
            old_x, old_y = current_pts[idx]
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            
            if (new_x, new_y) == (old_x, old_y):
                continue
            
            # Check duplicate
            if any(p[0] == new_x and p[1] == new_y for i, p in enumerate(current_pts) if i != idx):
                continue
                
            current_pts[idx] = (new_x, new_y)
            
            new_triples = count_triples(current_pts)
            
            if new_triples == -1: # Invalid (4 collinear)
                current_pts[idx] = (old_x, old_y)
                no_improve += 1
                continue
                
            if new_triples >= current_triples:
                current_triples = new_triples
                no_improve = 0
                if new_triples > best_triples:
                    best_triples = new_triples
                    best_pts = current_pts[:]
            else:
                # Simulated annealing accept
                if random.random() < math.exp((new_triples - current_triples) / temp):
                    current_triples = new_triples
                else:
                    current_pts[idx] = (old_x, old_y)
                no_improve += 1
            
            temp = max(0.1, temp * 0.98)
            
        # If we found a good one, maybe refine further?
        # For now, we just keep the best found so far.

    if best_pts is None:
        # Fallback to parabola (valid, but score 20)
        return [[x, x*x] for x in range(-10, 10)]
        
    return [[p[0], p[1]] for p in best_pts]
