"""Rewritten program for minimum maximal no-isosceles set on [0,16)^2."""
N = 16
import random
import time
import itertools
import math

def construct():
    """
    Construct a maximal no-isosceles set (no three points form an isosceles triangle)
    that is minimal in size. Uses greedy construction followed by local search
    (swap/replace moves) and random restarts to escape local minima.
    """
    deadline = time.time() + 280  # safety margin before 330s kill
    
    def is_iso(a, b, c):
        ab2 = (a[0]-b[0])**2 + (a[1]-b[1])**2
        ac2 = (a[0]-c[0])**2 + (a[1]-c[1])**2
        bc2 = (b[0]-c[0])**2 + (b[1]-c[1])**2
        return ab2 == ac2 or ab2 == bc2 or ac2 == bc2
    
    def is_valid_set(points):
        """Check that no three points in the set form an isosceles triangle."""
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_iso(points[i], points[j], points[k]):
                        return False
        return True
    
    def is_maximal(points):
        """Check that no grid point can be added without creating an isosceles triangle."""
        point_set = set(points)
        for p in itertools.product(range(N), repeat=2):
            if p in point_set:
                continue
            # Check if adding p creates any isosceles triangle
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    if is_iso(p, points[i], points[j]):
                        break
                else:
                    continue
                break
            else:
                return False  # p can be added, so not maximal
        return True
    
    def greedy_construct(order):
        """Greedy construction following the given order of points."""
        chosen = []
        chosen_set = set()
        for p in order:
            if p in chosen_set:
                continue
            ok = True
            for i in range(len(chosen)):
                for j in range(i+1, len(chosen)):
                    if is_iso(p, chosen[i], chosen[j]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                chosen.append(p)
                chosen_set.add(p)
        return chosen
    
    def try_add_point(points, p):
        """Try to add point p to the set, returning new set if valid."""
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                if is_iso(p, points[i], points[j]):
                    return None
        return points + [p]
    
    def local_search(points):
        """
        Local search: try to remove points while maintaining maximality,
        then try to re-add different points. Also try swap moves.
        """
        if time.time() > deadline - 10:
            return points
        
        points = list(points)
        improved = True
        max_iters = 500
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline - 15:
            improved = False
            iters += 1
            n = len(points)
            
            # Try removing each point
            for i in range(n):
                candidate = points[:i] + points[i+1:]
                if is_valid_set(candidate):
                    # Check if still maximal
                    if is_maximal(candidate):
                        points = candidate
                        improved = True
                        break
            if improved:
                continue
            
            # Try swap: remove one point, add another
            for i in range(len(points)):
                remaining = points[:i] + points[i+1:]
                for p in itertools.product(range(N), repeat=2):
                    if p in remaining:
                        continue
                    new_set = remaining + [p]
                    if is_valid_set(new_set) and is_maximal(new_set):
                        if len(new_set) < len(points):
                            points = new_set
                            improved = True
                            break
                if improved:
                    break
            if improved:
                continue
            
            # Try adding a point (if not maximal, make it maximal)
            if not is_maximal(points):
                # Find a point that can be added
                for p in itertools.product(range(N), repeat=2):
                    if p in points:
                        continue
                    new_set = try_add_point(points, p)
                    if new_set is not None:
                        points = new_set
                        improved = True
                        break
        
        return points
    
    best = None
    best_len = float('inf')
    
    rng = random.Random(42)
    
    while time.time() < deadline - 30:
        # Generate a random order for greedy construction
        all_points = list(itertools.product(range(N), repeat=2))
        rng.shuffle(all_points)
        
        # Greedy construction
        points = greedy_construct(all_points)
        
        # Ensure maximality by adding points
        if not is_maximal(points):
            # Add points until maximal
            changed = True
            while changed:
                changed = False
                for p in all_points:
                    if p in points:
                        continue
                    new_set = try_add_point(points, p)
                    if new_set is not None:
                        points = new_set
                        changed = True
                        break
        
        # Local search to minimize
        points = local_search(points)
        
        if len(points) < best_len:
            best = points
            best_len = len(points)
        
        # Early termination if we find a very good solution
        if best_len <= 10:
            break
    
    if best is None:
        # Fallback: simple greedy
        best = greedy_construct(list(itertools.product(range(N), repeat=2)))
        # Ensure maximal
        if not is_maximal(best):
            for p in itertools.product(range(N), repeat=2):
                if p in best:
                    continue
                new_set = try_add_point(best, p)
                if new_set is not None:
                    best = new_set
    
    return best
