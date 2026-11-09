"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s kill
    MAX_XY = 100
    
    def elapsed():
        return time.time() - start_time
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    def are_collinear(p, q, r):
        return cross(p, q, r) == 0
    
    def get_lines(points):
        """Return dict: frozenset of point indices -> list of indices on that line"""
        n = len(points)
        lines = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = (i, j)
                # Find all points collinear with i and j
                members = [i, j]
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            members.append(k)
                if len(members) >= 2:
                    fs = frozenset(members)
                    lines[fs] = members
        return lines
    
    def has_four_collinear(points):
        """Check if any 4 points are collinear"""
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        if are_collinear(points[i], points[j], points[k]):
                            if are_collinear(points[i], points[j], points[l]):
                                return True
        return False
    
    def count_four_collinear(points):
        """Count number of 4-collinear subsets"""
        n = len(points)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        if are_collinear(points[i], points[j], points[k]):
                            if are_collinear(points[i], points[j], points[l]):
                                count += 1
        return count
    
    def compute_score(points):
        """Compute largest subset with no three collinear (exact for small n, greedy for large)"""
        n = len(points)
        # Build conflict graph: edge between i,j if there exists k such that i,j,k collinear
        # We want max independent set in this graph
        # For n=20, use branch and bound
        # First, find all collinear triples
        collinear_with = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        collinear_with[i].add(j)
                        collinear_with[i].add(k)
                        collinear_with[j].add(i)
                        collinear_with[j].add(k)
                        collinear_with[k].add(i)
                        collinear_with[k].add(j)
        
        # Branch and bound for maximum independent set
        best = [0]
        remaining = set(range(n))
        
        def branch(bound, current_size, candidates):
            if not candidates:
                if current_size > best[0]:
                    best[0] = current_size
                return
            if current_size + len(candidates) <= best[0]:
                return
            if elapsed() > DEADLINE - 5:
                return
            # Pick vertex with highest degree in candidates
            v = max(candidates, key=lambda x: len(collinear_with[x] & candidates))
            # Branch: include v
            new_candidates = candidates - {v} - collinear_with[v]
            branch(bound, current_size + 1, new_candidates)
            # Branch: exclude v
            branch(bound, current_size, candidates - {v})
        
        branch(0, 0, set(range(n)))
        return best[0]
    
    def compute_score_fast(points):
        """Faster score: greedy + local improvement"""
        n = len(points)
        collinear_with = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        collinear_with[i].add(j)
                        collinear_with[i].add(k)
                        collinear_with[j].add(i)
                        collinear_with[j].add(k)
                        collinear_with[k].add(i)
                        collinear_with[k].add(j)
        
        # Greedy: repeatedly pick vertex with fewest conflicts
        selected = []
        remaining = set(range(n))
        while remaining:
            # Pick vertex with min conflicts within remaining
            best_v = min(remaining, key=lambda x: len(collinear_with[x] & remaining))
            selected.append(best_v)
            remaining -= collinear_with[best_v]
            remaining.discard(best_v)
        
        base_score = len(selected)
        
        # Try to improve: for each unselected vertex, see if we can swap
        unselected = set(range(n)) - set(selected)
        improved = True
        while improved:
            improved = False
            for u in list(unselected):
                # Try adding u: remove all selected that conflict with u
                conflicts = collinear_with[u] & set(selected)
                if len(conflicts) < 1:  # adding u without removing any
                    # Actually we need to check if u conflicts with any selected
                    if not conflicts:
                        selected.append(u)
                        unselected.discard(u)
                        improved = True
                        break
            # Try removing a selected vertex that allows adding 2 unselected
            for s in list(selected):
                # If we remove s, which unselected can we add?
                new_unselected = unselected | {s}
                # Greedily add from new_unselected
                added = []
                temp_remaining = set(new_unselected)
                while temp_remaining:
                    best_v = min(temp_remaining, key=lambda x: len(collinear_with[x] & temp_remaining))
                    added.append(best_v)
                    temp_remaining -= collinear_with[best_v]
                    temp_remaining.discard(best_v)
                if len(added) > 1:  # net gain
                    # Apply
                    selected.remove(s)
                    for a in added:
                        if a not in selected:
                            selected.append(a)
                            unselected.discard(a)
                    improved = True
                    break
        
        return len(selected)
    
    def generate_initial():
        """Generate points on a grid-like structure with some collinear triples"""
        points = []
        # Use a 5x4 grid subset
        xs = [-4, -2, 0, 2, 4]
        ys = [-3, -1, 1, 3]
        for x in xs:
            for y in ys:
                points.append([x, y])
        # Scale to fit
        return points
    
    def random_point():
        return [random.randint(-MAX_XY, MAX_XY), random.randint(-MAX_XY, MAX_XY)]
    
    def valid_configuration(points):
        """Check: 20 distinct points, |x|,|y| <= 100, no four collinear"""
        if len(points) != 20:
            return False
        if len(set(tuple(p) for p in points)) != 20:
            return False
        for p in points:
            if abs(p[0]) > MAX_XY or abs(p[1]) > MAX_XY:
                return False
        if has_four_collinear(points):
            return False
        return True
    
    def local_search(points, time_limit):
        """Local search to minimize score while maintaining validity"""
        if not valid_configuration(points):
            return points
        current_score = compute_score_fast(points)
        best_points = [p[:] for p in points]
        best_score = current_score
        last_improve = time.time()
        
        while time.time() - start_time < time_limit:
            # Pick a random point to move
            idx = random.randint(0, 19)
            old = points[idx][:]
            # Try random new positions
            found = False
            for _ in range(50):
                new_pt = random_point()
                if new_pt == old:
                    continue
                points[idx] = new_pt
                if valid_configuration(points):
                    s = compute_score_fast(points)
                    if s < best_score:
                        best_score = s
                        best_points = [p[:] for p in points]
                        last_improve = time.time()
                        found = True
                        break
                    elif s == best_score and random.random() < 0.3:
                        # Accept equal score with some probability (plateau)
                        if time.time() - last_improve > 5:
                            best_points = [p[:] for p in points]
                            last_improve = time.time()
                            found = True
                            break
            if not found:
                points[idx] = old
            if best_score <= 10:
                break
            if time.time() - last_improve > 15:
                # Restart with perturbation
                for _ in range(3):
                    idx2 = random.randint(0, 19)
                    old2 = points[idx2][:]
                    points[idx2] = random_point()
                    if not valid_configuration(points):
                        points[idx2] = old2
                last_improve = time.time()
        
        if best_score < current_score:
            points[:] = best_points
        return points
    
    def generate_line_based():
        """Generate points by placing them on specific lines to create collinear triples"""
        random.seed(42)
        # Choose 6 lines, each with 3-4 points, ensuring no 4 on any line
        # Lines defined by two points
        lines = []
        used_points = set()
        
        # Create lines with 3 points each (18 points) + 2 extra
        # Use lines with small integer slopes
        line_defs = [
            ([(-6, -4), (-3, -2), (0, 0)]),       # slope 2/3
            ([(2, 1), (4, 3), (6, 5)]),            # slope 1
            ([(-5, 2), (-2, 4), (1, 6)]),          # slope 2/3
            ([(3, -3), (5, -1), (7, 1)]),          # slope 1
            ([(-4, -5), (-1, -3), (2, -1)]),       # slope 2/3
            ([(0, -4), (3, -2), (6, 0)]),          # slope 2/3
        ]
        
        points = []
        for line_pts in line_defs:
            for p in line_pts:
                if tuple(p) not in used_points:
                    points.append(list(p))
                    used_points.add(tuple(p))
        
        # Add 2 more points not on any existing line
        while len(points) < 20:
            p = [random.randint(-10, 10), random.randint(-10, 10)]
            if tuple(p) not in used_points:
                # Check it doesn't create a 4-collinear
                points.append(p)
                used_points.add(tuple(p))
        
        return points
    
    # Main search
    best_points = None
    best_score = 20
    
    # Try multiple initial configurations
    initial_configs = []
    
    # Config 1: line-based
    initial_configs.append(generate_line_based())
    
    # Config 2: grid-based with perturbations
    grid_pts = []
    for x in range(-4, 5, 2):
        for y in range(-3, 4, 2):
            grid_pts.append([x, y])
    initial_configs.append(grid_pts[:20])
    
    # Config 3: random
    random_pts = []
    while len(random_pts) < 20:
        p = random_point()
        if p not in random_pts:
            random_pts.append(p)
    initial_configs.append(random_pts)
    
    for config in initial_configs:
        if elapsed() > DEADLINE - 10:
            break
        # Validate
        if not valid_configuration(config):
            # Try to fix
            continue
        
        s = compute_score_fast(config)
        if s < best_score:
            best_score = s
            best_points = [p[:] for p in config]
        
        # Local search
        time_left = DEADLINE - 10 - elapsed()
        if time_left < 5:
            break
        search_limit = min(start_time + time_left * 0.3, start_time + 30)
        config = local_search(config, search_limit)
        
        s = compute_score_fast(config)
        if s < best_score:
            best_score = s
            best_points = [p[:] for p in config]
    
    # Final validation
    if best_points is None or not valid_configuration(best_points):
        # Fallback: return a valid configuration
        # Use a simple valid configuration
        fallback = []
        for i in range(20):
            x = -10 + i
            y = i * i % 20 - 10
            fallback.append([x, y])
        # Ensure validity
        if not valid_configuration(fallback):
            # Very simple: points on a curve but perturbed
            fallback = [[x, x*x + x] for x in range(-10, 10)]
        return fallback
    
    return best_points
# EVOLVE-BLOCK-END
