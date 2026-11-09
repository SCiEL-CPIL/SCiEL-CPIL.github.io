import random
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Helper: normalize direction vector for collinearity checks
    def _norm(dx, dy):
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            return (0, 0)
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        return (dx, dy)

    # Check if any 4 points are collinear (hard constraint)
    def _has_four_collinear(pts):
        n = len(pts)
        for i in range(n):
            xi, yi = pts[i]
            seen = {}
            for j in range(i + 1, n):
                dx = pts[j][0] - xi
                dy = pts[j][1] - yi
                key = _norm(dx, dy)
                c = seen.get(key, 0) + 1
                if c >= 3:
                    return True
                seen[key] = c
        return False

    # Count collinear triples (surrogate for score; more triples => smaller max no-three subset)
    def _count_triples(pts):
        n = len(pts)
        line_map = {}
        for i in range(n):
            xi, yi = pts[i]
            for j in range(i + 1, n):
                dx = pts[j][0] - xi
                dy = pts[j][1] - yi
                key = _norm(dx, dy)
                # Use line equation: dy*x - dx*y = c
                c = dy * xi - dx * yi
                line_key = (key[0], key[1], c)
                if line_key not in line_map:
                    line_map[line_key] = []
                line_map[line_key].append(j)
        
        total = 0
        for line_pts in line_map.values():
            k = len(line_pts)
            if k >= 2:
                total += k * (k - 1) // 2
        return total

    # Exact branch and bound for largest subset with no three collinear
    def _max_no_three_collinear(pts):
        n = len(pts)
        # Precompute: for each pair, which other points are collinear with them
        # collinear_with[i][j] = set of k != i,j such that i,j,k collinear
        collinear_with = [[set() for _ in range(n)] for _ in range(n)]
        for i in range(n):
            xi, yi = pts[i]
            for j in range(i + 1, n):
                dx = pts[j][0] - xi
                dy = pts[j][1] - yi
                g = gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx //= g
                dy //= g
                for k in range(n):
                    if k == i or k == j:
                        continue
                    dxk = pts[k][0] - xi
                    dyk = pts[k][1] - yi
                    # Check if (dxk, dyk) is parallel to (dx, dy)
                    if dx * dyk - dy * dxk == 0:
                        collinear_with[i][j].add(k)
                        collinear_with[j][i].add(k)

        # Branch and bound
        best = [0]
        
        def backtrack(remaining, current_size, last_two):
            if time.time() > deadline - 5:
                return
            if not remaining:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Upper bound: current_size + len(remaining)
            if current_size + len(remaining) <= best[0]:
                return
            
            # Try to add as many as possible greedily first for a better bound
            # Simple bound: current_size + len(remaining)
            
            # Choose the point to branch on (first in remaining)
            # We can choose which point to try adding first
            # Strategy: try adding points, and if it conflicts, skip
            
            # For efficiency, pick a point that is "most constrained"
            # But for simplicity, just use the first one
            p = remaining[0]
            rest = remaining[1:]
            
            # Option 1: Don't include p
            backtrack(rest, current_size, last_two)
            
            # Option 2: Include p
            # Check if p is collinear with last_two
            if last_two:
                i, j = last_two
                if p in collinear_with[i][j]:
                    return  # Can't include p
            # Also need to check if including p creates a triple with any two in remaining
            # Actually, the constraint is: no three collinear in the final set.
            # If we include p, we must ensure that for any two points q, r in the final set,
            # they are not collinear with p.
            # This is complex. Let's use a different approach.
            
            # Alternative: maintain the set of points chosen.
            # When adding p, we must remove from 'remaining' any point q such that
            # q is collinear with p and some other chosen point.
            # This is getting complex for branch and bound.
            
            # Let's use a simpler but effective heuristic:
            # The score is the size of the largest subset with no 3 collinear.
            # This is equivalent to the independence number of a 3-uniform hypergraph.
            # Branch and bound on hypergraphs is hard.
            
            # Let's just use a greedy upper bound:
            # The number of points we can still add is at most the number of points
            # that don't form a triple with any two already chosen.
            # But this requires knowing the chosen set.
            
            # Given the time constraint, let's just do a simple search
            # and rely on the surrogate (triple count) for optimization.
            # The exact score is only needed for the final answer, but we can
            # approximate it or compute it quickly for small n.
            
            # For now, let's just return a greedy estimate if exact is too slow.
            # Actually, let's implement a proper branch and bound with memoization
            # or a simpler recursive approach.
            
            # Re-implementing backtrack properly:
            # State: (frozen remaining, last_two, current_size)
            # But last_two is not sufficient because we need to know all chosen points.
            # Let's pass the set of chosen points.
            pass

        # Given the complexity, let's use a simpler method:
        # The maximum subset with no 3 collinear can be found by trying all subsets?
        # No, 2^20 is too large.
        
        # Let's use the fact that if we have many collinear triples, the max subset is small.
        # We can use a greedy algorithm to find a lower bound, and a simple upper bound.
        
        # For the purpose of this task, the exact score is computed by the evaluator.
        # We just need to minimize it. The surrogate (triple count) is a good proxy.
        # Let's focus on maximizing the triple count.
        
        return 0  # Placeholder, not used in optimization

    # Initial random valid configuration
    best_points = None
    best_triple_count = -1

    # Generate initial points
    while time.time() < deadline - 10:
        pts = []
        used = set()
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in used:
                used.add((x, y))
                pts.append((x, y))
        
        if not _has_four_collinear(pts):
            triple_count = _count_triples(pts)
            if best_points is None or triple_count > best_triple_count:
                best_points = pts
                best_triple_count = triple_count
                
        if best_points is None:
            continue

        # Local Search: Move one point to increase triple count
        current_points = best_points[:]
        current_triples = best_triple_count
        
        # Try to improve current best
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            
            idx = random.randint(0, 19)
            
            # Try moving this point
            for _ in range(20):
                new_x = random.randint(-100, 100)
                new_y = random.randint(-100, 100)
                
                # Check if point already exists (excluding idx)
                existing = set()
                for k in range(20):
                    if k != idx:
                        existing.add((current_points[k][0], current_points[k][1]))
                
                if (new_x, new_y) in existing:
                    continue
                
                new_points = current_points[:]
                new_points[idx] = (new_x, new_y)
                
                if _has_four_collinear(new_points):
                    continue
                
                new_triples = _count_triples(new_points)
                
                # We want to MAXIMIZE triples to minimize the max independent set
                if new_triples > current_triples:
                    current_points = new_points
                    current_triples = new_triples
                    break
                elif new_triples == current_triples:
                    # Accept equal with low prob to escape local minima
                    if random.random() < 0.1:
                        current_points = new_points
                        break

        if current_triples > best_triple_count:
            best_points = current_points
            best_triple_count = current_triples

    return [[p[0], p[1]] for p in best_points]
