"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    # Generate Pythagorean triples for distance generation
    def gen_pyth_triples(max_n=200):
        triples = set()
        for m in range(2, max_n):
            for n in range(1, m):
                if (m - n) % 2 == 1 and math.gcd(m, n) == 1:
                    a = m*m - n*n
                    b = 2*m*n
                    c = m*m + n*n
                    if c <= 10**6:
                        triples.add((a, b, c))
                        triples.add((b, a, c))
        return triples

    def is_square(n):
        if n < 0:
            return False
        r = int(math.isqrt(n))
        return r * r == n

    def dist_sq(p1, p2):
        """Return (num, den) for squared distance between two points (x1,y1),(x2,y2) with D=1."""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy

    def check_collinear(points):
        """Check no three points are collinear. Returns True if valid."""
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    if (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1) != 0:
                        continue
                    else:
                        return False
        return True

    def check_concyclic(points):
        """Check no four points are concyclic. Returns True if valid."""
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        x1, y1 = points[i]
                        x2, y2 = points[j]
                        x3, y3 = points[k]
                        x4, y4 = points[l]
                        # Circle through first 3 points, check if 4th is on it
                        # Using determinant form
                        d = (x1*(y2*y3 - y2*y4 - y1*y3 + y1*y4)
                           - y1*(x2*y3 - x2*y4 - x1*y3 + x1*y4)
                           + (x1*x1 + y1*y1)*(x2*y3 - x2*y4 - x1*y3 + x1*y4)
                           - (x2*x2 + y2*y2)*(x1*y3 - x1*y4 - x3*y1 + x3*y4)
                           + (x3*x3 + y3*y3)*(x1*y2 - x1*y4 - x2*y1 + x2*y4)
                           - (x4*x4 + y4*y4)*(x1*y2 - x1*y3 - x2*y1 + x2*y3))
                        if d != 0:
                            continue
                        else:
                            return False
        return True

    # Start with the known 5-point set
    best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_count = 5

    # Try to add points using lattice points with rational distances
    # Strategy: for each existing point, find lattice points at Pythagorean distances
    # that don't create collinear or concyclic violations

    def try_add_point(points, candidate):
        """Try adding candidate to points. Returns True if valid."""
        new_points = points + [candidate]
        # Check all pairwise distances are rational (for D=1, integer distances squared)
        n = len(new_points)
        for i in range(n):
            for j in range(i+1, n):
                dsq = dist_sq(new_points[i], new_points[j])
                if not is_square(dsq):
                    return False
        # Check no three collinear
        if not check_collinear(new_points):
            return False
        # Check no four concyclic
        if not check_concyclic(new_points):
            return False
        return True

    # Generate candidate points
    def generate_candidates(existing_points, max_coord=500):
        """Generate candidate points at rational distances from existing points."""
        candidates = set()
        triples = gen_pyth_triples(300)
        
        for (px, py) in existing_points:
            for (a, b, c) in triples:
                # Points at distance c from (px, py)
                # Using the triple components
                candidates.add((px + a, py + b))
                candidates.add((px + a, py - b))
                candidates.add((px - a, py + b))
                candidates.add((px - a, py - b))
                candidates.add((px + b, py + a))
                candidates.add((px + b, py - a))
                candidates.add((px - b, py + a))
                candidates.add((px - b, py - a))
        
        # Filter to reasonable range and remove duplicates with existing
        existing_set = set(existing_points)
        filtered = []
        for (x, y) in candidates:
            if -max_coord <= x <= max_coord and -max_coord <= y <= max_coord:
                if (x, y) not in existing_set:
                    filtered.append((x, y))
        return filtered

    # Incremental search: try to build up from the 5-point set
    current = list(best_points)
    
    while time.time() < deadline:
        if len(current) >= 12:
            break
        
        # Generate candidates
        cands = generate_candidates(current, max_coord=600)
        
        # Sort candidates by "goodness" - prefer those far from existing points
        def candidate_score(cand):
            min_dist = float('inf')
            for p in current:
                d = dist_sq(cand, p)
                if d < min_dist:
                    min_dist = d
            return -min_dist  # prefer farther
        
        cands.sort(key=candidate_score)
        
        # Try adding candidates
        added = False
        for cand in cands:
            if time.time() > deadline:
                break
            if try_add_point(current, cand):
                current.append(cand)
                added = True
                if len(current) > best_count:
                    best_points = list(current)
                    best_count = len(current)
                break  # Restart candidate generation with updated set
        
        if not added:
            # Try with a different existing point as anchor or different range
            # Try random perturbation of the search
            import random
            random.seed(len(current) * 7919)
            cands = generate_candidates(current, max_coord=800)
            random.shuffle(cands)
            for cand in cands[:200]:
                if time.time() > deadline:
                    break
                if try_add_point(current, cand):
                    current.append(cand)
                    if len(current) > best_count:
                        best_points = list(current)
                        best_count = len(current)
                    break
            else:
                # Stuck - try a different starting configuration
                # Try some known configurations or restart
                break

    # Format the result
    points = best_points
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}
# EVOLVE-BLOCK-END
