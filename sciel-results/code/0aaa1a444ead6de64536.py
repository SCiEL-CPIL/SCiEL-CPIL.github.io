"""Rewritten program for integral point sets in general position."""
# EVOLVE-BLOCK-START
import math
import time
from itertools import combinations
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    def is_perfect_square(n):
        if n < 0:
            return False
        if n == 0:
            return True
        s = int(math.isqrt(n))
        return s * s == n
    
    def dist_sq(p, q):
        return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
    
    def check_integer_distances(points):
        for i, j in combinations(range(len(points)), 2):
            if not is_perfect_square(dist_sq(points[i], points[j])):
                return False
        return True
    
    def no_three_collinear(points):
        n = len(points)
        for i, j, k in combinations(range(n), 3):
            p, q, r = points[i], points[j], points[k]
            # Cross product (q-p) x (r-p)
            if (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]) == 0:
                return False
        return True
    
    def no_four_concyclic(points):
        n = len(points)
        for i, j, k, l in combinations(range(n), 4):
            p, q, r, s = points[i], points[j], points[k], points[l]
            # Circle through p, q, r; check if s is on it
            # Using determinant form
            a = p[0]; b = p[1]; c = p[0]**2 + p[1]**2
            d = q[0]; e = q[1]; f = q[0]**2 + q[1]**2
            g = r[0]; h = r[1]; i_val = r[0]**2 + r[1]**2
            j_val = s[0]; k_val = s[1]; l_val = s[0]**2 + s[1]**2
            
            # 4x4 determinant
            def det4(x1,y1,z1, x2,y2,z2, x3,y3,z3, x4,y4,z4):
                return (x1*(y2*(z3 - z4) - y3*(z2 - z4) + y4*(z2 - z3))
                        - y1*(x2*(z3 - z4) - x3*(z2 - z4) + x4*(z2 - z3))
                        + z1*(x2*(y3 - y4) - x3*(y2 - y4) + x4*(y2 - y3))
                        - 1*(x2*(y3*z4 - y4*z3) - x3*(y2*z4 - y4*z2) + x4*(y2*z3 - y3*z2)))
            
            val = det4(a, b, c, d, e, f, g, h, i_val, j_val, k_val, l_val)
            if val == 0:
                return False
        return True
    
    def validate(points):
        if len(points) < 3:
            return False
        if len(set(points)) != len(points):
            return False
        if not check_integer_distances(points):
            return False
        if not no_three_collinear(points):
            return False
        if not no_four_concyclic(points):
            return False
        return True
    
    def format_output(points, D=1):
        return {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in points]}
    
    # Strategy: Use rational parametrization to generate points with rational pairwise distances
    # Key insight: if we place points on a circle with rational center and radius, 
    # and use rational parametrization, pairwise distances can be rational
    
    # Generate candidate points using rational parametrization of circles
    # For D=1, we work in integer/rational coordinates
    
    # Method 1: Use known constructions
    # Points on a circle centered at origin with rational radius
    # Parametrize: (r*(1-t^2)/(1+t^2), r*2t/(1+t^2)) for rational t
    
    def generate_rational_points_on_circle(r_num, r_den, t_values):
        """Generate points on circle of radius r_num/r_den using rational parametrization."""
        points = []
        r = Fraction(r_num, r_den)
        for t in t_values:
            # Point on unit circle: ((1-t^2)/(1+t^2), 2t/(1+t^2))
            # Scale by r
            x = r * (1 - t**2) / (1 + t**2)
            y = r * 2 * t / (1 + t**2)
            # Convert to integers if possible
            if x.denominator == 1 and y.denominator == 1:
                points.append((int(x), int(y)))
            else:
                # Keep as fractions for now, scale later
                points.append((x, y))
        return points
    
    # Method 2: Search for integer points with integer pairwise distances
    # Build a larger pool and search for cliques
    
    # Generate pool of integer points
    pool = []
    for x in range(-60, 61):
        for y in range(-60, 61):
            pool.append((x, y))
    
    n_pool = len(pool)
    
    # Build adjacency: edge if integer distance
    adj = [set() for _ in range(n_pool)]
    dist_cache = {}
    
    for i in range(n_pool):
        if time.time() > deadline - 40:
            break
        for j in range(i + 1, n_pool):
            d = dist_sq(pool[i], pool[j])
            if is_perfect_square(d):
                adj[i].add(j)
                adj[j].add(i)
    
    # Backtracking search for clique
    def search_clique(target_size, candidates, current, depth, best):
        if depth == target_size:
            pts = [pool[idx] for idx in current]
            if no_three_collinear(pts) and no_four_concyclic(pts):
                best.append(pts[:])
            return True
        if time.time() > deadline - 15:
            return False
        
        # Find common neighbors of all points in current
        if depth == 0:
            common = set(candidates)
        else:
            common = adj[current[-1]].copy()
            for idx in current[:-1]:
                common &= adj[idx]
            common &= set(candidates)
        
        # Sort by degree for better pruning
        common = sorted(common, key=lambda x: len(adj[x]), reverse=True)
        
        for idx in common:
            if idx in current:
                continue
            # Prune
            remaining_count = len([x for x in common if x > idx])
            if remaining_count < target_size - depth - 1:
                break
            current.append(idx)
            found = search_clique(target_size, candidates, current, depth + 1, best)
            current.pop()
            if found and len(best) >= 2:
                return True
        return False
    
    # Try to find 7-point clique first (ambitious but worth trying)
    found_7 = []
    if time.time() < deadline - 50:
        degrees = [(len(adj[i]), i) for i in range(n_pool)]
        degrees.sort(reverse=True)
        top_indices = [i for _, i in degrees[:300]]
        search_clique(7, top_indices, [], 0, found_7)
    
    if found_7:
        pts = found_7[0]
        if validate(pts):
            return format_output(pts)
    
    # Try 6-point clique
    found_6 = []
    if time.time() < deadline - 25:
        degrees = [(len(adj[i]), i) for i in range(n_pool)]
        degrees.sort(reverse=True)
        top_indices = [i for _, i in degrees[:300]]
        search_clique(6, top_indices, [], 0, found_6)
    
    if found_6:
        pts = found_6[0]
        if validate(pts):
            return format_output(pts)
    
    # Try 5-point clique
    found_5 = []
    if time.time() < deadline - 10:
        degrees = [(len(adj[i]), i) for i in range(n_pool)]
        degrees.sort(reverse=True)
        top_indices = [i for _, i in degrees[:300]]
        search_clique(5, top_indices, [], 0, found_5)
    
    if found_5:
        pts = found_5[0]
        if validate(pts):
            return format_output(pts)
    
    # Fallback: hardcoded 6-point set
    hardcoded_6 = [(0, 0), (25, 0), (7, 24), (0, 39), (24, 42), (7, 72)]
    if validate(hardcoded_6):
        return format_output(hardcoded_6)
    
    # Fallback: 5-point set
    fallback_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    if validate(fallback_5):
        return format_output(fallback_5)
    
    # Last resort: 3-point set
    return format_output([(0, 0), (3, 0), (0, 4)])
# EVOLVE-BLOCK-END
