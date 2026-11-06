"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from collections import defaultdict

# EVOLVE-BLOCK-START

def _collinearity_stats(points):
    """
    Given a list of 20 points, compute:
    - count4: number of lines with >= 4 points
    - count3: number of lines with exactly 3 points
    - max_line: maximum number of points on any line
    """
    n = len(points)
    # Map from normalized line to set of point indices
    line_map = defaultdict(set)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            # Normalize the direction
            g = math.gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            else:
                # This shouldn't happen for distinct points
                continue
            # Ensure consistent sign: dx > 0 or (dx == 0 and dy > 0)
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            # Line is determined by point i and direction (dx, dy)
            # But we need to identify the line uniquely.
            # The line can be represented by (dx, dy, c) where c = dy*xi - dx*yi
            # Actually, for a line with direction (dx, dy), the normal is (dy, -dx) or (-dy, dx).
            # Let's use the form: A*x + B*y = C where (A, B) is normal to (dx, dy).
            # Normal can be (dy, -dx). So C = dy*xi - dx*yi.
            # We need to normalize (A, B, C) so that gcd(A, B, C) = 1 and sign is consistent.
            A = dy
            B = -dx
            C = A * xi + B * yi  # Wait, the line equation is A(x - xi) + B(y - yi) = 0 => Ax + By = A*xi + B*yi
            # Let's re-verify: direction (dx, dy). Normal (A, B) = (dy, -dx).
            # Dot product of normal and direction: dy*dx + (-dx)*dy = 0. Correct.
            # Line: dy*(x - xi) - dx*(y - yi) = 0 => dy*x - dx*y = dy*xi - dx*yi.
            # So A = dy, B = -dx, C = dy*xi - dx*yi.
            A = dy
            B = -dx
            C = A * xi + B * yi
            # Normalize (A, B, C)
            g_abc = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g_abc > 0:
                A //= g_abc
                B //= g_abc
                C //= g_abc
            # Sign consistency: make A positive, or if A is 0 make B positive, or if A,B are 0 make C positive (impossible for line)
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            line_map[key].add(i)
            line_map[key].add(j)
    
    count4 = 0
    count3 = 0
    max_line = 0
    for key, pts in line_map.items():
        size = len(pts)
        if size >= 4:
            count4 += 1
        elif size == 3:
            count3 += 1
        if size > max_line:
            max_line = size
    return count4, count3, max_line

def _is_valid(points):
    """Check if points have no 4 collinear."""
    n = len(points)
    line_map = defaultdict(set)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            g = math.gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            A = dy
            B = -dx
            C = A * xi + B * yi
            g_abc = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g_abc > 0:
                A //= g_abc
                B //= g_abc
                C //= g_abc
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            line_map[key].add(i)
            line_map[key].add(j)
            if len(line_map[key]) >= 4:
                return False
    return True

def _score_subset(points):
    """
    Exact branch and bound for largest subset with no three collinear.
    This is the independence number of the 3-uniform hypergraph of collinear triples.
    """
    n = len(points)
    # Precompute all triples that are collinear
    # A triple is collinear if any pair is on a line with the third.
    # Actually, we can just check if any 3 points are collinear.
    # For efficiency, let's find all sets of points on each line.
    line_map = defaultdict(set)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            g = math.gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            A = dy
            B = -dx
            C = A * xi + B * yi
            g_abc = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g_abc > 0:
                A //= g_abc
                B //= g_abc
                C //= g_abc
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            line_map[key].add(i)
            line_map[key].add(j)
    
    # Collect all forbidden sets (subsets of size >= 3 on a line)
    # Actually, we just need to know which triples are collinear.
    # A subset is valid if it contains no collinear triple.
    # We can use a standard maximum independent set algorithm on the conflict graph?
    # No, it's a hypergraph. But we can convert to a graph:
    # Two vertices are adjacent if they are in a collinear triple? No.
    # The condition is: no 3 vertices are collinear.
    # This is equivalent to: for every line, the subset contains at most 2 points from that line.
    
    # Let's use a simple branch and bound.
    # Order points. Try to include or exclude.
    # We can prune if the current count + remaining points < best found.
    
    # Precompute for each point, the set of other points that are collinear with it and a third point?
    # Actually, it's easier to just check validity incrementally.
    
    # Let's try a different approach: since n=20 is small, we can use a MILP or just a clever search.
    # But we need it to be fast.
    # Let's use the property: a subset is valid iff for every line, it has <= 2 points.
    
    # We can compute the maximum subset by iterating over all subsets? 2^20 = 1M. That's feasible.
    # But checking each subset takes O(n^2) or O(n^3). 1M * 20^3 = 8e9. Too slow.
    
    # We need a better way.
    # Let's use the branch and bound on the lines.
    # Or, we can use the fact that the score is likely small (e.g., 10-15).
    
    # Let's implement a simple branch and bound:
    # Sort points.
    # At each step, decide to include or exclude point i.
    # Maintain the set of included points.
    # Check if adding point i creates a collinear triple with any two existing points.
    # If so, we cannot include i.
    # Prune if current_size + (n - i) <= best.
    
    # To check if adding i creates a collinear triple:
    # For each pair (j, k) in included, check if i, j, k are collinear.
    # This is O(k^2) per step, where k is current size.
    # Total complexity is roughly O(2^n * n^2). 2^20 * 400 = 4e8. Might be too slow in Python.
    
    # Let's optimize:
    # Precompute for each point i, the set of pairs (j, k) such that i, j, k are collinear.
    # Actually, for each point i, and each other point j, find all k such that i, j, k are collinear.
    # This is the line containing i and j.
    
    # Let's precompute lines for each pair.
    # line_for_pair[(i, j)] = set of k such that i, j, k are collinear.
    
    line_for_pair = {}
    for i in range(n):
        for j in range(i + 1, n):
            line_for_pair[(i, j)] = set()
            # Find all k on the line through i and j
            # We already have line_map. We can use it.
            # But line_map is built from pairs.
            # Let's just use the line_map we built earlier.
            # For a pair (i, j), find the line key.
            xi, yi = points[i]
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            g = math.gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            A = dy
            B = -dx
            C = A * xi + B * yi
            g_abc = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g_abc > 0:
                A //= g_abc
                B //= g_abc
                C //= g_abc
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            # line_map[key] is the set of points on this line
            line_for_pair[(i, j)] = line_map.get(key, set())
    
    # Now, in the search:
    # When we consider adding point i to the current set S:
    # We need to check if there exist j, k in S such that i, j, k are collinear.
    # This is equivalent to: for any j in S, if i and j are on a line that contains some k in S, then it's invalid.
    # Actually, it's simpler: for any pair (j, k) in S, if i is on the line through j and k, then invalid.
    # Or: for any j in S, let L be the line through i and j. If L contains any k in S (other than i, j), then invalid.
    # Since i is not in S yet, we just check if L contains any point in S.
    # Wait, if i and j are in the new set, and k is in S, and i, j, k are collinear, then the triple (i, j, k) is in the new set.
    # So we need to check: for each j in S, is there a k in S such that i, j, k are collinear?
    # This can be checked by: for each j in S, look at the line through i and j. If that line contains any point from S (other than j), then it's invalid.
    # Actually, if the line through i and j contains k in S, then i, j, k are collinear.
    # So we can precompute for each i, and each j, the set of points on the line through i and j.
    # Then, for a given S, adding i is invalid if for any j in S, (line_through_i_j & S) is non-empty (excluding j, but j is in S, so we need to check if there's another point).
    # Actually, if line_through_i_j contains j and k, and j, k are in S, then adding i makes i, j, k collinear.
    # So we need to check if (line_through_i_j - {j}) & S is non-empty.
    
    # Let's implement the search.
    best = 0
    # We can use a recursive function or a stack.
    # To speed up, we can order points by some heuristic.
    # Let's just use the natural order for now.
    
    def search(idx, current_set, current_size):
        nonlocal best
        if current_size + (n - idx) <= best:
            return
        if idx == n:
            if current_size > best:
                best = current_size
            return
        
        # Option 1: Exclude point idx
        search(idx + 1, current_set, current_size)
        
        # Option 2: Include point idx
        # Check if valid
        valid = True
        for j in current_set:
            # Check line through idx and j
            key = line_for_pair.get((idx, j)) or line_for_pair.get((j, idx))
            if key is not None:
                # Check if any point in current_set (other than j) is on this line
                # Actually, we just need to check if current_set intersects with key, excluding j.
                # But key is a set of indices.
                # If there is any k in current_set such that k != j and k is in key, then invalid.
                # We can check: if (key & current_set) - {j} is not empty.
                # But key is a set, current_set is a set.
                # Let's just iterate.
                for k in current_set:
                    if k != j and k in key:
                        valid = False
                        break
            if not valid:
                break
        
        if valid:
            current_set.add(idx)
            search(idx + 1, current_set, current_size + 1)
            current_set.remove(idx)
    
    search(0, set(), 0)
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    random.seed(42)
    
    # Initial population: random points
    # We want to minimize the score (largest subset with no 3 collinear).
    # This means we want to maximize the "density" of collinear triples,
    # while ensuring no 4 are collinear.
    
    # Let's use Simulated Annealing.
    # State: list of 20 points.
    # Energy: score (we want to minimize).
    # But computing score is expensive.
    # Let's use a proxy: count3 (number of 3-collinearities).
    # Maximizing count3 should correlate with minimizing the score.
    # Also, we must maintain count4 == 0.
    
    # Let's start with a random valid configuration.
    # Generate random points, check validity, if not, perturb.
    
    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]
    
    def generate_random_config():
        pts = []
        while len(pts) < 20:
            p = random_point()
            if p not in pts:
                pts.append(p)
        return pts
    
    # Find an initial valid config
    config = generate_random_config()
    while not _is_valid(config):
        # Perturb a random point
        idx = random.randint(0, 19)
        config[idx] = random_point()
        # Ensure distinct
        if config.count(config[idx]) > 1:
            config[idx] = random_point()
    
    # Compute initial score
    current_score = _score_subset(config)
    current_count3, _, _ = _collinearity_stats(config)
    
    best_config = config[:]
    best_score = current_score
    best_count3 = current_count3
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    while T > min_T and time.time() < deadline:
        # Propose a move: move one point to a random location
        idx = random.randint(0, 19)
        old_point = config[idx]
        new_point = random_point()
        
        # Ensure distinct
        if new_point in config:
            continue
        
        config[idx] = new_point
        
        # Check validity
        if _is_valid(config):
            new_count3, _, _ = _collinearity_stats(config)
            # We want to maximize count3
            delta_count3 = new_count3 - current_count3
            
            if delta_count3 > 0 or random.random() < math.exp(delta_count3 / T):
                # Accept
                current_count3 = new_count3
                # Recompute score only if count3 improved significantly or periodically
                # To save time, we can just update the score if we accept a better count3
                # But the score is what we really care about.
                # Let's compute the score every time we accept a better count3.
                # Or, we can just track the best score found so far.
                # Actually, let's compute the score for the new config.
                new_score = _score_subset(config)
                if new_score < best_score:
                    best_score = new_score
                    best_config = config[:]
                    best_count3 = current_count3
                current_score = new_score
            else:
                # Reject
                config[idx] = old_point
        else:
            # Invalid, reject
            config[idx] = old_point
        
        T *= cooling
    
    # Return the best config found
    return best_config

# EVOLVE-BLOCK-END
