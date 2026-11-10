"""No-isosceles set: maximize number of grid points in [0, N)^2 with no isosceles triangle."""
import time
import random

N = 64
TIME_LIMIT = 290  # seconds, leave margin before 330s kill


def construct():
    start_time = time.time()
    
    # We'll use a randomized local search approach.
    # State: a set of points. We try to add points and remove conflicting ones.
    # Key insight: for each pair of points, the "forbidden" third points are those
    # that would form an isosceles triangle with the pair.
    
    # Precompute for each point pair, the set of points that would form an isosceles triangle.
    # Actually, let's think differently: for a set S, a point p is "safe" to add if for every
    # pair (a, b) in S, p does not form an isosceles triangle with a and b.
    # Equivalently: for every pair (a, b) in S, |pa| != |pb| (i.e., p is not equidistant from a and b).
    
    # So p is safe iff for all pairs (a,b) in S, dist(p,a) != dist(p,b).
    # This means p must NOT lie on the perpendicular bisector of any segment ab.
    
    # Efficient check: for a given set S and candidate p, check if there exist a, b in S
    # such that dist2(p,a) == dist2(p,b).
    
    # Let's use a different representation: maintain for each point p, the set of squared distances
    # from p to all points in S. If any distance value appears twice, p conflicts.
    
    # Actually, let's use: for each pair (a,b) in S, compute the perpendicular bisector line.
    # A point p is on that bisector iff dist2(p,a) == dist2(p,b).
    
    # For efficiency, let's maintain: for each candidate point p, the multiset of distances to points in S.
    # p is safe iff all distances are distinct.
    
    # Let's try a greedy + local search approach:
    # 1. Start with random point, greedily add safe points.
    # 2. Try local moves: swap a point with a non-point.
    
    def dist2(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy
    
    def is_safe(p, point_set):
        """Check if p can be added to point_set without creating an isosceles triangle."""
        # For each pair (a, b) in point_set, check if dist2(p,a) == dist2(p,b)
        # Equivalently: the set of distances from p to all points in point_set must have all distinct values.
        dists = set()
        for pt in point_set:
            d = dist2(p, pt)
            if d in dists:
                return False
            dists.add(d)
        return True
    
    def greedy_build(order):
        """Build a set by greedily adding points in the given order."""
        point_set = []
        for p in order:
            if is_safe(p, point_set):
                point_set.append(p)
        return point_set
    
    def try_local_search(point_set, time_budget):
        """Try to improve point_set via local swaps. Returns improved set."""
        ps = list(point_set)
        ps_set = set(ps)
        all_points = [(x, y) for x in range(N) for y in range(N)]
        non_points = [p for p in all_points if p not in ps_set]
        
        improved = True
        while improved and (time.time() - start_time) < time_budget:
            improved = False
            random.shuffle(non_points)
            for new_p in non_points:
                if (time.time() - start_time) >= time_budget:
                    break
                # Try swapping each point in ps with new_p
                for i, old_p in enumerate(ps):
                    # Check if removing old_p and adding new_p keeps the set valid
                    # We need to check: after removing old_p, is new_p safe?
                    # And after adding new_p, is the set still valid?
                    # Actually, we need to check that the new set (ps - {old_p} + {new_p}) is valid.
                    # This means: for all pairs in the new set, no isosceles triangle.
                    # Efficient way: check if new_p is safe with (ps - {old_p}),
                    # and that removing old_p doesn't break anything (it can't, since we're removing).
                    # Wait, that's not quite right. We need the entire set to be valid.
                    # But if ps was valid, and we remove old_p, the remaining set is still valid.
                    # Then we add new_p: we need new_p to be safe with the remaining set.
                    remaining = [pt for j, pt in enumerate(ps) if j != i]
                    if is_safe(new_p, remaining):
                        ps[i] = new_p
                        ps_set.discard(old_p)
                        ps_set.add(new_p)
                        non_points.remove(new_p)
                        non_points.append(old_p)
                        improved = True
                        break
                if improved:
                    break
        return ps
    
    # Main search: multiple restarts with greedy + local search
    random.seed(42)
    best_set = []
    
    while (time.time() - start_time) < TIME_LIMIT:
        # Generate a random order of all points
        all_points = [(x, y) for x in range(N) for y in range(N)]
        random.shuffle(all_points)
        
        # Greedy build
        candidate = greedy_build(all_points)
        
        if len(candidate) > len(best_set):
            best_set = list(candidate)
        
        # Local search improvement
        remaining_time = TIME_LIMIT - (time.time() - start_time)
        if remaining_time > 5:  # only do local search if we have time
            candidate = try_local_search(candidate, start_time + remaining_time - 2)
            if len(candidate) > len(best_set):
                best_set = list(candidate)
    
    return best_set
