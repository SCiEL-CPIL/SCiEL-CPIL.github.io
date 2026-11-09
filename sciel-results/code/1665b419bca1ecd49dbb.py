"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import defaultdict

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 280  # Keep 50s margin for 330s limit

    all_points = [(x, y) for x in range(N) for y in range(N)]
    num_points = N * N

    def sq_dist(p1, p2):
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

    def can_add_fast(S_set, dist_sets, p):
        """
        Check if point p can be added to set S without creating an isosceles triangle.
        dist_sets: dict mapping each point in S to a set of squared distances to other points in S.
        Returns True if p can be added.
        """
        if len(S_set) < 2:
            return True
        
        # Case 1: p is the apex. Check if any two points in S are equidistant from p.
        dists_from_p = set()
        for s in S_set:
            d = sq_dist(p, s)
            if d in dists_from_p:
                return False
            dists_from_p.add(d)
        
        # Cases 2 & 3: For each s in S, check if d(p,s) equals d(s,t) for some t in S, t != s.
        # This is equivalent to: for each s in S, check if d(p,s) is in dist_sets[s].
        for s in S_set:
            d_ps = sq_dist(p, s)
            if d_ps in dist_sets[s]:
                return False
        
        return True

    def add_point(S_set, dist_sets, p):
        """Add point p to S_set and update dist_sets."""
        S_set.add(p)
        dist_sets[p] = set()
        for s in S_set:
            if s != p:
                d = sq_dist(p, s)
                dist_sets[p].add(d)
                dist_sets[s].add(d)

    def remove_point(S_set, dist_sets, p):
        """Remove point p from S_set and update dist_sets."""
        for s in S_set:
            if s != p:
                d = sq_dist(p, s)
                dist_sets[s].discard(d)
        del dist_sets[p]
        S_set.discard(p)

    def greedy_construct(order):
        """Greedy construction given an order of points to try."""
        S = set()
        dist_sets = {}
        for p in order:
            if time.time() > deadline - 30:
                break
            if can_add_fast(S, dist_sets, p):
                add_point(S, dist_sets, p)
        return S, dist_sets

    # Multiple greedy restarts with different orderings
    best_set = set()
    best_dist_sets = {}
    best_size = 0

    # Try several different random orderings
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        # Create a shuffled order
        order = all_points[:]
        random.shuffle(order)
        
        S, dist_sets = greedy_construct(order)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            best_dist_sets = {k: v.copy() for k, v in dist_sets.items()}
    
    # Local search on the best solution
    S = best_set
    dist_sets = best_dist_sets
    no_improve_count = 0
    max_no_improve = 30

    while time.time() < deadline - 10:
        improved = False
        
        # Try to add a random point
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            p = random.choice(all_points)
            if p not in S:
                if can_add_fast(S, dist_sets, p):
                    add_point(S, dist_sets, p)
                    improved = True
                    no_improve_count = 0
                    break
        
        # Try to swap: remove one point, add another
        if not improved and len(S) > 0:
            for _ in range(50):
                if time.time() > deadline - 10:
                    break
                remove_p = random.choice(list(S))
                add_p = random.choice(all_points)
                if add_p in S or add_p == remove_p:
                    continue
                # Temporarily remove remove_p
                remove_point(S, dist_sets, remove_p)
                if can_add_fast(S, dist_sets, add_p):
                    add_point(S, dist_sets, add_p)
                    improved = True
                    no_improve_count = 0
                else:
                    # Restore remove_p
                    add_point(S, dist_sets, remove_p)
                break  # Only try one swap per iteration
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Random restart: rebuild with different randomization
                if time.time() > deadline - 30:
                    break
                order = all_points[:]
                random.shuffle(order)
                S, dist_sets = greedy_construct(order)
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S.copy()
                    best_dist_sets = {k: v.copy() for k, v in dist_sets.items()}
                no_improve_count = 0
        else:
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
                best_dist_sets = {k: v.copy() for k, v in dist_sets.items()}

    return list(best_set)
# EVOLVE-BLOCK-END
