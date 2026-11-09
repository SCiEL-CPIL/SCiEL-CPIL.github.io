import time
import random

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 330s limit

    # Pre-generate all grid points and index them
    points = [(x, y) for x in range(N) for y in range(N)]
    M = N * N
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # Pre-compute squared distances between all pairs
    # Using a dictionary of dictionaries or a flat list of lists.
    # Given M=4096, a 4096x4096 matrix is ~16M entries. In Python, this is heavy.
    # Better to compute on the fly or use a sparse structure?
    # Actually, distance calculation is very fast. Let's see if we can optimize the check.
    # The bottleneck is checking O(K^2) pairs for each new point P.
    # Optimization: For a set S, maintain a dictionary `dist_counts` or similar?
    # No, the condition is: no three points P, A, B in S form isosceles.
    # This means for any new point P added to S, for every pair (A, B) in S:
    # dist(P, A) != dist(P, B)
    # dist(P, A) != dist(A, B)
    # dist(P, B) != dist(A, B)
    
    # Let's use the strategy from Inspiration 1 which uses sets of distances.
    # For each point p in S, maintain a set D[p] of squared distances to other points in S.
    # When adding new point P:
    # 1. Check if any distance from P to existing points is repeated (apex at P).
    #    i.e., len({dist(P, s) for s in S}) == len(S)
    # 2. For each s in S, check if dist(P, s) is in D[s].
    #    If dist(P, s) == dist(s, t) for some t in S, then triangle P-s-t is isosceles with apex s.
    #    Note: If dist(P, s) == dist(P, t), that's case 1.
    #    If dist(P, s) == dist(s, t), that's case 2 (apex at s).
    #    Are there other cases?
    #    Triangle P, A, B. Sides: PA, PB, AB.
    #    Isosceles if PA=PB (apex P), PA=AB (apex A), or PB=AB (apex B).
    #    
    #    So the check for adding P to S is:
    #    - For all A, B in S:
    #       - dist(P, A) != dist(P, B)  (Apex P)
    #       - dist(P, A) != dist(A, B)  (Apex A)
    #       - dist(P, B) != dist(A, B)  (Apex B)
    #    
    #    The Inspiration 1 logic:
    #    `dists_from_p = set()`
    #    `for s in S:`
    #       `d = sq_dist(p, s)`
    #       `if d in dists_from_p: return False`  # Apex P case
    #       `dists_from_p.add(d)`
    #       `if d in D[s]: return False`          # Apex s case (dist(P,s) == dist(s,t) for some t)
    #    
    #    This covers Apex P and Apex s. Does it cover Apex t?
    #    If dist(P, B) == dist(A, B), then when we iterate s=A, we check `d = dist(P, A)`.
    #    We check `if d in D[A]`. D[A] contains dist(A, t) for all t in S.
    #    If dist(A, B) == dist(P, B), then dist(A, B) is in D[A] only if B is in S (which it is).
    #    Wait. The condition `if d in D[s]` checks if `dist(P, s)` equals any distance from `s` to another point in S.
    #    If `dist(P, A) == dist(B, A)`, then `dist(P, A)` is in `D[A]` because `B` is in `S`.
    #    So yes, it covers the case where A is the apex.
    #    By symmetry, it covers all apex cases.
    
    def sq_dist(i, j):
        x1, y1 = points[i]
        x2, y2 = points[j]
        dx = x1 - x2
        dy = y1 - y2
        return dx*dx + dy*dy

    def try_add(S, D, p):
        """
        Returns True if p can be added.
        Updates S and D if successful.
        S is a set of point indices.
        D is a dict: index -> set of squared distances to other points in S.
        """
        if p in S:
            return False
        
        valid = True
        dists_from_p = set()
        
        # Iterate through existing points in S
        for s in S:
            d = sq_dist(p, s)
            
            # Check Apex P: is this distance already seen from p?
            if d in dists_from_p:
                valid = False
                break
            
            dists_from_p.add(d)
            
            # Check Apex s: is this distance equal to a distance from s to another point?
            if d in D[s]:
                valid = False
                break
        
        if valid:
            S.add(p)
            D[p] = dists_from_p
            # Update D for all existing points s
            for s in S:
                if s != p:
                    D[s].add(sq_dist(p, s))
            return True
        return False

    def remove_point(S, D, p):
        """Remove point p from S and update D."""
        if p not in S:
            return
        # Calculate distances from p to all other points in S
        dists_to_remove = set()
        for s in S:
            if s != p:
                dists_to_remove.add(sq_dist(p, s))
        
        # Remove these distances from D[s] for all s in S
        for s in S:
            if s != p:
                D[s].difference_update(dists_to_remove)
        
        S.discard(p)
        del D[p]

    def greedy_construct(order_indices, time_limit):
        """
        Greedily add points from order_indices to an empty set.
        Returns (S, D).
        """
        S = set()
        D = {}
        
        for p in order_indices:
            if time.time() > time_limit:
                break
            try_add(S, D, p)
            
        return S, D

    best_S = set()
    best_D = {}
    best_size = 0

    # Phase 1: Random Greedy Restarts
    # Use a fixed seed for reproducibility if needed, but random restarts benefit from randomness.
    # The task says "search score is value/seed". Usually implies the seed is fixed by evaluator or we set it.
    # I'll use a fixed seed for local consistency, but the algorithm is stochastic.
    random.seed(42)
    
    t_phase1_end = start_time + 60.0
    
    while time.time() < t_phase1_end:
        order = list(range(M))
        random.shuffle(order)
        
        S, D = greedy_construct(order, deadline)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
            best_D = {k: set(v) for k, v in D.items()}
            
            # If we found a very large set, we might want to stop early?
            # Let's continue to see if we can do better.

    # Phase 2: Local Search / Perturbation on the best solution
    # We try to add points, and if stuck, remove some and try again.
    
    S = best_S.copy()
    D = {k: set(v) for k, v in best_D.items()}
    
    no_improve = 0
    max_no_improve = 50 # Reduce threshold to allow more restarts/perturbations
    
    # Pre-compute list of all points not in S? 
    # It changes as we add/remove.
    
    while time.time() < deadline - 5.0:
        improved = False
        
        # Try to add any point
        # To avoid bias, shuffle candidates
        # Generating candidates list [i for i in range(M) if i not in S] is O(M).
        # M=4096, so this is fast.
        
        candidates = [i for i in range(M) if i not in S]
        random.shuffle(candidates)
        
        added_any = False
        for p in candidates:
            if try_add(S, D, p):
                improved = True
                added_any = True
                # Update best if improved
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = S.copy()
                    best_D = {k: set(v) for k, v in D.items()}
                break # Add one, then restart loop to re-evaluate candidates
        
        if not improved:
            # Stuck. Perturb.
            if len(S) > 0:
                # Remove 1 to 3 random points
                remove_count = random.randint(1, 3)
                if remove_count > len(S):
                    remove_count = len(S)
                to_remove = random.sample(list(S), remove_count)
                for r in to_remove:
                    remove_point(S, D, r)
                
                no_improve += 1
                if no_improve > max_no_improve:
                    # Full restart
                    order = list(range(M))
                    random.shuffle(order)
                    S, D = greedy_construct(order, deadline)
                    no_improve = 0
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = S.copy()
                        best_D = {k: set(v) for k, v in D.items()}
            else:
                # Empty set, restart
                order = list(range(M))
                random.shuffle(order)
                S, D = greedy_construct(order, deadline)
                no_improve = 0
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = S.copy()
                    best_D = {k: set(v) for k, v in D.items()}
        else:
            no_improve = 0

    # Convert best_S (set of indices) to list of points
    return [points[i] for i in best_S]
# EVOLVE-BLOCK-END
