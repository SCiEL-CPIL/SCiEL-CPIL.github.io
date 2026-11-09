import time
import random
import math
from itertools import combinations

N = 16
TOTAL = N * N  # 256

def construct():
    start_time = time.time()
    deadline = start_time + 295  # Safety margin before 330s kill
    
    # Index points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    point_to_idx = {p: i for i, p in enumerate(all_points)}
    
    # Precompute squared distances
    def d2(i, j):
        x1, y1 = all_points[i]
        x2, y2 = all_points[j]
        return (x1 - x2) ** 2 + (y1 - y2) ** 2
    
    # Precompute for each pair (i, j), the set of points k such that (i, j, k) is isosceles
    # iso[i][j] = set of k where d(i,k)==d(i,j) or d(j,k)==d(i,j) or d(i,k)==d(j,k)
    # Actually: (i,j,k) is isosceles if d(i,j)==d(i,k) or d(i,j)==d(j,k) or d(i,k)==d(j,k)
    
    # For efficiency, precompute distance from each point to all others
    dist = [[0] * TOTAL for _ in range(TOTAL)]
    for i in range(TOTAL):
        for j in range(TOTAL):
            dist[i][j] = d2(i, j)
    
    # For each point i, precompute which pairs (j, k) would form isosceles with i
    # iso_with[i] = set of (j, k) pairs such that (i, j, k) is isosceles
    # This is expensive: 256 * (256*255/2) ~ 8M checks
    # Let's instead precompute: for each pair (j, k), the set of i such that (i, j, k) is isosceles
    # iso_pair[j][k] = set of i
    
    # Actually, let's use a different approach for speed:
    # For each point i, and each other point j, precompute the "circle" of points equidistant from i as j
    # circle[i][j] = set of k such that d(i,k) == d(i,j)
    
    # Better: For checking if adding point p to set S creates an isosceles triangle,
    # we need: for all pairs (a, b) in S, (p, a, b) is not isosceles.
    # (p, a, b) is isosceles iff d(p,a)==d(p,b) or d(p,a)==d(a,b) or d(p,b)==d(a,b)
    # Note: d(a,b)==d(a,b) is always true, so the third condition is always satisfied?
    # Wait no. The three sides are d(p,a), d(p,b), d(a,b).
    # Isosceles means at least two are equal:
    #   d(p,a) == d(p,b)  [p is apex]
    #   d(p,a) == d(a,b)  [a is apex]
    #   d(p,b) == d(a,b)  [b is apex]
    
    # So for a fixed p and a fixed pair (a,b), we can check in O(1) with precomputed distances.
    
    # Strategy: Branch and bound with bitsets
    # Represent the set of "available" points (points that can still be added) as a bitset.
    # When we add a point p, we remove all points q such that for some pair (a,b) in current set,
    # (p, a, b) is isosceles... wait, that's not right.
    
    # Let me think differently. We want a maximal independent set in a 3-uniform hypergraph.
    # Hyperedges = isosceles triples. We want a maximal set with no hyperedge fully contained.
    # And we want to minimize the size.
    
    # Key insight: A set S is maximal no-isosceles iff for every point p not in S,
    # there exist a, b in S such that (p, a, b) is isosceles.
    
    # Approach: 
    # 1. Use randomized greedy to get an initial maximal set.
    # 2. Use local search: try to remove points while maintaining maximality.
    # 3. Use pair-removal: try removing two points at once.
    
    # For speed, precompute for each triple (i,j,k) whether it's isosceles.
    # 256^3 / 6 ≈ 2.8M triples. Feasible.
    # But storing all is expensive. Let's precompute a lookup.
    
    # Precompute: is_iso[i][j][k] using a flat array or dict
    # Actually, let's precompute for each pair (i,j), the set of k such that (i,j,k) is isosceles.
    # Store as a set of integers.
    
    iso_pairs = {}  # (i, j) -> set of k such that (i,j,k) is isosceles
    for i in range(TOTAL):
        for j in range(i+1, TOTAL):
            dik = dist[i][j]
            bad_ks = set()
            for k in range(TOTAL):
                if k == i or k == j:
                    continue
                if dist[i][k] == dik or dist[j][k] == dik or dist[i][k] == dist[j][k]:
                    bad_ks.add(k)
            iso_pairs[(i, j)] = bad_ks
            iso_pairs[(j, i)] = bad_ks
    
    def can_add(p, chosen_set, chosen_list):
        """Check if p can be added to chosen without creating isosceles triangle."""
        if p in chosen_set:
            return False
        for i in range(len(chosen_list)):
            for j in range(i+1, len(chosen_list)):
                a = chosen_list[i]
                b = chosen_list[j]
                if p in iso_pairs[(a, b)]:
                    return False
        return True
    
    def is_maximal(s_set, s_list):
        """Check if s is maximal: no point outside s can be added."""
        for q in range(TOTAL):
            if q in s_set:
                continue
            if can_add(q, s_set, s_list):
                return False
        return True
    
    def try_prune(s_list, max_time=None):
        """Try to remove points (single and pair) while maintaining maximality."""
        s_list = s_list[:]
        s_set = set(s_list)
        improved = True
        while improved and (max_time is None or time.time() < max_time):
            improved = False
            n = len(s_list)
            # Try single removals
            for i in range(n):
                if time.time() > deadline - 10:
                    return s_list
                p = s_list[i]
                s_temp = [x for j, x in enumerate(s_list) if j != i]
                s_temp_set = set(s_temp)
                if is_maximal(s_temp_set, s_temp):
                    s_list = s_temp
                    s_set = s_temp_set
                    improved = True
                    break
            # Try pair removals
            if not improved and n >= 2:
                for i in range(n):
                    if time.time() > deadline - 10:
                        return s_list
                    for j in range(i+1, n):
                        p1, p2 = s_list[i], s_list[j]
                        s_temp = [x for k, x in enumerate(s_list) if k != i and k != j]
                        s_temp_set = set(s_temp)
                        if is_maximal(s_temp_set, s_temp):
                            s_list = s_temp
                            s_set = s_temp_set
                            improved = True
                            break
                    if improved:
                        break
        return s_list
    
    best_set = None
    best_size = float('inf')
    
    # Multiple randomized greedy trials
    num_trials = 500
    for trial in range(num_trials):
        if time.time() > deadline - 15:
            break
        
        random.seed(12345 + trial * 7)
        
        chosen = []
        chosen_set = set()
        
        # Random permutation of points
        points_order = list(range(TOTAL))
        random.shuffle(points_order)
        
        for p in points_order:
            if time.time() > deadline - 15:
                break
            if p in chosen_set:
                continue
            if can_add(p, chosen_set, chosen):
                chosen.append(p)
                chosen_set.add(p)
        
        if chosen:
            pruned = try_prune(chosen)
            if len(pruned) < best_size:
                best_size = len(pruned)
                best_set = pruned[:]
    
    # Local search from best
    if best_set and time.time() < deadline - 20:
        current = best_set[:]
        current_set = set(current)
        
        for iteration in range(200):
            if time.time() > deadline - 10:
                break
            
            # Try to find a removal
            found = False
            n = len(current)
            
            # Single removal
            for i in range(n):
                if time.time() > deadline - 10:
                    break
                p = current[i]
                s_temp = [x for j, x in enumerate(current) if j != i]
                s_temp_set = set(s_temp)
                if is_maximal(s_temp_set, s_temp):
                    current = s_temp
                    current_set = s_temp_set
                    found = True
                    break
            
            # Pair removal
            if not found and n >= 2:
                for i in range(n):
                    if time.time() > deadline - 10:
                        break
                    for j in range(i+1, n):
                        p1, p2 = current[i], current[j]
                        s_temp = [x for k, x in enumerate(current) if k != i and k != j]
                        s_temp_set = set(s_temp)
                        if is_maximal(s_temp_set, s_temp):
                            current = s_temp
                            current_set = s_temp_set
                            found = True
                            break
                    if found:
                        break
            
            if not found:
                # Perturb: add a random valid point, then prune
                candidates = [q for q in range(TOTAL) if q not in current_set]
                valid_candidates = [q for q in candidates if can_add(q, current_set, current)]
                if valid_candidates:
                    q = random.choice(valid_candidates)
                    current.append(q)
                    current_set.add(q)
                    current = try_prune(current)
                    current_set = set(current)
            
            if len(current) < best_size:
                best_size = len(current)
                best_set = current[:]
    
    if best_set is None:
        # Fallback
        chosen = []
        chosen_set = set()
        for p in range(TOTAL):
            if can_add(p, chosen_set, chosen):
                chosen.append(p)
                chosen_set.add(p)
        return [all_points[i] for i in chosen]
    
    return [all_points[i] for i in best_set]
