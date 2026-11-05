import itertools
import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer for safety

    # Helper: Check if 4 points are collinear
    def are_collinear(p1, p2, p3, p4):
        def cross(a, b, c):
            return (b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])
        return cross(p1, p2, p3) == 0 and cross(p1, p2, p4) == 0

    def is_valid(points):
        n = len(points)
        # Check all lines defined by pairs
        # If any line has >= 4 points, invalid
        # To be efficient, just check all 4-combinations? 
        # C(20,4) = 4845. Fast enough.
        for i in range(n-3):
            for j in range(i+1, n-2):
                for k in range(j+1, n-1):
                    for l in range(k+1, n):
                        if are_collinear(points[i], points[j], points[k], points[l]):
                            return False
        return True

    def get_triples(points):
        """Return list of all collinear triples (i, j, k) with i<j<k"""
        n = len(points)
        triples = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = points[i], points[j], points[k]
                    if (p2[0]-p1[0])*(p3[1]-p1[1]) == (p3[0]-p1[0])*(p2[1]-p1[1]):
                        triples.append((i, j, k))
        return triples

    def calc_exact_score(points):
        """Exact B&B for largest subset with no 3 collinear"""
        n = len(points)
        triples = get_triples(points)
        
        if not triples:
            return n
            
        # Precompute for each vertex, the set of other vertices it forms a triple with
        # Actually, for B&B, we need to know: if we pick v, which pairs (u,w) are forbidden?
        # Forbidden pairs are those (u,w) such that (v,u,w) is a triple.
        
        # Let's use a recursive search with bitmask
        # mask: bitmask of available vertices
        # chosen: bitmask of chosen vertices (not strictly needed for size, but for conflict check?)
        # Actually, the constraint is: No 3 chosen vertices are collinear.
        # This is equivalent to: The set of chosen vertices is an independent set in the 3-uniform hypergraph.
        
        # Heuristic: Order vertices by degree (number of triples they are in)
        degree = [0] * n
        for i, j, k in triples:
            degree[i] += 1
            degree[j] += 1
            degree[k] += 1
            
        order = sorted(range(n), key=lambda x: -degree[x])
        # Remap indices to order positions? 
        # It's easier to just iterate in a fixed order in the recursion, 
        # but prune based on the global best.
        
        best_size = 0
        chosen = [False] * n
        
        # Precompute conflict pairs for each vertex v
        # conflict_pairs[v] is a list of (u, w) such that v, u, w are collinear and u < w
        conflict_pairs = [[] for _ in range(n)]
        for i, j, k in triples:
            # For i, pair (j,k) is forbidden if j,k are both chosen
            conflict_pairs[i].append((j, k))
            conflict_pairs[j].append((i, k))
            conflict_pairs[k].append((i, j))

        def backtrack(pos, current_size):
            nonlocal best_size
            if pos == n:
                if current_size > best_size:
                    best_size = current_size
                return
            
            # Pruning
            if current_size + (n - pos) <= best_size:
                return
            
            # Option 1: Skip vertex pos
            backtrack(pos + 1, current_size)
            
            # Option 2: Include vertex pos
            v = pos
            valid = True
            # Check if including v creates a collinear triple with any two already chosen vertices
            # Wait, the recursion is sequential. 
            # If we include v, we must ensure that there are no two vertices u, w in `chosen` 
            # such that v, u, w are collinear.
            # Since we process in order 0..n-1, `chosen` only contains vertices with index < pos.
            # So we only need to check pairs (u, w) where u, w < pos.
            
            # We can optimize this check.
            # For each pair (u, w) in conflict_pairs[v], if u < pos and w < pos and chosen[u] and chosen[w], then invalid.
            # Note: conflict_pairs[v] contains all pairs (u,w) collinear with v.
            # We only care about those where both u and w are already processed (index < pos).
            
            for u, w in conflict_pairs[v]:
                # We need to ensure u and w are both in `chosen`
                # Since we iterate pos from 0 to n-1, if u < pos and w < pos, they are either chosen or not.
                # If either u >= pos or w >= pos, they are not chosen yet (or will be chosen later).
                # But wait, if we choose v now, and later choose u and w, we will check that pair then.
                # So we only need to check against previously chosen vertices.
                if u < pos and w < pos:
                    if chosen[u] and chosen[w]:
                        valid = False
                        break
            
            if valid:
                chosen[v] = True
                backtrack(pos + 1, current_size + 1)
                chosen[v] = False

        backtrack(0, 0)
        return best_size

    def calc_heuristic_score(points):
        """Approximate score: 20 - max_disjoint_triples"""
        n = len(points)
        triples = get_triples(points)
        if not triples:
            return n
        
        # Greedy matching of triples
        # Sort triples by... maybe just iterate?
        # To maximize disjoint triples, we can use a simple greedy:
        # Pick a triple, mark its vertices used, remove all triples sharing vertices.
        # Repeat.
        
        # For better results, sort triples by "scarcity" (how many other triples share vertices with it)
        # Or just random order with multiple restarts?
        # Let's try a simple greedy with a few random restarts.
        
        max_disjoint = 0
        for _ in range(5): # 5 restarts
            used = [False] * n
            count = 0
            # Shuffle triples to break ties
            shuffled_triples = triples[:]
            random.shuffle(shuffled_triples)
            for t in shuffled_triples:
                i, j, k = t
                if not used[i] and not used[j] and not used[k]:
                    used[i] = True
                    used[j] = True
                    used[k] = True
                    count += 1
            max_disjoint = max(max_disjoint, count)
            
        return n - max_disjoint

    def generate_random():
        points = []
        while len(points) < 20:
            p = (random.randint(-100, 100), random.randint(-100, 100))
            if p not in points:
                points.append(p)
        return points

    best_points = None
    best_exact_score = 21
    best_heuristic_score = 21

    # Initial population
    candidates = []
    for _ in range(20):
        pts = generate_random()
        if is_valid(pts):
            h_score = calc_heuristic_score(pts)
            candidates.append((h_score, pts))
    
    if not candidates:
        # Fallback
        pts = [(i*10, (i*i) % 100 - 50) for i in range(20)]
        if not is_valid(pts):
            pts = [(i*5, i*3) for i in range(20)]
        return [[p[0], p[1]] for p in pts]

    # Sort by heuristic score
    candidates.sort(key=lambda x: x[0])
    
    # Local Search on the best candidate
    current_h, current_pts = candidates[0]
    
    # Check exact score of the initial best
    if time.time() < deadline - 10:
        e_score = calc_exact_score(current_pts)
        best_exact_score = e_score
        best_points = current_pts[:]

    # Search Loop
    while time.time() < deadline - 5:
        # Pick a candidate (best or random)
        if random.random() < 0.8:
            h, pts = candidates[0]
        else:
            h, pts = random.choice(candidates)
        
        pts = pts[:]
        
        # Move one point
        idx = random.randint(0, 19)
        old_p = pts[idx]
        new_p = (random.randint(-100, 100), random.randint(-100, 100))
        if new_p in pts:
            continue
        pts[idx] = new_p
        
        if not is_valid(pts):
            continue
        
        new_h = calc_heuristic_score(pts)
        
        # Accept if better or equal (with some probability)
        if new_h < h or (new_h == h and random.random() < 0.5):
            # Update candidate
            # Replace the candidate in the list? 
            # Simpler: just update the current best tracking
            # We'll maintain a list of top candidates
            pass # Placeholder
            
    # This simple loop isn't updating candidates well.
    # Let's do a proper hill climbing.
    
    # Reset
    best_pts = candidates[0][1][:]
    best_h = candidates[0][0]
    best_e = calc_exact_score(best_pts)
    
    for _ in range(1000): # Max iterations
        if time.time() > deadline - 5:
            break
            
        trial = best_pts[:]
        idx = random.randint(0, 19)
        new_p = (random.randint(-100, 100), random.randint(-100, 100))
        if new_p in trial:
            continue
        trial[idx] = new_p
        
        if not is_valid(trial):
            continue
        
        trial_h = calc_heuristic_score(trial)
        
        if trial_h < best_h:
            best_h = trial_h
            best_pts = trial[:]
            # Check exact score periodically
            if time.time() < deadline - 10:
                trial_e = calc_exact_score(trial)
                if trial_e < best_e:
                    best_e = trial_e
        elif trial_h == best_h and random.random() < 0.2:
            # Random walk
            best_pts = trial[:]
            if time.time() < deadline - 10:
                trial_e = calc_exact_score(trial)
                if trial_e < best_e:
                    best_e = trial_e

    return [[p[0], p[1]] for p in best_pts]
