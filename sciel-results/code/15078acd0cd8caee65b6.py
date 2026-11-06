import time
import random
import itertools

N = 16

def _dist_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def _is_isosceles(a, b, c):
    d1 = _dist_sq(a, b)
    d2 = _dist_sq(a, c)
    d3 = _dist_sq(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def construct():
    all_points = [(x, y) for x in range(N) for y in range(N)]
    best_set = None
    best_size = float('inf')
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s margin
    
    # Pre-compute distances between all pairs? 
    # There are 256 points. 256*256/2 = 32768 pairs.
    # This is small. Let's precompute a distance matrix or just a dict.
    # dist_dict[(p1, p2)] = dist_sq
    # But tuples are hashable.
    # Actually, computing on the fly is probably fine given the low N.
    
    while time.time() < deadline:
        S = []
        remaining = set(all_points)
        
        while remaining:
            # Find valid candidates
            valid = []
            for p in remaining:
                compatible = True
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        if _is_isosceles(p, S[i], S[j]):
                            compatible = False
                            break
                    if not compatible:
                        break
                if compatible:
                    valid.append(p)
            
            if not valid:
                break
            
            # Heuristic: Pick point with max "blocking degree"
            # Degree of v = number of u in remaining \ {v} that are blocked by v given S
            # u is blocked by v if exists a in S such that {u, v, a} is isosceles
            
            best_v = None
            best_deg = -1
            best_ties = []
            
            for v in valid:
                deg = 0
                for u in remaining:
                    if u == v:
                        continue
                    # Check if u is blocked by v given S
                    # u is blocked if exists a in S such that {u, v, a} is isosceles
                    # Note: if S is empty, v doesn't block u (unless we consider pairs in S, which are none).
                    # If S is empty, deg is 0 for all.
                    for a in S:
                        if _is_isosceles(u, v, a):
                            deg += 1
                            break
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                    best_ties = [v]
                elif deg == best_deg:
                    best_ties.append(v)
            
            # Randomly pick from ties
            chosen = random.choice(best_ties)
            S.append(chosen)
            remaining.remove(chosen)
            
            # Update remaining: remove points that are now blocked by S
            # A point q is blocked if exists a, b in S such that {q, a, b} is isosceles
            new_remaining = set()
            for q in remaining:
                compatible = True
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        if _is_isosceles(q, S[i], S[j]):
                            compatible = False
                            break
                    if not compatible:
                        break
                if compatible:
                    new_remaining.add(q)
            remaining = new_remaining
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S.copy()
            
    return best_set
