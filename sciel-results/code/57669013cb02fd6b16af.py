import random
import time
import itertools

N = 16
TOTAL_POINTS = N * N
POINT_IDS = list(range(TOTAL_POINTS))
COORDS = [(i // N, i % N) for i in POINT_IDS]

# Precompute isosceles triples
# iso_pairs[pair_id] = list of points that form isosceles triangle with the pair
# pair_id is index in list of all pairs
# We need efficient lookup: given a, b, find c.
# Let's store a dict: (a, b) -> list of c
# Or better, for each point p, list of (a, b) pairs that form isosceles with p?
# For coverage count, we need: for a pair (a, b), which points q are covered?
# So: pair_to_points[(a, b)] = [q1, q2, ...]

pair_to_points = {}

def _dist2(a, b):
    ax, ay = COORDS[a]
    bx, by = COORDS[b]
    return (ax - bx) ** 2 + (ay - by) ** 2

def _precompute():
    # Iterate over all pairs (a, b) with a < b
    for a in range(TOTAL_POINTS):
        for b in range(a + 1, TOTAL_POINTS):
            d_ab = _dist2(a, b)
            # Find c such that (a,b,c) is isosceles
            # Cases:
            # 1. d_ac == d_ab  => c on circle center a radius d_ab
            # 2. d_bc == d_ab  => c on circle center b radius d_ab
            # 3. d_ac == d_bc  => c on perpendicular bisector of ab
            # It's easier to just check all c != a, b
            # But that's O(N^3) = 256^3 = 16M. Might be slow in Python.
            # Let's optimize.
            # For a given pair (a, b), the possible c's are limited.
            # Instead, iterate over all triples? No.
            # Let's iterate c for each pair.
            # 256 * 256 / 2 * 256 = 8M checks. Each check is 3 dist2 calcs.
            # 24M dist2 calcs. Should be okay in 1-2 seconds.
            pass

# Better precomputation strategy:
# Iterate over all triples (a, b, c) with a < b < c.
# Check if isosceles.
# If yes, add c to pair_to_points[(a,b)], b to pair_to_points[(a,c)], a to pair_to_points[(b,c)].
# Number of triples: C(256, 3) ~ 2.8M.
# This is very fast.

def _precompute_isosceles():
    global pair_to_points
    pair_to_points = {}
    for a in range(TOTAL_POINTS):
        ax, ay = COORDS[a]
        for b in range(a + 1, TOTAL_POINTS):
            bx, by = COORDS[b]
            d_ab2 = (ax - bx) ** 2 + (ay - by) ** 2
            for c in range(b + 1, TOTAL_POINTS):
                cx, cy = COORDS[c]
                d_ac2 = (ax - cx) ** 2 + (ay - cy) ** 2
                d_bc2 = (bx - cx) ** 2 + (by - cy) ** 2
                if d_ab2 == d_ac2 or d_ab2 == d_bc2 or d_ac2 == d_bc2:
                    # Isosceles
                    pair_to_points.setdefault((a, b), []).append(c)
                    pair_to_points.setdefault((a, c), []).append(b)
                    pair_to_points.setdefault((b, c), []).append(a)

# Ensure symmetric keys in pair_to_points for easy lookup
def _get_covered_points(a, b):
    key1 = (min(a, b), max(a, b))
    return pair_to_points.get(key1, [])

def construct():
    _precompute_isosceles()
    
    best_size = TOTAL_POINTS
    best_set = None
    deadline = time.time() + 300
    
    # Try multiple random orders
    for _ in range(1000): # Max restarts, but limited by time
        if time.time() > deadline - 10:
            break
        
        random.shuffle(POINT_IDS)
        S = []
        S_set = set()
        
        # Greedy construction
        for p in POINT_IDS:
            # Check if p can be added
            # p conflicts if there exists a, b in S such that (a,b,p) is isosceles
            # i.e., p in _get_covered_points(a, b)
            can_add = True
            for a, b in itertools.combinations(S, 2):
                if p in _get_covered_points(a, b):
                    can_add = False
                    break
            if can_add:
                S.append(p)
                S_set.add(p)
        
        # Local search: try to remove points
        # Initialize cover counts for points NOT in S
        cover_count = {p: 0 for p in POINT_IDS if p not in S_set}
        
        # Calculate initial cover counts
        for a, b in itertools.combinations(S, 2):
            for q in _get_covered_points(a, b):
                if q not in S_set:
                    cover_count[q] += 1
        
        # Iterative removal
        changed = True
        while changed:
            changed = False
            # Try removing each point in S
            # Randomize order of removal attempts
            S_copy = S[:]
            random.shuffle(S_copy)
            
            for p in S_copy:
                if p not in S_set:
                    continue
                
                # Check if p is covered by S \ {p}
                # We need to check if there exists a, b in S \ {p} such that p in _get_covered_points(a, b)
                # This is O(|S|^2) per removal check.
                # Also need to check if any q becomes uncovered.
                
                # 1. Check if p is covered by S \ {p}
                p_covered = False
                for a, b in itertools.combinations(S, 2):
                    if a == p or b == p:
                        continue
                    if p in _get_covered_points(a, b):
                        p_covered = True
                        break
                if not p_covered:
                    continue
                
                # 2. Check if any q becomes uncovered
                # q loses coverage from pairs involving p.
                # Pairs are (p, b) for b in S \ {p}
                # For each such pair, q in _get_covered_points(p, b) loses 1 count.
                # If any count becomes 0, invalid.
                
                # We need to simulate the decrement.
                # Since we only remove one p, we can just check the current counts.
                # A point q is currently covered by k pairs in S.
                # If we remove p, q is covered by k - m pairs, where m is number of pairs (p, b) in S that cover q.
                # We need k - m > 0 for all q not in S \ {p}.
                # Note: p is now in the "uncovered" set (since it's removed), but we already checked p is covered by S \ {p}.
                # Wait, p is covered by S \ {p}, so p is fine.
                # We only care about q in POINT_IDS \ S (original S).
                # But p is removed, so the new set of "outside" points is (POINT_IDS \ S) U {p}.
                # We checked p is covered.
                # For q in POINT_IDS \ S:
                #   new_count[q] = cover_count[q] - m_q
                #   where m_q = number of b in S \ {p} such that q in _get_covered_points(p, b)
                
                valid_removal = True
                # We need to calculate m_q for each q.
                # This is expensive if done naively.
                # But |S| is small.
                
                # Let's collect all q that are covered by pairs (p, b)
                affected_q = {}
                for b in S:
                    if b == p:
                        continue
                    for q in _get_covered_points(p, b):
                        if q in S_set:
                            continue # q is in S, so it's not "outside" and doesn't need coverage count
                        # q is outside S
                        affected_q[q] = affected_q.get(q, 0) + 1
                
                for q, m_q in affected_q.items():
                    if cover_count[q] - m_q <= 0:
                        valid_removal = False
                        break
                
                if valid_removal:
                    # Perform removal
                    S.remove(p)
                    S_set.remove(p)
                    
                    # Update cover counts
                    for q, m_q in affected_q.items():
                        cover_count[q] -= m_q
                    
                    # Add p to cover_count? No, p is now outside.
                    # But we don't need cover_count[p] for future removals unless p is removed again (impossible).
                    # Wait, if we remove another point later, p might be affected.
                    # So we should add p to cover_count and calculate its initial count.
                    # p is covered by S (new S).
                    # We need cover_count[p] to check if p remains covered if we remove other points.
                    # Actually, the logic above checks "is p covered by S \ {p}"?
                    # No, the logic checks if removing p is valid.
                    # After removal, p is outside.
                    # If we later remove another point r, we need to check if p is still covered.
                    # So we MUST maintain cover_count for ALL points not in S.
                    
                    # Calculate cover_count for p with respect to new S
                    p_count = 0
                    for a, b in itertools.combinations(S, 2):
                        if p in _get_covered_points(a, b):
                            p_count += 1
                    cover_count[p] = p_count
                    
                    changed = True
        
        # End local search
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
            
        # Early exit if we found a very small set?
        # Hard to know lower bound.
        
    return best_set if best_set is not None else []

# Run precomputation once at module level?
# No, construct() is the entry point.
# But precomputation takes time.
# We can cache it.

_precomputed = False
def _ensure_precomputed():
    global _precomputed
    if not _precomputed:
        _precompute_isosceles()
        _precomputed = True

def construct():
    _ensure_precomputed()
    
    best_size = TOTAL_POINTS
    best_set = None
    deadline = time.time() + 300
    
    for _ in range(1000):
        if time.time() > deadline - 5:
            break
        
        random.shuffle(POINT_IDS)
        S = []
        S_set = set()
        
        # Greedy
        for p in POINT_IDS:
            can_add = True
            for a, b in itertools.combinations(S, 2):
                if p in _get_covered_points(a, b):
                    can_add = False
                    break
            if can_add:
                S.append(p)
                S_set.add(p)
        
        # Local Search
        cover_count = {p: 0 for p in POINT_IDS if p not in S_set}
        for a, b in itertools.combinations(S, 2):
            for q in _get_covered_points(a, b):
                if q not in S_set:
                    cover_count[q] += 1
        
        changed = True
        while changed:
            changed = False
            S_copy = S[:]
            random.shuffle(S_copy)
            
            for p in S_copy:
                if p not in S_set:
                    continue
                
                # 1. Is p covered by S \ {p}?
                p_covered = False
                for a, b in itertools.combinations(S, 2):
                    if a == p or b == p:
                        continue
                    if p in _get_covered_points(a, b):
                        p_covered = True
                        break
                if not p_covered:
                    continue
                
                # 2. Do any outside points become uncovered?
                affected_q = {}
                for b in S:
                    if b == p:
                        continue
                    for q in _get_covered_points(p, b):
                        if q in S_set:
                            continue
                        affected_q[q] = affected_q.get(q, 0) + 1
                
                valid = True
                for q, m_q in affected_q.items():
                    if cover_count[q] - m_q <= 0:
                        valid = False
                        break
                
                if valid:
                    S.remove(p)
                    S_set.remove(p)
                    
                    for q, m_q in affected_q.items():
                        cover_count[q] -= m_q
                    
                    # Update cover_count for p
                    p_count = 0
                    for a, b in itertools.combinations(S, 2):
                        if p in _get_covered_points(a, b):
                            p_count += 1
                    cover_count[p] = p_count
                    
                    changed = True
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
            
    return best_set if best_set is not None else []
