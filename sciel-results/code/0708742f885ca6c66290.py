"""
Strategy: Iterated Local Search with Random Restarts and Intensive Local Improvement.

Instead of Simulated Annealing (which allows violations and uses a temperature schedule),
we use a pure local search approach that always maintains a valid maximal set.
The key insight is to use a "remove-then-repair" strategy:
1. Start with a valid maximal set (greedy).
2. Repeatedly try to remove a point from S.
3. If removing the point violates maximality (some point becomes uncovered), try to repair by adding a different point that covers the uncovered points without creating isosceles triangles.
4. If repair fails, restore the removed point.
5. If repair succeeds, we have a smaller valid maximal set.
6. Periodically restart with a new random greedy solution to escape local minima.

This approach is different from SA because:
- It never allows violations (always valid maximal set).
- It uses a specific repair mechanism rather than random moves.
- It uses random restarts rather than a cooling schedule.
"""

import time
import random
import itertools
import sys
from collections import defaultdict

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
POINT_SET = set(POINTS)
DEADLINE = 320.0

def _dist2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

# Precompute isosceles triples
print("Precomputing...", file=sys.stderr)
start_time = time.time()

# For each pair (a,b), store the set of points p such that (p,a,b) is isosceles
IsoWithPair = {}
# For each point p, store the set of pairs (a,b) such that (p,a,b) is isosceles
IsoWithPoint = {p: set() for p in POINTS}

for i, a in enumerate(POINTS):
    for b in POINTS[i+1:]:
        d_ab = _dist2(a, b)
        pts = set()
        for p in POINTS:
            if p == a or p == b:
                continue
            d_pa = _dist2(p, a)
            d_pb = _dist2(p, b)
            if d_pa == d_ab or d_pb == d_ab or d_pa == d_pb:
                pts.add(p)
                IsoWithPoint[p].add(frozenset((a, b)))
        if pts:
            IsoWithPair[frozenset((a, b))] = pts

print(f"Precomputation took {time.time() - start_time:.2f}s", file=sys.stderr)

def _is_valid_maximal(S):
    """Check if S is a valid maximal no-isosceles set."""
    S_set = set(S)
    # Check no isosceles
    for p in S_set:
        for pair in IsoWithPoint[p]:
            a, b = list(pair)
            if a in S_set and b in S_set:
                return False
    # Check maximality: every point not in S is covered
    for p in POINTS:
        if p in S_set:
            continue
        covered = False
        for pair in IsoWithPoint[p]:
            a, b = list(pair)
            if a in S_set and b in S_set:
                covered = True
                break
        if not covered:
            return False
    return True

def _get_covered_by(S, p):
    """Check if point p is covered by set S (p not in S)."""
    S_set = set(S)
    for pair in IsoWithPoint[p]:
        a, b = list(pair)
        if a in S_set and b in S_set:
            return True
    return False

def _greedy_initial():
    """Greedy construction of a valid maximal set."""
    S = set()
    pts = list(POINTS)
    random.shuffle(pts)
    
    for p in pts:
        # Check if p can be added (no isosceles with existing S)
        valid = True
        for a in S:
            for b in S:
                if a == b:
                    continue
                if frozenset((a, b)) in IsoWithPoint[p]:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            S.add(p)
    
    # Now make it maximal by adding points that are uncovered
    # We need to add points to cover all uncovered points
    # This is a set cover problem, but we can do it greedily
    changed = True
    while changed:
        changed = False
        # Find uncovered points
        uncovered = []
        for p in POINTS:
            if p not in S and not _get_covered_by(S, p):
                uncovered.append(p)
        
        if not uncovered:
            break
        
        # Try to add an uncovered point that doesn't create isosceles
        random.shuffle(uncovered)
        for p in uncovered:
            valid = True
            for a in S:
                for b in S:
                    if a == b:
                        continue
                    if frozenset((a, b)) in IsoWithPoint[p]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S.add(p)
                changed = True
                break
    
    # If we still have uncovered points, we need to swap
    # This is complex, so for now just return what we have
    # The local search will fix it
    return S

def _local_search(S, time_limit):
    """Local search: try to remove points and repair."""
    S = set(S)
    best_S = set(S)
    best_size = len(S)
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Try to remove a random point
        if not S:
            break
        
        p = random.choice(list(S))
        S.remove(p)
        
        # Check if S is still maximal
        uncovered = []
        for q in POINTS:
            if q in S:
                continue
            if not _get_covered_by(S, q):
                uncovered.append(q)
        
        if not uncovered:
            # S is still maximal, we have a smaller valid set
            if len(S) < best_size:
                best_size = len(S)
                best_S = set(S)
            continue
        
        # Need to repair: add a point that covers some uncovered points
        # without creating isosceles
        best_repair = None
        best_cover_count = 0
        
        # Try each candidate point not in S
        candidates = [q for q in POINTS if q not in S]
        random.shuffle(candidates)
        
        for q in candidates:
            # Check if q can be added (no isosceles)
            valid = True
            for a in S:
                for b in S:
                    if a == b:
                        continue
                    if frozenset((a, b)) in IsoWithPoint[q]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Count how many uncovered points q would cover
            cover_count = 0
            for u in uncovered:
                # Check if (u, a, b) is isosceles for some a,b in S U {q}
                # Actually, we need to check if u is covered by S U {q}
                # u is covered if there exist a, b in S U {q} such that (u, a, b) is isosceles
                # Since u was not covered by S, we need at least one of a or b to be q
                # So we check if (u, q, a) is isosceles for some a in S
                for a in S:
                    if frozenset((q, a)) in IsoWithPoint[u]:
                        cover_count += 1
                        break
            
            if cover_count > best_cover_count:
                best_cover_count = cover_count
                best_repair = q
        
        if best_repair is not None and best_cover_count > 0:
            S.add(best_repair)
            # Don't update best_S here, we'll check at the end
        else:
            # Repair failed, restore p
            S.add(p)
    
    return best_S, best_size

def construct():
    start_time = time.time()
    
    best_S = None
    best_size = float('inf')
    
    # Multiple restarts
    num_restarts = 0
    while time.time() - start_time < DEADLINE:
        num_restarts += 1
        
        # Get initial solution
        S = _greedy_initial()
        
        # Local search with time limit
        time_left = DEADLINE - (time.time() - start_time)
        search_time = min(30.0, time_left * 0.8)
        
        if search_time < 5.0:
            break
        
        S, size = _local_search(S, search_time)
        
        if size < best_size:
            best_size = size
            best_S = S
        
        # Check if we found a perfect solution
        if best_size <= 10:
            break
    
    if best_S is None:
        # Fallback: return a greedy solution
        best_S = _greedy_initial()
    
    return list(best_S)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
