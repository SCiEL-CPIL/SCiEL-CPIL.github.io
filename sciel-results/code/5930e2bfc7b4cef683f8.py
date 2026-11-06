import itertools
import random
import numpy as np

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
NUM_POINTS = len(POINTS)

# Precompute isosceles relationships
# conflict_pairs[q] = list of (i, j) such that (i, j, q) is isosceles
# We use 0-based indices for POINTS
# i, j, k are indices into POINTS

# Optimization: 
# dist_sq[i][j]
dist_sq = np.zeros((NUM_POINTS, NUM_POINTS), dtype=int)
for i in range(NUM_POINTS):
    for j in range(i+1, NUM_POINTS):
        d = (POINTS[i][0]-POINTS[j][0])**2 + (POINTS[i][1]-POINTS[j][1])**2
        dist_sq[i][j] = d
        dist_sq[j][i] = d

# conflict_pairs[q] is a list of tuples (i, j)
conflict_pairs = [[] for _ in range(NUM_POINTS)]

# For each q, find pairs (i, j) such that d(i,j) == d(i,q) or d(i,j) == d(j,q) or d(i,q) == d(j,q)
# Actually, it's easier to iterate pairs (i,j) and find q?
# Or iterate i, j and check conditions for q.
# Let's iterate i, j.
for i in range(NUM_POINTS):
    for j in range(i+1, NUM_POINTS):
        d_ij = dist_sq[i][j]
        for k in range(NUM_POINTS):
            if k == i or k == j:
                continue
            d_ik = dist_sq[i][k]
            d_jk = dist_sq[j][k]
            # Isosceles if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk
            if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                # (i, j) blocks k
                conflict_pairs[k].append((i, j))

def is_maximal(S):
    """
    Check if S is a maximal no-isosceles set.
    S is a set of indices.
    Maximal means: for every q not in S, there exists a pair (a, b) in S 
    such that (a, b, q) is isosceles.
    """
    # Convert S to a set of frozensets for fast lookup? 
    # Or just check if any pair in conflict_pairs[q] is subset of S.
    S_set = set(S)
    
    for q in range(NUM_POINTS):
        if q in S_set:
            continue
        
        # Check if q is blocked by any pair in S
        blocked = False
        for a, b in conflict_pairs[q]:
            if a in S_set and b in S_set:
                blocked = True
                break
        if not blocked:
            return False
    return True

def construct():
    best_size = float('inf')
    best_set = None
    
    # Random seed for reproducibility in tests? 
    # The task implies we just need to return a good set.
    # We will run multiple restarts.
    
    num_restarts = 50
    
    for r in range(num_restarts):
        # 1. Greedy Construction
        # Random order
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        
        S = set()
        
        # We need to check if adding p to S creates an isosceles triangle.
        # This is equivalent to: does there exist q in S such that (p, q, r) is isosceles for some r in S?
        # No, the condition is that S U {p} contains NO isosceles triangle.
        # So, for all pairs (a, b) in S, (p, a, b) must NOT be isosceles.
        # Also for all a in S, we don't form a triangle with just p and a (need 3 points).
        # So we just need to check: for all a, b in S with a != b, is (p, a, b) isosceles?
        # If yes, we cannot add p.
        
        # To speed up, we can precompute for each p, the set of pairs (a,b) that conflict with p.
        # Let conflict_with[p] be a set of pairs (a,b) such that (p,a,b) is isosceles.
        # Then p can be added to S if S contains no pair from conflict_with[p].
        
        # Let's build conflict_with
        # conflict_with[p] is list of (a,b)
        # We already have conflict_pairs[q] which is list of (i,j) blocking q.
        # Note: conflict_pairs[q] contains (i,j) if (i,j,q) is isosceles.
        # This is symmetric. (i,j,q) is isosceles iff (i,q,j) is isosceles.
        # So conflict_with[p] is exactly conflict_pairs[p].
        
        # Check if S contains any pair from conflict_pairs[p]
        def can_add(p, S_set):
            for a, b in conflict_pairs[p]:
                if a in S_set and b in S_set:
                    return False
            return True

        for p in order:
            if can_add(p, S):
                S.add(p)
        
        # Now S is a maximal no-isosceles set?
        # Yes, because we considered all points in random order.
        # If a point q was not added, it's because when we considered it, 
        # it conflicted with some pair in the current S.
        # Since we only add points, the set S only grows.
        # So if q conflicted with S_at_that_time, it conflicts with final S.
        # Thus, q is blocked by final S.
        # So S is maximal.
        
        # 2. Pruning Phase
        # Try to remove points while maintaining maximality.
        # We do this iteratively.
        
        changed = True
        while changed:
            changed = False
            # Shuffle order of removal to avoid bias
            points_to_check = list(S)
            random.shuffle(points_to_check)
            
            for p in points_to_check:
                if p not in S:
                    continue
                
                S_new = S - {p}
                if len(S_new) < len(S): # p was in S
                    if is_maximal(S_new):
                        S = S_new
                        changed = True
                        break # Restart check as S changed significantly
        
        # 3. Optional: Try to remove 2 points at once if 1 fails?
        # This might be too slow. Let's stick to 1 for now.
        # But maybe a simple local search swap?
        # If we can't remove p, maybe we can remove p and add q? No, size doesn't decrease.
        # We want to minimize size.
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S.copy()
            
            # Early stop if we find a very small set?
            # Theoretical lower bound?
            # For N=16, grid 16x16.
            # If best_size <= 10, we might stop.
            if best_size <= 12:
                break

    if best_set is None:
        # Fallback: return a valid maximal set (the last one generated or any)
        # This should not happen
        return [POINTS[i] for i in range(10)] # Dummy
        
    return [POINTS[i] for i in best_set]
