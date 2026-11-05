import random
import time
import itertools
import sys

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = N * N

# Precompute isosceles conflicts
# conflict_pairs[k] is a set of (i, j) indices such that points[i], points[j], points[k] form an isosceles triangle
# Note: We use indices 0..255 for speed
conflict_pairs = [set() for _ in range(NUM_POINTS)]

# Map point to index
point_to_idx = {p: i for i, p in enumerate(POINTS)}

# Precomputation
# For each k, find all pairs (i, j) such that {i, j, k} is isosceles
# Optimization: Iterate over all triples? 256^3 is too slow in Python.
# Iterate over k (256), and for each k, iterate over pairs (i, j) (32k). Total 8M checks.
# Each check is 3 distance comparisons. 24M ops. Feasible in < 1s with PyPy/Cython, but pure Python might take 5-10s.
# Let's optimize the check.

def dist2(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

# Faster precomputation
# For each pair (i, j), find all k such that {i, j, k} is isosceles.
# There are ~32k pairs. For each pair, there are limited k's that form isosceles triangles.
# Geometric property: Given base ij, the locus of k is the perpendicular bisector of ij.
# We can just check all k for each pair. 32k * 256 = 8M.

pair_conflicts = [set() for _ in range(NUM_POINTS * NUM_POINTS)] # Too big? 65k sets.
# Actually, we only need to know for a given k, which pairs block it.
# Let's stick to conflict_pairs[k] = set of (i, j) tuples.

# To speed up precomputation, we can use numpy if available, but standard library is safer for "standard library" constraint.
# However, the prompt says "NumPy and SciPy are available".

import numpy as np

# Vectorized precomputation
coords = np.array(POINTS)
# Distances squared matrix
# D[i, j] = dist2(POINTS[i], POINTS[j])
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
D = np.sum(diff ** 2, axis=2)

# conflict_pairs[k] contains (i, j) if D[i,k] == D[j,k] or D[i,k] == D[i,j] or D[j,k] == D[i,j]
# This is O(N^3) = 16M operations. NumPy can handle this.

# We will store for each k, a list of pairs (i, j) that form an isosceles triangle with k.
# To make repair fast, we want to check: "Is x blocked by S?"
# x is blocked if there exists (i, j) in S x S such that (i, j) in conflict_pairs[x].

# Let's build a list of conflicts for each k.
# conflict_pairs[k] = list of (i, j)
conflict_pairs = [set() for _ in range(NUM_POINTS)]

# Iterate over k
for k in range(NUM_POINTS):
    # d_ik = D[k, :]
    # d_ij = D
    # Condition: d_ik == d_jk OR d_ik == d_ij OR d_jk == d_ij
    # Note: i, j, k are distinct.
    
    # We can just loop i, j in Python for simplicity if NumPy precomp is complex to extract.
    # 8M iterations in Python is ~2-5 seconds. Acceptable for startup.
    pass

# Let's do it in Python loop for clarity and correctness, relying on the 300s budget.
# 8M iterations * ~10 ops = 80M ops. Python does ~50M ops/s. So ~2s.

start_time = time.time()

for k in range(NUM_POINTS):
    dk = D[k, :] # Distances from k to all
    for i in range(NUM_POINTS):
        if i == k: continue
        for j in range(i + 1, NUM_POINTS):
            if j == k: continue
            # Check isosceles
            d_ik = dk[i]
            d_jk = dk[j]
            d_ij = D[i, j]
            if d_ik == d_jk or d_ik == d_ij or d_jk == d_ij:
                conflict_pairs[k].add((i, j))
                conflict_pairs[k].add((j, i)) # Symmetric, but we store ordered pairs for easy lookup? 
                # Actually, for "is x blocked by S", we check if any pair in S is in conflict_pairs[x].
                # S is a set of indices.
                # We can store conflict_pairs[k] as a set of frozensets or sorted tuples.
                # Let's use sorted tuples (min(i,j), max(i,j)).

# Re-do with sorted tuples to save space and allow set intersection.
conflict_pairs = [set() for _ in range(NUM_POINTS)]
for k in range(NUM_POINTS):
    dk = D[k, :]
    for i in range(NUM_POINTS):
        if i == k: continue
        for j in range(i + 1, NUM_POINTS):
            if j == k: continue
            d_ik = dk[i]
            d_jk = dk[j]
            d_ij = D[i, j]
            if d_ik == d_jk or d_ik == d_ij or d_jk == d_ij:
                pair = (i, j) if i < j else (j, i)
                conflict_pairs[k].add(pair)

# Helper to check if x is blocked by set S (set of indices)
def is_blocked(x, S):
    # S is a set of indices
    # Check if any pair in S is in conflict_pairs[x]
    # Efficient: Iterate over pairs in S? |S| is small (~10-20). Pairs ~ 100-400.
    # conflict_pairs[x] is a set of ~100-200 pairs.
    # Intersection is fast.
    if not S:
        return False
    # Create set of pairs in S
    # Optimization: Precompute pairs of S?
    # For now, just check membership.
    for i in S:
        for j in S:
            if i == j: continue
            pair = (i, j) if i < j else (j, i)
            if pair in conflict_pairs[x]:
                return True
    return False

# This is O(|S|^2) per check. |S| ~ 15. 225 checks.
# To check maximality of S: Check all x not in S. 256 - 15 = 241 checks.
# 241 * 225 = 54k ops. Very fast.

def is_maximal(S):
    S_set = set(S)
    for x in range(NUM_POINTS):
        if x in S_set:
            continue
        if not is_blocked(x, S_set):
            return False
    return True

def greedy_construct(order):
    chosen = set()
    for p in order:
        # Check if p is blocked by chosen
        if not is_blocked(p, chosen):
            chosen.add(p)
    # Ensure maximality (greedy might not be maximal if order was bad? 
    # No, greedy adding until no more can be added IS maximal by construction 
    # IF we check all points. We iterate through all points in 'order'.
    # If a point is not added, it is blocked by the *current* chosen set.
    # But adding later points might unblock it? No, adding points only adds constraints.
    # So if it was blocked, it remains blocked.
    # Thus, greedy produces a maximal set.
    return chosen

def repair(S, points_to_add_candidates):
    # S is a set of indices.
    # We want to add points to S to make it maximal.
    # Greedy Set Cover approach.
    S = set(S)
    while True:
        # Find unblocked points
        unblocked = []
        for x in range(NUM_POINTS):
            if x in S:
                continue
            if not is_blocked(x, S):
                unblocked.append(x)
        
        if not unblocked:
            return S
        
        # Find point y not in S that blocks the most unblocked points
        best_y = None
        max_blocked = 0
        # Candidates for y: any point not in S
        # To speed up, we can just pick the first one that blocks something?
        # Or iterate all.
        for y in range(NUM_POINTS):
            if y in S:
                continue
            # How many unblocked points does y block?
            # y blocks x if (y, z) in conflict_pairs[x] for some z in S?
            # No, y is added to S. The new set is S U {y}.
            # x is blocked by S U {y} if:
            # 1. x is blocked by S (already false)
            # 2. x is blocked by y and some z in S.
            # i.e., (y, z) in conflict_pairs[x] for some z in S.
            
            count = 0
            for x in unblocked:
                # Check if y blocks x given S
                # Need z in S such that (y, z) in conflict_pairs[x]
                # Precompute for y: which x does y block given S?
                # This is expensive to do for all y.
                
                # Simpler: Just try to add y, and see if it reduces unblocked count?
                # Too slow.
                
                # Heuristic: Pick a random unblocked point x, and pick a y that blocks it.
                # Or just pick the first y that blocks *any* unblocked point.
                pass
        
        # Simple Heuristic Repair:
        # Pick a random unblocked point x.
        # Find a point y not in S such that y blocks x (i.e., there exists z in S such that (y, z) in conflict_pairs[x]).
        # If no such y, pick any y not in S? (If no y blocks x, then x can never be blocked? 
        # No, if S is empty, no y blocks x. But S is not empty).
        
        # Let's just add a random point not in S that is "compatible" (doesn't form isosceles with S).
        # Wait, we must maintain validity.
        # y must not be blocked by S? No, y must be *valid* with S.
        # y is valid with S if y is NOT blocked by S?
        # No. y is valid with S if adding y doesn't create an isosceles triangle.
        # This is exactly: y is NOT blocked by S.
        
        # So we can only add points y that are NOT blocked by S.
        # But we want to block unblocked points x.
        
        # Strategy:
        # 1. Find all y not in S such that y is valid with S (i.e., not blocked by S).
        # 2. Among these y, pick the one that blocks the most unblocked x.
        
        valid_candidates = []
        for y in range(NUM_POINTS):
            if y in S: continue
            if not is_blocked(y, S):
                valid_candidates.append(y)
        
        if not valid_candidates:
            # We are stuck? This shouldn't happen if S is a subset of a valid set.
            break
            
        best_y = random.choice(valid_candidates)
        # Better: Pick the one that blocks the most.
        max_score = -1
        for y in valid_candidates:
            score = 0
            for x in unblocked:
                # Check if y blocks x
                # (y, z) in conflict_pairs[x] for some z in S
                # We can precompute this?
                # For now, just check.
                for z in S:
                    pair = (y, z) if y < z else (z, y)
                    if pair in conflict_pairs[x]:
                        score += 1
                        break
            if score > max_score:
                max_score = score
                best_y = y
        
        S.add(best_y)

def construct():
    best_set = None
    best_size = float('inf')
    
    deadline = time.time() + 290  # Safety margin
    
    # Initial Greedy
    random.seed(42)
    order = list(range(NUM_POINTS))
    random.shuffle(order)
    current_set = greedy_construct(order)
    best_set = set(current_set)
    best_size = len(current_set)
    
    # Local Search
    while time.time() < deadline:
        # Try to remove a point and repair
        # Pick a random point to remove
        if not current_set:
            break
            
        p = random.choice(list(current_set))
        new_set = set(current_set)
        new_set.remove(p)
        
        # Repair new_set
        repaired_set = repair(new_set, None)
        
        if len(repaired_set) < best_size:
            best_set = repaired_set
            best_size = len(repaired_set)
            current_set = repaired_set
        elif len(repaired_set) == len(current_set):
            # Accept with some probability to escape local minima?
            # Or just accept if it's different?
            if random.random() < 0.5:
                current_set = repaired_set
        else:
            # Revert
            pass
            
        # Also try random restarts if stuck?
        # For now, just iterate.
        
    return [POINTS[i] for i in best_set]

# Run construct to ensure it works
if __name__ == "__main__":
    result = construct()
    print(len(result))
