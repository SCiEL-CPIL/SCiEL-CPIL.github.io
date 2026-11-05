import time
import random
import math
import numpy as np

N = 64
TOTAL = N * N

# Precompute squared distances
xs = np.repeat(np.arange(N), N)
ys = np.tile(np.arange(N), N)

dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
dist2 = (dx * dx + dy * dy).astype(np.int64)

# Precompute modular parameters for the lattice strategy
# We look for sets of the form {(x, y) : y = a*x^2 + b*x + c mod N}
# Or more generally, points on parabolas.
# A known construction for no-isosceles sets uses quadratic residues.
# Let's try a grid of parabolas: y = k*x^2 mod N for various k, x.
# However, we need to ensure no three points form an isosceles triangle.
# Isosceles triangle condition: d1^2 == d2^2 or d1^2 == d3^2 or d2^2 == d3^2.
# This is equivalent to saying that for any point P, the distances to all other points in the set must be distinct.
# AND for any pair of points P, Q, the distance d(P,Q) must not be equal to any other distance d(R,S) 
# Wait, the condition "no three points forming an isosceles triangle" means:
# For any distinct A, B, C in the set, d(A,B) != d(A,C) and d(A,B) != d(B,C) and d(A,C) != d(B,C).
# This implies that for any point A, all distances from A to other points in the set must be distinct.
# It ALSO implies that for any pair A, B, the distance d(A,B) cannot be equal to d(A,C) for any C.
# Actually, if d(A,B) == d(A,C), then triangle ABC is isosceles at A.
# If d(A,B) == d(B,C), then triangle ABC is isosceles at B.
# So the condition is:
# 1. For every A in S, the set {d(A, B) : B in S, B != A} has size |S|-1.
# 2. This condition is sufficient. If for every A, distances to others are distinct, can we still have an isosceles triangle?
#    Suppose A, B, C form an isosceles triangle.
#    Case 1: AB = AC. Then d(A,B) == d(A,C). But for point A, distances to B and C must be distinct. Contradiction.
#    Case 2: BA = BC. Then d(B,A) == d(B,C). But for point B, distances to A and C must be distinct. Contradiction.
#    Case 3: CA = CB. Then d(C,A) == d(C,B). But for point C, distances to A and B must be distinct. Contradiction.
# So the condition "for every point in the set, distances to all other points in the set are distinct" is NECESSARY AND SUFFICIENT.

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    best_set = []
    best_size = 0
    
    # Strategy:
    # 1. Generate a strong initial set using a structured approach.
    #    We try to find a set where for each point, distances are distinct.
    #    A simple heuristic: pick points such that their coordinates are "spread out".
    #    Let's try a greedy approach but with a smarter ordering: 
    #    Sort points by some property? Or just use the lattice idea.
    
    # Let's try to construct a set based on the property that x-coordinates are distinct 
    # and y-coordinates are a function of x, e.g., y = x^2 mod N.
    # If we take the set S = {(x, x^2 mod N) for x in 0..N-1}, is it valid?
    # Let's check.
    # Points: P_x = (x, x^2 mod N).
    # Distance squared between P_x and P_y:
    # d^2 = (x-y)^2 + (x^2-y^2 mod N)^2.
    # This doesn't look obviously distinct.
    
    # Alternative: Use a known "no-three-in-line" style construction? No, that's collinearity.
    # This is "no-isosceles".
    
    # Let's stick to a robust random restart greedy, but optimize the inner loop heavily.
    # The current program scores 9.5. The best inspiration scores 10.25.
    # The gap suggests that better initialization or a different search method is needed.
    
    # Let's try:
    # 1. Fast Greedy: Use numpy to check validity of a new point against the current set in O(1) vectorized ops?
    #    Actually, checking if dists are distinct and not in ud[i] is the bottleneck.
    #    
    # New Idea: 
    # Instead of maintaining `ud` (distance sets for each point), which is hard to vectorize,
    # we can maintain the set of selected points and check the condition from scratch for the new point?
    # No, that's O(K) per point, which is what we do.
    
    # Let's try a different angle: 
    # The condition is that the distance matrix of the selected points must have distinct entries in each row (excluding diagonal).
    # 
    # Let's try to use a "backtracking" or "repair" heuristic.
    # Start with a random set of size M.
    # If invalid, remove a point involved in a violation.
    # Then try to add a new point.
    
    # Let's try to improve the greedy by using a "conflict graph" approach.
    # When we can't add a point, we know which existing points caused the conflict.
    
    # Actually, the most promising path from inspirations is "structured initialization".
    # Let's try to initialize with points that are "far apart" in a modular sense.
    
    # Let's try a grid of points (x, y) where x is in a set X and y is in a set Y.
    # If we pick X and Y such that the differences x_i - x_j and y_i - y_j produce unique sums of squares?
    
    # Let's try a specific construction:
    # Take points (x, 0) for x in 0..k-1. 
    # Distances are |x_i - x_j|. These are distinct for a given x_i? No, |x_i - x_j| can repeat.
    # e.g., x=0, dist to 1 is 1, dist to -1 is 1. But we are in [0, N).
    # So for x=0, dist to 1 is 1, to 2 is 2... distinct.
    # For x=1, dist to 0 is 1, to 2 is 1. NOT distinct.
    # So a line is bad.
    
    # What about a curve? y = x^2.
    # Let's test y = x^2 mod N for N=64.
    
    def is_valid_set(indices):
        if len(indices) < 3:
            return True
        # Build distance matrix for these indices
        # This is slow, so only use for verification or small sets.
        pass

    # Let's go with a highly optimized greedy with random restarts, but add a "shake" phase.
    # If greedy gets stuck, we remove a random point and try to re-add points.
    
    all_indices = list(range(TOTAL))
    
    for attempt in range(2000):
        if time.time() > deadline:
            break
        
        # Deterministic-ish start to ensure diversity
        random.shuffle(all_indices)
        
        si = []
        ud = []
        
        for p_idx in all_indices:
            if not si:
                si.append(p_idx)
                ud.append(set())
                continue
            
            k = len(si)
            
            # Vectorized distance calculation
            # dist2[p_idx, si] gives distances to all selected points
            dists = dist2[p_idx, si]
            
            # Check 1: All distances from p to si must be distinct
            # We can check this by comparing the sorted array to itself with a shift?
            # Or simply: if len(set(dists)) != k, fail.
            # set(dists) is O(k).
            if len(set(dists.tolist())) != k:
                continue
            
            # Check 2: For each existing point i, dists[i] is not in ud[i]
            # This is the hard part to vectorize.
            # We can do a loop, but it's Python-level.
            valid = True
            for i in range(k):
                if dists[i] in ud[i]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Add p
            si.append(p_idx)
            ud.append(set(dists.tolist()))
            
            # Update ud for existing points
            for i in range(k):
                ud[i].add(dists[i])
        
        # Refinement: Local Search
        # Try to swap points or add/remove.
        # Let's try a simple hill climbing:
        # 1. Try to add any rejected point.
        # 2. If that fails, try to remove a random point and see if we can add two new ones?
        
        # Current refinement in seed is just adding.
        # Let's try "kick" moves: remove a random point, then try to add random points.
        
        selected_set = set(si)
        unselected = [p for p in all_indices if p not in selected_set]
        random.shuffle(unselected)
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Try to add
            for p_idx in unselected:
                if p_idx in selected_set:
                    continue
                
                k = len(si)
                dists = dist2[p_idx, si].tolist()
                
                if len(set(dists)) != k:
                    continue
                
                valid = True
                for i in range(k):
                    if dists[i] in ud[i]:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                si.append(p_idx)
                selected_set.add(p_idx)
                ud.append(set(dists))
                for i in range(k):
                    ud[i].add(dists[i])
                unselected.remove(p_idx)
                improved = True
                break
        
        if len(si) > best_size:
            best_size = len(si)
            best_set = [(idx // N, idx % N) for idx in si]
            
        if time.time() > deadline - 30:
            break
            
    return best_set
