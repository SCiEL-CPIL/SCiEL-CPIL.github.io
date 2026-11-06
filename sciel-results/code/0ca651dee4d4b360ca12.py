import numpy as np
import time
import random
import math

N = 64
# Pre-compute all grid points as a numpy array for vectorized ops
# Shape: (4096, 2)
ALL_POINTS = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
NUM_POINTS = len(ALL_POINTS)

def squared_dist_vec(p1, p2):
    """Vectorized squared distance between p1 (N, 2) and p2 (M, 2) -> (N, M) matrix"""
    return np.sum((p1[:, np.newaxis, :] - p2[np.newaxis, :, :]) ** 2, axis=2)

def is_valid_set(points):
    """
    Check if a set of points (numpy array of shape (k, 2)) contains no isosceles triangles.
    Returns True if valid.
    """
    k = len(points)
    if k < 3:
        return True
    
    # Compute all pairwise squared distances: (k, k)
    dists = squared_dist_vec(points, points)
    
    # We need to check for any triangle (i, j, l) with i < j < l
    # such that d(i,j) == d(i,l) or d(i,j) == d(j,l) or d(i,l) == d(j,l)
    
    # To do this efficiently without O(k^3) python loops:
    # For each pair (i, j), we check if there exists a third point l (l != i, l != j)
    # such that dist(i, l) == dist(i, j) OR dist(j, l) == dist(i, j) OR dist(i, l) == dist(j, l).
    
    # Let's iterate over pairs (i, j) where i < j.
    # There are k(k-1)/2 pairs. For each pair, we check the rest of the points.
    # This is O(k^3) in worst case, but with numpy vectorization inside the loop,
    # it might be faster than pure python O(k^3).
    
    # Alternative: 
    # For each point i, look at the vector of distances from i to all others: d_i
    # If there are any duplicate values in d_i (excluding self), then there is an isosceles triangle with i as the apex.
    # Wait, that only checks d(i,a) == d(i,b).
    # We also need to check d(i,a) == d(a,b) and d(i,b) == d(a,b).
    
    # Let's stick to the pair iteration but optimize the inner check.
    # For a pair (i, j), let d_ij = dists[i, j].
    # We need to check if any other point l has:
    # 1. dists[i, l] == d_ij  (Apex at i? No, d(i,l)==d(i,j) means i is apex of triangle i-l-j? No, d(i,l)=d(i,j) implies isosceles with apex i if l!=j. Yes.)
    # 2. dists[j, l] == d_ij  (Apex at j)
    # 3. dists[i, l] == dists[j, l] (Apex at l)
    
    # Actually, the condition "no three points forming an isosceles triangle" is equivalent to:
    # For all distinct i, j, l:
    # d(i,j) != d(i,l)
    # d(i,j) != d(j,l)
    # d(i,l) != d(j,l)
    
    # Let's check this.
    # If we iterate over all pairs (i, j), we can check the third condition against all other points l.
    # Condition 1: d(i,j) == d(i,l). This is equivalent to checking if the value d(i,j) appears in the row i of the distance matrix (excluding j and i).
    # Condition 2: d(i,j) == d(j,l). This is equivalent to checking if the value d(i,j) appears in the row j of the distance matrix (excluding i and j).
    # Condition 3: d(i,l) == d(j,l). This is equivalent to checking if there is a column l where dists[i, l] == dists[j, l].
    
    # This can be done with numpy operations per pair, but iterating k^2/2 pairs in python is slow if k is large.
    # However, k is likely small (< 100). 100^2/2 = 5000 pairs.
    # For each pair, we do a few numpy vector ops on arrays of size k.
    # 5000 * (few ops on size 100) is very fast.
    
    for i in range(k):
        for j in range(i + 1, k):
            d_ij = dists[i, j]
            
            # Condition 1: Is there an l != i, j such that dists[i, l] == d_ij?
            # Exclude i and j from row i.
            row_i = dists[i, :].copy()
            row_i[i] = -1
            row_i[j] = -1
            if np.any(row_i == d_ij):
                return False
                
            # Condition 2: Is there an l != i, j such that dists[j, l] == d_ij?
            row_j = dists[j, :].copy()
            row_j[i] = -1
            row_j[j] = -1
            if np.any(row_j == d_ij):
                return False
                
            # Condition 3: Is there an l != i, j such that dists[i, l] == dists[j, l]?
            # Compare row i and row j, excluding columns i and j.
            # We can just compare the whole rows and mask out i, j.
            mask = np.ones(k, dtype=bool)
            mask[i] = False
            mask[j] = False
            if np.any(dists[i, mask] == dists[j, mask]):
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    best_set = []
    best_size = 0
    
    # Strategy:
    # 1. Generate a few random "structured" initial sets.
    #    Instead of fully random points, we can try to pick points from a sparse grid
    #    or use a greedy approach with a strong heuristic.
    # 2. Use a vectorized local search (Simulated Annealing or Hill Climbing) to improve the set.
    #    Move: Remove one point, add another.
    #    Check: Is the new set valid?
    
    # Let's start with a greedy construction that is faster by using a "forbidden" set logic?
    # No, let's just do random restarts with a very fast validity check for small sets,
    # then local search.
    
    # Actually, the bottleneck is the validity check.
    # If we use the `is_valid_set` function above, it's O(k^3) but with numpy.
    # For k=50, 50^3 = 125,000 operations. In numpy, this is sub-millisecond.
    # So we can afford to check validity frequently.
    
    # Initial construction: Greedy
    # We'll try to build a set by adding points one by one.
    # To speed up, we can pre-filter candidates?
    
    # Let's try a different initial strategy:
    # Pick a random subset of the grid, e.g., points where x % 2 == 0 and y % 2 == 0?
    # This reduces the number of points to 1024.
    # Then run greedy on this subset?
    
    # Better: Just random restarts with a "fast" greedy.
    # Fast greedy:
    # Maintain the current set S.
    # To add a point p, we need to check if it forms an isosceles triangle with any pair in S.
    # We can pre-compute the distance matrix of S.
    # When adding p, we compute d(p, s) for all s in S.
    # Then we check:
    # 1. Are any d(p, s) equal to any existing d(s1, s2) in S?
    #    This is hard to check fast without storing all pairwise distances of S.
    #    But we can store a set of all pairwise squared distances in S.
    #    Let D_S be the set of all d(s_i, s_j) for s_i, s_j in S, i < j.
    #    When adding p:
    #    - Compute d(p, s) for all s in S.
    #    - If any d(p, s) is in D_S, is it necessarily invalid?
    #      If d(p, s) == d(s1, s2), does that form an isosceles triangle?
    #      Triangle (p, s, s1) has sides d(p,s), d(p,s1), d(s,s1).
    #      If d(p,s) == d(s1,s2), it doesn't directly imply isosceles unless s2 is involved.
    #      Wait, the condition is that NO three points form an isosceles triangle.
    #      If d(p, s) == d(s1, s2), and s, s1, s2 are in S, it doesn't involve p in the triangle (s1, s2, ?).
    #      The triangle involving p must be (p, a, b) where a, b in S.
    #      So we need to check if for any pair (a, b) in S:
    #      d(p, a) == d(p, b)
    #      d(p, a) == d(a, b)
    #      d(p, b) == d(a, b)
    
    #    So, if we maintain D_S (set of pairwise distances in S), we can check:
    #    - For each a in S, let d_pa = d(p, a).
    #    - If d_pa is in D_S, is it invalid?
    #      If d_pa == d(a, b) for some b in S, then triangle (p, a, b) has sides d_pa, d_pb, d_ab.
    #      Since d_pa == d_ab, it is isosceles. So YES, if d_pa is in D_S, it is invalid?
    #      Wait. D_S contains d(a, b) for ALL pairs a, b in S.
    #      So if d(p, a) == d(a, b) for some b, then d(p, a) is in D_S.
    #      So checking if d(p, a) is in D_S covers the case d(p, a) == d(a, b).
    #      Similarly, if d(p, b) is in D_S, it covers d(p, b) == d(a, b).
    #      What about d(p, a) == d(p, b)?
    #      This is not covered by D_S. We need to check if any two distances from p to S are equal.
    
    #    So the check for adding p is:
    #    1. Compute vector d_p of distances from p to all points in S.
    #    2. If there are duplicates in d_p, invalid.
    #    3. If any value in d_p is in D_S, invalid.
    
    #    This is O(k) to compute d_p, O(k log k) or O(k) to check duplicates (if we use a set),
    #    and O(k) to check membership in D_S (if D_S is a set).
    #    Total O(k) per insertion!
    #    This is MUCH faster than O(k^2).
    
    #    Let's implement this greedy.
    
    def fast_greedy_construct():
        current_set = []
        D_S = set() # Set of all pairwise squared distances in current_set
        
        # Shuffle all points
        indices = np.arange(NUM_POINTS)
        np.random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            p = ALL_POINTS[idx]
            
            if not current_set:
                current_set.append(p)
                continue
            
            # Compute distances from p to all points in current_set
            # current_set is a list of numpy arrays (2,)
            # Convert to numpy array for vectorized distance
            S = np.array(current_set)
            d_p = squared_dist_vec(p[np.newaxis, :], S)[0] # Shape (k,)
            
            # Check for duplicates in d_p
            # If len(set(d_p)) < len(d_p), there are duplicates
            if len(np.unique(d_p)) < len(d_p):
                continue
                
            # Check if any d_p value is in D_S
            # We can do this by checking intersection
            # Since d_p is an array, we can loop through it?
            # Or convert D_S to a numpy array? D_S changes.
            # For small k, looping through d_p and checking in D_S (python set) is fast.
            valid = True
            for d in d_p:
                if d in D_S:
                    valid = False
                    break
            
            if valid:
                # Add p to current_set
                current_set.append(p)
                # Update D_S
                # New distances are d_p
                # Add all d_p to D_S
                for d in d_p:
                    D_S.add(d)
                    
        return current_set

    # Run multiple restarts of fast_greedy_construct
    # Then apply local search to the best result found so far.
    
    best_set = fast_greedy_construct()
    best_size = len(best_set)
    
    # Local Search: Simulated Annealing
    # State: current_set (list of points)
    # Move: Remove one point, add a random point from the grid (or a candidate pool).
    # Accept if valid and size increases (or with probability if size same? No, we want max size).
    # Actually, we can only accept if the new set is valid.
    # If valid and size > current size, accept.
    # If valid and size == current size, accept with some probability (to escape local minima)?
    # But we want to maximize size. So we should prefer size increase.
    # However, getting stuck at a local maximum of size k is likely.
    # To increase size, we might need to swap a point to allow a different point to be added later.
    # So "swap" moves (remove one, add one) that keep size constant are useful for rearrangement.
    
    # Let's try:
    # 1. Start with best_set from greedy.
    # 2. Try random swaps (remove i, add j).
    #    - Check if new set is valid.
    #    - If valid, accept the swap (even if size is same).
    #    - This explores the space of valid sets of the same size.
    # 3. Periodically, try to add a point (remove none).
    #    - If valid, size increases.
    # 4. If stuck, try "remove two, add one" or "remove one, add two"?
    #    "Remove one, add two" increases size by 1.
    #    This is a powerful move.
    
    # Let's implement a local search that tries:
    # - Add a random point (if valid) -> size +1
    # - Swap a random point (if valid) -> size same
    # - Remove a random point (if valid? No, removing always valid, but size -1. Only do if we want to reduce complexity? No.)
    # - Remove one, add two (if valid) -> size +1
    
    # To make this fast, we need a fast validity check for the new set.
    # The `is_valid_set` function is O(k^3) with numpy.
    # For k=100, it's fast enough.
    
    def try_add_point(current_set, p):
        # Check if adding p to current_set is valid
        # We can reuse the logic from fast_greedy_construct but for a specific p
        if not current_set:
            return True, current_set + [p]
        
        S = np.array(current_set)
        d_p = squared_dist_vec(p[np.newaxis, :], S)[0]
        
        if len(np.unique(d_p)) < len(d_p):
            return False, current_set
            
        # Check against D_S? We don't have D_S here.
        # We can just call is_valid_set on the new set.
        # Since k is small, this is fine.
        new_set = current_set + [p]
        if is_valid_set(np.array(new_set)):
            return True, new_set
        return False, current_set

    def try_swap_point(current_set, idx_remove, p_add):
        # Remove current_set[idx_remove], add p_add
        new_set = current_set[:idx_remove] + current_set[idx_remove+1:]
        new_set.append(p_add)
        if is_valid_set(np.array(new_set)):
            return True, new_set
        return False, current_set

    def try_remove_one_add_two(current_set, idx_remove, p1, p2):
        new_set = current_set[:idx_remove] + current_set[idx_remove+1:]
        new_set.append(p1)
        new_set.append(p2)
        if is_valid_set(np.array(new_set)):
            return True, new_set
        return False, current_set

    # Main loop for local search
    current_set = best_set[:]
    current_size = len(current_set)
    
    # We will run this for a while
    # To speed up, we can maintain a list of "candidate" points to try adding.
    # Initially, all points.
    
    # Let's just do random trials.
    
    num_iterations = 0
    while time.time() < deadline:
        num_iterations += 1
        
        # Randomly choose a move type
        move_type = random.random()
        
        if move_type < 0.1:
            # Try to add a point
            # Pick a random point from the grid
            idx = random.randint(0, NUM_POINTS - 1)
            p = ALL_POINTS[idx]
            # Check if p is already in current_set
            # To speed up, we can keep a set of indices in current_set
            # But for now, just check.
            # Actually, if p is in current_set, adding it is redundant/invalid (duplicate points?)
            # The task implies distinct points.
            # Let's assume distinct.
            
            # Optimization: Only try points not in current_set.
            # We can maintain a list of available points.
            # For now, just pick random and check.
            
            valid, new_set = try_add_point(current_set, p)
            if valid:
                current_set = new_set
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set[:]
                    
        elif move_type < 0.6:
            # Try to swap a point
            # Pick a random index to remove
            idx_remove = random.randint(0, current_size - 1)
            # Pick a random point to add
            idx_add = random.randint(0, NUM_POINTS - 1)
            p_add = ALL_POINTS[idx_add]
            
            # Check if p_add is already in current_set (and not the one being removed)
            # Simple check:
            # If p_add is in current_set, and it's not at idx_remove, then we are just moving it?
            # No, we remove one and add another. If they are the same point, no change.
            # If p_add is already in the set, we are effectively moving it to a new position?
            # No, the set is a set. If we remove A and add B, and B is already in the set,
            # the size stays the same, but we have replaced A with B (which was already there? No, B is a point).
            # Wait, if B is already in the set, removing A and adding B results in a set with B (still there) and A gone.
            # So the set changes.
            
            valid, new_set = try_swap_point(current_set, idx_remove, p_add)
            if valid:
                current_set = new_set
                # Size is same
                
        else:
            # Try to remove one, add two
            if current_size > 0:
                idx_remove = random.randint(0, current_size - 1)
                idx_add1 = random.randint(0, NUM_POINTS - 1)
                idx_add2 = random.randint(0, NUM_POINTS - 1)
                
                p1 = ALL_POINTS[idx_add1]
                p2 = ALL_POINTS[idx_add2]
                
                # Ensure p1 != p2 and not in set?
                # is_valid_set will handle duplicates?
                # If p1 == p2, the set will have a duplicate?
                # The problem says "list of (x, y) grid points". Usually implies distinct.
                # If we add the same point twice, is it valid?
                # "No three points forming an isosceles triangle".
                # If we have duplicate points, distance is 0.
                # Triangle with sides 0, 0, d is degenerate. Is it isosceles?
                # Usually, distinct points are assumed.
                # Let's enforce distinctness in the move.
                
                if p1[0] == p2[0] and p1[1] == p2[1]:
                    continue
                    
                # Check if p1 or p2 is the one being removed?
                # If p1 is the one being removed, we are effectively just adding p2.
                # That's fine.
                
                valid, new_set = try_remove_one_add_two(current_set, idx_remove, p1, p2)
                if valid:
                    current_set = new_set
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_set[:]
                        
        # Safety check: if current_size < best_size, we might have degraded?
        # No, we only accept valid moves.
        # Add move increases size.
        # Swap keeps size.
        # Remove-Add-2 increases size.
        # So current_size never decreases.
        # Wait, "Remove one, add two" increases size by 1.
        # "Add" increases size by 1.
        # "Swap" keeps size.
        # So we are monotonically increasing the size of `current_set`.
        # This is good. It means we will eventually find a local maximum.
        # To escape local maxima, we might need to allow size-decreasing moves?
        # Or "Remove two, add one" (size -1) to rearrange?
        # If we are stuck at size K, and no Add or Remove-Add-2 works, we are at a local max.
        # To escape, we might need to reduce size to a "better" configuration.
        # Let's add a "Remove one" move (size -1) with low probability, only if stuck.
        
        # How to detect stuck?
        # If we haven't increased size in the last 1000 iterations.
        # Let's add a counter.
        
        # (Simplification: Just run for time. The monotonic increase might get stuck.)
        # Let's add a "Restart" if stuck?
        # Or allow "Remove one" moves.
        
        # Let's add a "Remove one" move with probability 0.1 if we haven't improved in a while.
        # For now, let's stick to the above.
        
    return [(int(x), int(y)) for x, y in best_set]

if __name__ == "__main__":
    # Quick test
    s = construct()
    print(len(s))
