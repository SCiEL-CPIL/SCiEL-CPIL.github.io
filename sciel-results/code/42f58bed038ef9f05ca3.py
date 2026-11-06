import random
import time
import math
import sys

N = 64

def construct():
    # Precompute all grid points
    points = [(x, y) for x in range(N) for y in range(N)]
    num_points = len(points)
    
    # Map point to index
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # Precompute squared distances between all pairs
    # dist[i][j] = squared distance between points[i] and points[j]
    # Using a list of lists for faster access than dict
    dist = [[0] * num_points for _ in range(num_points)]
    for i in range(num_points):
        xi, yi = points[i]
        for j in range(i + 1, num_points):
            xj, yj = points[j]
            dx = xi - xj
            dy = yi - yj
            d = dx*dx + dy*dy
            dist[i][j] = d
            dist[j][i] = d
            
    def is_valid_set(point_indices):
        """
        Check if the set of point indices forms a no-isosceles set.
        Returns True if valid.
        """
        n = len(point_indices)
        if n < 3:
            return True
            
        # For each point i in the set, compute distances to all other points in the set
        # d_p[i] is not needed globally, we need to check triples.
        
        # Efficient check:
        # 1. For each point p in set, check if it is equidistant to any two other points q, r.
        #    This means for fixed p, the set of distances {d(p, x) for x in set, x != p} must have unique values.
        # 2. For each point p in set, and for each other point q in set, 
        #    check if d(p, q) == d(q, r) for any r in set, r != p, r != q.
        #    This is equivalent to: for fixed q, the set of distances {d(q, x) for x in set, x != q} 
        #    must not contain d(p, q) if p is the apex? No.
        
        # Let's stick to the O(n^2) check with precomputed distance matrix.
        # For each pair (i, j) in set:
        #   d_ij = dist[i][j]
        #   For each k in set, k != i, k != j:
        #      d_ik = dist[i][k]
        #      d_jk = dist[j][k]
        #      If d_ik == d_jk: Isosceles (apex at k) -> Invalid
        #      If d_ik == d_ij: Isosceles (apex at i) -> Invalid
        #      If d_jk == d_ij: Isosceles (apex at j) -> Invalid
        
        # This is O(n^3) if done naively.
        # Optimization:
        # For each i in set:
        #   Compute dict: dist_from_i = {j: dist[i][j] for j in set, j != i}
        #   If len(dist_from_i) < len(set)-1: Duplicate distance from i to two others. Invalid. (Apex at i)
        #   For each j in set, j != i:
        #       d_ij = dist[i][j]
        #       # Check if there exists k != i, k != j such that dist[j][k] == d_ij (Apex at j)
        #       # And dist[i][k] == d_ij (Apex at i, already checked by uniqueness? No, uniqueness of dist_from_i checks d(i,k1)==d(i,k2). 
        #       # It does NOT check d(i,k) == d(i,j). 
        #       # Wait. If d(i,k) == d(i,j), then i is apex. This is caught by "len(dist_from_i) < n-1" if k and j are distinct.
        #       # So "Apex at i" is fully handled by uniqueness of distances from i.
        #       
        #       # Now check Apex at j: d(j,k) == d(j,i) = d_ij.
        #       # We need to know if any k (k!=i, k!=j) has dist[j][k] == d_ij.
        #       # We can precompute for each j, the set of distances to other points in the set?
        #       # That's expensive to update.
        
        # Alternative O(n^2) check:
        # For each i in set:
        #    seen_dists_i = set()
        #    for j in set:
        #        if i == j: continue
        #        d_ij = dist[i][j]
        #        if d_ij in seen_dists_i:
        #            return False # Apex at i
        #        seen_dists_i.add(d_ij)
        
        # This handles "Apex at i" for all i.
        # Does it handle "Apex at j" where d(j,k) == d(j,i)?
        # If d(j,k) == d(j,i), then for point j, the distances to i and k are equal.
        # So when we iterate i=j, we will see d(j,i) and d(j,k) are equal.
        # So the single loop "For each i, check uniqueness of distances from i to all others" is SUFFICIENT.
        
        # Let's verify.
        # Isosceles triangle ABC.
        # Case 1: AB = AC. Apex A. Distances from A to B and C are equal. Caught when i=A.
        # Case 2: BA = BC. Apex B. Distances from B to A and C are equal. Caught when i=B.
        # Case 3: CA = CB. Apex C. Distances from C to A and B are equal. Caught when i=C.
        
        # Yes! The condition "No three points form an isosceles triangle" is equivalent to:
        # For every point P in the set, the distances from P to all other points in the set are distinct.
        
        # So the check is simply:
        # For each p in set:
        #    d_p = [dist[p][q] for q in set if q != p]
        #    if len(set(d_p)) != len(d_p): return False
        # return True
        
        # Complexity: O(n^2) per check.
        # For n=30, 900 ops. Very fast.
        
        for p in point_indices:
            seen = set()
            for q in point_indices:
                if p == q:
                    continue
                d = dist[p][q]
                if d in seen:
                    return False
                seen.add(d)
        return True

    # Initial Greedy Solution
    def greedy_initial():
        current = []
        random.shuffle(points)
        for p in points:
            idx = point_to_idx[p]
            new_set = current + [idx]
            if is_valid_set(new_set):
                current.append(idx)
        return current

    # Simulated Annealing
    best_set = greedy_initial()
    best_size = len(best_set)
    
    # We can try multiple initializations if time permits, but SA should explore well.
    current_set = best_set[:]
    current_size = best_size
    
    start_time = time.time()
    time_limit = 280 # Leave 20s buffer
    
    T = 10.0
    cooling = 0.999
    
    # To speed up validity check, we can keep the set as a list of indices.
    # We need to generate moves.
    
    while time.time() - start_time < time_limit:
        # Generate move
        move_type = random.random()
        
        if move_type < 0.3:
            # Try to add a random point
            p_idx = random.randint(0, num_points - 1)
            if p_idx not in current_set:
                new_set = current_set + [p_idx]
                if is_valid_set(new_set):
                    # Always accept size increase
                    current_set = new_set
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_set[:]
        elif move_type < 0.6:
            # Try to remove a random point
            if current_size > 0:
                rem_idx = random.randint(0, current_size - 1)
                p_idx = current_set[rem_idx]
                new_set = current_set[:]
                new_set.pop(rem_idx)
                # Always valid
                # Accept with probability based on energy (size decrease)
                delta_E = 1.0 # size decreases by 1, energy increases by 1 (since E = -size)
                if random.random() < math.exp(-delta_E / T):
                    current_set = new_set
                    current_size -= 1
        else:
            # Try to swap
            if current_size > 0:
                rem_pos = random.randint(0, current_size - 1)
                p_idx = current_set[rem_pos]
                
                # Pick a random point to swap in
                q_idx = random.randint(0, num_points - 1)
                if q_idx != p_idx and q_idx not in current_set:
                    new_set = current_set[:]
                    new_set[rem_pos] = q_idx
                    if is_valid_set(new_set):
                        # Size is same, so delta_E = 0. Always accept? 
                        # Or accept with prob 1 to allow random walk?
                        # Yes, accept to explore.
                        current_set = new_set
                        # current_size unchanged
                        
        T *= cooling
        if T < 0.01:
            T = 0.01
            
    # Convert indices back to points
    return [points[i] for i in best_set]

# Run construct to get the result
# The function construct is the entry point.

