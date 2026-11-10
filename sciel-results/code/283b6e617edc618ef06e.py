"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 64

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 320  # Safety margin before 330s kill
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    n_points = len(all_points)
    
    # Precompute squared distances between all pairs
    dist_sq = [[0] * n_points for _ in range(n_points)]
    for i in range(n_points):
        xi, yi = all_points[i]
        for j in range(i + 1, n_points):
            xj, yj = all_points[j]
            d = (xi - xj) ** 2 + (yi - yj) ** 2
            dist_sq[i][j] = d
            dist_sq[j][i] = d
    
    # Build conflict graph: two points conflict if they can form an isosceles triangle with some third point
    # For each pair (i, j), check if there exists k such that dist(i,k) == dist(j,k) or dist(i,k) == dist(i,j) or dist(j,k) == dist(i,j)
    # Actually, three points form an isosceles triangle if any two of the three pairwise distances are equal.
    # Conflict between i and j means: adding both i and j to a set would create an isosceles triangle with some existing point k.
    # But for maximum independent set on conflict graph, we need: edge between i and j if they CANNOT both be in the solution.
    # Two points i, j cannot both be in the solution if there exists a k (k != i, k != j) such that {i, j, k} forms an isosceles triangle.
    # Actually, the constraint is global: no THREE points form an isosceles triangle. So the conflict graph should have an edge between
    # i and j if i and j are part of any isosceles triple. But that's not quite right for MIS either.
    # 
    # Let me reconsider. The constraint is: no three selected points form an isosceles triangle.
    # This is NOT a pairwise constraint in the simple sense. It's a 3-uniform hypergraph constraint.
    # 
    # Alternative approach: Greedy with local improvement.
    
    def is_valid_set(point_set):
        """Check if no three points in the set form an isosceles triangle."""
        n = len(point_set)
        # Map point to index for O(1) lookup
        point_to_idx = {p: i for i, p in enumerate(point_set)}
        
        for i in range(n):
            pi = point_set[i]
            # Collect distances from pi to all other points
            dists_from_i = {}
            for j in range(n):
                if i == j:
                    continue
                pj = point_set[j]
                d = (pi[0] - pj[0]) ** 2 + (pi[1] - pj[1]) ** 2
                if d not in dists_from_i:
                    dists_from_i[d] = []
                dists_from_i[d].append(j)
            
            # If any distance appears 2 or more times, then pi with those two points forms isosceles
            for d, pts in dists_from_i.items():
                if len(pts) >= 2:
                    return False
        
        # Also need to check cases where the equal distances are not from the same vertex
        # Actually, the above check covers all cases: for each vertex, if two other vertices are at the same distance,
        # then those three form an isosceles triangle with the first vertex as the apex.
        # But we also need to check if the base is at the same distance... no, the apex check is sufficient.
        # If triangle has two equal sides, the apex is the vertex where the two equal sides meet.
        # So checking for each vertex if two others are at the same distance covers all isosceles triangles.
        
        return True
    
    def try_add_point(current_set, point, point_to_idx):
        """Check if adding point to current_set keeps it valid."""
        # For each existing point in current_set, check if the new point creates an isosceles triangle
        # For each pair (existing1, existing2), check if {existing1, existing2, point} is isosceles
        # Also for each existing point, check if {existing, point, existing2} for some existing2 is isosceles
        
        n = len(current_set)
        px, py = point
        
        # Check if point conflicts with any pair in current_set
        # For each pair (i, j) in current_set, check if {current_set[i], current_set[j], point} is isosceles
        for i in range(n):
            xi, yi = current_set[i]
            di = (px - xi) ** 2 + (py - yi) ** 2
            for j in range(i + 1, n):
                xj, yj = current_set[j]
                dj = (px - xj) ** 2 + (py - yj) ** 2
                dij = (xi - xj) ** 2 + (yi - yj) ** 2
                # Check if any two of di, dj, dij are equal
                if di == dj or di == dij or dj == dij:
                    return False
        return True
    
    # Greedy construction with randomization
    def greedy_construct(order, deadline):
        result = []
        result_set = set()
        for point in order:
            if time.time() > deadline:
                break
            if try_add_point(result, point, None):
                result.append(point)
                result_set.add(point)
        return result
    
    # Local improvement: try to swap points
    def local_improve(point_set, deadline):
        point_set = list(point_set)
        n = len(point_set)
        improved = True
        # Limit iterations to prevent infinite loops or excessive time
        max_iters = 20 
        iter_count = 0
        while improved and iter_count < max_iters:
            iter_count += 1
            if time.time() > deadline:
                break
            improved = False
            # Try removing each point
            for i in range(n):
                if time.time() > deadline:
                    return point_set
                p = point_set[i]
                new_set = point_set[:i] + point_set[i+1:]
                # Try adding a random point not in the set
                # Optimization: Don't build the full candidate list every time if it's huge
                # But all_points is fixed. 
                # We pick a random subset of candidates to try
                # Instead of filtering all_points (O(N)), we can just pick random points from all_points
                # and check if they are in the set.
                found = False
                # Try 50 random candidates
                for _ in range(50):
                    if time.time() > deadline:
                        return point_set
                    q = random.choice(all_points)
                    if q in set(new_set):
                        continue
                    if try_add_point(new_set, q, None):
                        point_set = new_set + [q]
                        n = len(point_set)
                        improved = True
                        found = True
                        break
                if improved:
                    break
        return point_set
    
    # Run multiple greedy passes with different random orders
    best_set = []
    
    # Reduce trials to avoid timeout, rely on better time management per trial
    # 10 trials is often enough for this size if the heuristic is decent
    num_trials = 20
    
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break
        
        # Random order
        order = all_points[:]
        random.shuffle(order)
        
        result = greedy_construct(order, deadline)
        
        # Local improvement
        result = local_improve(result, deadline)
        
        if len(result) > len(best_set):
            best_set = result
        
        # If we have enough time, do more trials
        if time.time() > deadline - 5:
            break
    
    # Final local improvement on best_set
    if time.time() < deadline - 2:
        best_set = local_improve(best_set, deadline)
    
    return best_set
# EVOLVE-BLOCK-END
