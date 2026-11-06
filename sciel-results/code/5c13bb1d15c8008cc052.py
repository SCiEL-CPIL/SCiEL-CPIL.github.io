import random
import time
import numpy as np

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 285.0
    
    # Precompute all points and squared distances
    all_points = [(x, y) for x in range(N) for y in range(N)]
    n_points = len(all_points)
    
    # Compute squared distance matrix
    xs = np.array([p[0] for p in all_points])
    ys = np.array([p[1] for p in all_points])
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist_matrix = dx * dx + dy * dy
    
    # Build conflict graph: two points conflict if they form an isosceles triangle with any third point
    # For each pair (i,j), check if there exists k such that:
    #   dist(i,k) == dist(j,k)  (isosceles with apex k)
    #   dist(i,k) == dist(i,j)  (isosceles with apex i or j)
    #   dist(j,k) == dist(i,j)  (isosceles with apex i or j)
    
    # We'll use a different approach: build the conflict set for each point
    # Point i conflicts with point j if there exists k != i,j such that:
    #   dist(i,k) == dist(j,k) OR dist(i,k) == dist(i,j) OR dist(j,k) == dist(i,j)
    
    # This is O(n^3) which is too slow for n=4096. Let's use a smarter approach.
    # 
    # Alternative: Use the fact that for a set S, the constraint is that for any three distinct points
    # a,b,c in S, no two of the three distances are equal.
    #
    # Let's use a branch-and-bound on the independent set problem with a different representation.
    # We'll maintain for each point the set of distances to other selected points, and check constraints.
    
    def is_valid_set(points):
        """Check if a set of points has no isosceles triangles."""
        if len(points) < 3:
            return True
        n = len(points)
        # Compute all pairwise squared distances
        for i in range(n):
            for j in range(i + 1, n):
                d_ij = dist_matrix[points[i], points[j]]
                for k in range(i + 1, n):
                    if k == j:
                        continue
                    d_ik = dist_matrix[points[i], points[k]]
                    d_jk = dist_matrix[points[j], points[k]]
                    if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                        return False
        return True
    
    def greedy_with_pruning(seed):
        """
        Greedy construction with aggressive pruning using distance tracking.
        """
        random.seed(seed)
        shuffled = list(range(n_points))
        random.shuffle(shuffled)
        
        selected = []
        # dist_to_selected[i] = set of squared distances from point i to selected points
        dist_to_selected = [set() for _ in range(n_points)]
        
        for idx in shuffled:
            if time.time() > deadline - 5.0:
                break
            
            # Check if adding idx would create an isosceles triangle
            valid = True
            new_dists = []
            
            for sel in selected:
                d = dist_matrix[idx, sel]
                new_dists.append(d)
                
                # Check if sel already has a point at this distance
                if d in dist_to_selected[sel]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check if idx has duplicate distances
            if len(new_dists) != len(set(new_dists)):
                continue
            
            # Accept idx
            selected.append(idx)
            dist_to_selected[idx] = set(new_dists)
            
            # Update dist_to_selected for all existing selected points
            for i, sel in enumerate(selected[:-1]):
                dist_to_selected[sel].add(new_dists[i])
        
        return selected
    
    def local_improve(selected, seed):
        """
        Try to improve a solution by removing and re-adding points.
        """
        random.seed(seed)
        current = list(selected)
        current_len = len(current)
        
        if current_len < 3:
            return current
        
        # Build quick lookup
        current_set = set(current)
        dist_map = {}
        for i in current:
            dists = set()
            for j in current:
                if i != j:
                    dists.add(dist_matrix[i, j])
            dist_map[i] = dists
        
        # Try to find points that can be swapped
        improvements = 0
        max_iterations = 50
        
        for _ in range(max_iterations):
            if time.time() > deadline - 10.0:
                break
            
            # Try adding a random unselected point
            unselected = [i for i in range(n_points) if i not in current_set]
            if not unselected:
                break
            
            random.shuffle(unselected)
            improved = False
            
            for candidate in unselected[:20]:  # Try a few candidates
                # Check if we can add candidate
                valid = True
                new_dists = []
                for sel in current:
                    d = dist_matrix[candidate, sel]
                    new_dists.append(d)
                    if d in dist_map[sel]:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                if len(new_dists) != len(set(new_dists)):
                    continue
                
                # Check if adding candidate creates isosceles with any existing pair
                # We need to check all triples involving candidate
                for i in range(len(current)):
                    for j in range(i + 1, len(current)):
                        a = current[i]
                        b = current[j]
                        d_ab = dist_matrix[a, b]
                        d_ac = dist_matrix[a, candidate]
                        d_bc = dist_matrix[b, candidate]
                        if d_ab == d_ac or d_ab == d_bc or d_ac == d_bc:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Add candidate
                    current.append(candidate)
                    current_set.add(candidate)
                    dist_map[candidate] = set(new_dists)
                    for i, sel in enumerate(current[:-1]):
                        dist_map[sel].add(new_dists[i])
                    improved = True
                    improvements += 1
                    break
            
            if not improved:
                # Try removing a point and adding another
                random.shuffle(current)
                for remove_idx in range(min(3, len(current))):
                    to_remove = current[remove_idx]
                    
                    # Try to find a replacement
                    unselected = [i for i in range(n_points) if i not in current_set]
                    random.shuffle(unselected)
                    
                    for candidate in unselected[:30]:
                        # Temporarily remove to_remove
                        temp_current = [c for c in current if c != to_remove]
                        temp_set = set(temp_current)
                        
                        # Check if we can add candidate to temp_current
                        valid = True
                        new_dists = []
                        for sel in temp_current:
                            d = dist_matrix[candidate, sel]
                            new_dists.append(d)
                            # Check against original dist_map minus to_remove's contribution
                            if d in dist_map[sel] and d != dist_matrix[sel, to_remove]:
                                valid = False
                                break
                        if not valid:
                            continue
                        if len(new_dists) != len(set(new_dists)):
                            continue
                        
                        # Full check
                        test_set = temp_current + [candidate]
                        if len(test_set) < 3:
                            # Accept
                            current.remove(to_remove)
                            current_set.remove(to_remove)
                            del dist_map[to_remove]
                            current.append(candidate)
                            current_set.add(candidate)
                            dist_map[candidate] = set(new_dists)
                            for i, sel in enumerate(current[:-1]):
                                if sel != candidate:
                                    d = dist_matrix[sel, candidate]
                                    if d not in dist_map[sel]:
                                        dist_map[sel].add(d)
                            improvements += 1
                            break
                        
                        # Check all triples
                        n_test = len(test_set)
                        valid_full = True
                        for i in range(n_test):
                            for j in range(i + 1, n_test):
                                a = test_set[i]
                                b = test_set[j]
                                d_ab = dist_matrix[a, b]
                                for k in range(j + 1, n_test):
                                    c = test_set[k]
                                    d_ac = dist_matrix[a, c]
                                    d_bc = dist_matrix[b, c]
                                    if d_ab == d_ac or d_ab == d_bc or d_ac == d_bc:
                                        valid_full = False
                                        break
                                if not valid_full:
                                    break
                            if not valid_full:
                                break
                        
                        if valid_full:
                            # Accept swap
                            current.remove(to_remove)
                            current_set.remove(to_remove)
                            del dist_map[to_remove]
                            current.append(candidate)
                            current_set.add(candidate)
                            dist_map[candidate] = set(new_dists)
                            for i, sel in enumerate(current[:-1]):
                                if sel != candidate:
                                    d = dist_matrix[sel, candidate]
                                    if d not in dist_map[sel]:
                                        dist_map[sel].add(d)
                            improvements += 1
                            break
                    
                    if improvements > 0:
                        break
        
        return current
    
    best_set = []
    best_len = 0
    
    seed = 0
    while time.time() < deadline - 10.0:
        # Run greedy
        result = greedy_with_pruning(seed)
        
        if len(result) > best_len:
            best_set = result
            best_len = len(result)
        
        # Try local improvement on the best result
        if best_len > 0 and time.time() < deadline - 20.0:
            improved = local_improve(best_set, seed + 1000)
            if len(improved) > best_len:
                best_set = improved
                best_len = len(improved)
        
        seed += 1
    
    return [all_points[i] for i in best_set]
# EVOLVE-BLOCK-END
