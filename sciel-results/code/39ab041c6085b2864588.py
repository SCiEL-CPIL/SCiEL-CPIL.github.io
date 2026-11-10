import time
import random
import math

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    best_set = []
    best_size = 0
    
    def greedy_select(points_order):
        selected = []
        # dist_from[i] is a set of squared distances from selected[i] to all other selected points
        dist_from = []
        
        for p in points_order:
            px, py = p
            valid = True
            d_p = []
            
            # Calculate distances from P to all selected points
            for i in range(len(selected)):
                ax, ay = selected[i]
                d_sq = (px - ax)**2 + (py - ay)**2
                d_p.append(d_sq)
            
            # Check 1: P is apex. All d_p must be unique.
            if len(set(d_p)) != len(d_p):
                valid = False
            else:
                # Check 2: For each A, d(P,A) must not be equal to d(A,B) for any B
                for i in range(len(selected)):
                    d_pa = d_p[i]
                    # dist_from[i] contains distances from selected[i] to all OTHER selected points
                    if d_pa in dist_from[i]:
                        valid = False
                        break
                        
            if valid:
                selected.append(p)
                # Update dist_from
                new_dist_set = set()
                for i in range(len(selected) - 1):
                    new_dist_set.add(d_p[i])
                dist_from.append(new_dist_set)
                
                # Update existing dist_from sets with new distance to P
                for i in range(len(selected) - 1):
                    dist_from[i].add(d_p[i])
                    
        return selected, dist_from

    def local_search(selected, dist_from, time_limit=5.0):
        """
        Attempt to improve the set by removing and re-adding points.
        Returns the best set found.
        """
        end_time = time.time() + time_limit
        current_set = selected[:]
        current_dist = [s.copy() for s in dist_from]
        best_local_set = current_set[:]
        best_local_size = len(current_set)
        
        # Map point -> index in current_set for quick lookup? 
        # Actually, we just need to know if a point is in the set.
        current_set_set = set(current_set)
        
        # We will try to remove a point and then try to add a different one.
        # Or simply remove a "bad" point and let greedy fill it back?
        # Better: Remove a point, then try to add ALL remaining points (greedy) to see if we get bigger.
        # This is expensive. 
        # Alternative: Swap. Remove P, add Q.
        
        # Let's try a simpler perturbation:
        # 1. Pick a random point in the set to remove.
        # 2. Try to add random points from the pool that are not in the set.
        
        # To be efficient, we need to know which points are NOT in the set.
        # all_points is 4096. current_set is ~100-200.
        # We can maintain a list of "available" points.
        
        # Let's implement a "kick" move:
        # Remove k random points from the set.
        # Then run a limited greedy pass on the remaining available points to fill up.
        
        available_points = [p for p in all_points if p not in current_set_set]
        
        while time.time() < end_time:
            # Pick a random point to remove
            if not current_set:
                break
                
            idx_to_remove = random.randint(0, len(current_set) - 1)
            removed_point = current_set[idx_to_remove]
            
            # Remove it from data structures
            # This is tricky with the dist_from list.
            # If we remove index i, we need to remove dist_from[i] and remove the distance to i from all other dist_from sets.
            
            # Let's just rebuild from scratch if the set is small? No, that's slow.
            # Let's implement the removal carefully.
            
            # 1. Remove the distance from removed_point to all other points in their dist sets.
            for i in range(len(current_set)):
                if i == idx_to_remove:
                    continue
                ax, ay = current_set[i]
                rx, ry = removed_point
                d_sq = (ax - rx)**2 + (ay - ry)**2
                if d_sq in current_dist[i]:
                    current_dist[i].discard(d_sq)
            
            # 2. Remove the point and its dist set
            current_set.pop(idx_to_remove)
            current_dist.pop(idx_to_remove)
            current_set_set.discard(removed_point)
            
            # 3. Try to add new points greedily from available_points
            # We need to check validity against the current_set.
            # To make this faster, we can shuffle available_points and try a few.
            # Or just iterate through a random subset.
            
            # Let's try to add up to 3 points to compensate for the 1 removed.
            attempts = 0
            max_attempts = 10
            while attempts < max_attempts and time.time() < end_time:
                # Pick a random candidate
                if not available_points:
                    break
                candidate = random.choice(available_points)
                attempts += 1
                
                # Check if candidate can be added
                px, py = candidate
                valid = True
                d_p = []
                
                for i in range(len(current_set)):
                    ax, ay = current_set[i]
                    d_sq = (px - ax)**2 + (py - ay)**2
                    d_p.append(d_sq)
                
                if len(set(d_p)) != len(d_p):
                    continue
                    
                for i in range(len(current_set)):
                    if d_p[i] in current_dist[i]:
                        valid = False
                        break
                        
                if valid:
                    # Add it
                    current_set.append(candidate)
                    new_dist_set = set(d_p)
                    current_dist.append(new_dist_set)
                    current_set_set.add(candidate)
                    
                    # Update existing dist sets
                    for i in range(len(current_set) - 1):
                        current_dist[i].add(d_p[i])
                        
                    # Remove from available
                    try:
                        available_points.remove(candidate)
                    except ValueError:
                        pass
                    
                    # If we added one, we might have balanced the size.
                    # Break to try removing another point?
                    # Or continue adding?
                    # Let's break to keep the loop tight.
                    break
            
            if len(current_set) > best_local_size:
                best_local_size = len(current_set)
                best_local_set = current_set[:]
                
        return best_local_set

    # Run multiple restarts
    while time.time() < deadline:
        random.shuffle(all_points)
        current_set, current_dist = greedy_select(all_points)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
        # If we have time, do local search
        if time.time() < deadline - 10:
            improved_set = local_search(current_set, current_dist, time_limit=2.0)
            if len(improved_set) > best_size:
                best_size = len(improved_set)
                best_set = improved_set[:]
                
    return best_set
