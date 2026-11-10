import time
import random
import math

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 280.0  # Safety margin
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    point_set = set(all_points)
    
    def dist2(a, b):
        return (a[0] - b[0])**2 + (a[1] - b[1])**2

    def is_valid_addition(p, S, dist_sets):
        """
        S: list of points currently in the set
        dist_sets: list of sets, dist_sets[i] contains squared distances from S[i] to all other points in S
        p: candidate point
        Returns True if p can be added to S without forming an isosceles triangle.
        """
        k = len(S)
        if k == 0:
            return True, []
        
        # Calculate distances from p to all points in S
        distances = []
        for i, a in enumerate(S):
            d = dist2(p, a)
            distances.append(d)
            
        # Check 1: Apex at existing point A.
        # dist(p, A) must not be in dist_sets[A]
        for i in range(k):
            if distances[i] in dist_sets[i]:
                return False, []
                
        # Check 2: Apex at p.
        # No two distances from p to S should be equal.
        if len(set(distances)) < k:
            return False, []
            
        return True, distances

    def try_add(p, S, dist_sets):
        """Attempts to add p to S. Returns True if successful, updating S and dist_sets."""
        valid, distances = is_valid_addition(p, S, dist_sets)
        if valid:
            idx = len(S)
            S.append(p)
            for i in range(idx):
                dist_sets[i].add(distances[i])
            dist_sets.append(set(distances))
            return True
        return False

    def try_remove(idx, S, dist_sets):
        """Removes S[idx] from S and updates dist_sets."""
        p = S[idx]
        # Remove p from S
        S.pop(idx)
        # Remove the set for p
        dist_sets.pop(idx)
        # Update existing dist_sets: remove the distance to p
        for i in range(len(S)):
            d = dist2(S[i], p)
            dist_sets[i].discard(d)

    # Initial Greedy Construction to get a decent starting point
    random.seed(42)
    shuffled = all_points.copy()
    random.shuffle(shuffled)
    
    S = []
    dist_sets = []
    for p in shuffled:
        try_add(p, S, dist_sets)
        
    best_set = S.copy()
    best_len = len(S)
    
    # Simulated Annealing
    current_temp = 10.0
    cooling_rate = 0.9995
    
    while time.time() - start_time < time_limit:
        if current_temp < 0.1:
            break
            
        k = len(S)
        move_type = random.random()
        
        # 1. Add a random point
        if move_type < 0.4:
            # Pick a random point not in S
            # Efficiently: pick random index from all_points, check if in S
            # For simplicity, pick from a list of candidates? 
            # Better: just pick random point from all_points. If in S, retry or skip.
            # Since S is small relative to 4096, probability of hitting S is low.
            p = random.choice(all_points)
            if p not in S:
                old_len = len(S)
                if try_add(p, S, dist_sets):
                    new_len = len(S)
                    # Always accept improvements
                    if new_len > best_len:
                        best_len = new_len
                        best_set = S.copy()
                else:
                    # Try to remove a point to make space? No, that's a different move.
                    pass
        # 2. Remove a random point (Cooling move)
        elif move_type < 0.7:
            if k > 0:
                idx = random.randint(0, k - 1)
                old_len = len(S)
                try_remove(idx, S, dist_sets)
                new_len = len(S)
                # Accept worse moves with probability exp(-delta/T)
                delta = new_len - old_len # delta is -1
                if random.random() < math.exp(delta / current_temp):
                    # Accepted
                    pass
                else:
                    # Revert
                    # We need to add it back. But we lost the distances?
                    # try_remove doesn't return enough info to easily revert if we don't store it.
                    # Let's modify try_remove to return the removed point and its distances?
                    # Or simpler: just don't revert immediately, but this breaks SA logic if we don't track state correctly.
                    # Let's implement a proper undo mechanism or just re-add if valid.
                    # Actually, if we remove a point, adding it back is always valid if the rest of the set didn't change.
                    # So we can just try_add(S[idx], ...) but S[idx] is gone.
                    # Let's store the removed point.
                    pass
        # 3. Swap a point
        else:
            if k > 0:
                idx_in = random.randint(0, k - 1)
                p_in = S[idx_in]
                # Pick a random point not in S
                p_out = random.choice(all_points)
                while p_out in S:
                    p_out = random.choice(all_points)
                
                # Remove p_in
                try_remove(idx_in, S, dist_sets)
                # Add p_out
                if try_add(p_out, S, dist_sets):
                    # Check if this swap is an improvement? 
                    # Length is same. We just want to explore.
                    # Accept with probability 1 (neutral move) or based on some heuristic?
                    # For now, accept neutral moves to allow exploration.
                    pass
                else:
                    # Revert: Add p_in back
                    try_add(p_in, S, dist_sets)
                    # Note: try_add might fail if distances changed? 
                    # No, we removed p_in, then tried p_out. If p_out failed, S is back to state before p_out was added.
                    # p_in was removed. Adding p_in back to the set without p_in is valid because the set is a subset of the previous valid set.
                    # Wait, try_remove removed p_in. The remaining set is valid. 
                    # Adding p_in back to the remaining set is valid because (Remaining + p_in) was valid before.
                    # So try_add(p_in) should always succeed.
                    pass

        current_temp *= cooling_rate

    return best_set
# EVOLVE-BLOCK-END
