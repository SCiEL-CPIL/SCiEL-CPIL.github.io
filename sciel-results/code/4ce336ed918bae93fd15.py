import numpy as np
import time
import random

N = 64
# Precompute all points as a matrix of shape (4096, 2)
POINTS = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
NUM_POINTS = N * N

# Precompute squared distances between all pairs of points? 
# 4096 x 4096 matrix of int32 is ~64MB. Acceptable.
# D[i, j] = squared distance between POINTS[i] and POINTS[j]
# We only need upper triangle or full matrix for fast lookup.
DISTS = np.zeros((NUM_POINTS, NUM_POINTS), dtype=np.int64)
# Fill distance matrix
# Vectorized calculation
diff = POINTS[:, np.newaxis, :] - POINTS[np.newaxis, :, :]
DISTS = np.sum(diff ** 2, axis=2)

def construct():
    start_time = time.time()
    deadline = start_time + 285.0
    
    # State: boolean array of selected points
    # We will use Simulated Annealing
    # Objective: Maximize number of points subject to no isosceles triangles.
    
    # Initialize with a random subset or greedy
    # Let's start with an empty set and add points greedily to get a decent start
    # Or just start with a small random set.
    
    # Helper: Check if adding point p to set S is valid
    # Isosceles condition:
    # 1. For any two existing points a, b in S, dist(p, a) != dist(p, b) (p is apex)
    # 2. For any existing point a in S, and any new point p, 
    #    we must ensure that for any other existing point b, 
    #    dist(a, p) != dist(a, b) AND dist(p, a) != dist(p, b) ... wait.
    
    # Let's re-verify the condition for "No three points forming an isosceles triangle".
    # A triangle (a, b, c) is isosceles if any two sides are equal.
    # Sides: d(a,b), d(b,c), d(c,a).
    # Isosceles if d(a,b)==d(a,c) OR d(a,b)==d(b,c) OR d(b,c)==d(a,c).
    
    # When adding p to S:
    # New triangles are (p, a, b) for all a, b in S.
    # Conditions for (p, a, b) NOT to be isosceles:
    # 1. d(p, a) != d(p, b)  (p is apex)
    # 2. d(p, a) != d(a, b)  (a is apex) -> d(a, p) != d(a, b)
    # 3. d(p, b) != d(a, b)  (b is apex) -> d(b, p) != d(b, a)
    
    # So, to check if p can be added to S:
    # Let D_p = [d(p, s) for s in S]
    # 1. All values in D_p must be distinct.
    # 2. For each s in S, let d_ps = d(p, s).
    #    For every other t in S, d_ps != d(s, t).
    #    This means d(p, s) must NOT be in the set of distances from s to any other point in S.
    
    # Data structure for fast checking:
    # selected: list of indices
    # dist_sets: dict mapping index s -> set of squared distances from s to all other selected points
    
    def is_valid_addition(p_idx, selected_indices, dist_sets):
        if not selected_indices:
            return True
        
        # 1. Distances from p to all selected
        d_p = DISTS[p_idx, selected_indices]
        
        # Check distinctness of d_p
        if len(d_p) != len(np.unique(d_p)):
            return False
        
        # 2. For each selected s, check if d(p, s) is already in dist_sets[s]
        # d_p[i] is distance between p and selected_indices[i]
        # We need to check for each i: d_p[i] not in dist_sets[selected_indices[i]]
        
        # Vectorized check?
        # dist_sets is a list of sets. 
        # We can iterate. This is O(K) where K is size of S.
        # Given K ~ 20-40, this is fast.
        
        for i, s_idx in enumerate(selected_indices):
            if d_p[i] in dist_sets[s_idx]:
                return False
                
        return True

    def add_point(p_idx, selected_indices, dist_sets):
        d_p = DISTS[p_idx, selected_indices]
        # Update dist_sets
        dist_sets[p_idx] = set(d_p)
        for i, s_idx in enumerate(selected_indices):
            dist_sets[s_idx].add(d_p[i])
        selected_indices.append(p_idx)
        
    def remove_point(p_idx, selected_indices, dist_sets):
        # Get distances from p to others
        others = [s for s in selected_indices if s != p_idx]
        d_p = DISTS[p_idx, others]
        
        # Remove p's distances from others' sets
        for i, s_idx in enumerate(others):
            dist_sets[s_idx].discard(d_p[i])
            
        # Remove p from selected
        selected_indices.remove(p_idx)
        # Remove p's set
        if p_idx in dist_sets:
            del dist_sets[p_idx]

    # Initialize
    random.seed(42)
    current_selected = []
    current_dist_sets = {}
    
    # Greedy initialization
    all_indices = list(range(NUM_POINTS))
    random.shuffle(all_indices)
    for p_idx in all_indices:
        if is_valid_addition(p_idx, current_selected, current_dist_sets):
            add_point(p_idx, current_selected, current_dist_sets)
            
    best_selected = current_selected[:]
    best_dist_sets = {k: set(v) for k, v in current_dist_sets.items()}
    best_size = len(current_selected)
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    # We want to maximize size.
    # Moves:
    # 1. Add a random unselected point if valid. (Size +1)
    # 2. Remove a random selected point and add a different random point. (Size 0 or -1)
    #    Actually, standard SA for packing usually tries to swap to escape local minima.
    #    If we only add, we get stuck.
    #    Let's allow:
    #    - Add: if valid, accept (improvement).
    #    - Swap: Remove p_old, Add p_new.
    #      If p_new is valid in (S \ {p_old}), we can add it.
    #      Size remains same.
    #      This allows rearranging the set to allow future additions.
    #    - Remove: Remove p_old. Size -1.
    #      Accept with probability exp(-1/T) (since it's worse).
    
    # To make it efficient, we need to quickly find valid additions.
    # We'll just pick random candidates.
    
    unselected_mask = np.ones(NUM_POINTS, dtype=bool)
    for idx in current_selected:
        unselected_mask[idx] = False
        
    # Precompute list of unselected indices for fast sampling
    unselected_indices = np.where(unselected_mask)[0]
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to Add
            if len(unselected_indices) == 0:
                continue
            p_new = random.choice(unselected_indices)
            
            if is_valid_addition(p_new, current_selected, current_dist_sets):
                # Accept
                add_point(p_new, current_selected, current_dist_sets)
                unselected_mask[p_new] = False
                unselected_indices = np.where(unselected_mask)[0]
                
                if len(current_selected) > best_size:
                    best_size = len(current_selected)
                    best_selected = current_selected[:]
                    best_dist_sets = {k: set(v) for k, v in current_dist_sets.items()}
            else:
                # Reject (or accept with prob 0 since it's invalid)
                pass
                
        elif move_type < 0.9:
            # Try Swap: Remove p_old, Add p_new
            if len(current_selected) == 0 or len(unselected_indices) == 0:
                continue
            
            p_old = random.choice(current_selected)
            p_new = random.choice(unselected_indices)
            
            if p_old == p_new:
                continue
            
            # Check if p_new is valid in S \ {p_old}
            # Temporarily remove p_old
            # To avoid modifying state, we can simulate the check.
            # is_valid_addition(p_new, S \ {p_old}, dist_sets \ {p_old} adjusted)
            
            # Let's just do it with modification and rollback if rejected.
            # But "valid" check is strict. If valid, we swap.
            # If not valid, we don't swap.
            # This is a deterministic improvement/neutral move.
            # To escape local optima, we might need to allow invalid swaps?
            # No, the constraint is hard. We must maintain validity.
            # So swaps must result in a valid set.
            
            # Check validity of p_new in S_without_p_old
            # We need to construct the "virtual" set of distances.
            # Distances from p_new to all s in S \ {p_old} must be distinct and not conflict with existing distances in S \ {p_old}.
            
            others = [s for s in current_selected if s != p_old]
            if not others:
                # S becomes empty, p_new is always valid
                valid = True
            else:
                d_new = DISTS[p_new, others]
                if len(d_new) != len(np.unique(d_new)):
                    valid = False
                else:
                    for i, s_idx in enumerate(others):
                        # Check if d_new[i] is in dist_sets[s_idx] EXCLUDING p_old's contribution?
                        # dist_sets[s_idx] contains d(s_idx, p_old).
                        # We are removing p_old, so d(s_idx, p_old) is no longer in the set of "other points".
                        # So we check if d_new[i] is in (dist_sets[s_idx] - {d(s_idx, p_old)})
                        d_old_to_s = DISTS[s_idx, p_old]
                        if d_new[i] in dist_sets[s_idx]:
                            # If it's there, is it ONLY from p_old?
                            # If d_new[i] == d_old_to_s, then after removing p_old, it might be gone.
                            # But if d_new[i] == d_old_to_s, and there is another t in S with d(s_idx, t) == d_old_to_s,
                            # then d_new[i] is still in the set.
                            # We need to check if there is ANY t in S \ {p_old} such that d(s_idx, t) == d_new[i].
                            # This is expensive to check naively.
                            # However, dist_sets[s_idx] is a set. It doesn't tell us multiplicity.
                            # We can just check:
                            # Is there a t in others such that DISTS[s_idx, t] == d_new[i]?
                            # This is O(K).
                            found_conflict = False
                            for t in others:
                                if t != s_idx and DISTS[s_idx, t] == d_new[i]:
                                    found_conflict = True
                                    break
                            if found_conflict:
                                valid = False
                                break
                            # If not found, it means the only point at that distance was p_old (or none).
                            # If it was p_old, it's removed, so no conflict.
                            # If it was none, no conflict.
                            # Wait, if d_new[i] is in dist_sets[s_idx], it MUST have come from some point in S.
                            # If that point was p_old, and we removed p_old, is it still in the set of distances to OTHERS?
                            # No.
                            # So if d_new[i] is in dist_sets[s_idx], we check if any OTHER point in S (excluding p_old) has that distance.
                            # If yes -> conflict.
                            # If no -> no conflict (the distance value is now "free" for s_idx).
                            
            if valid:
                # Perform Swap
                remove_point(p_old, current_selected, current_dist_sets)
                add_point(p_new, current_selected, current_dist_sets)
                
                unselected_mask[p_old] = True
                unselected_mask[p_new] = False
                unselected_indices = np.where(unselected_mask)[0]
                
                # Size is same, so no best update unless we track structure?
                # We only care about size.
                
        else:
            # Try Remove (to unblock future additions?)
            # Rarely useful unless stuck.
            if len(current_selected) > 0:
                p_old = random.choice(current_selected)
                remove_point(p_old, current_selected, current_dist_sets)
                unselected_mask[p_old] = True
                unselected_indices = np.where(unselected_mask)[0]
                
        T *= cooling
        
        # Check time
        if time.time() > deadline - 5:
            break
            
    # Final check: return best set found
    # Convert indices to coordinates
    return [tuple(POINTS[idx]) for idx in best_selected]

# EVOLVE-BLOCK-END
