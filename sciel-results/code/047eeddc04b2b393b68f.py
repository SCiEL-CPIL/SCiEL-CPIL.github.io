import numpy as np
import random
import time

# Configuration
N = 64
GRID_SIZE = N * N
TIME_LIMIT = 290.0  # Safety margin before 300s hard kill

# Pre-compute coordinates as numpy arrays
X = np.arange(N, dtype=np.int32)
Y = np.arange(N, dtype=np.int32)
XX, YY = np.meshgrid(X, Y, indexing='ij')
COORDS = np.stack([XX.ravel(), YY.ravel()], axis=1)
COORDS = COORDS.astype(np.int32)

# Pre-compute squared distances matrix? 
# 64*64 = 4096 points. 4096x4096 int32 matrix is 16MB. This is fine.
# However, calculating it once is O(N^4) if done naively, but we can do it vectorized.
# Let's precompute DIST2[i, j] = (x_i - x_j)^2 + (y_i - y_j)^2
# We only need this for points in the set, but having the full matrix allows fast lookups.
# Actually, for Simulated Annealing, we need to check if adding point k to set S creates an isosceles triangle.
# This requires checking all pairs (a, b) in S.
# If S is size K, checking a new point k involves O(K^2) distance checks.
# Vectorized: 
#   d_ka = DIST2[k, S] (vector of length K)
#   d_ab = DIST2[S, S] (matrix KxK)
#   We need to check if any d_ka[i] == d_ka[j] for i!=j (k is apex)
#   OR if d_ka[i] == d_ab[i, j] (a is apex)
#   OR if d_ka[j] == d_ab[i, j] (b is apex)
#
# Let's precompute the full distance squared matrix.
print("Precomputing distances...")
start_prep = time.time()
DIST2 = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
# Vectorized distance calculation
# COORDS is (4096, 2)
# diff = COORDS[:, np.newaxis, :] - COORDS[np.newaxis, :, :] -> (4096, 4096, 2)
# This creates a 4096*4096*2 int32 array, which is ~128MB. Might be okay, but let's be careful.
# Alternatively, loop over rows?
# Let's try the vectorized way first, if memory is an issue, we'll fall back.
diff = COORDS[:, np.newaxis, :] - COORDS[np.newaxis, :, :]
DIST2 = np.sum(diff**2, axis=2)
print(f"Precomputation took {time.time() - start_prep:.2f}s")

def count_isosceles(points_idx):
    """
    Count the number of isosceles triangles in the set of points.
    points_idx: numpy array of indices into COORDS.
    Returns: (count, is_valid) where is_valid is True if count == 0.
    """
    K = len(points_idx)
    if K < 3:
        return 0, True
    
    # Extract distance matrix for the subset
    # D_sub[i, j] = DIST2[points_idx[i], points_idx[j]]
    D_sub = DIST2[np.ix_(points_idx, points_idx)]
    
    # Triangles are defined by triples (i, j, k) with i < j < k.
    # We need to check if any of the three sides are equal.
    # Sides: d_ij, d_ik, d_jk
    
    # To vectorize this, we can iterate over one index, say k (the largest in the triple),
    # and check all pairs (i, j) with i < j < k.
    # But K can be up to ~100-200. K^3 is 1e6 - 8e6. 
    # A pure Python loop over K^3 is too slow.
    # A vectorized check for a fixed k:
    #   Let S_k be the subset of points with index < k in the sorted list?
    #   No, the order in points_idx is arbitrary.
    #   
    # Let's use the property: 
    # For a fixed vertex v, let d_v be the vector of distances from v to all other vertices in the set.
    # If there are duplicate distances in d_v, then v is the apex of an isosceles triangle.
    # This covers the case where the two equal sides meet at v.
    # 
    # Algorithm:
    # 1. For each v in the set:
    #    a. Get distances from v to all other points in the set.
    #    b. Check for duplicates. If any, increment count.
    #    c. However, this counts each isosceles triangle with a unique apex once.
    #       An equilateral triangle has 3 apexes? No, in an equilateral triangle, all 3 vertices are apexes of isosceles triangles (with the base being the other two sides).
    #       Wait, the definition is "no three points forming an isosceles triangle".
    #       An equilateral triangle IS an isosceles triangle.
    #       If we count duplicates of distances from v, we find triangles where v is the apex (two sides equal).
    #       Does this cover all isosceles triangles?
    #       Yes. Any isosceles triangle has at least one vertex where the two adjacent sides are equal.
    #       So if we sum over all v the number of duplicate distance pairs, we count the number of isosceles triangles (with multiplicity if equilateral).
    #       We just need count > 0 to be invalid.
    
    count = 0
    # To speed up, we can vectorize the "duplicate check" for all v at once?
    # D_sub is KxK.
    # For each row i, we want to know if there are any duplicate values in D_sub[i, :j] and D_sub[i, j+1:]?
    # Actually, we just need to know if ANY row has a duplicate value (excluding the diagonal 0).
    
    # Let's iterate rows in Python but use numpy for the row check?
    # K is likely < 200. 200 iterations of numpy operations is fast.
    for i in range(K):
        # Get distances from point i to all other points
        # Exclude self (distance 0)
        d_i = D_sub[i, :].copy()
        d_i[i] = -1 # Mark self
        
        # Check for duplicates among the remaining K-1 distances
        # We can sort and check for adjacent equals, or use a set.
        # Since K is small, sorting is O(K log K).
        if K > 2:
            # Remove the -1
            d_valid = d_i[d_i != -1]
            if len(d_valid) > 1:
                # Check for duplicates
                # np.unique returns unique values and counts
                _, counts = np.unique(d_valid, return_counts=True)
                if np.any(counts > 1):
                    count += 1 # At least one isosceles triangle with apex i
                    # We could count exactly how many, but we only care if > 0 for validity.
                    # For SA, we want the exact count to gauge "how bad" it is.
                    # Let's count the number of pairs with equal distances.
                    # This is sum((count-1) choose 2)? No.
                    # If count of distance d is c, the number of pairs is c choose 2.
                    # So add sum(c*(c-1)//2 for c in counts if c>1)
                    count += int(np.sum(counts[counts > 1] * (counts[counts > 1] - 1) // 2))
                    # Wait, the previous line already added 1. Let's just do the exact count.
                    # Reset count for this i to 0 before adding.
                    # Actually, let's just return the total count.
                    # The logic above: if any duplicate, it's invalid.
                    # For SA, we want the objective function to be the number of isosceles triangles.
                    # Let's refine the counting.
    
    # The loop above is a bit messy. Let's rewrite the counting logic cleanly.
    # We want the total number of isosceles triangles.
    # An isosceles triangle is defined by a set of 3 points {a,b,c} such that at least two distances are equal.
    # Summing over apexes:
    # For each vertex v, let N_v(d) be the number of other vertices at distance d from v.
    # The number of isosceles triangles with apex v is sum_d (N_v(d) choose 2).
    # Sum this over all v.
    # Note: An equilateral triangle is counted 3 times (once for each apex).
    # This is fine, as long as we are consistent. We just want to minimize this sum.
    
    total_count = 0
    for i in range(K):
        d_i = D_sub[i, :].copy()
        d_i[i] = -1
        d_valid = d_i[d_i != -1]
        if len(d_valid) > 1:
            _, counts = np.unique(d_valid, return_counts=True)
            # Sum of c*(c-1)//2
            total_count += int(np.sum(counts * (counts - 1) // 2))
            
    return total_count, total_count == 0

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Initial state: Start with a small random set or a structured one.
    # Let's start with a random set of size 10.
    indices = np.random.choice(GRID_SIZE, size=10, replace=False)
    current_set = set(indices)
    current_indices = np.array(list(current_set))
    
    # Evaluate initial state
    current_cost, _ = count_isosceles(current_indices)
    best_set = current_set.copy()
    best_cost = current_cost
    
    # Simulated Annealing
    T = 100.0
    T_min = 0.1
    cooling_rate = 0.999
    
    # We need to be able to add/remove/move points efficiently.
    # Maintaining a sorted numpy array is expensive for insert/delete.
    # Let's maintain a list of indices and convert to numpy array for evaluation.
    # Or maintain a boolean mask of size GRID_SIZE?
    # A boolean mask allows O(1) add/remove.
    # But extracting the indices for evaluation requires np.nonzero(mask), which is O(N).
    # N=4096. This is fast.
    
    mask = np.zeros(GRID_SIZE, dtype=bool)
    for i in indices:
        mask[i] = True
    
    # Helper to get indices from mask
    def get_indices(m):
        return np.nonzero(m)[0]
    
    current_indices = get_indices(mask)
    current_cost, _ = count_isosceles(current_indices)
    
    # SA Loop
    while time.time() < deadline:
        # Generate a move
        move_type = random.random()
        new_mask = mask.copy()
        
        if move_type < 0.4:
            # Add a point
            # Choose a random point not in the set
            # To be fast, we can just pick a random index and check if it's in the set.
            # Or maintain a list of available points?
            # With N=4096, picking random and checking mask is O(1).
            attempts = 0
            while attempts < 100:
                idx = random.randint(0, GRID_SIZE - 1)
                if not mask[idx]:
                    new_mask[idx] = True
                    break
                attempts += 1
            if attempts >= 100:
                continue # Failed to find a point to add? Unlikely.
                
        elif move_type < 0.7:
            # Remove a point
            if len(current_indices) > 0:
                # Pick a random point from the current set
                # We can pick a random index from current_indices
                idx_to_remove = current_indices[random.randint(0, len(current_indices) - 1)]
                new_mask[idx_to_remove] = False
            else:
                continue
                
        elif move_type < 0.9:
            # Move a point: Remove one, Add another
            if len(current_indices) > 0:
                idx_to_remove = current_indices[random.randint(0, len(current_indices) - 1)]
                new_mask[idx_to_remove] = False
                
                attempts = 0
                while attempts < 100:
                    idx_to_add = random.randint(0, GRID_SIZE - 1)
                    if not mask[idx_to_add] and idx_to_add != idx_to_remove:
                        new_mask[idx_to_add] = True
                        break
                    attempts += 1
                if attempts >= 100:
                    new_mask[idx_to_remove] = True # Revert
                    continue
            else:
                continue
                
        else:
            # Swap: Remove one from set, Add one from complement (effectively same as Move if we don't care about order, but here it's explicit)
            # Actually, Move and Swap are similar. Let's stick to Add, Remove, Move.
            # Let's add a "Perturb" move: Randomly flip 3-5 points.
            num_flips = random.randint(3, 5)
            for _ in range(num_flips):
                idx = random.randint(0, GRID_SIZE - 1)
                new_mask[idx] = not new_mask[idx]
                
        # Evaluate new state
        new_indices = get_indices(new_mask)
        new_cost, new_valid = count_isosceles(new_indices)
        
        # Acceptance criterion
        delta = new_cost - current_cost
        if delta <= 0 or random.random() < np.exp(-delta / T):
            # Accept
            mask = new_mask
            current_indices = new_indices
            current_cost = new_cost
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_set = set(new_indices)
                if best_cost == 0:
                    # We found a valid set. Can we make it bigger?
                    # The objective is to MAXIMIZE the number of points.
                    # My SA minimizes the number of isosceles triangles.
                    # If cost is 0, we have a valid set.
                    # We should now try to ADD points to increase the size while keeping cost 0.
                    # This is a different optimization goal.
                    # Let's modify the objective:
                    # We want to maximize Size - Lambda * Cost?
                    # Or, if Cost == 0, we want to maximize Size.
                    # If Cost > 0, we want to minimize Cost.
                    # 
                    # Let's use a composite objective:
                    # Score = -len(set) * 100 + cost
                    # We want to minimize this score.
                    # This encourages larger sets if they are valid (cost 0).
                    # If cost is high, the penalty dominates.
                    
                    pass # We'll handle this in the acceptance logic below.
                    
        # Cooling
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Optional: Restart with higher T?
            # For now, just keep going.
            
    # Final result: We need to return a VALID set.
    # The best_set found might have cost > 0 if we couldn't reduce it to 0.
    # We should try to repair the best_set or find a valid subset.
    # If best_cost > 0, we need to remove points until it's valid.
    
    result_indices = list(best_set)
    result_indices = np.array(result_indices)
    
    # If the best set is not valid, try to repair it by removing points.
    cost, valid = count_isosceles(result_indices)
    if not valid:
        # Greedy removal
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            # Find a point involved in an isosceles triangle and remove it
            # This is hard to do efficiently without re-evaluating.
            # Let's just try removing random points until valid.
            for i in range(len(result_indices)):
                if time.time() > deadline - 10:
                    break
                # Try removing point i
                new_indices = np.delete(result_indices, i)
                c, v = count_isosceles(new_indices)
                if v:
                    result_indices = new_indices
                    changed = True
                    break
                elif c < cost:
                    # Even if not valid, if it reduces cost, keep it?
                    # No, we want a valid set.
                    # Let's just accept any removal that reduces cost, and hope we hit 0.
                    result_indices = new_indices
                    cost = c
                    changed = True
                    break
            # If no single removal helps, try removing 2?
            # Or just accept the best valid subset found so far.
            # Actually, if we start from a valid set and only accept moves that keep cost 0 or reduce cost,
            # we might end up with a valid set.
            
    # Ensure we return a valid set.
    # If the current result is not valid, we must find a valid subset.
    # A simple valid set is any set of size < 3.
    # But we want the largest.
    
    # Let's do a final check.
    final_cost, final_valid = count_isosceles(result_indices)
    if not final_valid:
        # Fallback: Return a small valid set
        # e.g., 2 points
        return [COORDS[i] for i in result_indices[:2]]
        
    return [COORDS[i] for i in result_indices]

# To make the objective function work correctly for SA, we need to adjust the acceptance logic.
# The current logic minimizes cost. But we want to maximize size if cost is 0.
# Let's redefine the objective function.
# Objective = cost * 1000 - len(set)
# We want to MINIMIZE this.
# If cost is 0, Objective = -len(set). Minimizing this means maximizing len(set).
# If cost is 1, Objective = 1000 - len(set). This is much larger than any valid set's objective.
# So the SA will first try to get cost to 0, then try to increase size.

def get_objective(points_idx):
    cost, valid = count_isosceles(points_idx)
    return cost * 1000 - len(points_idx)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Initial state
    mask = np.zeros(GRID_SIZE, dtype=bool)
    # Start with a random set of size 20
    init_indices = np.random.choice(GRID_SIZE, size=20, replace=False)
    for i in init_indices:
        mask[i] = True
        
    current_indices = np.nonzero(mask)[0]
    current_obj = get_objective(current_indices)
    
    best_obj = current_obj
    best_mask = mask.copy()
    
    T = 100.0
    T_min = 0.1
    cooling_rate = 0.999
    
    while time.time() < deadline:
        new_mask = mask.copy()
        move_type = random.random()
        
        if move_type < 0.4:
            # Add
            idx = random.randint(0, GRID_SIZE - 1)
            if not mask[idx]:
                new_mask[idx] = True
            else:
                continue
        elif move_type < 0.7:
            # Remove
            if len(current_indices) > 0:
                idx = current_indices[random.randint(0, len(current_indices) - 1)]
                new_mask[idx] = False
            else:
                continue
        else:
            # Move
            if len(current_indices) > 0:
                idx_remove = current_indices[random.randint(0, len(current_indices) - 1)]
                new_mask[idx_remove] = False
                attempts = 0
                while attempts < 100:
                    idx_add = random.randint(0, GRID_SIZE - 1)
                    if not mask[idx_add] and idx_add != idx_remove:
                        new_mask[idx_add] = True
                        break
                    attempts += 1
                if attempts >= 100:
                    new_mask[idx_remove] = True
                    continue
            else:
                continue
                
        new_indices = np.nonzero(new_mask)[0]
        new_obj = get_objective(new_indices)
        
        delta = new_obj - current_obj
        if delta <= 0 or random.random() < np.exp(-delta / T):
            mask = new_mask
            current_indices = new_indices
            current_obj = new_obj
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_mask = mask.copy()
                
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    result_indices = np.nonzero(best_mask)[0]
    return [COORDS[i] for i in result_indices]

if __name__ == "__main__":
    # Test
    res = construct()
    print(f"Found set of size {len(res)}")
