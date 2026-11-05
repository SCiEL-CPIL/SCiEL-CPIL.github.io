import time
import random
import math
import numpy as np

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 295  # Safety margin
    
    # Precompute squared distances
    all_points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    dx = all_points[:, 0].reshape(-1, 1) - all_points[:, 0].reshape(1, -1)
    dy = all_points[:, 1].reshape(-1, 1) - all_points[:, 1].reshape(1, -1)
    dist_matrix = dx * dx + dy * dy
    
    # Helper to calculate penalty for adding point j to set S_indices
    # Returns the number of isosceles triangles formed by j and pairs in S_indices
    def calc_penalty(j, S_indices):
        if len(S_indices) < 2:
            return 0
        
        S_arr = np.array(S_indices)
        # Distances from j to all points in S
        dists_j = dist_matrix[j, S_arr]
        
        # 1. Count triangles with apex at j
        # Group by distance, count pairs
        unique_dists, counts = np.unique(dists_j, return_counts=True)
        penalty_j = 0
        for c in counts:
            if c > 1:
                penalty_j += c * (c - 1) // 2
        
        # 2. Count triangles with apex at a in S
        # For each a in S, count b in S such that d(a,b) == d(a,j)
        # dists_from_a[k, l] = d(S_arr[k], S_arr[l])
        # dists_j[k] = d(j, S_arr[k])
        
        # Vectorized calculation
        # We need to compare each row of dists_from_a with the corresponding value in dists_j
        # dists_from_a is |S| x |S|
        # This might be memory heavy if |S| is large, but |S| is small.
        dists_from_a = dist_matrix[S_arr[:, None], S_arr[None, :]]
        
        # For each k, we want sum(dists_from_a[k, :] == dists_j[k])
        # Create a mask
        # dists_j[None, :] is 1 x |S|
        # dists_from_a is |S| x |S|
        # We want to compare dists_from_a[k, l] with dists_j[k]
        # This is not a direct broadcast comparison.
        
        # We can do:
        # matches = (dists_from_a == dists_j[:, None]) # |S| x |S|
        # counts_a = np.sum(matches, axis=1)
        
        matches = (dists_from_a == dists_j[:, None])
        counts_a = np.sum(matches, axis=1)
        penalty_a = np.sum(counts_a)
        
        return penalty_j + penalty_a

    # Greedy Initialization
    def greedy_init():
        indices = np.arange(N * N)
        np.random.shuffle(indices)
        
        current_set = []
        
        for idx in indices:
            if time.time() > deadline - 10:
                break
            j = int(idx)
            if calc_penalty(j, current_set) == 0:
                current_set.append(j)
                
        return current_set

    best_set = greedy_init()
    best_size = len(best_set)
    
    # Simulated Annealing
    # T starts high, cools down
    # We use a simple cooling schedule
    T = 1.0
    cooling = 0.999
    
    # We will run SA for the remaining time
    while time.time() < deadline - 5:
        if not best_set:
            break
            
        # Pick random i to remove, j to add
        i_idx = random.randint(0, len(best_set) - 1)
        i = best_set[i_idx]
        
        # Pick j not in best_set
        # Efficiently: pick random index from 0 to N*N-1, check if in set
        # Since set is small, we can use a set for O(1) lookup
        # But we need to maintain a list for SA.
        # Let's maintain a set of active points for fast rejection.
        
        # To avoid maintaining a separate set, we can just pick random j and check.
        # Or, we can pick from the list of inactive points.
        # Inactive list is large (4000).
        
        # Let's just pick random j in [0, N*N)
        j = random.randint(0, N * N - 1)
        
        # Check if j is already in best_set
        # This is O(|S|).
        if j in best_set:
            continue
            
        # Form S_temp
        # We don't want to create a new list every time if possible, but for clarity:
        # S_temp is best_set without i
        
        # Calculate penalty for adding j to S_temp
        # We need S_temp as a list for calc_penalty
        # Creating S_temp is O(|S|)
        
        # Optimization: calc_penalty takes a list.
        # Let's pass best_set and exclude i in the calculation?
        # No, calc_penalty uses numpy arrays.
        
        # Create S_temp list
        # This is O(|S|)
        S_temp = [p for p in best_set if p != i]
        
        penalty = calc_penalty(j, S_temp)
        
        # Acceptance criterion
        accepted = False
        if penalty == 0:
            accepted = True
        else:
            # Boltzmann factor
            # If penalty is large, exp is small
            # T is temperature
            # We want to accept bad moves with some probability
            # Delta E = penalty (since current penalty is 0)
            prob = math.exp(-penalty / T)
            if random.random() < prob:
                accepted = True
                
        if accepted:
            # Update best_set
            best_set[i_idx] = j
            # Note: The size remains the same.
            # We don't change best_size here, but we might find a better config later?
            # Actually, SA with swaps keeps size constant.
            # To increase size, we need "Add" moves.
            
            # Let's add "Add" moves to increase size.
            # If we just swap, we stay at the same size.
            # We need to try to add a point if possible.
            
        # Try to Add a point (increase size)
        # This is a different move type.
        # Pick random j not in best_set.
        # Check if penalty is 0.
        # If yes, add it.
        
        # We can alternate between Swap and Add.
        # Or just try Add. If it fails, try Swap.
        
        # Let's try Add first.
        # If Add succeeds, size increases.
        # If Add fails, try Swap to escape local maximum.
        
        # Actually, the loop above was Swap.
        # Let's restructure:
        # 1. Try to Add a random point.
        # 2. If fail, try to Swap.
        
        # The code above did Swap.
        # Let's modify the loop to prioritize Add.
        
        # For now, this loop does Swap.
        # We need to add a mechanism to grow the set.
        
        # Let's add a separate loop or condition.
        
        # Cool down
        T *= cooling
        if T < 0.01:
            T = 0.01 # Keep some randomness

    # Final Greedy Polish
    # Try to add points greedily to the current best_set
    # This is fast and can pick up easy wins
    indices = np.arange(N * N)
    np.random.shuffle(indices)
    
    for idx in indices:
        if time.time() > deadline - 2:
            break
        j = int(idx)
        if j in best_set:
            continue
        if calc_penalty(j, best_set) == 0:
            best_set.append(j)
            
    # Convert indices back to (x, y)
    final_points = [tuple(all_points[i]) for i in best_set]
    return final_points
# EVOLVE-BLOCK-END
