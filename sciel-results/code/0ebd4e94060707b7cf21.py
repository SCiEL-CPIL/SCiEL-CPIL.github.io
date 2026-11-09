import time
import random
import numpy as np

N = 64
M = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 285  # Safety margin for 300s limit

    # Precompute points
    points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    
    # Precompute distance matrix D where D[i, j] = squared distance
    # M = 4096, so D is 4096x4096 int32 ~ 64MB. Safe.
    try:
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        D = np.sum(diff**2, axis=2).astype(np.int32)
        # Free memory
        del diff
    except MemoryError:
        D = None

    # --- Bitset-based distance representation ---
    # Max squared distance is (N-1)^2 + (N-1)^2 = 2*(63^2) = 7938
    # We need bitset size for 0..7938.
    # Number of 64-bit words needed: ceil(7939 / 64) = 124
    NUM_WORDS = 124
    
    # Precompute a bitset for each possible distance value? No, that's expensive memory.
    # Instead, for each point i, we maintain a bitset of distances to the current set S.
    # Let's precompute a function to get the bitset for a specific distance value.
    # Actually, it's better to precompute the bit index for each distance.
    # dist_to_bit[dist] = (word_index, bit_mask)
    
    max_dist = 2 * (N - 1) ** 2
    dist_word = np.zeros(max_dist + 1, dtype=np.int32)
    dist_bit = np.zeros(max_dist + 1, dtype=np.uint64)
    for d in range(max_dist + 1):
        dist_word[d] = d // 64
        dist_bit[d] = 1 << (d % 64)

    # State:
    # S_indices: list of indices in points
    # dist_bitsets: list of numpy arrays of uint64, one for each point in S
    #   dist_bitsets[i][w] is the bitset of distances from S_indices[i] to all other points in S
    
    def add_point(p_idx, S_indices, dist_bitsets):
        """
        Attempts to add p_idx to the set.
        Returns True if added, False otherwise.
        Modifies S_indices and dist_bitsets in place if successful.
        """
        k = len(S_indices)
        
        if k == 0:
            S_indices.append(p_idx)
            # Initialize bitset for the new point (empty)
            new_bitset = np.zeros(NUM_WORDS, dtype=np.uint64)
            dist_bitsets.append(new_bitset)
            return True

        # 1. Compute distances from p_idx to all points in S
        # d_vals is an array of length k
        d_vals = D[p_idx, S_indices]
        
        # Check if all distances are distinct
        # If any duplicate, invalid.
        # Using a set is O(k log k) or O(k). k is small (< 100 usually), so this is fine.
        # But we can also use the bitset logic.
        # Let's use a temporary bitset to check for duplicates.
        temp_bitset = np.zeros(NUM_WORDS, dtype=np.uint64)
        
        for d in d_vals:
            w = dist_word[d]
            b = dist_bit[d]
            if temp_bitset[w] & b:
                return False # Duplicate distance found
            temp_bitset[w] |= b

        # 2. Check if adding p_idx creates an isosceles triangle with any pair in S.
        # For each point i in S, the distance d_i = D[p_idx, S_indices[i]] must NOT be in dist_bitsets[i].
        # If it is, then there exists some j in S such that D[S_indices[i], S_indices[j]] == d_i.
        # This forms an isosceles triangle with apex S_indices[i].
        
        for i in range(k):
            d_val = d_vals[i]
            w = dist_word[d_val]
            b = dist_bit[d_val]
            if dist_bitsets[i][w] & b:
                return False

        # 3. Update state
        S_indices.append(p_idx)
        
        # Update existing bitsets
        # For each existing point i, add distance d_vals[i] to its bitset
        for i in range(k):
            d_val = d_vals[i]
            w = dist_word[d_val]
            b = dist_bit[d_val]
            dist_bitsets[i][w] |= b

        # Create bitset for the new point
        # The new point's bitset contains distances to all points in S (including itself? No, distance to self is 0, but we don't care about self-distance for isosceles checks with distinct points, except that 0 is never a distance between distinct points).
        # Actually, the bitset for point i stores distances to *other* points in S.
        # So for the new point, we store d_vals (distances to the old S).
        new_bitset = temp_bitset.copy() # temp_bitset already has all d_vals set
        dist_bitsets.append(new_bitset)

        return True

    def remove_point(idx_in_S, S_indices, dist_bitsets):
        """
        Removes the point at position idx_in_S from the set.
        Modifies S_indices and dist_bitsets in place.
        """
        p_idx = S_indices[idx_in_S]
        k = len(S_indices)
        
        # We need to remove p_idx from all other points' bitsets.
        # And remove the bitset of p_idx itself.
        
        # Compute distances from p_idx to all other points
        # Note: S_indices still contains p_idx at idx_in_S
        other_indices = [S_indices[j] for j in range(k) if j != idx_in_S]
        
        if not other_indices:
            S_indices.pop(idx_in_S)
            dist_bitsets.pop(idx_in_S)
            return

        d_vals = D[p_idx, other_indices]
        
        # Update bitsets of other points
        # We need to clear the bit corresponding to d_vals[j] in dist_bitsets[other_point_idx]
        # This is tricky because the indices in dist_bitsets change if we just pop.
        # Better approach:
        # 1. Identify the bitset to remove: dist_bitsets[idx_in_S]
        # 2. For every other point i, clear the bit for distance D[S_indices[i], p_idx].
        
        for i in range(k):
            if i == idx_in_S:
                continue
            d_val = D[S_indices[i], p_idx]
            w = dist_word[d_val]
            b = dist_bit[d_val]
            dist_bitsets[i][w] &= ~b

        # Remove the point from lists
        S_indices.pop(idx_in_S)
        dist_bitsets.pop(idx_in_S)

    best_S_indices = []
    best_dist_bitsets = []
    best_size = 0

    def update_best(S_indices, dist_bitsets):
        nonlocal best_S_indices, best_dist_bitsets, best_size
        if len(S_indices) > best_size:
            best_size = len(S_indices)
            best_S_indices = S_indices.copy()
            best_dist_bitsets = [b.copy() for b in dist_bitsets]

    # Initial greedy constructions with different orderings
    strategies = [
        lambda: list(range(M)),
        lambda: [i for i in range(M) if (i // N + i % N) % 2 == 0], # Checkerboard
        lambda: [i for i in range(M) if (i // N) == (i % N)], # Diagonal
    ]

    for strat in strategies:
        S_indices = []
        dist_bitsets = []
        candidates = strat()
        random.shuffle(candidates)
        
        for p in candidates:
            if time.time() > deadline:
                break
            if add_point(p, S_indices, dist_bitsets):
                pass # Added
        
        update_best(S_indices, dist_bitsets)

    # Local Search / Simulated Annealing style improvement
    # We will try to improve the best set found so far.
    
    current_S_indices = best_S_indices.copy()
    current_dist_bitsets = [b.copy() for b in best_dist_bitsets]
    
    # Precompute list of all points not in current set
    current_set = set(current_S_indices)
    all_points = list(range(M))
    candidates_to_add = [p for p in all_points if p not in current_set]
    
    # We'll use a loop that tries to add random points, and if stuck, removes one and tries again.
    # This is a simple hill-climbing with restarts.
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        if restart == 0:
            # Start with current best
            S_indices = current_S_indices.copy()
            dist_bitsets = [b.copy() for b in current_dist_bitsets]
        else:
            # Perturb: remove a few random points from best
            S_indices = best_S_indices.copy()
            dist_bitsets = [b.copy() for b in best_dist_bitsets]
            
            # Remove 1-3 random points
            remove_count = random.randint(1, min(3, len(S_indices)))
            if remove_count > 0:
                # Get indices to remove
                remove_idx = random.sample(range(len(S_indices)), remove_count)
                # Sort descending to remove safely
                remove_idx.sort(reverse=True)
                for idx in remove_idx:
                    remove_point(idx, S_indices, dist_bitsets)
        
        # Try to add points
        # Shuffle candidates
        remaining = [p for p in all_points if p not in set(S_indices)]
        random.shuffle(remaining)
        
        improved = False
        for p in remaining:
            if time.time() > deadline:
                break
            if add_point(p, S_indices, dist_bitsets):
                improved = True
                # If we added a point, we might want to re-shuffle or just continue?
                # Continuing is fine as the set grows.
        
        if len(S_indices) > best_size:
            update_best(S_indices, dist_bitsets)
            # If we improved, maybe restart the adding process from this new state?
            # For now, just record best.
            
        # If we didn't improve after trying all candidates, we can try a different perturbation
        # The loop above handles restarts.

    # Return the best set found
    return [tuple(points[i]) for i in best_S_indices]
