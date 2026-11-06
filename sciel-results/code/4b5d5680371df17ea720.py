import numpy as np
import random
import time

N = 64
NUM_POINTS = N * N
DEADLINE = time.time() + 290  # Safety margin before 300s limit

# Pre-compute coordinates
points = np.array([(x, y) for y in range(N) for x in range(N)], dtype=np.int32)

# Pre-compute full distance matrix (4096x4096)
def _precompute_distances():
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sum(diff**2, axis=2).astype(np.int32)

D_grid = _precompute_distances()

def construct():
    start_time = time.time()
    best_set = []
    best_size = 0
    
    # We will run multiple restarts of a greedy + local search algorithm
    # The local search uses a fast compatibility check based on maintaining distance sets
    
    while time.time() < DEADLINE - 5:  # Safety margin
        
        # Randomize the order of points for greedy phase
        all_indices = list(range(NUM_POINTS))
        random.shuffle(all_indices)
        
        S = []
        S_set = set()
        # dist_sets[s_idx] is a set of squared distances from s_idx to all other points in S
        dist_sets = {}
        
        def is_compatible_fast(p_idx):
            if not S:
                return True
            
            # Get distances from p_idx to all points in S
            # S is a list of indices
            d_to_S = D_grid[p_idx, S]
            
            # Check 1: d(p, S_i) == d(p, S_j)
            # If any duplicate in d_to_S, return False
            if len(np.unique(d_to_S)) < len(S):
                return False
            
            # Check 2 & 3: d(p, S_i) == d(S_i, S_j)
            # For each S_i, check if d(p, S_i) is in dist_sets[S_i]
            for i, s_idx in enumerate(S):
                d_ps = d_to_S[i]
                if d_ps in dist_sets[s_idx]:
                    return False
            
            return True

        def add_point(p_idx):
            S.append(p_idx)
            S_set.add(p_idx)
            
            new_dist_set = set()
            for s_idx in S[:-1]: # All points except the one just added
                d = D_grid[p_idx, s_idx]
                dist_sets[s_idx].add(d)
                new_dist_set.add(d)
            dist_sets[p_idx] = new_dist_set

        def remove_point(p_idx):
            # Find index of p_idx in S
            # Since S is a list, we can use index() but it's O(N).
            # For efficiency in local search, we might want to keep S as a set for removal?
            # But is_compatible needs ordered access or at least iteration.
            # Let's assume S is small enough (<100) that list operations are fine.
            
            if p_idx not in S_set:
                return
            
            # Remove from S list
            try:
                pos = S.index(p_idx)
                S.pop(pos)
            except ValueError:
                return
            
            S_set.remove(p_idx)
            
            p_dist_set = dist_sets.pop(p_idx)
            for s_idx in S:
                d = D_grid[p_idx, s_idx]
                dist_sets[s_idx].discard(d)

        # Greedy Phase
        for idx in all_indices:
            if is_compatible_fast(idx):
                add_point(idx)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
        # Local Search Phase
        pool = [i for i in all_indices if i not in S_set]
        
        # Run local search for a fixed number of steps or time
        # We alternate between trying to add and swapping
        local_search_deadline = time.time() + 10
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < local_search_deadline and len(pool) > 0:
            if no_improve_count > max_no_improve:
                break
            
            # 1. Try to add a random point from pool
            if pool:
                p_idx = random.choice(pool)
                if is_compatible_fast(p_idx):
                    add_point(p_idx)
                    pool.remove(p_idx)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_set = S[:]
                    no_improve_count = 0
                    continue
            
            # 2. Try to swap: remove one from S, add one from pool
            # To be efficient, we can try a few random swaps
            swapped = False
            if S and pool:
                # Try up to 10 random swaps
                for _ in range(10):
                    rem_idx = random.choice(S)
                    add_idx = random.choice(pool)
                    
                    # Remove rem_idx
                    remove_point(rem_idx)
                    pool.append(rem_idx)
                    
                    # Try to add add_idx
                    if is_compatible_fast(add_idx):
                        add_point(add_idx)
                        if add_idx in pool:
                            pool.remove(add_idx)
                        swapped = True
                        if len(S) > best_size:
                            best_size = len(S)
                            best_set = S[:]
                        no_improve_count = 0
                        break
                    else:
                        # Restore rem_idx
                        add_point(rem_idx)
                        pool.remove(rem_idx)
                        # Note: add_point(rem_idx) might fail if rem_idx is not compatible with current S?
                        # Wait, rem_idx was in S before removal.
                        # After removal, S is smaller.
                        # Is rem_idx always compatible with S \ {rem_idx}?
                        # Yes, because S was valid, so any subset is valid.
                        # So add_point(rem_idx) should succeed.
                        # However, our add_point function doesn't check compatibility, it just adds.
                        # We must ensure we only call add_point if we know it's compatible.
                        # In the swap logic:
                        # 1. Remove A. S is valid.
                        # 2. Check if P is compatible with S.
                        # 3. If yes, add P.
                        # 4. If no, we must restore A.
                        # Since S (without A) is valid, A is compatible with S (without A).
                        # So add_point(A) is safe.
            
            if not swapped:
                no_improve_count += 1

    # Final verification
    if best_set:
        # Verify best_set
        if len(best_set) > 2:
             K = len(best_set)
             D_sub = D_grid[np.ix_(best_set, best_set)]
             valid = True
             for i in range(K):
                 for j in range(i + 1, K):
                     d_ij = D_sub[i, j]
                     for k in range(j + 1, K):
                         d_ik = D_sub[i, k]
                         d_jk = D_sub[j, k]
                         if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                             valid = False
                             break
                     if not valid: break
                 if not valid: break
             if not valid:
                 # Fallback to a small valid set if something went wrong
                 return [(0, 0)]
         
    return [tuple(points[i]) for i in best_set]
