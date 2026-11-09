import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 310  # Safety margin before 330s limit

    N = 64
    M = N * N
    
    # Precompute all points
    points = [(x, y) for x in range(N) for y in range(N)]
    coords = np.array(points, dtype=np.int32)
    
    # Precompute squared distance matrix
    # dist[i][j] = squared distance between points[i] and points[j]
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sum(diff**2, axis=-1)
    
    # Map each unique squared distance value to a bit index
    # Max squared distance is (N-1)^2 + (N-1)^2 = 2*(N-1)^2 = 2*63^2 = 7938
    # We can use the distance value itself as the index if we use a large enough bitset
    # Or we can compress. Since max dist is ~8000, a Python int bitset of 8000 bits is fine.
    # Actually, we can just use the distance value as the bit position.
    # Let's verify max distance.
    max_dist = int(np.max(dist))
    
    def try_add(S_set, dist_to_S, p):
        """
        Try to add point p to set S_set.
        S_set: set of indices in S
        dist_to_S: dict mapping index i in S to an integer bitset of distances to other points in S
        Returns True if can be added, False otherwise.
        """
        if p in S_set:
            return False
        
        S_list = list(S_set)
        if not S_list:
            return True
        
        # Get distances from p to all points in S
        S_arr = np.array(S_list, dtype=np.int32)
        d_vals = dist[p, S_arr]
        
        # Step 1: Check for duplicate distances from p to S
        # If any two points in S are at the same distance from p, it's isosceles.
        if len(set(d_vals)) < len(d_vals):
            return False
        
        # Step 2: Check conflicts
        # For each q in S, the distance d = dist[p, q] must NOT be in dist_to_S[q].
        # dist_to_S[q] is a bitset. We can check if bit d is set.
        for idx, q in enumerate(S_list):
            d = d_vals[idx]
            if (dist_to_S[q] >> d) & 1:
                return False
        
        return True

    def add_point(S_set, dist_to_S, p):
        """
        Add point p to S_set and update dist_to_S.
        Assumes try_add has already checked validity.
        """
        S_set.add(p)
        dist_to_S[p] = 0
        for q in S_set:
            if q == p:
                continue
            d = dist[p, q]
            # Set bit d in dist_to_S[p] and dist_to_S[q]
            mask = 1 << d
            dist_to_S[p] |= mask
            dist_to_S[q] |= mask

    def remove_point(S_set, dist_to_S, p):
        """
        Remove point p from S_set and update dist_to_S.
        """
        if p not in S_set:
            return
        for q in S_set:
            if q == p:
                continue
            d = dist[p, q]
            # Clear bit d in dist_to_S[q]
            mask = 1 << d
            dist_to_S[q] &= ~mask
        del dist_to_S[p]
        S_set.discard(p)

    def greedy_construct():
        S_set = set()
        dist_to_S = {}
        candidates = list(range(M))
        random.shuffle(candidates)
        
        for p in candidates:
            if time.time() > deadline - 20:
                break
            if try_add(S_set, dist_to_S, p):
                add_point(S_set, dist_to_S, p)
        
        return S_set, dist_to_S

    def local_search(S_set, dist_to_S):
        S_set = set(S_set)
        dist_to_S = {k: v for k, v in dist_to_S.items()}  # ints are immutable, so no deep copy needed
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            S_list = list(S_set)
            random.shuffle(S_list)
            
            # Try adding points not in S
            candidates = [i for i in range(M) if i not in S_set]
            random.shuffle(candidates)
            
            for p in candidates:
                if time.time() > deadline - 10:
                    break
                if try_add(S_set, dist_to_S, p):
                    add_point(S_set, dist_to_S, p)
                    improved = True
                    break
            
            if improved:
                continue
            
            # If no addition, try a swap
            if S_list:
                # Try a few swaps
                swaps_tried = 0
                max_swaps = 100
                while swaps_tried < max_swaps and time.time() < deadline - 10:
                    q_to_remove = random.choice(S_list)
                    remove_point(S_set, dist_to_S, q_to_remove)
                    
                    candidates = [i for i in range(M) if i not in S_set]
                    random.shuffle(candidates)
                    
                    added = False
                    for p in candidates:
                        if time.time() > deadline - 10:
                            break
                        if p == q_to_remove:
                            continue
                        if try_add(S_set, dist_to_S, p):
                            add_point(S_set, dist_to_S, p)
                            added = True
                            break
                    
                    if not added:
                        # Revert
                        add_point(S_set, dist_to_S, q_to_remove)
                        swaps_tried += 1
                    else:
                        improved = True
                        break
                if not improved:
                    break
            else:
                break
                
        return S_set, dist_to_S

    best_S_set = set()
    best_dist_to_S = {}
    
    # Multiple restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        S_set, dist_to_S = greedy_construct()
        S_set, dist_to_S = local_search(S_set, dist_to_S)
        
        if len(S_set) > len(best_S_set):
            best_S_set = S_set.copy()
            best_dist_to_S = {k: v for k, v in dist_to_S.items()}
    
    # Final polish on the best set found
    if time.time() < deadline - 5:
        best_S_set, best_dist_to_S = local_search(best_S_set, best_dist_to_S)

    return [points[i] for i in best_S_set]

# For testing
if __name__ == "__main__":
    res = construct()
    print(len(res))
