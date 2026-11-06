import time
import random
import numpy as np

N = 64
GRID_SIZE = N * N
# Max squared distance in [0, 64)^2 is (63^2 + 63^2) = 7938
MAX_DIST = 7938

# Precompute points
points_arr = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
ALL_POINTS = [tuple(p) for p in points_arr]

xs = points_arr[:, 0].astype(np.int64)
ys = points_arr[:, 1].astype(np.int64)
dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
DIST = (dx * dx + dy * dy)  # shape (GRID_SIZE, GRID_SIZE)

# Precompute conflict bitsets
# conflict[i] is a bitmask of points j such that adding j to a set containing i 
# might create an isosceles triangle with some k.
# Actually, a simpler static conflict definition:
# Two points i, j are "conflicting" if there exists a point k such that 
# d(i,k) == d(j,k) OR d(i,k) == d(i,j) OR d(j,k) == d(i,j).
# This is a necessary condition for i and j to be in a set that causes a conflict?
# No, if i and j are in the set, and k is in the set, then it's a conflict.
# But we don't know k in advance.
#
# Let's use the dynamic check but optimize it using precomputed distance lists.
# For each point i, precompute the list of (distance, point_index) pairs.
# Or better, for each point i, a dictionary: distance -> bitmask of points at that distance.
# This allows fast intersection checks.

# Let's try a different bitset approach:
# For each point i, and each distance d, let B[i, d] be the bitmask of points at distance d from i.
# Memory: 4096 * 7939 * (4096/8) bytes = 4096 * 7939 * 512 bytes ~ 16 GB. Too much.
#
# Alternative:
# For each point i, store a dictionary: {distance: bitmask_of_points}
# Number of unique distances per point is small (~100-200).
# Total memory: 4096 * 200 * 512 bytes ~ 400 MB. Acceptable.

dist_bitmasks = [None] * GRID_SIZE
for i in range(GRID_SIZE):
    d_dict = {}
    d_row = DIST[i]
    for j in range(GRID_SIZE):
        if i == j: continue
        d = d_row[j]
        if d not in d_dict:
            d_dict[d] = 0
        d_dict[d] |= (1 << j)
    dist_bitmasks[i] = d_dict

def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    best_set = []
    best_size = 0
    best_mask = 0
    
    def is_valid_addition(mask, k):
        # Check if adding k to the set represented by mask is valid.
        # 1. No two points in S have the same distance to k.
        #    This means: for any distance d, the number of points in S at distance d from k must be <= 1.
        #    Let B_k_d be the bitmask of points at distance d from k.
        #    We need (mask & B_k_d).bit_count() <= 1 for all d.
        #    Since we are adding k, we check the existing mask.
        #    Actually, if there are already 2 points in S at distance d from k, it's invalid.
        #    If there is 1 point in S at distance d from k, it's OK for now, but we must ensure 
        #    that this point (say a) doesn't form an isosceles with k and another point b in S.
        
        # Let's refine the condition.
        # A set S is valid if for all distinct i, j, k in S:
        # d(i,j) != d(i,k) and d(i,j) != d(j,k) and d(i,k) != d(j,k).
        
        # When adding k to S:
        # For all i, j in S:
        # d(i,j) != d(i,k)  =>  d(i,k) is not in {d(i, s) for s in S}
        # d(i,j) != d(j,k)  =>  d(j,k) is not in {d(j, s) for s in S}
        # d(i,k) != d(j,k)  =>  d(i,k) != d(j,k) for all i, j in S.
        
        # Check 3: d(i,k) != d(j,k) for all i, j in S.
        # This is equivalent to: all distances from k to points in S are distinct.
        # We can check this by iterating through distances from k to points in S.
        # Let S_k = [s for s in S].
        # dists_k = [DIST[k, s] for s in S_k]
        # if len(dists_k) != len(set(dists_k)): return False
        
        # Check 1 & 2: For all i in S, d(i,k) is not in {d(i, s) for s in S}.
        # Note that d(i, i) = 0, which is not equal to d(i,k) > 0.
        # So we just need to check if d(i,k) is in the set of distances from i to S.
        
        # Efficient check:
        # For each i in S:
        #   d_ik = DIST[i, k]
        #   We need to check if there exists j in S such that DIST[i, j] == d_ik.
        #   This is equivalent to: (mask & dist_bitmasks[i][d_ik]) != 0.
        #   BUT, dist_bitmasks[i][d_ik] includes k itself if k is at that distance.
        #   Wait, dist_bitmasks[i] is precomputed for ALL points.
        #   So dist_bitmasks[i][d_ik] will have bit k set.
        #   We are checking against `mask`, which does NOT contain k yet.
        #   So if (mask & dist_bitmasks[i][d_ik]) != 0, it means there is some j in S (j != k) 
        #   such that d(i, j) == d(i, k). This is a conflict.
        
        # So the check is:
        # 1. Distances from k to S are distinct.
        # 2. For all i in S, (mask & dist_bitmasks[i][DIST[i, k]]) == 0.
        
        # How to do this fast?
        # We need to iterate over points in S.
        # If S is represented as a list, we can iterate.
        # If S is represented as a bitmask, we need to extract indices.
        # Extracting indices from a 4096-bit integer is slow in Python.
        # So let's maintain S as a list AND a bitmask.
        
        # Wait, the function signature takes `mask`. I should pass the list too or reconstruct it.
        # Let's change the helper to take `S_list` and `mask`.
        
        pass # Placeholder, will implement in the nested function below

    def is_valid_addition_impl(S_list, mask, k):
        m = len(S_list)
        if m < 2:
            return True
            
        # Check 1: Distances from k to S are distinct
        # We can do this by checking if adding k creates a duplicate distance.
        # Instead of creating a list of distances, we can use a set.
        # But for m < 100, a simple loop or set is fine.
        # Let's use a set for distances.
        dists_seen = set()
        for i in S_list:
            d = DIST[k, i]
            if d in dists_seen:
                return False
            dists_seen.add(d)
            
        # Check 2: For all i in S, d(i, k) is not a distance from i to any j in S.
        # i.e., (mask & dist_bitmasks[i][DIST[i, k]]) == 0
        for i in S_list:
            d_ik = DIST[i, k]
            # dist_bitmasks[i] is a dict: {distance: bitmask}
            # If d_ik is not in the dict, it means no point is at that distance from i?
            # No, dist_bitmasks[i] contains distances to ALL points in the grid.
            # So d_ik MUST be in dist_bitmasks[i] because k is in the grid.
            # Actually, dist_bitmasks[i] was built from DIST[i, :].
            # So d_ik is a key.
            conflict_mask = dist_bitmasks[i].get(d_ik, 0)
            if (mask & conflict_mask) != 0:
                return False
                
        return True

    def greedy_init():
        S = []
        mask = 0
        indices = list(range(GRID_SIZE))
        random.shuffle(indices)
        
        for k in indices:
            if time.time() > deadline - 5:
                break
            if is_valid_addition_impl(S, mask, k):
                S.append(k)
                mask |= (1 << k)
                
        return S, mask

    def local_search(S, mask):
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            m = len(S)
            
            # Try to add random points
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                k = random.randint(0, GRID_SIZE - 1)
                if not (mask & (1 << k)) and is_valid_addition_impl(S, mask, k):
                    S.append(k)
                    mask |= (1 << k)
                    improved = True
            
            # Try swaps
            if m > 0:
                for _ in range(50):
                    if time.time() > deadline - 5:
                        break
                    idx = random.randint(0, m - 1)
                    a = S[idx]
                    # Remove a
                    S.pop(idx)
                    mask &= ~(1 << a)
                    
                    # Try to add 20 random points
                    added = False
                    for _ in range(20):
                        k = random.randint(0, GRID_SIZE - 1)
                        if not (mask & (1 << k)) and is_valid_addition_impl(S, mask, k):
                            S.append(k)
                            mask |= (1 << k)
                            added = True
                            improved = True
                            break
                    
                    if not added:
                        # Revert
                        S.append(a)
                        mask |= (1 << a)
                        
        return S, mask

    # Main loop
    while time.time() < deadline - 10:
        S, mask = greedy_init()
        S, mask = local_search(S, mask)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            best_mask = mask

    if len(best_set) == 0:
        best_set = [0]
        
    return [ALL_POINTS[i] for i in best_set]
