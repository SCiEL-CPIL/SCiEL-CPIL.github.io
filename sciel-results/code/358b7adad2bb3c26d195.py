N = 64
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 290
    
    n_pts = N * N
    all_points = [(x, y) for y in range(N) for x in range(N)]
    pts_arr = np.array(all_points, dtype=np.int32)
    
    # Precompute squared distances
    # diff = pts_arr[:, np.newaxis, :] - pts_arr[np.newaxis, :, :]
    # sq_dists = (diff ** 2).sum(axis=2).astype(np.int64)
    # This creates a 4096x4096 int64 array. 16MB. Fast to access.
    diff = pts_arr[:, np.newaxis, :] - pts_arr[np.newaxis, :, :]
    sq_dists = (diff ** 2).sum(axis=2).astype(np.int64)
    
    def try_greedy_random(rng):
        """Greedy construction with random ordering."""
        S = set()
        dist_sets = {}
        
        order = list(range(n_pts))
        rng.shuffle(order)
        
        for idx in order:
            if time.time() > deadline:
                break
            
            valid = True
            dists_to_idx = set()
            for j in S:
                d = sq_dists[idx, j]
                if d in dists_to_idx:
                    valid = False
                    break
                dists_to_idx.add(d)
                if d in dist_sets.get(j, set()):
                    valid = False
                    break
            
            if not valid:
                continue
            
            S.add(idx)
            dist_sets[idx] = set(dists_to_idx)
            for j in S:
                if j == idx:
                    continue
                dist_sets[j].add(sq_dists[idx, j])
        
        return S, dist_sets

    def is_valid_addition_to_temp(idx, S_temp, dist_sets_temp):
        """
        Check if idx can be added to S_temp.
        S_temp is a set of indices.
        dist_sets_temp is a dict of sets of distances.
        Returns True if valid.
        """
        dists_to_idx = set()
        for j in S_temp:
            d = sq_dists[idx, j]
            if d in dists_to_idx:
                return False
            dists_to_idx.add(d)
            if d in dist_sets_temp.get(j, set()):
                return False
        return True

    def add_point_to_temp(idx, S_temp, dist_sets_temp):
        """Add idx to S_temp and update dist_sets_temp in place."""
        dists_to_idx = set()
        for j in S_temp:
            d = sq_dists[idx, j]
            dists_to_idx.add(d)
            dist_sets_temp[j].add(d)
        
        S_temp.add(idx)
        dist_sets_temp[idx] = dists_to_idx

    def remove_point_from_temp(idx, S_temp, dist_sets_temp):
        """Remove idx from S_temp and update dist_sets_temp in place."""
        saved_dists = dist_sets_temp.pop(idx)
        for j in S_temp:
            d = sq_dists[idx, j]
            dist_sets_temp[j].discard(d)
        S_temp.remove(idx)

    def simulated_annealing(S, dist_sets, rng):
        """
        SA loop with 1-for-2 swaps.
        """
        current_S = S
        current_dist = dist_sets
        current_size = len(current_S)
        best_S = current_S.copy()
        best_dist = {k: v.copy() for k, v in current_dist.items()}
        best_size = current_size
        
        # Temperature decay
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        while time.time() < deadline - 5:
            if T < T_min:
                T = 1.0 # Restart if stuck? Or just keep going with flat acceptance for swaps
            
            # Pick a random point to remove
            if not current_S:
                break
            u = rng.choice(list(current_S))
            
            # Create a temporary copy of the state
            # To avoid deep copying the whole dist_sets dict (expensive),
            # we will modify and revert.
            
            # Save state of u and its neighbors
            # u's dist set
            u_dists = current_dist[u].copy()
            # Neighbors' dist sets that contain u's distance
            affected_neighbors = {}
            for j in current_S:
                if j == u:
                    continue
                d = sq_dists[u, j]
                if d in current_dist[j]:
                    if j not in affected_neighbors:
                        affected_neighbors[j] = []
                    affected_neighbors[j].append(d)
            
            # Remove u
            current_S.remove(u)
            del current_dist[u]
            for j, ds_list in affected_neighbors.items():
                for d in ds_list:
                    current_dist[j].discard(d)
            
            # Try to add 2 points (v, w)
            # Generate random candidates from S^c
            # We can precompute a list of all points not in current_S?
            # current_S changes, so it's dynamic.
            # Using random sampling from 0..n_pts-1 is faster.
            
            success_1_for_2 = False
            success_1_for_1 = False
            
            # Try up to 10 random pairs (v, w)
            # This is expensive: 10 * (check v + check w)
            # check is O(K)
            
            # Optimization: Try 1-for-1 first if 1-for-2 fails?
            # Or just try 1-for-2.
            
            # Let's try a few random v's. For each v, try a few w's.
            
            for _ in range(5): # 5 v candidates
                v = rng.randint(0, n_pts - 1)
                if v in current_S:
                    continue
                
                # Check if v can be added
                if not is_valid_addition_to_temp(v, current_S, current_dist):
                    continue
                
                # v is valid. Add it temporarily.
                # Save state for v
                v_dists = set()
                v_affected = {}
                for j in current_S:
                    d = sq_dists[v, j]
                    v_dists.add(d)
                    if d not in v_affected:
                        v_affected[j] = []
                    v_affected[j].append(d)
                
                # Apply v
                current_S.add(v)
                current_dist[v] = v_dists
                for j, ds_list in v_affected.items():
                    for d in ds_list:
                        current_dist[j].add(d)
                
                # Now try to add w
                for _ in range(5): # 5 w candidates
                    w = rng.randint(0, n_pts - 1)
                    if w in current_S:
                        continue
                    
                    # Check if w can be added to (current_S which includes v)
                    if not is_valid_addition_to_temp(w, current_S, current_dist):
                        continue
                    
                    # w is valid!
                    # We have a 1-for-2 swap.
                    # Commit the changes for w.
                    w_dists = set()
                    w_affected = {}
                    for j in current_S:
                        if j == w:
                            continue
                        d = sq_dists[w, j]
                        w_dists.add(d)
                        if d not in w_affected:
                            w_affected[j] = []
                        w_affected[j].append(d)
                        
                    current_S.add(w)
                    current_dist[w] = w_dists
                    for j, ds_list in w_affected.items():
                        for d in ds_list:
                            current_dist[j].add(d)
                    
                    # Update best
                    if len(current_S) > best_size:
                        best_size = len(current_S)
                        best_S = current_S.copy()
                        best_dist = {k: v.copy() for k, v in current_dist.items()}
                    
                    success_1_for_2 = True
                    break # Break w loop
                
                if success_1_for_2:
                    break # Break v loop
                
                # If 1-for-2 failed, revert v
                # Remove v
                current_S.remove(v)
                del current_dist[v]
                for j, ds_list in v_affected.items():
                    for d in ds_list:
                        current_dist[j].discard(d)
            
            if not success_1_for_2:
                # Try 1-for-1 swap (u out, v in) to navigate
                # We already have u removed.
                # Try 5 random v's.
                for _ in range(5):
                    v = rng.randint(0, n_pts - 1)
                    if v in current_S:
                        continue
                    
                    if not is_valid_addition_to_temp(v, current_S, current_dist):
                        continue
                    
                    # Valid swap.
                    # Commit v.
                    v_dists = set()
                    v_affected = {}
                    for j in current_S:
                        d = sq_dists[v, j]
                        v_dists.add(d)
                        if d not in v_affected:
                            v_affected[j] = []
                        v_affected[j].append(d)
                        
                    current_S.add(v)
                    current_dist[v] = v_dists
                    for j, ds_list in v_affected.items():
                        for d in ds_list:
                            current_dist[j].add(d)
                    
                    success_1_for_1 = True
                    break
                
                if not success_1_for_1:
                    # Revert u
                    current_S.add(u)
                    current_dist[u] = u_dists
                    for j, ds_list in affected_neighbors.items():
                        for d in ds_list:
                            current_dist[j].add(d)
            
            # Cooling
            T *= cooling
            if T < T_min:
                T = T_min

        return best_S, best_dist

    best_S = set()
    best_dist = {}
    best_size = 0
    
    rng = random.Random(42)
    
    # Phase 1: Greedy
    while time.time() < deadline - 100:
        S, dist_sets = try_greedy_random(rng)
        if len(S) > best_size:
            best_S = S.copy()
            best_dist = {k: v.copy() for k, v in dist_sets.items()}
            best_size = len(S)
    
    # Phase 2: SA
    if best_size > 0:
        best_S, best_dist = simulated_annealing(best_S, best_dist, rng)
        best_size = len(best_S)
    
    result = [all_points[i] for i in best_S]
    return result

# EVOLVE-BLOCK-END
