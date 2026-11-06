import time
import random
import numpy as np

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 270.0
    
    # Precompute all grid points
    xs = np.arange(N)
    ys = np.arange(N)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    all_points = np.column_stack([XX.ravel(), YY.ravel()])  # shape (4096, 2)
    n_pts = N * N
    
    # Precompute squared distances between all pairs
    # Vectorized computation of full distance matrix
    # all_points shape (4096, 2)
    # diff shape (4096, 4096, 2)
    # This might be memory heavy (4096*4096*2*8 bytes ~ 268 MB for float64).
    # Let's use uint16/int32 carefully.
    # d = (x1-x2)^2 + (y1-y2)^2
    # Max value 63^2+63^2 = 7938. Fits in uint16.
    
    # To save memory and time, compute in chunks or use scipy if available, 
    # but standard numpy broadcasting is usually fine for 4kx4k.
    # Let's try direct broadcasting with int32 to avoid overflow in intermediate steps.
    
    p = all_points.astype(np.int32)
    # (4096, 1, 2) - (1, 4096, 2) -> (4096, 4096, 2)
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dist_mat = (diff ** 2).sum(axis=2).astype(np.uint16)
    
    best_set = []
    best_size = 0
    
    def try_add(p_idx, S):
        """
        Checks if p_idx can be added to S.
        S is a list of indices.
        Returns (True, new_S) or (False, None).
        """
        k = len(S)
        if k == 0:
            return True, [p_idx]
        
        S_arr = np.array(S, dtype=np.int32)
        dists = dist_mat[p_idx, S_arr]
        
        # Check 1: Distances from p to all points in S must be unique
        if len(np.unique(dists)) != k:
            return False, None
        
        # Check 2: For each point s_i in S, dist(p, s_i) must not equal dist(s_i, s_j) for any j in S
        # dist_mat[s_i, S_arr] gives distances from s_i to all points in S (including s_i itself which is 0)
        # Since dists[i] > 0 (p not in S), we just check if dists[i] is in the row.
        
        for i in range(k):
            s_i = S_arr[i]
            # Check if dists[i] is in dist_mat[s_i, S_arr]
            # Using np.isin is efficient for small arrays
            if np.isin(dists[i], dist_mat[s_i, S_arr]):
                return False, None
        
        return True, S + [p_idx]

    def greedy_from_order(order, S, time_limit):
        for idx in order:
            if time.time() > time_limit:
                break
            success, new_S = try_add(idx, S)
            if success:
                S = new_S
        return S

    def local_search(S, time_limit):
        """
        Attempts to improve S by removing one point and adding up to two others.
        """
        improved = True
        k = len(S)
        while improved and time.time() < time_limit:
            improved = False
            if k == 0:
                break
            
            # Try removing each point in S (shuffled)
            indices_to_try = list(range(k))
            random.shuffle(indices_to_try)
            
            in_set_mask = np.zeros(n_pts, dtype=bool)
            for pt in S:
                in_set_mask[pt] = True
            candidates = np.where(~in_set_mask)[0]
            
            for remove_idx in indices_to_try:
                if time.time() > time_limit:
                    break
                
                rem_S = [S[i] for i in range(k) if i != remove_idx]
                
                # Try to add 2 points from candidates to rem_S
                # We can pick random pairs or just iterate
                # To keep it fast, we try a limited number of random candidates
                # Or better: try to find ANY point that can be added to rem_S, then another.
                
                # First, find points that can be added to rem_S
                addable = []
                # Shuffle candidates for diversity
                cand_sample = candidates.copy()
                random.shuffle(cand_sample)
                
                # Limit checks to avoid slowing down if k is large
                check_limit = min(len(cand_sample), 200)
                
                for c in cand_sample[:check_limit]:
                    success, _ = try_add(c, rem_S)
                    if success:
                        addable.append(c)
                        if len(addable) >= 2:
                            break
                
                if len(addable) >= 2:
                    # Add both
                    S = rem_S + addable[:2]
                    k = len(S)
                    improved = True
                    # Update in_set_mask for next iteration? 
                    # It's easier to recompute at start of while loop or just break and let outer loop handle?
                    # The current structure breaks inner loop but continues outer.
                    # We should update in_set_mask to reflect new S if we continue, 
                    # but since we break the 'for remove_idx' loop, we go to next 'while' iteration.
                    # So in_set_mask is stale. We should recompute it at the start of the while loop.
                    break
                elif len(addable) == 1:
                    # Adding 1 point doesn't increase size. 
                    # But maybe swapping helps? 
                    # The task is to maximize size. Swapping 1 for 1 keeps size same.
                    # We only care about size increase.
                    pass
        
        return S

    num_restarts = 1000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        S = []
        
        if restart % 5 == 0:
            order = np.random.permutation(n_pts)
        elif restart % 5 == 1:
            order = np.arange(n_pts)
        elif restart % 5 == 2:
            order = np.arange(n_pts).reshape(N, N).T.ravel()
        elif restart % 5 == 3:
            order = []
            for d in range(2*N - 1):
                for x in range(max(0, d-N+1), min(N, d+1)):
                    y = d - x
                    if 0 <= y < N:
                        order.append(x * N + y)
            order = np.array(order)
        else:
            order = np.random.permutation(n_pts)
        
        S = greedy_from_order(order, S, deadline)
        
        if time.time() < deadline - 5:
            order2 = np.random.permutation(n_pts)
            S = greedy_from_order(order2, S, deadline)
        
        if time.time() < deadline - 10 and len(S) >= 5:
            S = local_search(S, deadline - 5)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = [(int(all_points[pt][0]), int(all_points[pt][1])) for pt in S]
            
    if best_size > 0 and time.time() < deadline - 5:
        polished_S = local_search(best_set_indices_to_list(best_set), deadline)
        if len(polished_S) > best_size:
            best_size = len(polished_S)
            best_set = [(int(all_points[pt][0]), int(all_points[pt][1])) for pt in polished_S]

    return best_set

def best_set_indices_to_list(best_set):
    return [x * N + y for x, y in best_set]
