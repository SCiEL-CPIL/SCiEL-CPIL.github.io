import numpy as np
import time

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    n = distance_matrix.shape[0]
    
    # Helper to calculate tour length using vectorized operations
    def get_tour_length(t):
        idx = np.array(t)
        next_idx = np.roll(idx, -1)
        return np.sum(distance_matrix[idx, next_idx])

    # 2-Opt improvement with early termination
    def two_opt(t):
        t = list(t)
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for i in range(n - 1):
                if time.time() > deadline:
                    return t
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    a = t[i]
                    b = t[i + 1]
                    c = t[j]
                    d = t[(j + 1) % n]
                    
                    delta = (distance_matrix[a, c] + distance_matrix[b, d]) - (distance_matrix[a, b] + distance_matrix[c, d])
                    if delta < -1e-10:
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                        improved = True
        return t

    # Or-Opt move: remove a segment of 1-3 nodes and reinsert it elsewhere
    def or_opt(t):
        t = list(t)
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for seg_len in [1, 2, 3]:
                for i in range(n):
                    if time.time() > deadline:
                        return t
                    if i + seg_len >= n:
                        continue
                    # Extract segment
                    seg = t[i:i + seg_len]
                    # Remove segment
                    remaining = t[:i] + t[i + seg_len:]
                    if len(remaining) < 2:
                        continue
                    # Try inserting at each position in remaining
                    best_pos = -1
                    best_delta = float('inf')
                    for k in range(len(remaining)):
                        # Skip if inserting right back where it was
                        if k == i and i + seg_len <= len(remaining) + seg_len:
                            # Check if this is the original position
                            if k == 0 or (i > 0 and k == i):
                                # Only skip if it's exactly the original spot
                                if i == 0 and k == 0:
                                    continue
                                elif i > 0 and k == i:
                                    continue
                        prev = remaining[(k - 1) % len(remaining)]
                        next_ = remaining[k % len(remaining)]
                        # Old edges: prev->next_
                        # New edges: prev->seg[0], seg[-1]->next_
                        old_cost = distance_matrix[prev, next_]
                        new_cost = distance_matrix[prev, seg[0]] + distance_matrix[seg[-1], next_]
                        # Also need to account for internal segment edges staying same
                        delta = new_cost - old_cost
                        if delta < best_delta:
                            best_delta = delta
                            best_pos = k
                    if best_pos >= 0 and best_delta < -1e-10:
                        # Reconstruct tour: insert seg at best_pos in remaining
                        new_tour = remaining[:best_pos] + seg + remaining[best_pos:]
                        # Verify length actually improved
                        if get_tour_length(new_tour) < get_tour_length(t) - 1e-10:
                            t = new_tour
                            improved = True
        return t

    # Combined local search: 2-Opt then Or-Opt
    def local_search(t):
        t = two_opt(t)
        t = or_opt(t)
        t = two_opt(t)  # Another 2-Opt pass after Or-Opt
        return t

    # Nearest Neighbor construction
    def nearest_neighbor(start_node):
        visited = np.zeros(n, dtype=bool)
        visited[start_node] = True
        tour = [start_node]
        current = start_node
        
        for _ in range(n - 1):
            if time.time() > deadline:
                break
            dists = distance_matrix[current, :].copy()
            dists[visited] = np.inf
            next_node = np.argmin(dists)
            tour.append(next_node)
            visited[next_node] = True
            current = next_node
        return tour

    # Farthest Insertion construction
    def farthest_insertion():
        dists_from_0 = distance_matrix[0, :]
        farthest_node = np.argmax(dists_from_0)
        tour = [0, farthest_node]
        remaining = [i for i in range(n) if i not in tour]
        
        while remaining:
            if time.time() > deadline:
                break
            max_min_dist = -1
            point_to_insert = -1
            for k in remaining:
                min_dist_to_tour = np.min(distance_matrix[k, tour])
                if min_dist_to_tour > max_min_dist:
                    max_min_dist = min_dist_to_tour
                    point_to_insert = k
            
            best_pos = -1
            best_increase = float('inf')
            for i in range(len(tour)):
                prev_node = tour[i]
                next_node = tour[(i + 1) % len(tour)]
                increase = (distance_matrix[prev_node, point_to_insert] + 
                            distance_matrix[point_to_insert, next_node] - 
                            distance_matrix[prev_node, next_node])
                if increase < best_increase:
                    best_increase = increase
                    best_pos = i + 1
            
            tour.insert(best_pos, point_to_insert)
            remaining.remove(point_to_insert)
            
        return tour

    # Main search loop
    best_tour = None
    best_len = float('inf')
    
    # 1. Nearest Neighbor from 0
    tour_nn = nearest_neighbor(0)
    tour_nn = local_search(tour_nn)
    len_nn = get_tour_length(tour_nn)
    if len_nn < best_len:
        best_len = len_nn
        best_tour = tour_nn

    # 2. Farthest Insertion
    tour_fi = farthest_insertion()
    tour_fi = local_search(tour_fi)
    len_fi = get_tour_length(tour_fi)
    if len_fi < best_len:
        best_len = len_fi
        best_tour = tour_fi

    # 3. Random Restarts with time budget
    rng = np.random.default_rng(42)
    num_restarts = 50  # More restarts, but time-limited
    
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        
        rand_tour = list(range(n))
        rng.shuffle(rand_tour)
        improved_rand = local_search(rand_tour)
        len_rand = get_tour_length(improved_rand)
        if len_rand < best_len:
            best_len = len_rand
            best_tour = improved_rand

    # Fallback if something went wrong
    if best_tour is None:
        best_tour = list(range(n))
        best_tour = local_search(best_tour)
        best_len = get_tour_length(best_tour)

    # Construct Heuristic Matrix
    safe_dist = distance_matrix.copy()
    np.fill_diagonal(safe_dist, 1.0)
    
    eta = 1.0 / safe_dist
    
    # Boost edges in the best found tour with a stronger factor
    boost_factor = 5000.0
    
    for i in range(n):
        u = best_tour[i]
        v = best_tour[(i + 1) % n]
        eta[u, v] = boost_factor
    
    # Ensure diagonal is not attractive
    np.fill_diagonal(eta, 1e-9)
    
    # Ensure all values are positive and finite
    eta = np.maximum(eta, 1e-9)
    eta = np.where(np.isfinite(eta), eta, 1e-9)
    
    return eta
