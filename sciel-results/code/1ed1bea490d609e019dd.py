import numpy as np
import time

# Global state
_current_tour = []
_start_time = None
_TIME_LIMIT = 145.0

def _reset_state():
    global _current_tour, _start_time
    _current_tour = []
    _start_time = time.time()

def _apply_2opt(tour, distance_matrix, max_passes=3):
    """
    Apply 2-opt improvement.
    """
    n = len(tour)
    if n < 4:
        return tour
    
    t = tour[:]
    improved = True
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                
                if j == n - 1:
                    old_cost = distance_matrix[t[i], t[i+1]] + distance_matrix[t[j], t[0]]
                    new_cost = distance_matrix[t[i], t[j]] + distance_matrix[t[i+1], t[0]]
                else:
                    old_cost = distance_matrix[t[i], t[i+1]] + distance_matrix[t[j], t[j+1]]
                    new_cost = distance_matrix[t[i], t[j]] + distance_matrix[t[i+1], t[j+1]]
                
                if new_cost < old_cost - 1e-8:
                    t[i+1:j+1] = t[i+1:j+1][::-1]
                    improved = True
                    break
            
            if improved:
                break
        
    return t

def _apply_oropt(tour, distance_matrix, max_passes=2):
    """
    Apply Or-opt (move 1 city).
    """
    n = len(tour)
    if n < 4:
        return tour
    
    t = tour[:]
    improved = True
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        
        for i in range(n):
            city = t[i]
            prev_city = t[i-1] if i > 0 else t[n-1]
            next_city = t[(i+1) % n]
            
            # Remove city
            if i == 0:
                t_no_city = t[1:]
            elif i == n-1:
                t_no_city = t[:-1]
            else:
                t_no_city = t[:i] + t[i+1:]
            
            min_cost = float('inf')
            best_pos = -1
            
            for k in range(len(t_no_city)):
                pos_prev = t_no_city[k]
                pos_next = t_no_city[(k+1) % len(t_no_city)]
                
                # If we are inserting back into the original gap, skip?
                # The original gap is between prev_city and next_city.
                # If pos_prev == prev_city and pos_next == next_city, it's the same spot.
                # We allow it, cost will be equal.
                
                orig_cost = distance_matrix[prev_city, city] + distance_matrix[city, next_city] + distance_matrix[pos_prev, pos_next]
                new_cost = distance_matrix[prev_city, next_city] + distance_matrix[pos_prev, city] + distance_matrix[city, pos_next]
                
                if new_cost < min_cost:
                    min_cost = new_cost
                    best_pos = k
                    best_prev = pos_prev
                    best_next = pos_next
            
            if best_pos != -1:
                pos_prev = t_no_city[best_pos]
                pos_next = t_no_city[(best_pos+1) % len(t_no_city)]
                
                orig_cost = distance_matrix[prev_city, city] + distance_matrix[city, next_city] + distance_matrix[pos_prev, pos_next]
                new_cost = distance_matrix[prev_city, next_city] + distance_matrix[pos_prev, city] + distance_matrix[city, pos_next]
                
                if new_cost < orig_cost - 1e-8:
                    t.pop(i)
                    idx = t.index(pos_prev)
                    t.insert(idx + 1, city)
                    improved = True
                    break
        
    return t

def _cheapest_insertion_complete(tour_start, remaining_nodes, distance_matrix):
    """
    Complete the tour using Cheapest Insertion.
    """
    t = tour_start[:]
    rem = list(remaining_nodes)
    
    while rem:
        n_rem = len(rem)
        n_tour = len(t)
        
        if n_tour == 1:
            dists = distance_matrix[t[0], rem]
            best_idx = np.argmin(dists)
            best_node = rem[best_idx]
            t.append(best_node)
            rem.pop(best_idx)
            continue
            
        best_total_cost = float('inf')
        best_node_idx = -1
        best_insert_pos = -1
        
        for r_idx, node in enumerate(rem):
            for k in range(n_tour):
                u = t[k]
                v = t[(k+1) % n_tour]
                delta = distance_matrix[u, node] + distance_matrix[node, v] - distance_matrix[u, v]
                if delta < best_total_cost:
                    best_total_cost = delta
                    best_node_idx = r_idx
                    best_insert_pos = k
        
        if best_node_idx != -1:
            best_node = rem.pop(best_node_idx)
            t.insert(best_insert_pos + 1, best_node)
            
    return t

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _current_tour, _start_time
    
    if _start_time is None:
        _start_time = time.time()
        _current_tour = [current_node]
    
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    if len(unvisited_nodes) == 0:
        return -1
    
    if len(unvisited_nodes) == 1:
        next_node = int(unvisited_nodes[0])
        _current_tour.append(next_node)
        return next_node
    
    best_candidate = -1
    best_score = float('inf')
    
    for candidate in unvisited_nodes:
        sim_tour = _current_tour + [candidate]
        
        rem_mask = np.ones(len(unvisited_nodes), dtype=bool)
        idx_c = np.where(unvisited_nodes == candidate)[0][0]
        rem_mask[idx_c] = False
        remaining_nodes = unvisited_nodes[rem_mask]
        
        if time.time() - _start_time > _TIME_LIMIT - 10:
            curr = candidate
            rem = list(remaining_nodes)
            while rem:
                dists = distance_matrix[curr, rem]
                next_idx = np.argmin(dists)
                next_node = rem[next_idx]
                sim_tour.append(next_node)
                curr = next_node
                rem.pop(next_idx)
        else:
            sim_tour = _cheapest_insertion_complete(sim_tour, remaining_nodes, distance_matrix)
            
        if time.time() - _start_time > _TIME_LIMIT - 5:
            n = len(sim_tour)
            score = 0.0
            for i in range(n - 1):
                score += distance_matrix[sim_tour[i], sim_tour[i+1]]
            score += distance_matrix[sim_tour[-1], sim_tour[0]]
        else:
            optimized_tour = _apply_2opt(sim_tour, distance_matrix, max_passes=3)
            optimized_tour = _apply_oropt(optimized_tour, distance_matrix, max_passes=2)
            
            n = len(optimized_tour)
            score = 0.0
            for i in range(n - 1):
                score += distance_matrix[optimized_tour[i], optimized_tour[i+1]]
            score += distance_matrix[optimized_tour[-1], optimized_tour[0]]
        
        if score < best_score:
            best_score = score
            best_candidate = candidate
            
    _current_tour.append(best_candidate)
    return int(best_candidate)
