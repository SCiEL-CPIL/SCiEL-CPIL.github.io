import numpy as np
import time

# Global deadline guard
_DEADLINE = time.time() + 145  # Safety margin for 150s limit

def _lower_bound_estimate(current, unvisited, destination, dist_matrix):
    """
    Estimate the cost to complete the tour from 'current' visiting all 'unvisited' and returning to 'destination'.
    Uses a simple lower bound: 
    1. Min cost to enter the set of unvisited from current.
    2. Min cost to leave the set of unvisited to destination.
    3. Min spanning tree cost on unvisited nodes (approximated by sum of min 2 edges per node / 2).
    """
    if len(unvisited) == 0:
        return dist_matrix[current, destination]
    
    # Min edge from current to any unvisited
    min_in = np.min(dist_matrix[current, unvisited])
    # Min edge from any unvisited to destination
    min_out = np.min(dist_matrix[unvisited, destination])
    
    # Approximate MST cost for unvisited nodes
    if len(unvisited) == 1:
        mst_cost = 0
    else:
        sub_dist = dist_matrix[np.ix_(unvisited, unvisited)]
        np.fill_diagonal(sub_dist, np.inf)
        # Get 2 smallest per row
        mins = np.partition(sub_dist, 1, axis=1)[:, :2]
        mst_cost = np.sum(mins) / 2.0
        
    return min_in + min_out + mst_cost

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _DEADLINE
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Time check: if we're running low on time, fall back to simple nearest neighbor
    if time.time() > _DEADLINE - 1.0:
        dists = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists)
        return int(unvisited_nodes[best_idx])
    
    if n_unvisited == 2:
        a = int(unvisited_nodes[0])
        b = int(unvisited_nodes[1])
        cost_a = distance_matrix[current_node, a] + distance_matrix[a, b] + distance_matrix[b, destination_node]
        cost_b = distance_matrix[current_node, b] + distance_matrix[b, a] + distance_matrix[a, destination_node]
        return a if cost_a < cost_b else b

    # Beam Search Strategy
    # State: (path_nodes, cost_so_far, last_node)
    # We maintain a beam of the best 'width' paths of length 'depth'.
    
    BEAM_WIDTH = 20
    DEPTH = 4
    
    # Initial beam: one path starting at current_node
    beam = [(0.0, [], current_node)]
    
    for step in range(DEPTH):
        if not beam:
            break
            
        next_beam = []
        
        for cost_so_far, path, last in beam:
            visited_set = set(path)
            
            for node in unvisited_nodes:
                if node in visited_set:
                    continue
                
                new_cost = cost_so_far + distance_matrix[last, node]
                next_beam.append((new_cost, path + [node], node))
        
        if not next_beam:
            break
            
        next_beam.sort(key=lambda x: x[0])
        beam = next_beam[:BEAM_WIDTH]
        
        # Early termination if time is tight
        if time.time() > _DEADLINE - 2.0:
             if beam and len(beam[0][1]) > 0:
                 return int(beam[0][1][0])
             else:
                 dists = distance_matrix[current_node, unvisited_nodes]
                 best_idx = np.argmin(dists)
                 return int(unvisited_nodes[best_idx])

    # After DEPTH steps, select the path with the lowest estimated total cost
    best_total_cost = float('inf')
    best_first_node = -1
    
    for cost_so_far, path, last in beam:
        visited_set = set(path)
        remaining = [n for n in unvisited_nodes if n not in visited_set]
        remaining_arr = np.array(remaining)
        
        lb = _lower_bound_estimate(last, remaining_arr, destination_node, distance_matrix)
        total_est = cost_so_far + lb
        
        if total_est < best_total_cost:
            best_total_cost = total_est
            if len(path) > 0:
                best_first_node = int(path[0])
            else:
                best_first_node = int(unvisited_nodes[0])
                
    if best_first_node == -1:
        dists = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists)
        best_first_node = int(unvisited_nodes[best_idx])
        
    return best_first_node
