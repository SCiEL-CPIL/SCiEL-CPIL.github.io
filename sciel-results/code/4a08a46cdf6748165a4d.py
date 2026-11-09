import numpy as np
import time

_start_time = None

def _get_time_remaining():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, 145.0 - elapsed)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Time guard
    if _get_time_remaining() < 1.0:
        # Fallback to simple NN
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n = len(unvisited_nodes)
    if n == 0:
        return -1
    if n == 1:
        return int(unvisited_nodes[0])

    best_node = -1
    best_score = float('inf')

    # If few nodes left, compute exact cost of completing the tour
    if n <= 5:
        for i in range(n):
            v = unvisited_nodes[i]
            remaining = np.delete(unvisited_nodes, i)
            if len(remaining) == 0:
                total_score = distance_matrix[current_node, v] + distance_matrix[v, destination_node]
            elif n == 2:
                other = remaining[0]
                total_score = distance_matrix[current_node, v] + distance_matrix[v, other] + distance_matrix[other, destination_node]
            elif n == 3:
                w1, w2 = remaining
                path1 = distance_matrix[v, w1] + distance_matrix[w1, w2] + distance_matrix[w2, destination_node]
                path2 = distance_matrix[v, w2] + distance_matrix[w2, w1] + distance_matrix[w1, destination_node]
                total_score = distance_matrix[current_node, v] + min(path1, path2)
            else:
                # For n=4,5 use greedy completion estimate
                cost = distance_matrix[current_node, v]
                rem = remaining
                while len(rem) > 0:
                    next_node = rem[np.argmin(distance_matrix[v, rem])]
                    cost += distance_matrix[v, next_node]
                    v = next_node
                    rem = np.delete(rem, np.argmin(distance_matrix[v, rem]))
                cost += distance_matrix[v, destination_node]
                total_score = cost
                
            if total_score < best_score:
                best_score = total_score
                best_node = v
                
        return int(best_node)

    # For n > 5, use global connectivity-aware heuristic
    # Score = d(current, v) + 0.5 * (min_edge_to_unvisited(v) + min_edge_to_destination_or_unvisited(v))
    # This approximates the degree-2 completion cost: each node in the final tour
    # has degree 2. When we pick v next, v will have one edge from current.
    # The other edge of v should go to its nearest neighbor in the remaining set
    # or to destination if v is likely to be the last node.
    
    # Precompute min distances from each unvisited node to all other unvisited nodes
    # and to destination
    min_to_unvisited = np.full(n, float('inf'))
    min_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # For each candidate v, we want to estimate the "future cost" of connecting v
    # to the rest of the tour. In a degree-2 constraint, v needs one more connection.
    # The best such connection is min(min_to_unvisited[v], min_to_dest[v])
    # But min_to_unvisited[v] is the min distance from v to any other unvisited node.
    
    for i in range(n):
        v = unvisited_nodes[i]
        cost_to_v = distance_matrix[current_node, v]
        
        # Find min distance from v to any other unvisited node
        if n > 1:
            dists_v = distance_matrix[v, unvisited_nodes]
            dists_v[i] = float('inf')
            min_to_other_unvisited = np.min(dists_v)
        else:
            min_to_other_unvisited = float('inf')
        
        # The "future cost" for v is the min of connecting to another unvisited or to destination
        future_cost = min(min_to_other_unvisited, min_to_dest[i])
        
        # Total score: immediate cost + estimated future connection cost
        total_score = cost_to_v + 0.5 * future_cost
        
        if total_score < best_score:
            best_score = total_score
            best_node = v
            
    return int(best_node)
