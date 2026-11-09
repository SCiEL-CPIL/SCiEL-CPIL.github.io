"""Direct heuristic for TSP next-node selection."""
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s hard kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node for a TSP tour construction.
    
    Phase-aware scoring:
    - Large n: distance + destination bias + 2nd-nearest connectivity penalty
    - Medium n: 2-step lookahead + destination bias
    - Small n: full nearest-neighbor lookahead through remaining nodes
    """
    # Time guard
    if time.time() - _START_TIME > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        u, v = unvisited_nodes
        cost_u = distance_matrix[current_node, u] + distance_matrix[u, v] + distance_matrix[v, destination_node]
        cost_v = distance_matrix[current_node, v] + distance_matrix[v, u] + distance_matrix[u, destination_node]
        return int(unvisited_nodes[0] if cost_u <= cost_v else unvisited_nodes[1])
    
    # Adaptive destination weight
    alpha = 0.30 * (1.0 - n_unvisited / 55.0) + 0.05
    alpha = max(alpha, 0.05)
    alpha = min(alpha, 0.35)
    
    d_curr = distance_matrix[current_node, unvisited_nodes]
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    if n_unvisited <= 10:
        # Full NN lookahead
        lookahead_costs = np.zeros(n_unvisited)
        for i in range(n_unvisited):
            remaining = list(unvisited_nodes)
            remaining.pop(i)
            cur = unvisited_nodes[i]
            tour_cost = distance_matrix[current_node, unvisited_nodes[i]]
            visited_local = set([unvisited_nodes[i]])
            while remaining:
                d_to_rem = np.array([distance_matrix[cur, r] for r in remaining])
                j = int(np.argmin(d_to_rem))
                nxt = remaining[j]
                tour_cost += d_to_rem[j]
                remaining.pop(j)
                cur = nxt
            tour_cost += distance_matrix[cur, destination_node]
            lookahead_costs[i] = tour_cost
        
        min_lh = np.min(lookahead_costs)
        max_lh = np.max(lookahead_costs)
        if max_lh > min_lh:
            lookahead_norm = (lookahead_costs - min_lh) / (max_lh - min_lh)
        else:
            lookahead_norm = np.zeros(n_unvisited)
        
        gamma = 0.4
        scores = d_curr + alpha * d_dest + gamma * lookahead_norm * (max_lh - min_lh)
        
    elif n_unvisited <= 25:
        # 2-step lookahead: d(curr,i) + min_j d(i,j) + alpha * d(i,dest)
        sub_dm = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        # For each candidate i, the nearest other unvisited node
        np.fill_diagonal(sub_dm, np.inf)
        nearest_other = np.min(sub_dm, axis=1)  # min distance from each unvisited to any other unvisited
        
        # 2-step cost: d(curr,i) + nearest_other[i] + alpha * d_dest[i]
        scores = d_curr + nearest_other + alpha * d_dest
        
    else:
        # Large n: use 2nd nearest neighbor as connectivity penalty
        sub_dm = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        np.fill_diagonal(sub_dm, np.inf)
        # Two smallest distances per row
        # Use argpartition for efficiency
        idx = np.argpartition(sub_dm, 1, axis=1)[:, :2]
        two_smallest = np.take_along_axis(sub_dm, idx, axis=1)
        second_nearest = np.max(two_smallest, axis=1)
        
        min_c = np.min(second_nearest)
        max_c = np.max(second_nearest)
        if max_c > min_c:
            connect_norm = (second_nearest - min_c) / (max_c - min_c)
        else:
            connect_norm = np.zeros(n_unvisited)
        
        beta = 0.12
        scores = d_curr + alpha * d_dest + beta * connect_norm * (max_c - min_c)
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
