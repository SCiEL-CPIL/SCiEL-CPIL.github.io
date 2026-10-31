"""Traveling salesman constructive heuristic: regret-based selection with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next city using regret-based scoring with a time-bounded local refinement.
    
    Regret-based selection prioritizes nodes that would become expensive to visit later,
    capturing global tour structure. We combine:
      - Regret: second_min_dist - min_dist from current node (high regret = urgent)
      - Connection quality: how well-connected the candidate is to remaining nodes
      - Destination proximity: bias toward the final destination to close the tour
    
    A deadline guard ensures we return within the time budget.
    """
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return int(current_node)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Quick path: if only a few nodes left, do exhaustive evaluation
    if n_unvisited <= 8:
        return _exact_small_case(current_node, destination_node, unvisited_nodes, distance_matrix)
    
    # Compute base distances from current node to all unvisited
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Compute distances from destination to all unvisited (for tour-closing bias)
    dists_from_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Regret computation: for each candidate, how much worse is the 2nd-best option?
    # This is the classic regret-2 heuristic: prefer nodes where the "alternative" 
    # (best other node to go to) is much worse than going to this node now.
    
    # For efficiency with moderate n, compute regret in a vectorized way
    # regret[i] = second_min(dists_from_current excluding i) - min(dists_from_current excluding i)
    # But actually, regret is about: if I don't pick node i now, what's my best alternative?
    # So regret[i] = (best distance from current to any node != i) - dists_from_current[i]
    # Wait, that's not quite right. Standard regret-k: 
    #   regret(i) = sum of (k-th best - i-th best) ... 
    # Simpler: regret(i) = second_best_overall - best_overall_if_i_is_excluded
    # Actually the standard regret-2 is:
    #   For each candidate i, regret(i) = (2nd smallest dist from current) - (smallest dist from current)
    #   but that's the same for all i. 
    # 
    # Correct regret-2: regret(i) = min_{j != i} dist(current, j) - dist(current, i)
    #   i.e., how much worse is my best alternative compared to going to i?
    #   If i is the nearest, regret = 2nd_nearest - nearest (positive, good)
    #   If i is far, regret = nearest - far (negative, bad)
    
    # Compute min and second min distances
    sorted_dists = np.sort(dists_from_current)
    min_dist = sorted_dists[0]
    second_min_dist = sorted_dists[1]
    
    # Regret for each candidate: how much do I "lose" by not picking this one?
    # regret[i] = min_{j != i} dists[j] - dists[i]
    # If i is the argmin, regret[i] = second_min - min (positive)
    # If i is not the argmin, regret[i] = min - dists[i] (negative)
    
    # Vectorized: for each i, the min over j!=i is:
    #   if i is the unique argmin: second_min_dist
    #   else: min_dist
    min_overall = min_dist
    argmin_idx = np.argmin(dists_from_current)
    
    regret = np.empty(n_unvisited)
    regret[:] = min_overall - dists_from_current  # default: min - dists[i]
    regret[argmin_idx] = second_min_dist - min_dist  # special case for the argmin
    
    # Connection quality: average distance from candidate to all other unvisited nodes
    # Lower average = better connected = prefer
    if n_unvisited > 2:
        # dist_matrix among unvisited nodes
        unvisited_idx = unvisited_nodes
        sub_matrix = distance_matrix[np.ix_(unvisited_idx, unvisited_idx)]
        # Average distance from each node to others (excluding self)
        avg_conn = sub_matrix.sum(axis=1) / (n_unvisited - 1)
    else:
        avg_conn = np.zeros(n_unvisited)
    
    # Normalize scores to comparable scales
    regret_range = regret.max() - regret.min()
    if regret_range > 1e-12:
        regret_norm = (regret - regret.min()) / regret_range
    else:
        regret_norm = np.ones(n_unvisited) / n_unvisited
    
    conn_range = avg_conn.max() - avg_conn.min()
    if conn_range > 1e-12:
        conn_norm = (avg_conn.min() + avg_conn.max() - avg_conn) / conn_range  # invert: lower dist = higher score
    else:
        conn_norm = np.ones(n_unvisited) / n_unvisited
    
    # Destination proximity: prefer nodes closer to destination (helps close tour)
    dest_range = dists_from_dest.max() - dists_from_dest.min()
    if dest_range > 1e-12:
        dest_norm = (dists_from_dest.min() + dists_from_dest.max() - dists_from_dest) / dest_range
    else:
        dest_norm = np.ones(n_unvisited) / n_unvisited
    
    # Composite score: weighted combination
    # Regret is the primary signal (structural), connection and destination are secondary
    weight_regret = 0.5
    weight_conn = 0.3
    weight_dest = 0.2
    
    composite = weight_regret * regret_norm + weight_conn * conn_norm + weight_dest * dest_norm
    
    # Check deadline before any expensive refinement
    if time.time() > deadline - 5.0:
        return int(unvisited_nodes[np.argmax(composite)])
    
    # Local refinement: for the top-k candidates, do a 1-step lookahead
    # Evaluate: dist(current, candidate) + min_{j in unvisited\{candidate}} dist(candidate, j)
    # This captures the immediate cost of the choice
    k_top = min(5, n_unvisited)
    top_k_indices = np.argsort(composite)[-k_top:]
    
    best_score = -np.inf
    best_node = unvisited_nodes[np.argmax(composite)]
    
    for idx in top_k_indices:
        if time.time() > deadline - 2.0:
            break
        cand = unvisited_nodes[idx]
        # Immediate cost
        immediate = distance_matrix[current_node, cand]
        # Best next step from candidate (to any remaining unvisited)
        if n_unvisited > 1:
            mask = np.ones(n_unvisited, dtype=bool)
            mask[idx] = False
            remaining = unvisited_nodes[mask]
            next_costs = distance_matrix[cand, remaining]
            best_next = np.min(next_costs) if len(remaining) > 0 else 0.0
        else:
            best_next = 0.0
        
        # Also consider distance to destination if this might be the last
        if n_unvisited == 1:
            lookahead = immediate + distance_matrix[cand, destination_node]
        else:
            lookahead = immediate + best_next
        
        # Add a small destination bias for the last few nodes
        if n_unvisited <= 3:
            lookahead += 0.1 * distance_matrix[cand, destination_node]
        
        if lookahead < best_score:
            best_score = lookahead
            best_node = cand
    
    return int(best_node)


def _exact_small_case(current_node, destination_node, unvisited_nodes, distance_matrix):
    """For small cases, evaluate all candidates with 2-step lookahead."""
    n = len(unvisited_nodes)
    best_score = np.inf
    best_node = unvisited_nodes[0]
    
    for i in range(n):
        cand = unvisited_nodes[i]
        cost = distance_matrix[current_node, cand]
        
        if n == 1:
            cost += distance_matrix[cand, destination_node]
        elif n == 2:
            other = unvisited_nodes[1 - i]
            cost += distance_matrix[cand, other]
            cost += distance_matrix[other, destination_node]
        else:
            # 2-step lookahead: best next from cand
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            remaining = unvisited_nodes[mask]
            next_costs = distance_matrix[cand, remaining]
            cost += np.min(next_costs)
        
        if cost < best_score:
            best_score = cost
            best_node = cand
    
    return int(best_node)
# EVOLVE-BLOCK-END
