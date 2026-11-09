"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None


def _get_time_remaining():
    """Return seconds remaining before the 150s limit (with safety margin)."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, 145.0 - elapsed)


def _mst_lower_bound(nodes, distance_matrix):
    """
    Compute a lower bound on the cost to visit all remaining nodes and return to destination.
    Uses MST over the remaining nodes plus the two cheapest edges to/from the boundary.
    """
    n = len(nodes)
    if n == 0:
        return 0.0
    if n == 1:
        return distance_matrix[nodes[0], destination_node] + distance_matrix[destination_node, nodes[0]]
    
    # Extract submatrix for nodes
    submat = distance_matrix[np.ix_(nodes, nodes)]
    
    # Compute MST weight using Prim's algorithm
    in_mst = [False] * n
    in_mst[0] = True
    min_dist = submat[0].copy()
    mst_weight = 0.0
    
    for _ in range(n - 1):
        # Find minimum distance node not in MST
        min_dist[in_mst] = np.inf
        u = np.argmin(min_dist)
        mst_weight += min_dist[u]
        in_mst[u] = True
        min_dist = np.minimum(min_dist, submat[u])
    
    # Add boundary costs: cheapest edge from current to any node, and cheapest edge from any node to destination
    # But we need to be careful: the MST already connects all nodes.
    # A better lower bound: MST weight + min edge from current to set + min edge from set to destination
    # However, this might double-count. Let's use: MST + min(current->set) + min(set->dest) - max_edge_in_MST?
    # Actually, a simple valid lower bound is just the MST weight itself (since any path visiting all nodes must have cost >= MST).
    # But we also need to account for the start and end points.
    
    # The tour segment is: current -> [permutation of nodes] -> destination
    # This is a path, not a cycle. The MST of the nodes is a lower bound on the cost to connect them.
    # But we also need to connect current to the first node and the last node to destination.
    
    # A tighter lower bound:
    # 1. MST of the nodes
    # 2. + min over nodes of d(current, node)
    # 3. + min over nodes of d(node, destination)
    # This is valid because the path must enter the set from current and leave to destination.
    
    min_entry = np.min(distance_matrix[current_node, nodes])
    min_exit = np.min(distance_matrix[nodes, destination_node])
    
    return mst_weight + min_entry + min_exit


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Time guard: if we're running low on time, fall back to simple nearest neighbor
    if _get_time_remaining() < 0.5:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    candidates = unvisited_nodes
    n = len(candidates)
    
    if n == 0:
        return destination_node
    
    if n == 1:
        return int(candidates[0])
    
    # Use MST-based lower bound for structural guidance
    # For each candidate c:
    # 1. Cost to reach c: d(current, c)
    # 2. Cost to complete the tour from c: MST of remaining nodes (excluding c) + min edge from c to remaining + min edge from remaining to dest
    #    This captures global connectivity better than local 1-step lookahead.
    
    best_score = float('inf')
    best_node = candidates[0]
    
    # Pre-compute distances from current to all candidates
    dist_from_current = distance_matrix[current_node, candidates]
    
    for i, c in enumerate(candidates):
        cost_in = dist_from_current[i]
        
        if n == 2:
            # Exact calculation for 2 remaining nodes
            other_node = candidates[1 - i]
            # Path: current -> c -> other -> destination
            cost_total = distance_matrix[current_node, c] + distance_matrix[c, other_node] + distance_matrix[other_node, destination_node]
            total_score = cost_total
        elif n == 3:
            # Exact calculation for 3 remaining nodes (try both orders for the other two)
            others = np.delete(candidates, i)
            a, b = others[0], others[1]
            # Order 1: c -> a -> b -> dest
            cost1 = distance_matrix[c, a] + distance_matrix[a, b] + distance_matrix[b, destination_node]
            # Order 2: c -> b -> a -> dest
            cost2 = distance_matrix[c, b] + distance_matrix[b, a] + distance_matrix[a, destination_node]
            cost_total = distance_matrix[current_node, c] + min(cost1, cost2)
            total_score = cost_total
        else:
            # For larger n, use MST lower bound on remaining nodes
            remaining = np.delete(candidates, i)
            
            # MST of remaining nodes
            if len(remaining) == 0:
                mst_cost = 0.0
                min_exit = 0.0
            elif len(remaining) == 1:
                mst_cost = 0.0
                min_exit = distance_matrix[remaining[0], destination_node]
            else:
                # Compute MST of remaining
                submat = distance_matrix[np.ix_(remaining, remaining)]
                in_mst = [False] * len(remaining)
                in_mst[0] = True
                min_dist = submat[0].copy()
                mst_cost = 0.0
                for _ in range(len(remaining) - 1):
                    min_dist[in_mst] = np.inf
                    u = np.argmin(min_dist)
                    mst_cost += min_dist[u]
                    in_mst[u] = True
                    min_dist = np.minimum(min_dist, submat[u])
                min_exit = np.min(distance_matrix[remaining, destination_node])
            
            # Cost from c to enter the remaining set
            min_entry_from_c = np.min(distance_matrix[c, remaining]) if len(remaining) > 0 else 0.0
            
            # Total estimated cost: d(current,c) + min(d(c,remaining)) + MST(remaining) + min(d(remaining,dest))
            # This is a lower bound, but we use it as a heuristic score.
            # To make it a better heuristic, we can weight the MST term.
            total_score = cost_in + min_entry_from_c + mst_cost + min_exit
        
        if total_score < best_score:
            best_score = total_score
            best_node = c
            
    return int(best_node)
# EVOLVE-BLOCK-END
