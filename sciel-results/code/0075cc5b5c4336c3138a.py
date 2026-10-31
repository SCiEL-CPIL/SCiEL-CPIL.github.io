"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 145.0  # seconds; safety margin before 150s kill
_START_TIME = time.time()


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """Pick next city: hybrid of nearest-neighbor, destination bias, and isolation penalty."""
    # Deadline guard: if we're close to the limit, fall back to fast NN
    if time.time() > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n_unvisited = len(unvisited_nodes)
    if n_unvisited <= 1:
        return int(unvisited_nodes[0])

    # Distances from current node to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]  # shape (n_unvisited,)
    # Distances from destination to all unvisited
    dist_to_dest = distance_matrix[destination_node, unvisited_nodes]  # shape (n_unvisited,)

    # Destination bias grows as we approach the end:
    # When few nodes remain, heavily favor nodes close to destination
    if n_unvisited <= 5:
        dest_weight = 0.8
    elif n_unvisited <= 10:
        dest_weight = 0.4
    elif n_unvisited <= 20:
        dest_weight = 0.2
    else:
        dest_weight = 0.05

    # Isolation penalty: for each candidate i, estimate how "isolated" the remaining
    # nodes would be if we pick i. Approximate by checking the min distance from
    # each remaining node to any other remaining node (excluding the candidate).
    # To keep this fast, we only do a full isolation check when n_unvisited is small enough.
    if n_unvisited <= 25:
        # Compute a cheap isolation score for each candidate
        # For candidate i, the remaining set is unvisited_nodes minus i.
        # Approximate: sum over remaining nodes of (min distance to another remaining node)
        # This is O(n^2) which is fine for n <= 25.
        sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]  # (n, n)
        # For each candidate i, compute the max over j != i of min_{k != i, k != j} sub_dist[j, k]
        # Simplify: use the 2nd smallest distance from each node j to any other node in the set
        # Actually, let's use a simpler proxy: for candidate i, look at the max over
        # remaining nodes j of the minimum distance from j to any node in remaining \ {j}
        # This is expensive. Let's use a lighter proxy:
        # For each candidate i, compute the average of the 2nd-smallest distance from
        # each remaining node to the rest.
        
        # Precompute: for each node j in unvisited, the two smallest distances to other unvisited nodes
        # sorted_dists[j] = [d1, d2] where d1 <= d2 are the two smallest
        min1 = np.full(n_unvisited, np.inf)
        min2 = np.full(n_unvisited, np.inf)
        for j in range(n_unvisited):
            row = sub_dist[j].copy()
            row[j] = np.inf
            idx = np.argmin(row)
            min1[j] = row[idx]
            row[idx] = np.inf
            min2[j] = np.min(row)
        
        # For candidate i, the remaining nodes are all except i.
        # The isolation cost is approximated by the max of min2[j] for j != i,
        # but we need to account for the fact that node i is removed, so min1[j] might
        # have been to node i. Use min2[j] as a safe upper bound on the new min distance.
        # Actually, if min1[j] was to i, then the new min is min2[j]. If not, it's still min1[j].
        # To be safe and fast, use min2[j] for all j != i.
        
        # Precompute: for each i, the max of min2[j] for j != i
        # This is the max of min2 excluding position i.
        # Sort min2 to find the top values
        sorted_min2 = np.sort(min2)
        # The max excluding i: if min2[i] is the global max, use the 2nd largest; else use global max
        global_max = sorted_min2[-1]
        second_max = sorted_min2[-2]
        
        isolation_costs = np.full(n_unvisited, global_max)
        # Where min2[i] == global_max, the max excluding i is second_max
        # (if there are ties, this is still fine as an approximation)
        mask = (min2 == global_max)
        isolation_costs[mask] = second_max
        
        # Normalize isolation costs to [0, 1]
        iso_range = isolation_costs.max() - isolation_costs.min()
        if iso_range > 1e-9:
            isolation_costs = (isolation_costs - isolation_costs.min()) / iso_range
        else:
            isolation_costs = np.zeros(n_unvisited)
    else:
        isolation_costs = np.zeros(n_unvisited)

    # Normalize dist_from_current to [0, 1]
    d_range = dist_from_current.max() - dist_from_current.min()
    if d_range > 1e-9:
        dist_norm = (dist_from_current - dist_from_current.min()) / d_range
    else:
        dist_norm = np.zeros(n_unvisited)

    # Normalize dist_to_dest to [0, 1]
    dd_range = dist_to_dest.max() - dist_to_dest.min()
    if dd_range > 1e-9:
        dest_norm = (dist_to_dest - dist_to_dest.min()) / dd_range
    else:
        dest_norm = np.zeros(n_unvisited)

    # Weighted score: lower is better
    # Primary: distance from current. Secondary: distance to destination. Tertiary: isolation.
    iso_weight = 0.1 if n_unvisited <= 25 else 0.0
    score = dist_norm + dest_weight * dest_norm + iso_weight * isolation_costs

    best_idx = np.argmin(score)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
