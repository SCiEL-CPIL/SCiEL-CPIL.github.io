import numpy as np
import time

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a 2-step lookahead heuristic.
    
    Strategy:
    Instead of using a global average distance (centrality) which is a blunt heuristic,
    we estimate the local cost of inserting a node by looking at its two nearest 
    neighbors among the unvisited nodes. This approximates the 2-opt improvement 
    and avoids picking nodes that are isolated or create long "spikes" in the 
    remaining tour.
    
    Score for candidate u:
    1. Immediate cost: dist(current, u)
    2. Future cost estimate: sum of distances from u to its 2 nearest unvisited neighbors.
    3. Destination bias: if few nodes remain, add penalty for dist(u, destination).
    
    This is a local geometric heuristic that adapts to the local density of points.
    """
    if len(unvisited_nodes) == 0:
        return -1
        
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    n_total = len(distance_matrix)
    n_remaining = len(unvisited_nodes)
    
    # 1. Immediate distances from current node to all candidates
    # dists_from_current shape: (n_remaining,)
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Compute local neighborhood costs for each candidate
    # We need the distance from each candidate to its 2 nearest neighbors in the unvisited set.
    # sub_matrix is (n_remaining, n_remaining)
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Set diagonal to infinity so a node is not its own neighbor
    # Note: sub_matrix is a view or copy? np.ix_ usually returns a copy or advanced indexing result.
    # We need to be careful not to modify the original distance_matrix.
    # Let's create a copy for safety if we modify it, but we don't need to modify the original.
    # We can just use np.partition to find the 2 smallest values.
    
    # For each row i (candidate i), we want the sum of the 2 smallest distances in that row.
    # The diagonal is 0. We must exclude the diagonal.
    
    # Efficient way:
    # 1. Set diagonal to a large number temporarily in a copy? 
    #    Or just use np.argpartition.
    
    # Let's use np.argpartition to find indices of the 2 smallest distances.
    # k=2 means we want the 2 smallest.
    
    # Since N is small (<=50), we can just sort or use partition.
    # sub_matrix shape (N_rem, N_rem)
    
    # We need to ignore the self-distance (diagonal).
    # Create a mask or just set diagonal to inf in a temporary array.
    # For N=50, creating a temporary array is negligible.
    
    temp_sub = sub_matrix.copy()
    np.fill_diagonal(temp_sub, np.inf)
    
    # Find the indices of the 2 smallest distances for each row
    # np.argpartition(temp_sub, kth=1, axis=1) returns indices of the 2 smallest elements.
    # It doesn't sort them, but we just need their values.
    
    if n_remaining > 2:
        # kth=1 means the element at index 1 is the 2nd smallest.
        # This partitions the array such that elements < index 1 are smaller.
        # We want the 2 smallest.
        partition_indices = np.argpartition(temp_sub, kth=1, axis=1)
        # Get the values at these indices
        # partition_indices shape: (N_rem, 2)
        # We need to gather the values.
        # row_indices = np.arange(n_remaining)
        # neighbor_dists = temp_sub[row_indices[:, None], partition_indices]
        
        row_indices = np.arange(n_remaining)
        neighbor_dists = temp_sub[row_indices[:, None], partition_indices]
        
        # Sum of the 2 nearest neighbor distances
        local_cost = np.sum(neighbor_dists, axis=1)
    else:
        # If only 2 nodes left, the "2 nearest neighbors" for a node is just the other node (and itself which is inf).
        # Actually if n_remaining=2, temp_sub has one 0 (diag) and one d. 
        # Wait, if n_remaining=2, we return early? No, we return early if len==1.
        # If len==2, we pick one, then the other.
        # For n_remaining=2, the cost of picking u is dist(current, u). The next node is the only other one.
        # The "local cost" concept is less critical. 
        # Let's just set local_cost to 0 or the distance to the other node.
        # Using the partition logic:
        # temp_sub for 2x2 with diag inf:
        # [[inf, d12], [d21, inf]]
        # argpartition kth=1:
        # Row 0: indices [1, 0] -> values [d12, inf]. Sum = d12 + inf? No, we should handle this.
        # Better: if n_remaining == 2, local_cost is just the distance to the other node.
        local_cost = temp_sub.sum(axis=1) # This will be d12 + inf? No, fill_diagonal set it to inf.
        # Actually, for N=2, the 2nd smallest is inf.
        # Let's just use the sum of all non-inf elements?
        # Simpler: if n_remaining <= 2, local_cost = 0. The immediate cost dominates.
        local_cost = np.zeros(n_remaining)

    # 3. Destination bias
    # We want to ensure we don't get stuck far from the destination.
    # The current program uses a linear ramp. 
    # Let's use a stronger bias late in the game.
    # If n_remaining is small, the "local_cost" (2-NN) is less reliable because the set is small.
    # We should rely more on dist_to_destination.
    
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Weight for destination bias
    # If n_remaining > n_total * 0.2, weight is low.
    # If n_remaining < n_total * 0.2, weight increases.
    frac_rem = n_remaining / n_total
    
    # Let's use a quadratic ramp for destination to make it kick in harder near the end.
    w_dest = (1.0 - frac_rem) ** 2
    
    # Weight for local cost
    # We want local cost to be significant in the middle.
    # Let's scale local_cost. The distances are in the same units.
    # local_cost is sum of 2 distances. dist_from_current is 1 distance.
    # So local_cost is roughly 2x the scale of a single edge.
    # We should divide local_cost by 2 or use a weight.
    
    w_local = 0.5 # Scale factor for the 2-NN sum to make it comparable to single edge distances
    
    scores = dists_from_current + w_local * local_cost + w_dest * dists_to_dest
    
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
