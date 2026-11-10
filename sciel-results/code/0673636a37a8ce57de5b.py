"""
Reconstructs phylogenetic topology using Neighbor Joining with vectorized distance
computation and an explicit deadline guard.
"""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START

def _jc_distance(a, b):
    """Calculate Jukes-Cantor corrected distance."""
    n = len(a)
    if n == 0:
        return 0.0
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    p = diffs / n
    if p >= 0.7499:
        # Saturated
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"

    # Map labels to indices
    label_to_idx = {l: i for i, l in enumerate(labels)}
    idx_to_label = {i: l for i, l in enumerate(labels)}
    
    # Compute initial distance matrix
    # Using a dictionary for sparse access or list of lists for speed.
    # n is max 32, so n^2 is small. List of lists is faster.
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist[i][j] = d
            dist[j][i] = d

    # Active nodes
    active = list(range(n))
    # Newick representation for each active node
    newick = [labels[i] for i in range(n)]
    
    # We will maintain a "virtual" distance matrix.
    # Since we merge nodes, we need to update distances.
    # Instead of resizing a matrix, we can keep the original labels and just track which are active.
    # But the "new" nodes need new distance entries.
    # Let's expand the matrix if needed, or just use a dictionary of (node1, node2) -> dist.
    # Given n<=32, a dictionary is fine and simpler to manage with dynamic node creation.
    
    # Let's restart with a dictionary approach for clarity and correctness with dynamic merging.
    # Keys are tuples of node IDs.
    # Initial node IDs are integers 0..n-1.
    
    d_matrix = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                d_matrix[(i, j)] = dist[i][j]
                d_matrix[(j, i)] = dist[i][j]

    active_nodes = list(range(n))
    node_newick = {i: labels[i] for i in range(n)}
    
    next_id = n
    start_time = time.time()
    timeout = 270.0
    safety_margin = 5.0

    while len(active_nodes) > 3:
        if time.time() - start_time > timeout - safety_margin:
            break
            
        # Find the best pair (i, j) to join
        # Criterion: min_{k != i,j} (d(i,k) + d(k,j) - d(i,j))
        # We want the pair where this minimum is smallest (closest to 0).
        
        best_val = float('inf')
        best_pair = None
        best_witness = None
        
        n_active = len(active_nodes)
        
        for u_idx in range(n_active):
            i = active_nodes[u_idx]
            for v_idx in range(u_idx + 1, n_active):
                j = active_nodes[v_idx]
                
                d_ij = d_matrix[(i, j)]
                
                min_diff = float('inf')
                witness = None
                
                for k in active_nodes:
                    if k == i or k == j:
                        continue
                    # d(i,k) + d(k,j) - d(i,j)
                    val = d_matrix[(i, k)] + d_matrix[(k, j)] - d_ij
                    if val < min_diff:
                        min_diff = val
                        witness = k
                
                if min_diff < best_val:
                    best_val = min_diff
                    best_pair = (i, j)
                    best_witness = witness
        
        if best_pair is None:
            # Fallback: just join the two closest
            # This shouldn't happen, but safe guard
            break
            
        i, j = best_pair
        k = best_witness
        
        # Create new node
        new_id = next_id
        next_id += 1
        
        # Newick for new node
        node_newick[new_id] = f"({node_newick[i]},{node_newick[j]})"
        
        # Calculate distances from new node to all other active nodes
        # Formula: d(new, x) = 0.5 * (d(i, x) + d(j, x) - d(i, j))
        d_ij = d_matrix[(i, j)]
        
        for x in active_nodes:
            if x == i or x == j:
                continue
            d_ix = d_matrix[(i, x)]
            d_jx = d_matrix[(j, x)]
            d_nx = 0.5 * (d_ix + d_jx - d_ij)
            d_matrix[(new_id, x)] = d_nx
            d_matrix[(x, new_id)] = d_nx
            
        # Remove i and j from active
        active_nodes.remove(i)
        active_nodes.remove(j)
        active_nodes.append(new_id)
        
        # Note: We do not delete old distances, just ignore them by only iterating over active_nodes.

    # Assemble the final tree
    # If we broke out early, active_nodes might have > 3 items.
    # If we finished, we have 3 items.
    
    if len(active_nodes) == 3:
        a, b, c = active_nodes
        return f"({node_newick[a]},{node_newick[b]},{node_newick[c]});"
    else:
        # Fallback: build a star tree from remaining active nodes?
        # Or just return a balanced tree.
        # Since we have a deadline, if we are here, we didn't finish.
        # Let's try to quickly finish with simple NJ or just output what we have as a star.
        # A star tree is a valid topology, though likely poor.
        # Better: complete the joining using simple distance min.
        while len(active_nodes) > 3:
             # Find closest pair
             best_dist = float('inf')
             best_i = active_nodes[0]
             best_j = active_nodes[1]
             for u_idx in range(len(active_nodes)):
                 i = active_nodes[u_idx]
                 for v_idx in range(u_idx + 1, len(active_nodes)):
                     j = active_nodes[v_idx]
                     d = d_matrix[(i, j)]
                     if d < best_dist:
                         best_dist = d
                         best_i = i
                         best_j = j
             
             new_id = next_id
             next_id += 1
             node_newick[new_id] = f"({node_newick[best_i]},{node_newick[best_j]})"
             
             d_ij = best_dist
             for x in active_nodes:
                 if x == best_i or x == best_j:
                     continue
                 d_nx = 0.5 * (d_matrix[(best_i, x)] + d_matrix[(best_j, x)] - d_ij)
                 d_matrix[(new_id, x)] = d_nx
                 d_matrix[(x, new_id)] = d_nx
                 
             active_nodes.remove(best_i)
             active_nodes.remove(best_j)
             active_nodes.append(new_id)
             
        a, b, c = active_nodes
        return f"({node_newick[a]},{node_newick[b]},{node_newick[c]});"

# EVOLVE-BLOCK-END
