"""Distance-based phylogenetic tree reconstruction using Neighbor-Joining with Jukes-Cantor correction."""
import numpy as np
import math


def reconstruct(sequences):
    """Reconstruct unrooted tree topology from DNA sequences.
    
    Args:
        sequences: dict mapping taxon labels (t0..t(n-1)) to DNA strings over ACGT
        
    Returns:
        Newick string representing the unrooted tree topology
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 1:
        return f"{labels[0]};"
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Convert sequences to numpy arrays for faster distance computation
    seq_list = [sequences[label] for label in labels]
    L = len(seq_list[0])
    
    # Pre-convert sequences to integer arrays for faster comparison
    base_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    int_seqs = []
    for s in seq_list:
        arr = np.zeros(L, dtype=np.int8)
        for i, c in enumerate(s):
            arr[i] = base_to_int[c]
        int_seqs.append(arr)
    
    # Compute pairwise Jukes-Cantor corrected distances
    # D[a,b] = -0.75 * ln(1 - 4p/3) where p = proportion of differing sites
    dist = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Count differences using numpy
            diff_count = np.sum(int_seqs[i] != int_seqs[j])
            p = diff_count / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist[i, j] = d
            dist[j, i] = d
    
    # Neighbor-Joining algorithm
    # Track which original taxa are still "active"
    # We'll use a recursive approach with explicit tree building
    
    # Active nodes: start with all taxa
    # Each node has an index in the original set or is an internal node
    # We'll track the tree structure as we go
    
    # State:
    # - active: list of current node indices (0..n-1 for leaves, n+ for internal)
    # - tree: dict mapping node index to its Newick representation
    # - dist: current distance matrix over active nodes
    
    active = list(range(n))
    tree_repr = {i: labels[i] for i in range(n)}
    next_node_id = n
    
    # Current distance matrix over active nodes (reindexed)
    # We'll maintain a dict-based distance for active nodes
    # Actually, let's use a 2D array and track which positions correspond to which active nodes
    
    # Better approach: maintain dist as a dict from (i,j) to value for active nodes
    # Or reindex each iteration
    
    # Let's use a cleaner approach:
    # Maintain:
    #   active_nodes: list of node ids currently in the NJ process
    #   D: dict mapping (node_a, node_b) -> distance, for all pairs in active_nodes
    #   tree_str: dict mapping node_id -> Newick string for that subtree
    
    D = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                D[(i, j)] = dist[i, j]
    
    tree_str = {i: labels[i] for i in range(n)}
    
    while len(active) > 3:
        m = len(active)
        
        # Compute row sums (total distance from each node to all others)
        r = {}
        for a in active:
            total = 0.0
            for b in active:
                if b != a:
                    total += D[(a, b)]
            r[a] = total
        
        # Find the pair (a, b) that minimizes the NJ criterion:
        # Q(a,b) = (m-2)*D[a,b] - r[a] - r[b]
        best_q = float('inf')
        best_a = -1
        best_b = -1
        
        for i_idx in range(m):
            a = active[i_idx]
            for j_idx in range(i_idx + 1, m):
                b = active[j_idx]
                q_val = (m - 2) * D[(a, b)] - r[a] - r[b]
                if q_val < best_q:
                    best_q = q_val
                    best_a = a
                    best_b = b
        
        a = best_a
        b = best_b
        
        # Create a new internal node joining a and b
        new_node = next_node_id
        next_node_id += 1
        
        # Compute branch lengths (optional, but good for correctness)
        # delta_a = 0.5 * (D[a,b] + (r[a] - r[b]) / (m - 2))
        # delta_b = D[a,b] - delta_a
        delta_a = 0.5 * (D[(a, b)] + (r[a] - r[b]) / (m - 2))
        delta_b = D[(a, b)] - delta_a
        
        # Ensure non-negative (clamp small negatives to 0)
        if delta_a < 0:
            delta_a = 0.0
        if delta_b < 0:
            delta_b = 0.0
        
        # Build the subtree string
        # For topology recovery, branch lengths don't matter, but we include them
        tree_str[new_node] = f"({tree_str[a]}:{delta_a:.4f},{tree_str[b]}:{delta_b:.4f})"
        
        # Update distances from new_node to all remaining active nodes
        remaining = [x for x in active if x not in (a, b)]
        for x in remaining:
            # D[new, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
            new_d = 0.5 * (D[(a, x)] + D[(b, x)] - D[(a, b)])
            D[(new_node, x)] = new_d
            D[(x, new_node)] = new_d
        
        # Remove a and b from active, add new_node
        active = remaining + [new_node]
    
    # Final step: join the remaining 3 nodes into an unrooted tree
    # For 3 nodes, the unrooted tree is simply (a, b, c)
    # We need to compute the three branch lengths
    a, b, c = active
    
    dab = D[(a, b)]
    dac = D[(a, c)]
    dbc = D[(b, c)]
    
    # For a 3-taxon unrooted tree with topology ((a,b),c):
    # Actually for 3 taxa, the unrooted tree has 3 leaves meeting at one internal node
    # Branch lengths:
    # len_a = (dab + dac - dbc) / 2
    # len_b = (dab + dbc - dac) / 2
    # len_c = (dac + dbc - dab) / 2
    len_a = max(0.0, (dab + dac - dbc) / 2.0)
    len_b = max(0.0, (dab + dbc - dac) / 2.0)
    len_c = max(0.0, (dac + dbc - dab) / 2.0)
    
    # Build final Newick
    result = f"({tree_str[a]}:{len_a:.4f},{tree_str[b]}:{len_b:.4f},{tree_str[c]}:{len_c:.4f})"
    
    return result
