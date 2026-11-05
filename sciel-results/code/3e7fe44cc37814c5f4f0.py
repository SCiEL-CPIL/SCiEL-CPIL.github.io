"""Distance-based phylogenetic reconstruction using Jukes-Cantor corrected distances
and Neighbor-Joining algorithm."""
import math
import numpy as np


def _jc_distance(a, b):
    """Compute Jukes-Cantor corrected distance between two DNA sequences."""
    n = len(a)
    if n == 0:
        return 0.0
    # Count mismatches
    mismatches = 0
    for x, y in zip(a, b):
        if x != y:
            mismatches += 1
    p = mismatches / n
    if p >= 0.75:
        return 10.0  # Large distance cap
    if p <= 0.0:
        return 0.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def reconstruct(sequences):
    """Reconstruct unrooted tree topology using Neighbor-Joining.
    
    Args:
        sequences: dict mapping taxon labels (e.g., 't0', 't1', ...) to DNA strings
        
    Returns:
        Newick string representing the unrooted tree
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    if n_taxa <= 1:
        return labels[0] + ";" if labels else ""
    if n_taxa == 2:
        return "(" + ",".join(labels) + ");"
    if n_taxa == 3:
        return "(" + ",".join(labels) + ");"
    
    # Build distance matrix using NumPy for efficiency
    seq_list = [sequences[label] for label in labels]
    n = n_taxa
    dist = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(seq_list[i], seq_list[j])
            dist[i, j] = d
            dist[j, i] = d
    
    # Neighbor-Joining algorithm
    # active: list of currently active node indices
    active = list(range(n))
    # tree_repr: for each active node, store the Newick subtree string
    tree_repr = {i: labels[i] for i in range(n)}
    
    # We'll maintain a dynamic distance matrix. Start with the full matrix.
    # As we join nodes, we create new nodes and update distances.
    
    while len(active) > 3:
        m = len(active)
        # Compute row sums (Q-matrix criterion)
        r = np.zeros(m, dtype=np.float64)
        for i in range(m):
            for j in range(m):
                if i != j:
                    r[i] += dist[active[i], active[j]]
        
        # Find pair (a, b) minimizing Q(a,b) = (m-2)*D(a,b) - r[a] - r[b]
        best_q = float('inf')
        best_a, best_b = 0, 1
        for i in range(m):
            for j in range(i + 1, m):
                q = (m - 2) * dist[active[i], active[j]] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_a, best_b = i, j
        
        ai = active[best_a]
        bj = active[best_b]
        
        # Compute branch lengths (optional, but we include them for completeness)
        # d_a = 0.5 * D(ai, bj) + r[ai]/(2*(m-2)) - r[bj]/(2*(m-2))
        # d_b = D(ai, bj) - d_a
        # But since we only need topology, we can skip explicit branch lengths
        # and just create a new internal node.
        
        # Create new node
        new_idx = n + (n - len(active))  # Unique index for new internal node
        new_label = f"({tree_repr[ai]},{tree_repr[bj]})"
        tree_repr[new_idx] = new_label
        
        # Update distance matrix: compute distances from new node to all other active nodes
        # D(new, x) = 0.5 * (D(ai, x) + D(bj, x) - D(ai, bj))
        for k in range(m):
            if k != best_a and k != best_b:
                x = active[k]
                dist[new_idx, x] = 0.5 * (dist[ai, x] + dist[bj, x] - dist[ai, bj])
                dist[x, new_idx] = dist[new_idx, x]
        
        # Remove ai and bj from active, add new_idx
        active = [x for x in active if x != ai and x != bj]
        active.append(new_idx)
    
    # Now we have 3 active nodes. Build the final unrooted tree.
    # For 3 nodes, the unrooted tree is a star with 3 leaves.
    # We need to construct a proper Newick string.
    
    if len(active) == 3:
        a, b, c = active
        # The unrooted tree on 3 taxa is ((a,b),c) in any rooting, but as unrooted:
        # We represent it as (a,(b,c)) or similar. Actually, for 3 taxa, any binary
        # unrooted tree is the same topology (a star). Let's just output:
        return "(" + tree_repr[a] + "," + tree_repr[b] + "," + tree_repr[c] + ");"
    
    # This shouldn't happen, but just in case
    return "(" + ",".join(tree_repr[x] for x in active) + ");"
