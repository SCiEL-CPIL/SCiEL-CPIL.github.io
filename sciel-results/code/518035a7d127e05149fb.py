"""
Phylogenetic topology reconstruction using Neighbor Joining with Jukes-Cantor correction.
Optimized for speed and accuracy on simulated JC data.
"""
import math
import time

# EVOLVE-BLOCK-START

def _jc_distance_matrix(sequences, labels):
    """Compute pairwise JC-corrected distance matrix efficiently."""
    n = len(labels)
    m = len(sequences[labels[0]]) if n > 0 else 0
    
    # Pre-convert sequences to numpy-like arrays for faster comparison
    # Using a simple approach: for each pair, count differences
    dist = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        si = sequences[labels[i]]
        for j in range(i + 1, n):
            sj = sequences[labels[j]]
            diffs = 0
            for k in range(m):
                if si[k] != sj[k]:
                    diffs += 1
            p = diffs / m if m > 0 else 0.0
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist[i][j] = d
            dist[j][i] = d
    
    return dist


def _neighbor_joining(dist, labels):
    """
    Perform Neighbor Joining algorithm.
    Returns Newick string.
    """
    n = len(labels)
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    
    # Current set of active nodes
    active = list(range(n))
    # Current tree representation for each node
    trees = {i: labels[i] for i in range(n)}
    # Current distance matrix (will be updated)
    d = [row[:] for row in dist]
    
    # Counter for new internal nodes
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        
        # Compute row sums
        r = {i: 0.0 for i in active}
        for i in active:
            for j in active:
                if i != j:
                    r[i] += d[i][j]
        
        # Find the pair (i, j) that minimizes the NJ criterion
        best_criterion = float('inf')
        best_i = -1
        best_j = -1
        
        for idx_i in range(m):
            i = active[idx_i]
            for idx_j in range(idx_i + 1, m):
                j = active[idx_j]
                # NJ criterion: (m-2)*d(i,j) - r(i) - r(j)
                criterion = (m - 2) * d[i][j] - r[i] - r[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i = i
                    best_j = j
        
        # Join best_i and best_j
        i, j = best_i, best_j
        
        # Compute branch lengths
        # l(i, new) = 0.5 * d(i,j) + (r(i) - r(j)) / (2 * (m - 2))
        # l(j, new) = 0.5 * d(i,j) - (r(i) - r(j)) / (2 * (m - 2))
        if m - 2 == 0:
            li = 0.0
            lj = 0.0
        else:
            li = 0.5 * d[i][j] + (r[i] - r[j]) / (2.0 * (m - 2))
            lj = 0.5 * d[i][j] - (r[i] - r[j]) / (2.0 * (m - 2))
        
        # Create new node
        new_node = next_id
        next_id += 1
        
        # Build Newick subtree
        tree_i = trees[i]
        tree_j = trees[j]
        # We don't include branch lengths in the Newick for simplicity,
        # but the topology is what matters for RF distance
        new_tree = "(" + tree_i + "," + tree_j + ")"
        trees[new_node] = new_tree
        
        # Update distance matrix for the new node
        for k in active:
            if k != i and k != j:
                # d(new, k) = 0.5 * (d(i,k) + d(j,k) - d(i,j))
                d[new_node][k] = 0.5 * (d[i][k] + d[j][k] - d[i][j])
                d[k][new_node] = d[new_node][k]
        
        # Remove i and j from active, add new_node
        active = [x for x in active if x != i and x != j]
        active.append(new_node)
    
    # Now we have 3 nodes left
    if len(active) == 3:
        a, b, c = active
        # Build the final unrooted tree
        # The three leaves form a star-like structure, but we need to combine them
        # into a single unrooted tree. We can join two of them first, then join the third.
        # Actually, with 3 nodes, the topology is just a star or a comb.
        # Let's join a and b, then join the result with c.
        # But for an unrooted tree with 3 taxa, it's just (a,b,c)
        # We need to return a valid Newick string.
        
        # Combine all three into a single tree
        # The structure is ((a,b),c) or similar. For 3 taxa, any binary tree is fine.
        # Let's create: (a,b,c) -> but Newick requires binary or star.
        # Standard Newick for 3 taxa: (a,b,c); is a star.
        # Or ((a,b),c); is a binary tree.
        # For RF distance, both have the same splits for 3 taxa (no non-trivial splits).
        # So either works. Let's use a binary tree for consistency.
        
        # Join a and b
        ab_tree = "(" + trees[a] + "," + trees[b] + ")"
        # Join ab_tree with c
        final_tree = "(" + ab_tree + "," + trees[c] + ")"
        return final_tree + ";"
    elif len(active) == 2:
        a, b = active
        return "(" + trees[a] + "," + trees[b] + ");"
    else:
        return trees[active[0]] + ";"


def reconstruct(sequences):
    """
    Reconstruct phylogenetic topology from DNA sequences.
    
    Args:
        sequences: dict mapping taxon labels (t0, t1, ...) to DNA strings (ACGT)
    
    Returns:
        Newick string representing the unrooted tree topology
    """
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin before 180s limit
    
    # Extract and sort labels
    labels = sorted(sequences.keys(), key=lambda x: int(x[1:]))
    n = len(labels)
    
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    
    # Compute distance matrix
    dist = _jc_distance_matrix(sequences, labels)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback: return a star tree
        return "(" + ",".join(labels) + ");"
    
    # Perform Neighbor Joining
    result = _neighbor_joining(dist, labels)
    
    return result

# EVOLVE-BLOCK-END
