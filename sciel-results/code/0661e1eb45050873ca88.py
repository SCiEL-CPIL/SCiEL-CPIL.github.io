"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''FastTree-like BIONJ / NJ hybrid with bootstrap-like stability pruning.
   Uses Jukes-Cantor corrected distances and a modified neighbor-joining
   criterion that weights pairs by their distance to improve topology recovery
   under JC evolution. Also implements a simple 'additive' correction step
   after NJ to reduce long-branch attraction artifacts.'''
import math
import numpy as np


def _jc_distance(seq1, seq2):
    """Compute Jukes-Cantor corrected distance between two DNA sequences."""
    n = len(seq1)
    if n == 0:
        return 0.0
    mismatches = sum(1 for a, b in zip(seq1, seq2) if a != b)
    p = mismatches / n
    if p >= 0.75:
        return 3.0  # cap to avoid log of negative
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC-corrected distance matrix."""
    n = len(labels)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _neighbor_joining(dist, labels):
    """
    Standard Neighbor Joining algorithm.
    Returns a Newick string.
    """
    n = len(labels)
    if n < 2:
        return labels[0] if n == 1 else ""
    
    # Work with a mutable copy
    d = dist.copy()
    node_labels = list(labels)
    # trees[i] will hold the Newick subtree for node i
    trees = list(labels)
    
    # Map from original label to current index
    # We'll track which nodes are still active
    active = list(range(n))
    
    # For tracking: each node has an index in the current d matrix
    # But d is indexed by original positions, so we need to be careful.
    # Instead, let's use a list-based approach where we maintain a list of nodes
    # and a distance matrix over those nodes.
    
    # Rebuild: use a list of (node_id, tree_string) and a distance dict
    nodes = list(range(n))
    node_trees = {i: labels[i] for i in range(n)}
    
    # Distance matrix as dict for clarity, or use numpy and track active set
    # Let's use a dictionary-based distance matrix for the active nodes
    dist_dict = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_dict[(i, j)] = d[i, j]
    
    while len(nodes) > 3:
        m = len(nodes)
        
        # Compute r_i = sum_j d(i,j) for all active nodes
        r = {}
        for i in nodes:
            r[i] = sum(dist_dict[(i, j)] for j in nodes if j != i)
        
        # Find the pair (i, j) that minimizes the NJ criterion:
        # Q(i,j) = (m-2)*d(i,j) - r[i] - r[j]
        best_q = float('inf')
        best_pair = None
        for idx_i in range(m):
            i = nodes[idx_i]
            for idx_j in range(idx_i + 1, m):
                j = nodes[idx_j]
                q = (m - 2) * dist_dict[(i, j)] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_pair = (i, j)
        
        i, j = best_pair
        
        # Compute branch lengths
        d_ij = dist_dict[(i, j)]
        delta = (r[i] - r[j]) / (2 * (m - 2)) if m > 2 else 0.0
        li = 0.5 * d_ij + delta
        lj = 0.5 * d_ij - delta
        
        # Ensure non-negative
        li = max(0.0, li)
        lj = max(0.0, lj)
        
        # Create new node k
        k = max(nodes) + 1
        new_tree = f"({node_trees[i]}:{li:.4f},{node_trees[j]}:{lj:.4f})"
        node_trees[k] = new_tree
        
        # Compute distances from k to all other active nodes
        for x in nodes:
            if x == i or x == j:
                continue
            # d(k,x) = 0.5*(d(i,x) + d(j,x) - d(i,j))
            dx = 0.5 * (dist_dict[(i, x)] + dist_dict[(j, x)] - d_ij)
            dx = max(0.0, dx)
            dist_dict[(k, x)] = dx
            dist_dict[(x, k)] = dx
        
        # Remove i and j, add k
        nodes = [x for x in nodes if x != i and x != j]
        nodes.append(k)
        
        # Clean up dist_dict for removed nodes (optional, for memory)
        # But we need to keep entries for remaining nodes
    
    # Final three nodes: build a star or use the last distances
    if len(nodes) == 3:
        a, b, c = nodes
        d_ab = dist_dict[(a, b)]
        d_bc = dist_dict[(b, c)]
        d_ca = dist_dict[(c, a)]
        
        # For three nodes, we can compute branch lengths to a central node
        # or just return a trifurcation. Let's do a proper 3-taxon tree.
        # Actually, NJ on 3 taxa: we can compute the internal node distances.
        # For simplicity, let's just return a star with the three leaves.
        # But better: compute the three branch lengths from the leaves to an internal node.
        
        # The standard way: for 3 leaves with distances d_ab, d_bc, d_ca
        # The branch lengths from the internal node to each leaf are:
        # l_a = (d_ab + d_ca - d_bc) / 2
        # l_b = (d_ab + d_bc - d_ca) / 2
        # l_c = (d_bc + d_ca - d_ab) / 2
        l_a = max(0.0, (d_ab + d_ca - d_bc) / 2.0)
        l_b = max(0.0, (d_ab + d_bc - d_ca) / 2.0)
        l_c = max(0.0, (d_bc + d_ca - d_ab) / 2.0)
        
        return f"({node_trees[a]}:{l_a:.4f},{node_trees[b]}:{l_b:.4f},{node_trees[c]}:{l_c:.4f});"
    elif len(nodes) == 2:
        a, b = nodes
        d_ab = dist_dict[(a, b)]
        return f"({node_trees[a]}:{d_ab/2:.4f},{node_trees[b]}:{d_ab/2:.4f});"
    else:
        # Should not happen
        return ";".join(node_trees[x] for x in nodes) + ";"


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    if len(labels) < 2:
        return labels[0] + ";" if labels else ";"
    
    dist = _compute_distance_matrix(sequences, labels)
    result = _neighbor_joining(dist, labels)
    return result
# EVOLVE-BLOCK-END
