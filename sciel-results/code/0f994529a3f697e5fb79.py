"""Reconstruct phylogenetic topology from Jukes-Cantor simulated sequences."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 255.0  # seconds wall-clock; evaluator kills at 270s


def _jc_distance(p):
    """Jukes-Cantor correction for proportion of differences p."""
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC-corrected distance matrix using numpy."""
    n = len(labels)
    enc = {}
    for lab in labels:
        s = sequences[lab]
        arr = np.zeros(len(s), dtype=np.int8)
        for k, ch in enumerate(s):
            if ch == 'A':
                arr[k] = 0
            elif ch == 'C':
                arr[k] = 1
            elif ch == 'G':
                arr[k] = 2
            elif ch == 'T':
                arr[k] = 3
            else:
                arr[k] = 0
        enc[lab] = arr

    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            ai = enc[labels[i]]
            aj = enc[labels[j]]
            diffs = np.count_nonzero(ai != aj)
            p = diffs / len(ai)
            D[i, j] = D[j, i] = _jc_distance(p)
    return D


def _neighbor_joining(D, labels):
    """
    Perform neighbor joining on a distance matrix D.
    Returns a Newick string.
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    D = D.astype(np.float64).copy()
    names = list(labels)

    while len(names) > 3:
        m = len(names)
        r = D.sum(axis=1)
        Q = (m - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = idx
        if i > j:
            i, j = j, i

        new_name = f"({names[i]},{names[j]})"
        new_dist = 0.5 * (D[i, :] + D[j, :] - D[i, j])

        keep = [k for k in range(m) if k != i and k != j]
        new_m = m - 1
        new_D = np.zeros((new_m, new_m), dtype=np.float64)
        for a, ia in enumerate(keep):
            for b, ib in enumerate(keep):
                new_D[a, b] = D[ia, ib]
        
        new_node_idx = m - 2
        for a, ia in enumerate(keep):
            new_D[a, new_node_idx] = new_dist[ia]
            new_D[new_node_idx, a] = new_dist[ia]
        new_D[new_node_idx, new_node_idx] = 0.0

        new_names = [names[k] for k in keep] + [new_name]
        D = new_D
        names = new_names

    return f"({names[0]},{names[1]},{names[2]})"


def _parse_newick(newick):
    """
    Parse a Newick string into a tree structure.
    Returns a list of edges (u, v) and a list of leaf labels.
    """
    # Simple parser for unrooted trees
    # We will represent the tree as a set of edges
    edges = set()
    leaves = []
    
    def parse(s, pos):
        if pos >= len(s):
            return None, pos
        
        if s[pos] == '(':
            pos += 1
            # Parse children
            children = []
            while s[pos] != ')':
                child, pos = parse(s, pos)
                if child is not None:
                    children.append(child)
                if s[pos] == ',':
                    pos += 1
            pos += 1 # skip ')'
            
            # Check for label after ')'
            label = ""
            while pos < len(s) and s[pos] not in ',;':
                label += s[pos]
                pos += 1
            
            if label:
                # This is an internal node with a label? Or a leaf?
                # In our NJ output, internal nodes have no labels, leaves do.
                # But the parser might see a leaf as a child.
                # If label is present and no children, it's a leaf.
                # If label is present and children, it's an internal node with a label.
                # Our NJ output: (t0,t1) for 2 taxa, (t0,t1,t2) for 3.
                # For >3, it's nested.
                # Leaves are t0, t1, etc.
                # Internal nodes are just structures.
                
                # We need to create a node ID for this internal node
                node_id = f"n{len(leaves) + len(edges)}" # Unique ID
                leaves.append(node_id) # Wait, this is not a leaf
                # Let's use a counter for internal nodes
                return node_id, pos
            else:
                # No label, just a node
                node_id = f"n{len(leaves) + len(edges)}"
                return node_id, pos
        
        else:
            # It's a leaf
            label = ""
            while pos < len(s) and s[pos] not in ',;)':
                label += s[pos]
                pos += 1
            if label:
                leaves.append(label)
                return label, pos
            else:
                return None, pos

    # This parser is too complex for a simple Newick string.
    # Let's use a simpler approach: we know the topology from NJ.
    # We can just return the Newick string and use it for NNI.
    # But NNI requires a tree structure.
    # Let's implement a simple NNI on the Newick string? No.
    
    # Let's use a different approach:
    # We will not do NNI refinement due to complexity of parsing and scoring.
    # Instead, we will use a more accurate distance metric or a different algorithm.
    
    # Given the time constraint, I will stick to NJ but with a better distance matrix.
    # Or, I will implement a simple NNI refinement using the NJ Q-criterion.
    
    return None, 0


def reconstruct(sequences):
    """
    Reconstruct unrooted phylogenetic tree topology.
    Returns Newick string over labels t0..t(n-1).
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 3:
        if n == 1:
            return labels[0]
        elif n == 2:
            return f"({labels[0]},{labels[1]})"
        else:
            return f"({labels[0]},{labels[1]},{labels[2]})"

    # Compute distance matrix
    D = _compute_distance_matrix(sequences, labels)
    
    # Run NJ
    result = _neighbor_joining(D, labels)
    
    # Check time
    if time.time() > deadline - 5:
        return result + ";"

    # If time permits, we could try NNI refinement.
    # But for now, we return the NJ result.
    
    return result + ";"
# EVOLVE-BLOCK-END
