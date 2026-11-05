import math
import time
import numpy as np

def _jc_distance_matrix(sequences, labels):
    """Compute JC distance matrix using vectorized operations."""
    n = len(labels)
    # Encode sequences as 0-3 integers
    base_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([[base_to_int[c] for c in sequences[labels[i]]] for i in range(n)])
    
    # Compute pairwise differences
    # seqs has shape (n, L)
    # For each pair (i,j), count positions where seqs[i] != seqs[j]
    # Use broadcasting: (n, 1, L) vs (1, n, L)
    diff = (seqs[:, np.newaxis, :] != seqs[np.newaxis, :, :])
    # Count mismatches
    mismatches = diff.sum(axis=2)  # shape (n, n)
    
    # Compute JC distances
    L = seqs.shape[1]
    p = mismatches.astype(float) / L
    # Handle p >= 0.75 case
    p_clipped = np.minimum(p, 0.7499)
    dist = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    # Set diagonal to 0
    np.fill_diagonal(dist, 0.0)
    
    return dist

def _neighbor_joining(dist, labels):
    """
    Perform Neighbor Joining on distance matrix.
    Returns a Newick string.
    """
    n = len(labels)
    if n <= 3:
        if n == 1:
            return labels[0] + ";"
        elif n == 2:
            return f"({labels[0]},{labels[1]});"
        else:
            return f"({labels[0]},{labels[1]},{labels[2]});"
    
    # Work with indexed distance matrix
    # labels[0] -> index 0, labels[1] -> index 1, etc.
    D = dist.copy()  # (n, n)
    
    # Track which nodes are still active
    active = list(range(n))
    # Track the tree structure: each node has a list of (neighbor, length)
    # But for NJ, we just need to track the merging
    trees = {i: labels[i] for i in range(n)}
    
    node_count = n
    while len(active) > 3:
        m = len(active)
        
        # Compute row sums r
        r = np.zeros(m)
        for i, ai in enumerate(active):
            for j, aj in enumerate(active):
                if i != j:
                    r[i] += D[ai, aj]
        
        # Find pair (a, b) that minimizes the NJ criterion
        best_crit = float('inf')
        best_i, best_j = 0, 1
        for i in range(m):
            for j in range(i + 1, m):
                ai, aj = active[i], active[j]
                crit = (m - 2) * D[ai, aj] - r[i] - r[j]
                if crit < best_crit:
                    best_crit = crit
                    best_i, best_j = i, j
        
        ai, aj = active[best_i], active[best_j]
        
        # Create new node
        new_id = node_count
        node_count += 1
        
        # Compute branch lengths
        d_ab = D[ai, aj]
        # For each other node k, compute distance from new node to k
        new_dist = np.zeros(node_count)
        for k_idx, ak in enumerate(active):
            if ak in (ai, aj):
                continue
            new_dist[ak] = 0.5 * (D[ai, ak] + D[aj, ak] - d_ab)
        
        # Update distance matrix
        D[new_id, :] = new_dist
        D[:, new_id] = new_dist
        
        # Update tree
        trees[new_id] = f"({trees[ai]},{trees[aj]})"
        
        # Remove ai and aj from active, add new_id
        active = [x for x in active if x not in (ai, aj)]
        active.append(new_id)
    
    # Build final Newick
    if len(active) == 3:
        return f"({trees[active[0]]},{trees[active[1]]},{trees[active[2]]});"
    elif len(active) == 2:
        return f"({trees[active[0]]},{trees[active[1]]});"
    else:
        return trees[active[0]] + ";"

def _parse_newick_to_splits(newick, n_leaves):
    """
    Parse Newick string and extract all non-trivial splits.
    Returns a set of frozensets representing splits.
    """
    # Remove trailing semicolon
    newick = newick.strip()
    if newick.endswith(';'):
        newick = newick[:-1]
    
    # Find all leaf labels
    import re
    leaves = re.findall(r't\d+', newick)
    
    # Build a tree structure
    # We'll parse the Newick into a nested structure
    def parse(s):
        s = s.strip()
        if s.startswith('('):
            # Find matching closing paren
            depth = 0
            end_idx = -1
            for i, char in enumerate(s):
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                    if depth == 0:
                        end_idx = i
                        break
            
            inner = s[1:end_idx]
            # Split by commas at depth 0
            children = []
            current = ""
            depth = 0
            for char in inner:
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                if char == ',' and depth == 0:
                    children.append(current)
                    current = ""
                else:
                    current += char
            if current:
                children.append(current)
            
            parsed_children = [parse(c) for c in children]
            return parsed_children
        else:
            return s
    
    tree = parse(newick)
    
    # Extract splits
    splits = set()
    
    def extract_splits(node, parent_set):
        if isinstance(node, str):
            # Leaf
            return {node}
        else:
            # Internal node
            children_sets = []
            for child in node:
                child_set = extract_splits(child, parent_set)
                children_sets.append(child_set)
            
            # The split at this node
            if len(children_sets) >= 2:
                # Combine all children
                all_leaves = set()
                for cs in children_sets:
                    all_leaves.update(cs)
                
                # Each child defines a split
                for cs in children_sets:
                    if 0 < len(cs) < len(all_leaves):
                        # This is a non-trivial split
                        splits.add(frozenset(cs))
            
            return set().union(*children_sets)
    
    extract_splits(tree, None)
    
    return splits

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 160  # Safety margin before 180s limit
    
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    n = len(labels)
    
    if n <= 3:
        if n == 1:
            return labels[0] + ";"
        elif n == 2:
            return f"({labels[0]},{labels[1]});"
        else:
            return f"({labels[0]},{labels[1]},{labels[2]});"
    
    # Step 1: Compute distance matrix
    dist = _jc_distance_matrix(sequences, labels)
    
    # Step 2: Neighbor Joining
    newick = _neighbor_joining(dist, labels)
    
    # Check time
    if time.time() > deadline:
        return newick
    
    # Step 3: NNI Refinement
    # We'll do a simple NNI refinement using the split-based approach
    # For now, just return NJ result to ensure we don't timeout
    
    return newick
