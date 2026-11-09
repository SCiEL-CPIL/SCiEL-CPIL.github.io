"""Phylogenetic topology recovery using Quartet Consensus with Greedy Split Selection."""
import math
import time
import numpy as np
from itertools import combinations

def _jc_distance_matrix(seqs):
    """Compute JC-corrected distance matrix from list of numpy char arrays."""
    n = len(seqs)
    D = np.zeros((n, n))
    L = len(seqs[0])
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.sum(seqs[i] != seqs[j])
            p = diff / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            D[i, j] = d
            D[j, i] = d
    return D

def _quartet_vote(D, i, j, k, l):
    """
    Vote for a quartet topology based on distances.
    Returns a frozenset representing the split (the smaller side or canonical side).
    """
    # Topology 1: (i,j) vs (k,l) -> Split {i,j}
    s1 = D[i, j] + D[k, l]
    # Topology 2: (i,k) vs (j,l) -> Split {i,k}
    s2 = D[i, k] + D[j, l]
    # Topology 3: (i,l) vs (j,k) -> Split {i,l}
    s3 = D[i, l] + D[j, k]
    
    min_score = min(s1, s2, s3)
    
    if min_score == s1:
        split = frozenset([i, j])
    elif min_score == s2:
        split = frozenset([i, k])
    else:
        split = frozenset([i, l])
        
    # Canonicalize: ensure the smaller side is used, or lexicographically smaller if equal size
    # For compatibility checks, we just need a consistent representation.
    # We will represent splits by a bitmask in the main loop for speed.
    # Here we return the set of indices.
    return split

def _build_tree_from_splits(splits, labels, n):
    """
    Build a Newick string from a set of compatible splits.
    splits: set of frozensets (representing one side of the split).
    labels: list of label strings.
    n: number of taxa.
    """
    if n <= 3:
        if n == 1:
            return labels[0]
        elif n == 2:
            return f"({labels[0]},{labels[1]})"
        else:
            return f"({labels[0]},{labels[1]},{labels[2]})"
            
    # Map indices to labels for output
    # The splits are defined by indices 0..n-1
    
    # We need to recursively build the tree.
    # We need to find a split that divides the current set of leaves.
    # Since the splits are on the global set, we need to filter them.
    
    def build_subtree(leaf_indices):
        """Recursively build Newick for a subset of leaf indices."""
        if len(leaf_indices) == 1:
            return labels[leaf_indices[0]]
        elif len(leaf_indices) == 2:
            return f"({labels[leaf_indices[0]]},{labels[leaf_indices[1]]})"
        elif len(leaf_indices) == 3:
            return f"({labels[leaf_indices[0]]},{labels[leaf_indices[1]]},{labels[leaf_indices[2]]})"
        
        # Find a split that is compatible with the current leaf set
        # A split S (frozenset of indices) is valid for leaf_indices if:
        # 1. S is not a subset of leaf_indices (trivial)
        # 2. S is not disjoint from leaf_indices (trivial)
        # 3. S divides leaf_indices into two non-empty sets.
        # i.e., (S & leaf_indices) is not empty and (S & leaf_indices) is not leaf_indices
        
        valid_splits = []
        leaf_set = frozenset(leaf_indices)
        
        for s in splits:
            inter = s & leaf_set
            if inter and inter != leaf_set:
                # This split divides the current set
                # We prefer splits that are "deep" or "broad"? 
                # Any valid split works for a binary tree.
                # Let's pick one. To make it deterministic, pick the first one found.
                # Or pick the one that splits the set most evenly?
                # For simplicity, pick the first one.
                valid_splits.append(s)
                
        if not valid_splits:
            # No split found? This shouldn't happen if splits are complete.
            # Fallback to star
            return "(" + ",".join(labels[i] for i in leaf_indices) + ")"
        
        # Pick a split
        s = valid_splits[0]
        
        # The two sides of the split within the current leaf set
        side1 = s & leaf_set
        side2 = leaf_set - s
        
        # Convert to lists of indices
        idx1 = list(side1)
        idx2 = list(side2)
        
        # Recursively build subtrees
        newick1 = build_subtree(idx1)
        newick2 = build_subtree(idx2)
        
        return f"({newick1},{newick2})"
        
    all_indices = list(range(n))
    return build_subtree(all_indices)

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 170.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"
        
    # Convert sequences to numpy arrays
    seqs = [np.array(list(sequences[label]), dtype='U1') for label in labels]
    
    # Calculate distance matrix
    D = _jc_distance_matrix(seqs)
    
    # Calculate quartet votes
    # For n=32, number of quartets is ~36k. This is fast.
    split_counts = {}
    
    # Precompute bitmasks for all possible splits? No, just for the ones we see.
    # We will store splits as frozensets of indices.
    
    for i, j, k, l in combinations(range(n), 4):
        split = _quartet_vote(D, i, j, k, l)
        # Canonicalize the split: use the smaller side, or lex smaller if equal
        # For counting, we need a unique key.
        # frozenset([i,j]) is the same as frozenset([j,i]).
        # But we need to distinguish {i,j} from {i,k}.
        # The split is {i,j} vs {k,l}.
        # We can represent the split by the frozenset of the smaller side.
        # If sizes are equal, use the one with smaller min element.
        
        s1 = split
        s2 = frozenset([i, j, k, l]) - s1
        
        if len(s1) < len(s2):
            key = s1
        elif len(s2) < len(s1):
            key = s2
        else:
            # Equal size
            if min(s1) < min(s2):
                key = s1
            else:
                key = s2
                
        if key in split_counts:
            split_counts[key] += 1
        else:
            split_counts[key] = 1
            
    if time.time() > deadline:
        # Fallback to NJ if quartet calculation is too slow (unlikely for n<=32)
        return _nj_fallback(D, labels)
        
    # Sort splits by count (descending)
    sorted_splits = sorted(split_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Greedy selection of compatible splits
    selected_splits = []
    # Use bitmasks for fast compatibility check
    # n <= 32, so we can use Python integers as bitmasks.
    
    # Precompute bitmask for each candidate split
    candidate_splits = []
    for split, count in sorted_splits:
        mask = 0
        for idx in split:
            mask |= (1 << idx)
        candidate_splits.append((mask, split, count))
        
    all_mask = (1 << n) - 1
    
    for mask, split, count in candidate_splits:
        if len(selected_splits) >= n - 3:
            break
            
        # Check compatibility with all selected splits
        compatible = True
        for sel_mask, sel_split, sel_count in selected_splits:
            # Two splits cross if:
            # (mask & sel_mask) != 0
            # (mask & ~sel_mask) != 0
            # (~mask & sel_mask) != 0
            # (~mask & ~sel_mask) != 0
            
            inter = mask & sel_mask
            if inter == 0:
                continue # Disjoint, compatible
            if inter == mask:
                continue # Subset, compatible
            if inter == sel_mask:
                continue # Superset, compatible
                
            # Check if they cross
            # If inter is not 0, not mask, not sel_mask, they might cross.
            # They cross if the other three intersections are also non-empty.
            # 1. mask & sel_mask (inter) - already non-empty
            # 2. mask & ~sel_mask
            # 3. ~mask & sel_mask
            # 4. ~mask & ~sel_mask
            
            m_only = mask & (~sel_mask & all_mask)
            s_only = sel_mask & (~mask & all_mask)
            neither = (~mask & ~sel_mask & all_mask)
            
            if m_only != 0 and s_only != 0 and neither != 0:
                compatible = False
                break
                
        if compatible:
            selected_splits.append((mask, split, count))
            
    if time.time() > deadline:
        return _nj_fallback(D, labels)
        
    # Build tree from selected splits
    if selected_splits:
        # Convert selected splits back to frozensets
        splits_set = set([s[1] for s in selected_splits])
        newick = _build_tree_from_splits(splits_set, labels, n)
        return newick + ";"
    else:
        # Fallback to NJ
        return _nj_fallback(D, labels)

def _nj_fallback(D, labels):
    """Fallback to Neighbor Joining if quartet method fails or times out."""
    n = len(labels)
    if n <= 3:
        if n == 1:
            return labels[0] + ";"
        elif n == 2:
            return f"({labels[0]},{labels[1]});"
        else:
            return f"({labels[0]},{labels[1]},{labels[2]});"
            
    # Simple NJ implementation
    D = D.copy()
    nodes = list(range(n))
    tree = {}
    
    while len(nodes) > 3:
        k = len(nodes)
        row_sums = D.sum(axis=1)
        Q = (k - 2) * D - row_sums[:, None] - row_sums[None, :]
        
        # Find min Q
        min_val = np.inf
        min_i, min_j = 0, 0
        for i in range(k):
            for j in range(i + 1, k):
                if Q[i, j] < min_val:
                    min_val = Q[i, j]
                    min_i, min_j = i, j
                    
        a_idx, b_idx = min_i, min_j
        a_node, b_node = nodes[a_idx], nodes[b_idx]
        
        new_node = -1 - len(tree) # Unique ID
        tree[new_node] = (a_node, b_node)
        
        # Update distances
        keep_idxs = [k for k in range(k) if k not in (a_idx, b_idx)]
        new_k = k - 1
        new_D = np.zeros((new_k, new_k))
        
        # Copy keep D
        for i, ki in enumerate(keep_idxs):
            for j, kj in enumerate(keep_idxs):
                new_D[i, j] = D[ki, kj]
                
        # Calculate distances to new node
        for i, ki in enumerate(keep_idxs):
            dist = 0.5 * (D[ki, a_idx] + D[ki, b_idx] - D[a_idx, b_idx])
            new_D[i, new_k - 1] = dist
            new_D[new_k - 1, i] = dist
        new_D[new_k - 1, new_k - 1] = 0
        
        D = new_D
        nodes = [nodes[k] for k in keep_idxs] + [new_node]
        
    # Final 3 nodes
    if len(nodes) == 3:
        a, b, c = nodes
        root = -1 - len(tree)
        temp = -2 - len(tree)
        tree[root] = (a, temp)
        tree[temp] = (b, c)
        
        # Convert to Newick
        def to_newick(node):
            if node in tree:
                c1, c2 = tree[node]
                return f"({to_newick(c1)},{to_newick(c2)})"
            else:
                return labels[node]
                
        return to_newick(root) + ";"
        
    return "(" + ",".join(labels) + ");"
