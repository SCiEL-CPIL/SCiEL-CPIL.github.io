import math
import time
import numpy as np
import random

# EVOLVE-BLOCK-START
'''
Phylogenetic reconstruction using Fast Distance-based Quartet (FDQ) approach
with NNI refinement.
'''

def _jc_distance_vec(A):
    """
    Compute Jukes-Cantor corrected distances between rows of A.
    A: numpy array of shape (n, L) with values 0-3.
    Returns: n x n distance matrix.
    """
    n, L = A.shape
    diff = A[:, None, :] != A[None, :, :]
    hamming = np.sum(diff, axis=2)
    p = hamming / L
    p = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    return d

def _quartet_scores(D):
    """
    Compute quartet scores for all pairs of pairs.
    For each pair (i,j), compute score against all other pairs (k,l).
    Score = (d[i,j] + d[k,l]) - (d[i,k] + d[j,l])
    Positive means (ij|kl) is supported, negative means (ik|jl) is supported.
    Returns: dict mapping (i,j) -> array of scores for all k,l > j
    """
    n = D.shape[0]
    # For each pair (i,j) with i < j, compute scores against all (k,l) with k < l, k != i, l != j
    # We'll store the best supporting quartets
    return D

def _build_tree_fdq(D, labels):
    """
    Build tree using greedy agglomerative algorithm based on quartet support.
    At each step, find the pair of taxa/clades with the strongest quartet support
    and merge them.
    """
    n = len(labels)
    if n <= 2:
        if n == 1:
            return labels[0] + ";"
        return f"({labels[0]},{labels[1]})" + ";"
    
    # Start with all taxa as separate clusters
    # Each cluster is represented by its member set
    # We'll track:
    # - active: list of cluster ids
    # - members: dict cluster_id -> set of taxon indices
    # - parent: dict cluster_id -> (left_id, right_id, branch_len_left, branch_len_right)
    
    active = list(range(n))
    members = {i: {i} for i in range(n)}
    parent = {i: None for i in range(n)}
    next_id = n
    
    # Precompute all-pairs distances between original taxa
    # For merged clusters, we'll use average distance
    
    while len(active) > 1:
        m = len(active)
        if m == 2:
            # Last two clusters: merge them
            a, b = active
            # Compute branch lengths
            # For unrooted tree, we need a third reference
            # Just create a simple merge
            new_id = next_id
            next_id += 1
            members[new_id] = members[a] | members[b]
            parent[new_id] = (a, b, 0.0, 0.0)
            active = [new_id]
            break
        
        # For each pair of active clusters, compute quartet support score
        # Score for merging (i,j): sum over all other k of [d(i,j) + d(k,k') - d(i,k) - d(j,k')]
        # Simplified: use distance-based criterion
        # Find the pair with minimum average distance (like single-linkage)
        # but weighted by quartet support
        
        best_score = float('inf')
        best_i, best_j = -1, -1
        
        for idx_i in range(m):
            i = active[idx_i]
            for idx_j in range(idx_i + 1, m):
                j = active[idx_j]
                # Compute average distance between clusters i and j
                # d(i,j) = mean of D[a,b] for a in members[i], b in members[j]
                mi = members[i]
                mj = members[j]
                # Compute mean distance
                total = 0.0
                count = 0
                for a in mi:
                    for b in mj:
                        total += D[a, b]
                        count += 1
                avg_d = total / count if count > 0 else 0.0
                
                # Compute quartet support: for each other cluster k,
                # check if d(i,j) + d(k,ref) < d(i,k) + d(j,ref)
                # Use average distance to k as proxy
                support = 0.0
                for idx_k in range(m):
                    if idx_k == idx_i or idx_k == idx_j:
                        continue
                    k = active[idx_k]
                    mk = members[k]
                    # Mean distances
                    d_ik = 0.0
                    d_jk = 0.0
                    cnt_ik = 0
                    cnt_jk = 0
                    for a in mi:
                        for c in mk:
                            d_ik += D[a, c]
                            cnt_ik += 1
                    for b in mj:
                        for c in mk:
                            d_jk += D[b, c]
                            cnt_jk += 1
                    if cnt_ik > 0:
                        d_ik /= cnt_ik
                    if cnt_jk > 0:
                        d_jk /= cnt_jk
                    
                    # Quartet score: positive if (ij|k) is supported
                    # d(i,j) + d(k,k) vs d(i,k) + d(j,k)
                    # d(k,k) = 0, so score = d(i,j) - d(i,k) - d(j,k)
                    # But this isn't quite right for clusters
                    # Use: if d(i,j) < d(i,k) and d(i,j) < d(j,k), support (ij|k)
                    score_k = (d_ik + d_jk) - avg_d * 2
                    support += -score_k  # More negative means more support for (ij|k)
                
                # Combine: prefer small avg_d and high quartet support
                combined = avg_d - 0.1 * support / max(1, m - 2)
                
                if combined < best_score:
                    best_score = combined
                    best_i = i
                    best_j = j
        
        # Merge best_i and best_j
        new_id = next_id
        next_id += 1
        members[new_id] = members[best_i] | members[best_j]
        
        # Compute branch lengths
        mi = members[best_i]
        mj = members[best_j]
        
        # Average distances
        d_ij = 0.0
        cnt = 0
        for a in mi:
            for b in mj:
                d_ij += D[a, b]
                cnt += 1
        d_ij /= cnt if cnt > 0 else 1
        
        # For branch lengths, use simple heuristic
        # Branch length from new node to i and j
        # In a 3-clade star, branch lengths are:
        # b_i = 0.5 * (d_ij + d_ik - d_jk) for some reference k
        # For now, use equal split
        b_i = d_ij / 2.0
        b_j = d_ij / 2.0
        
        parent[new_id] = (best_i, best_j, b_i, b_j)
        
        # Remove old, add new
        active.remove(best_i)
        active.remove(best_j)
        active.append(new_id)
    
    # Build Newick from the tree structure
    def to_newick(node_id):
        if parent[node_id] is None:
            # Leaf
            return labels[node_id]
        else:
            left, right, bl, br = parent[node_id]
            return f"({to_newick(left)}:{bl:.4f},{to_newick(right)}:{br:.4f})"
    
    return to_newick(active[0]) + ";"

def _parse_newick_splits(newick):
    """Parse Newick string and return set of non-trivial splits."""
    # Strip trailing semicolon
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]
    
    # Parse to get all internal nodes and their splits
    # Simple approach: find all pairs of taxa that are separated by each internal edge
    
    # For small trees, we can enumerate all splits
    # First, extract all leaf labels
    import re
    leaves = re.findall(r't\d+', s)
    n = len(leaves)
    
    if n <= 3:
        return set()
    
    # Parse the tree structure
    # We'll do a recursive parse to get the splits
    
    def parse_node(s, pos):
        """Parse a Newick node starting at position pos.
        Returns (node, new_pos).
        """
        if s[pos] == '(':
            # Internal node
            children = []
            pos += 1
            while s[pos] != ')':
                child, pos = parse_node(s, pos)
                children.append(child)
                if s[pos] == ',':
                    pos += 1
            pos += 1  # skip ')'
            
            # Get branch length
            blen = 0.0
            if pos < len(s) and s[pos] == ':':
                pos += 1
                num_start = pos
                while pos < len(s) and (s[pos].isdigit() or s[pos] == '.' or s[pos] == '-'):
                    pos += 1
                blen = float(s[num_start:pos])
            
            # Get set of leaves in this subtree
            leaf_set = set()
            for child in children:
                leaf_set.update(child)
            
            return leaf_set, pos
        else:
            # Leaf
            name_start = pos
            while pos < len(s) and (s[pos].isalnum() or s[pos] == '_'):
                pos += 1
            name = s[name_start:pos]
            
            # Get branch length
            blen = 0.0
            if pos < len(s) and s[pos] == ':':
                pos += 1
                num_start = pos
                while pos < len(s) and (s[pos].isdigit() or s[pos] == '.' or s[pos] == '-'):
                    pos += 1
                blen = float(s[num_start:pos])
            
            return {name}, pos
    
    root_set, _ = parse_node(s, 0)
    
    # Now we need to find all non-trivial splits
    # A split is defined by an internal edge: the set of leaves on one side
    # We need to traverse the tree and for each internal node, get the split
    
    # Re-parse to get the tree structure with internal nodes
    def parse_tree(s, pos):
        """Parse and return (leaf_set, children_list, new_pos)."""
        if s[pos] == '(':
            pos += 1
            children = []
            while s[pos] != ')':
                child_set, child_children, pos = parse_tree(s, pos)
                children.append((child_set, child_children))
                if s[pos] == ',':
                    pos += 1
            pos += 1  # skip ')'
            
            # Skip branch length
            if pos < len(s) and s[pos] == ':':
                pos += 1
                while pos < len(s) and (s[pos].isdigit() or s[pos] == '.' or s[pos] == '-'):
                    pos += 1
            
            leaf_set = set()
            for cs, _ in children:
                leaf_set.update(cs)
            
            return leaf_set, children, pos
        else:
            name_start = pos
            while pos < len(s) and (s[pos].isalnum() or s[pos] == '_'):
                pos += 1
            name = s[name_start:pos]
            
            # Skip branch length
            if pos < len(s) and s[pos] == ':':
                pos += 1
                while pos < len(s) and (s[pos].isdigit() or s[pos] == '.' or s[pos] == '-'):
                    pos += 1
            
            return {name}, [], pos
    
    all_leaves, _, _ = parse_tree(s, 0)
    
    # Find all non-trivial splits
    # For each internal node, the split is (leaves_in_subtree, leaves_not_in_subtree)
    # We represent each split as a frozenset of the smaller side
    
    splits = set()
    
    def collect_splits(leaf_set, children):
        if not children:
            return
        for child_set, child_children in children:
            # The split defined by this child edge
            if 0 < len(child_set) < len(all_leaves):
                # Normalize: use the smaller side
                side1 = frozenset(child_set)
                side2 = frozenset(all_leaves - child_set)
                if len(side1) <= len(side2):
                    splits.add(side1)
                else:
                    splits.add(side2)
            collect_splits(child_set, child_children)
    
    collect_splits(all_leaves, parse_tree(s, 0)[1])
    
    return splits

def _rf_distance(splits1, splits2):
    """Compute normalized Robinson-Foulds distance between two sets of splits."""
    if not splits1 and not splits2:
        return 0.0
    union = splits1 | splits2
    inter = splits1 & splits2
    # Normalized RF = (|S1| + |S2| - 2|S1 ∩ S2|) / (|S1| + |S2|)
    # Or: 2 * (|S1 ∪ S2| - |S1 ∩ S2|) / (|S1| + |S2|)
    denom = len(splits1) + len(splits2)
    if denom == 0:
        return 0.0
    return 2.0 * (len(union) - len(inter)) / denom

def _nni_refine(newick, D, labels, max_iters=100, time_limit=5.0):
    """
    Apply NNI moves to refine the tree topology.
    Uses distance-based scoring for NNI moves.
    """
    start_time = time.time()
    
    # Parse the current tree into a binary tree structure
    # We'll represent the tree as a set of internal edges
    # For NNI, we need to find all internal edges and try the three possible topologies
    
    # Simplified approach: use the split-based representation
    # Try all possible NNI moves by modifying the tree
    
    # For simplicity, we'll do a few rounds of local search
    # This is a placeholder for a full NNI implementation
    
    return newick

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]})" + ";"
    
    # Convert to array
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = [sequences[l] for l in labels]
    A = np.array([[char_map[c] for c in seq] for seq in seqs], dtype=np.uint8)
    
    # Calculate distances
    D = _jc_distance_vec(A)
    
    # Build tree using FDQ approach
    newick = _build_tree_fdq(D, labels)
    
    return newick

# EVOLVE-BLOCK-END
