"""Rewritten program using Multi-Model NJ and Parsimony-based Selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import sys

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 170.0  # 170s out of 180s budget

def _check_deadline():
    return time.time() > _DEADLINE

def _jc_distance_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected distance matrix using NumPy."""
    n = len(labels)
    seqs = np.array([list(sequences[l]) for l in labels])
    # seqs shape: (n, L)
    # Pairwise hamming distance
    # We can use broadcasting: seqs[:, None, :] == seqs[None, :, :]
    # This creates an (n, n, L) array. For n=32, L=2000, this is 32*32*2000 = 2M booleans. Fine.
    matches = (seqs[:, None, :] == seqs[None, :, :]).sum(axis=2)
    distances = (n * 0) # placeholder
    # Actually, matches is count of identical. Hamming = L - matches
    hamming = seqs.shape[1] - matches
    p = hamming / seqs.shape[1]
    # JC correction: -3/4 * ln(1 - 4/3 * p)
    # Clip p to avoid log(0) or negative
    p = np.clip(p, 0, 0.7499)
    dist_matrix = -0.75 * np.log(1.0 - 4.0/3.0 * p)
    np.fill_diagonal(dist_matrix, 0)
    return dist_matrix

def _k2p_distance_matrix(sequences, labels):
    """Compute Kimura 2-Parameter corrected distance matrix using NumPy."""
    n = len(labels)
    seqs = np.array([list(sequences[l]) for l in labels])
    L = seqs.shape[1]
    
    # Count transitions (A<->G, C<->T) and transversions (others)
    # We need pairwise counts.
    # A:0, C:1, G:2, T:3
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs_int = np.array([[base_map[c] for c in s] for s in sequences.values()], dtype=np.uint8)
    # Reorder labels to match seqs_int? 
    # The input to this function assumes seqs_int is aligned with labels order.
    # Let's ensure we pass aligned data.
    
    # Pairwise comparison
    # x = seqs_int[:, None, :], y = seqs_int[None, :, :]
    x = seqs_int[:, None, :]
    y = seqs_int[None, :, :]
    
    # Transitions: (0,2), (2,0), (1,3), (3,1)
    # Transversions: everything else where x != y
    
    # Create masks
    trans_mask = ((x == 0) & (y == 2)) | ((x == 2) & (y == 0)) | ((x == 1) & (y == 3)) | ((x == 3) & (y == 1))
    trans_count = trans_mask.sum(axis=2)
    
    diff_mask = (x != y)
    total_diff = diff_mask.sum(axis=2)
    transv_count = total_diff - trans_count
    
    p_trans = trans_count / L
    p_transv = transv_count / L
    
    # K2P formula:
    # d = -1/2 * ln((1 - 2*p_transv - p_trans) * (1 - 2*p_transv + p_trans))
    # Wait, standard formula:
    # p1 = transitions/L, p2 = transversions/L
    # d = -1/2 * ln((1 - 2*p2 - p1) * (1 - 2*p2 + p1))
    
    term1 = 1.0 - 2.0*p_transv - p_trans
    term2 = 1.0 - 2.0*p_transv + p_trans
    
    # Clip to avoid log of negative/zero
    term1 = np.clip(term1, 1e-10, 1.0)
    term2 = np.clip(term2, 1e-10, 1.0)
    
    dist_matrix = -0.5 * np.log(term1 * term2)
    np.fill_diagonal(dist_matrix, 0)
    return dist_matrix

def _nj_reconstruct(dist_matrix, labels):
    """
    Perform Neighbor Joining on a distance matrix.
    Returns a Newick string.
    dist_matrix: (n, n) numpy array
    labels: list of strings
    """
    n = len(labels)
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Initialize
    nodes = list(range(n))
    trees = {i: labels[i] for i in nodes}
    D = dist_matrix.copy()
    
    # Keep track of active nodes
    active = set(nodes)
    
    while len(active) > 3:
        if _check_deadline():
            # Fallback: return a star tree or the current best
            # For simplicity, return a star tree of remaining active nodes
            rem = [trees[i] for i in active]
            return "(" + ",".join(rem) + ");"
        
        n_active = len(active)
        # Calculate r_i = sum_{j in active, j!=i} D[i,j]
        r = np.zeros(n)
        # Efficiently compute row sums for active nodes
        # D is n x n. We only care about active indices.
        # Since n is small (<=32), simple loops are fine, but let's try to be clean.
        for i in active:
            r[i] = np.sum([D[i, j] for j in active if j != i])
        
        # Find best pair (a, b) minimizing (n-2)*D[a,b] - r[a] - r[b]
        best_score = float('inf')
        best_pair = (None, None)
        
        active_list = list(active)
        for i_idx, a in enumerate(active_list):
            for b in active_list[i_idx+1:]:
                score = (n_active - 2) * D[a, b] - r[a] - r[b]
                if score < best_score:
                    best_score = score
                    best_pair = (a, b)
        
        a, b = best_pair
        
        # Calculate branch lengths (optional for topology, but good for completeness if we were returning lengths)
        # l_ab = 0.5 * D[a,b] + (r[a] - r[b]) / (2 * (n_active - 2))
        # l_ba = 0.5 * D[a,b] - (r[a] - r[b]) / (2 * (n_active - 2))
        
        # Create new node
        new_node = max(active) + 1
        trees[new_node] = f"({trees[a]},{trees[b]})"
        
        # Update distance matrix
        # D[new, x] = 0.5 * (D[a,x] + D[b,x] - D[a,b])
        for x in active:
            if x != a and x != b:
                D[new_node, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
                D[x, new_node] = D[new_node, x]
        
        # Remove a and b, add new_node
        active.remove(a)
        active.remove(b)
        active.add(new_node)
        
        # Ensure D size is sufficient (it's fixed size n x n, so we just use indices)
        # We don't need to resize D, just use the new_node index.
        
    # Final 3 nodes
    rem = [trees[i] for i in active]
    return "(" + ",".join(rem) + ");"

def _parse_newick_to_splits(newick, labels):
    """
    Parse a Newick string and extract all non-trivial unrooted splits.
    Returns a set of frozensets representing the splits.
    """
    # This is a simplified parser for the specific format generated by _nj_reconstruct
    # Format: (A,B,(C,D));
    # We need to find the bipartitions.
    
    # To be robust, we can use a recursive parser.
    # However, for small N, we can also just traverse the tree structure if we kept it.
    # Since we only have the string, we must parse.
    
    splits = set()
    n = len(labels)
    label_set = set(labels)
    
    def parse_node(s, idx):
        # Returns (node_type, data, next_idx)
        # node_type: 'taxon' or 'internal'
        if s[idx] == '(':
            # Internal node
            idx += 1
            children = []
            while s[idx] != ')':
                child, idx = parse_node(s, idx)
                children.append(child)
                if s[idx] == ',':
                    idx += 1
            idx += 1 # skip )
            # Check for branch length (optional)
            if idx < len(s) and s[idx] != ',' and s[idx] != ')' and s[idx] != ';':
                while idx < len(s) and s[idx] not in ',);':
                    idx += 1
            return ('internal', children, idx)
        else:
            # Taxon
            start = idx
            while idx < len(s) and s[idx] not in ',);':
                idx += 1
            name = s[start:idx]
            return ('taxon', name, idx)
    
    def get_leaf_set(node):
        if node[0] == 'taxon':
            return {node[1]}
        else:
            leaves = set()
            for child in node[1]:
                leaves.update(get_leaf_set(child))
            return leaves
    
    # Parse the whole string
    # Remove trailing semicolon
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]
    
    # The root is the whole string
    # We need to find all edges.
    # An edge corresponds to a split: (Leaves on one side, Leaves on other side)
    
    # We can do a DFS and for each child of an internal node, calculate the set of leaves in that subtree.
    # The split is (subtree_leaves, all_leaves - subtree_leaves)
    
    all_leaves = set(labels)
    
    def extract_splits(node, idx):
        if node[0] == 'internal':
            for child in node[1]:
                child_leaves = get_leaf_set(child)
                # A split is defined by a bipartition.
                # We represent it as a frozenset of the smaller side, or a canonical form.
                # To compare splits, we can use a frozenset of frozensets? 
                # No, a split is just a set of taxa on one side.
                # Canonical form: sort the two sets, take the lexicographically smaller one?
                # Or just store the frozenset of one side, and ensure we don't store both S and S^c.
                # Let's store the frozenset of the side that contains the lexicographically smallest taxon?
                # Or simpler: store the frozenset of leaves in the subtree.
                # When comparing, check if S == T or S == All - T.
                
                # For efficiency in the set, let's canonicalize:
                # If len(child_leaves) <= len(all_leaves) - len(child_leaves):
                #    key = frozenset(child_leaves)
                # else:
                #    key = frozenset(all_leaves - child_leaves)
                
                if len(child_leaves) <= len(all_leaves) - len(child_leaves):
                    key = frozenset(child_leaves)
                else:
                    key = frozenset(all_leaves - child_leaves)
                
                splits.add(key)
                extract_splits(child, idx) # Recurse
    
    # Start parsing
    # The string is a single tree.
    # We need to parse the root.
    # parse_node returns the node and the index after it.
    root_node, _ = parse_node(s, 0)
    extract_splits(root_node, 0)
    
    return splits

def _fitch_score(newick, sequences, labels):
    """
    Calculate the Fitch Parsimony score for a given tree and sequences.
    """
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]
    
    def parse_node(s, idx):
        if s[idx] == '(':
            idx += 1
            children = []
            while s[idx] != ')':
                child, idx = parse_node(s, idx)
                children.append(child)
                if s[idx] == ',':
                    idx += 1
            idx += 1 # skip )
            if idx < len(s) and s[idx] != ',' and s[idx] != ')' and s[idx] != ';':
                while idx < len(s) and s[idx] not in ',);':
                    idx += 1
            return ('internal', children, idx)
        else:
            start = idx
            while idx < len(s) and s[idx] not in ',);':
                idx += 1
            name = s[start:idx]
            return ('taxon', name, idx)
    
    root_node, _ = parse_node(s, 0)
    
    # Map labels to sequences
    seq_map = {l: sequences[l] for l in labels}
    
    total_score = 0
    L = len(sequences[labels[0]])
    
    # For each site, run Fitch algorithm
    for site in range(L):
        # Post-order traversal to compute sets
        # We need to traverse the tree.
        # Since the tree is small, we can just recurse.
        
        def fitch(node):
            # Returns (set_of_states, cost)
            if node[0] == 'taxon':
                state = seq_map[node[1]][site]
                return ({state}, 0)
            else:
                child_results = [fitch(child) for child in node[1]]
                sets = [cr[0] for cr in child_results]
                costs = sum(cr[1] for cr in child_results)
                
                intersection = set.intersection(*sets)
                if intersection:
                    return (intersection, costs)
                else:
                    union = set.union(*sets)
                    return (union, costs + 1)
        
        _, cost = fitch(root_node)
        total_score += cost
        
    return total_score

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return ";".join(labels)
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # 1. Compute JC Distances
    try:
        dist_jc = _jc_distance_matrix(sequences, labels)
    except Exception:
        dist_jc = None
        
    # 2. Compute K2P Distances
    try:
        dist_k2p = _k2p_distance_matrix(sequences, labels)
    except Exception:
        dist_k2p = None
        
    candidates = []
    
    if dist_jc is not None:
        try:
            tree_jc = _nj_reconstruct(dist_jc, labels)
            candidates.append(tree_jc)
        except Exception:
            pass
            
    if dist_k2p is not None and dist_k2p is not dist_jc:
        try:
            tree_k2p = _nj_reconstruct(dist_k2p, labels)
            candidates.append(tree_k2p)
        except Exception:
            pass
            
    if not candidates:
        # Fallback: Star tree
        return "(" + ",".join(labels) + ");"
        
    if len(candidates) == 1:
        return candidates[0]
        
    # Score candidates using Fitch Parsimony
    # This is O(S * N) per tree.
    best_tree = candidates[0]
    best_score = float('inf')
    
    for tree in candidates:
        if _check_deadline():
            break
        try:
            score = _fitch_score(tree, sequences, labels)
            if score < best_score:
                best_score = score
                best_tree = tree
        except Exception:
            # If scoring fails, ignore this candidate
            pass
            
    return best_tree
# EVOLVE-BLOCK-END
