"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Multi-Start Neighbor Joining with Randomized Tie-Breaking and Consensus.'''
import time
import math
import random
import numpy as np

# Global time tracking
_START_TIME = None
_TIME_LIMIT = 260.0  # Safety margin for 270s limit

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_left():
    _init_time()
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))

def _jc_distance(p):
    """Jukes-Cantor corrected distance."""
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC distances using NumPy for speed."""
    n = len(labels)
    # Map sequences to integer arrays
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([[seq_map[c] for c in sequences[label]] for label in labels])
    
    # Calculate differences: (n, n, L)
    # To save memory for large L, we can do it in chunks or just rely on NumPy efficiency for small n
    # n <= 32, L <= 2000. 32*32*2000 is ~2M elements, very manageable.
    diff = (seqs[:, np.newaxis, :] != seqs[np.newaxis, :, :])
    # Sum differences along the sequence axis
    num_diffs = np.sum(diff, axis=2)
    L = seqs.shape[1]
    p = num_diffs / L
    
    # Apply JC correction
    # Handle p >= 0.75
    p_clipped = np.clip(p, 0, 0.7499)
    dists = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    np.fill_diagonal(dists, 0.0)
    
    # Set very high distances for p >= 0.75 to avoid log(0) issues, though clipped handles it
    # Actually, if p > 0.75, the log argument becomes negative. 
    # The clip prevents that. But biologically, we might want to cap it.
    # The standard NJ handles this by just using a large number.
    # Let's replace inf or nan with a large value.
    dists = np.nan_to_num(dists, nan=3.0, posinf=3.0, neginf=0.0)
    
    return dists

def _nj_run(dists, labels, seed):
    """
    Run one instance of Neighbor Joining.
    dists: (n, n) numpy array
    labels: list of taxon labels
    seed: random seed for tie-breaking
    Returns: Newick string
    """
    rng = random.Random(seed)
    n = len(labels)
    
    # Work with a mutable copy of the distance matrix
    d = dists.copy()
    
    # Keep track of current nodes
    # active_nodes: list of indices into the original labels or new internal nodes
    # We will maintain a list of 'current' nodes and a mapping from current node id to Newick string
    # To simplify, let's use a dictionary for the tree structure and a list for active nodes.
    
    # State:
    # nodes: list of active node identifiers (strings)
    # tree: dict mapping node identifier to its Newick representation (or just the label for leaves)
    # d: current distance matrix, indexed by the order of 'nodes'
    
    nodes = [str(i) for i in range(n)]
    tree = {str(i): labels[i] for i in range(n)}
    
    # We need to manage the distance matrix carefully as nodes are added/removed.
    # Instead of modifying the 2D array in place which is complex with dynamic node sets,
    # we can maintain the matrix aligned with the 'nodes' list.
    
    current_d = d.copy()
    
    while len(nodes) > 3:
        n_curr = len(nodes)
        
        # Calculate row sums (Q matrix calculation requires r_i = sum_j d_ij)
        # r is a vector of length n_curr
        r = np.sum(current_d, axis=1)
        
        # Calculate Q matrix: Q_ij = (n-2)*d_ij - r_i - r_j
        # We only need to find the minimum Q_ij for i != j
        # Q is symmetric.
        
        # Efficient calculation of min Q:
        # Q_ij = (n-2)*d_ij - r_i - r_j
        # We can vectorize this.
        
        # Create a matrix of r_i and r_j
        # r_i is added to each row, r_j to each col
        # Q = (n-2)*D - r[:, None] - r[None, :]
        
        q_matrix = (n_curr - 2) * current_d - r[:, None] - r[None, :]
        
        # Set diagonal to infinity to avoid picking i==j
        np.fill_diagonal(q_matrix, np.inf)
        
        # Find the global minimum
        min_idx = np.unravel_index(np.argmin(q_matrix), q_matrix.shape)
        i, j = min_idx
        
        # Randomized Tie-Breaking:
        # If there are other pairs with Q values very close to the minimum, pick randomly among them.
        # This helps explore different topologies.
        threshold = 1e-6
        candidates = []
        # Iterate through all pairs to find candidates? O(n^2) which is fine for n<=32.
        # Or just check if the second minimum is close.
        # For simplicity and speed, let's just check a few neighbors or all pairs if n is small.
        # Given n<=32, O(n^2) check is ~1000 ops, negligible.
        
        min_q = q_matrix[i, j]
        
        # Collect all pairs with q <= min_q + threshold
        # We need to be careful not to include (j, i) if we already have (i, j) or vice versa, 
        # but since Q is symmetric and we pick unique pairs, we can just collect (a,b) where a<b.
        
        for a in range(n_curr):
            for b in range(a + 1, n_curr):
                if q_matrix[a, b] <= min_q + threshold:
                    candidates.append((a, b))
        
        if len(candidates) > 1:
            # Randomly select one
            i, j = rng.choice(candidates)
        
        # Join nodes i and j
        # Create new node
        new_node_id = f"n{len(nodes)}_{i}_{j}"
        
        # Newick string for the new node
        # The branch lengths in NJ are:
        # delta_i = 0.5 * d_ij
        # delta_j = 0.5 * d_ij
        # But for topology only, we just need the structure.
        # Newick: (tree[i], tree[j])
        new_tree_str = f"({tree[nodes[i]]},{tree[nodes[j]]})"
        
        # Calculate distances from new node to all other nodes k
        # d_new_k = 0.5 * (d_i_k + d_j_k - d_i_j)
        other_indices = [k for k in range(n_curr) if k != i and k != j]
        
        new_dists = {}
        d_ij = current_d[i, j]
        
        for k in other_indices:
            new_d = 0.5 * (current_d[i, k] + current_d[j, k] - d_ij)
            # Ensure non-negative? JC distances can be slightly negative due to error, but usually positive.
            # NJ usually assumes non-negative.
            new_dists[k] = max(0.0, new_d)
        
        # Update the node list and tree dict
        # Remove i and j
        # Add new_node_id
        
        # We need to rebuild the distance matrix for the next iteration.
        # The new set of nodes is: [nodes[k] for k in other_indices] + [new_node_id]
        
        new_nodes = [nodes[k] for k in other_indices] + [new_node_id]
        new_tree = {nodes[k]: tree[nodes[k]] for k in other_indices}
        new_tree[new_node_id] = new_tree_str
        
        # Build new distance matrix
        n_new = len(new_nodes)
        new_d = np.zeros((n_new, n_new))
        
        # Map old indices to new positions
        # The 'other_indices' correspond to the first part of new_nodes
        # The new_node_id corresponds to the last part
        
        # Distances between existing nodes (subset of old matrix)
        # We need to extract the submatrix for other_indices
        idx_arr = np.array(other_indices)
        sub_d = current_d[np.ix_(idx_arr, idx_arr)]
        new_d[:len(other_indices), :len(other_indices)] = sub_d
        
        # Distances from new node to existing nodes
        # The new node is at index len(other_indices) in new_nodes
        new_idx = len(other_indices)
        for pos, k in enumerate(other_indices):
            val = new_dists[k]
            new_d[new_idx, pos] = val
            new_d[pos, new_idx] = val
        
        # Distance from new node to itself is 0
        new_d[new_idx, new_idx] = 0.0
        
        # Update state
        nodes = new_nodes
        tree = new_tree
        current_d = new_d
        
    # Final 3 nodes: a, b, c
    # Unrooted tree on 3 taxa is just (a,b,c) or ((a,b),c) etc.
    # Standard Newick for 3 leaves: (a,b,c);
    # Since it's unrooted, any binary representation of the 3 leaves is topologically equivalent.
    # Let's just output (leaf1, leaf2, leaf3)
    
    # The 'nodes' list has 3 elements.
    # tree[nodes[0]], tree[nodes[1]], tree[nodes[2]]
    
    return "(" + ",".join(tree[n] for n in nodes) + ");"

def _normalize_newick(newick):
    """
    Normalize a Newick string to a canonical form for comparison.
    Removes branch lengths, sorts subtrees lexicographically to handle symmetry.
    """
    # Remove branch lengths
    # Pattern: (...,length)
    import re
    newick = re.sub(r'[:,](\d+\.?\d*)', '', newick)
    newick = newick.rstrip(';')
    
    # Recursive normalization
    def norm(s):
        s = s.strip()
        if s.startswith('('):
            # Find the matching parenthesis
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
            
            if end_idx == -1:
                return s
            
            inner = s[1:end_idx]
            label = s[end_idx+1:]
            
            # Split inner by commas, respecting parentheses
            parts = []
            depth = 0
            start = 0
            for i, char in enumerate(inner):
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                elif char == ',' and depth == 0:
                    parts.append(inner[start:i])
                    start = i + 1
            parts.append(inner[start:])
            
            # Normalize each part
            norm_parts = [norm(p) for p in parts]
            # Sort parts to canonicalize order
            norm_parts.sort()
            
            return "(" + ",".join(norm_parts) + ")" + label
        else:
            return s
    
    return norm(newick)

def reconstruct(sequences):
    _init_time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    if n_taxa < 3:
        # Trivial cases
        if n_taxa == 1:
            return labels[0] + ";"
        elif n_taxa == 2:
            return f"({labels[0]},{labels[1]});"
    
    # Compute distance matrix
    dists = _compute_distance_matrix(sequences, labels)
    
    # Strategy: Multi-start NJ
    # We want to run multiple iterations.
    # Budget check:
    # Single NJ is O(n^3). With n=32, it's very fast (< 1ms).
    # Distance matrix calc is O(n^2 * L). With n=32, L=2000, it's ~2M ops, very fast.
    # So we can run many restarts.
    
    # Let's run as many restarts as possible within the time limit.
    # Each restart: 1 NJ run + 1 normalization.
    
    candidate_trees = []
    
    # Number of restarts: 
    # Let's start with a fixed number, say 50, and if time permits, do more.
    # Or just loop until time is low.
    
    # To ensure diversity, we use different random seeds.
    # Also, we can perturb the distance matrix slightly? 
    # Randomized tie-breaking is usually sufficient for NJ.
    
    # Let's run at least 10 restarts to be safe.
    min_restarts = 10
    max_restarts = 1000 # Cap to avoid infinite loop if time check fails
    
    current_restart = 0
    
    while current_restart < max_restarts:
        if _time_left() < 5.0: # Safety margin
            break
            
        seed = current_restart
        tree_newick = _nj_run(dists, labels, seed)
        candidate_trees.append(tree_newick)
        
        current_restart += 1
        
        # If we have enough candidates, we can stop early if we find a consensus?
        # Consensus requires counting. 
        # Let's just collect all and pick the mode at the end.
        # But normalization is O(L) where L is length of newick.
        
    if not candidate_trees:
        # Fallback: Single deterministic NJ
        tree_newick = _nj_run(dists, labels, 0)
        candidate_trees.append(tree_newick)
        
    # Find the most frequent normalized tree
    # Normalize all
    normalized_trees = [_normalize_newick(t) for t in candidate_trees]
    
    # Count frequencies
    from collections import Counter
    counts = Counter(normalized_trees)
    
    # Pick the most common
    if counts:
        best_norm, count = counts.most_common(1)[0]
        # We need to return the original Newick string corresponding to this normalized tree
        # Since normalization is deterministic, we can just return one of the originals that maps to best_norm.
        # Or we can reconstruct from the normalized tree? 
        # The normalized tree is a valid Newick string (just sorted). 
        # The task accepts any valid Newick. 
        # So we can return best_norm directly? 
        # Wait, best_norm might have labels sorted in a way that is valid.
        # Let's check: _normalize_newick returns a valid Newick string.
        # Example: "(t0,t1,t2);" -> "(t0,t1,t2);"
        # If we return the normalized string, it is a valid tree.
        
        # However, the labels in the normalized string are the original labels.
        # So returning best_norm is safe.
        
        return best_norm + ";"
    else:
        # Should not happen
        return candidate_trees[0]

# EVOLVE-BLOCK-END
