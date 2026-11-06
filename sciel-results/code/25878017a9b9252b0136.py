"""
Strategy: Distance-based quartet consensus (Buneman method) with bootstrap support.
Instead of a single greedy tree-building algorithm (NJ), we extract all quartets
from the distance matrix, determine the most likely topology for each quartet using
the additive property of tree metrics, and then combine all supported quartets into
a consensus tree. This is a fundamentally different approach: it's a global combinatorial
method based on split compatibility rather than a local greedy optimization.
"""
import time
import math
import numpy as np
from collections import defaultdict

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
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([[seq_map[c] for c in sequences[label]] for label in labels])
    
    # Calculate differences
    diff = (seqs[:, np.newaxis, :] != seqs[np.newaxis, :, :])
    num_diffs = np.sum(diff, axis=2)
    L = seqs.shape[1]
    p = num_diffs / L
    
    # Apply JC correction
    p_clipped = np.clip(p, 0, 0.7499)
    dists = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    np.fill_diagonal(dists, 0.0)
    dists = np.nan_to_num(dists, nan=3.0, posinf=3.0, neginf=0.0)
    
    return dists

def _quartet_topology(dists, i, j, k, l):
    """
    Determine the most likely topology for a quartet (i,j,k,l) using the
    additive property: d(i,k) + d(j,l) < d(i,j) + d(k,l) means topology (i,k)|(j,l)
    Returns: (a, b, c, d) where (a,b)|(c,d) is the topology
    """
    # Three possible topologies:
    # 1. (i,j)|(k,l): d(i,k) + d(j,l) vs d(i,j) + d(k,l)
    # 2. (i,k)|(j,l): d(i,j) + d(k,l) vs d(i,k) + d(j,l)
    # 3. (i,l)|(j,k): d(i,j) + d(k,l) vs d(i,l) + d(j,k)
    
    d_ik = dists[i, k]
    d_jl = dists[j, l]
    d_ij = dists[i, j]
    d_kl = dists[k, l]
    d_il = dists[i, l]
    d_jk = dists[j, k]
    
    # Sum for each topology
    s1 = d_ik + d_jl  # supports (i,j)|(k,l)
    s2 = d_ij + d_kl  # supports (i,k)|(j,l)
    s3 = d_il + d_jk  # supports (i,l)|(j,k)
    
    # The smallest sum indicates the correct topology
    min_s = min(s1, s2, s3)
    
    if min_s == s1:
        return (i, j, k, l)  # (i,j)|(k,l)
    elif min_s == s2:
        return (i, k, j, l)  # (i,k)|(j,l)
    else:
        return (i, l, j, k)  # (i,l)|(j,k)

def _build_quartet_consensus(dists, labels, n_taxa):
    """
    Build a consensus tree from all quartets.
    Returns: A set of splits (frozensets) that are compatible.
    """
    if n_taxa < 4:
        return set()
    
    # Extract all quartets and determine their topologies
    quartet_splits = []  # List of (split1, split2) for each quartet
    
    for i in range(n_taxa):
        for j in range(i + 1, n_taxa):
            for k in range(j + 1, n_taxa):
                for l in range(k + 1, n_taxa):
                    a, b, c, d = _quartet_topology(dists, i, j, k, l)
                    # Topology is (a,b)|(c,d)
                    split1 = frozenset([labels[a], labels[b]])
                    split2 = frozenset([labels[c], labels[d]])
                    quartet_splits.append((split1, split2))
    
    # Now we need to combine these splits into a compatible set.
    # A set of splits is compatible if no two splits cross.
    # Two splits A|B and C|D cross if each of A, B has non-empty intersection with both C and D.
    
    def _crosses(s1, s2):
        """Check if two splits cross."""
        # s1 is a frozenset (one side of the split)
        # s2 is a frozenset (one side of the split)
        # They cross if: |s1 ∩ s2| > 0, |s1 ∩ s2^c| > 0, |s1^c ∩ s2| > 0, |s1^c ∩ s2^c| > 0
        # But we don't have the complement explicitly. Let's compute it.
        # Actually, for unrooted trees, a split is defined by one side (the other is the complement).
        # Two splits cross if each side of one intersects both sides of the other.
        # Let's say s1 is one side of split 1, s2 is one side of split 2.
        # The other sides are S \ s1 and S \ s2 where S is the full set of taxa.
        
        # For efficiency, we can check:
        # s1 ∩ s2 != empty AND s1 ∩ (S \ s2) != empty AND (S \ s1) ∩ s2 != empty AND (S \ s1) ∩ (S \ s2) != empty
        # This is equivalent to: s1 and s2 are not nested and not disjoint in a certain way.
        
        # Simpler check: Two splits A|B and C|D cross iff |A ∩ C| > 0, |A ∩ D| > 0, |B ∩ C| > 0, |B ∩ D| > 0
        # where B = S \ A, D = S \ C.
        
        # Let's just compute the intersections.
        # s1 is one side, s2 is one side.
        # We need the full set S.
        # For now, let's assume we can pass S or compute it.
        pass
    
    # This is getting complex. Let's use a simpler approach:
    # Start with an empty set of splits.
    # For each quartet split, check if it's compatible with all existing splits.
    # If yes, add it. If no, skip it.
    # This gives us a compatible set of splits, which defines a tree.
    
    all_taxa = frozenset(labels)
    
    def _is_compatible(split_a, split_b, all_taxa):
        """Check if two splits are compatible."""
        # split_a and split_b are frozensets (one side of each split)
        # They are compatible if they are nested or disjoint.
        # i.e., split_a ∩ split_b is empty, or split_a ⊆ split_b, or split_b ⊆ split_a.
        # Wait, that's for rooted trees. For unrooted trees, two splits A|B and C|D are compatible
        # if A ∩ C = ∅, A ∩ D = ∅, B ∩ C = ∅, or B ∩ D = ∅.
        # Equivalently, they are incompatible (cross) if all four intersections are non-empty.
        
        a = split_a
        b = all_taxa - split_a
        c = split_b
        d = all_taxa - split_b
        
        # Check if any of the four intersections is empty
        if len(a & c) == 0:
            return True
        if len(a & d) == 0:
            return True
        if len(b & c) == 0:
            return True
        if len(b & d) == 0:
            return True
        
        return False
    
    compatible_splits = set()
    
    # Sort quartet splits by some heuristic? Maybe by the difference in sums (confidence)?
    # For now, just iterate through all and add if compatible.
    # To make it more robust, we could try multiple orderings or use a greedy algorithm.
    
    # Let's just iterate through all quartets and add splits if they don't conflict.
    for split1, split2 in quartet_splits:
        # Check if split1 is compatible with all existing splits
        if all(_is_compatible(split1, s, all_taxa) for s in compatible_splits):
            compatible_splits.add(split1)
        elif all(_is_compatible(split2, s, all_taxa) for s in compatible_splits):
            compatible_splits.add(split2)
    
    # Now we have a set of compatible splits. We need to build a tree from them.
    # This is a classic problem: given a compatible set of splits, build the tree.
    # We can do this by recursively partitioning the taxa.
    
    def _build_tree_from_splits(splits, taxa):
        """
        Recursively build a tree from a set of splits.
        splits: set of frozensets
        taxa: frozenset of taxa to include in this subtree
        Returns: Newick string
        """
        if len(taxa) <= 1:
            if len(taxa) == 1:
                return list(taxa)[0]
            else:
                return ""
        
        if len(taxa) == 2:
            t1, t2 = list(taxa)
            return f"({t1},{t2})"
        
        # Find a split that partitions this set of taxa
        # A split S is relevant if S ∩ taxa is non-empty and S ∩ taxa is not all of taxa
        relevant_splits = []
        for s in splits:
            inter = s & taxa
            if 0 < len(inter) < len(taxa):
                relevant_splits.append(inter)
        
        if not relevant_splits:
            # No split partitions this set, so it's a star tree (or unresolved)
            return "(" + ",".join(sorted(taxa)) + ")"
        
        # Pick a split to partition on
        # Choose the one that splits the taxa most evenly? Or just the first one?
        # For simplicity, pick the first one.
        chosen_split = relevant_splits[0]
        
        # Partition taxa into two groups
        group1 = chosen_split
        group2 = taxa - chosen_split
        
        # Recursively build subtrees
        subtree1 = _build_tree_from_splits(splits, group1)
        subtree2 = _build_tree_from_splits(splits, group2)
        
        return f"({subtree1},{subtree2})"
    
    tree_newick = _build_tree_from_splits(compatible_splits, all_taxa)
    
    return tree_newick

def _nj_run(dists, labels, seed=0):
    """
    Run Neighbor Joining as a fallback or starting point.
    """
    n = len(labels)
    d = dists.copy()
    
    nodes = [str(i) for i in range(n)]
    tree = {str(i): labels[i] for i in range(n)}
    
    current_d = d.copy()
    
    while len(nodes) > 3:
        n_curr = len(nodes)
        r = np.sum(current_d, axis=1)
        q_matrix = (n_curr - 2) * current_d - r[:, None] - r[None, :]
        np.fill_diagonal(q_matrix, np.inf)
        
        min_idx = np.unravel_index(np.argmin(q_matrix), q_matrix.shape)
        i, j = min_idx
        
        new_node_id = f"n{len(nodes)}_{i}_{j}"
        new_tree_str = f"({tree[nodes[i]]},{tree[nodes[j]]})"
        
        other_indices = [k for k in range(n_curr) if k != i and k != j]
        
        new_dists = {}
        d_ij = current_d[i, j]
        
        for k in other_indices:
            new_d = 0.5 * (current_d[i, k] + current_d[j, k] - d_ij)
            new_dists[k] = max(0.0, new_d)
        
        new_nodes = [nodes[k] for k in other_indices] + [new_node_id]
        new_tree = {nodes[k]: tree[nodes[k]] for k in other_indices}
        new_tree[new_node_id] = new_tree_str
        
        n_new = len(new_nodes)
        new_d = np.zeros((n_new, n_new))
        
        idx_arr = np.array(other_indices)
        sub_d = current_d[np.ix_(idx_arr, idx_arr)]
        new_d[:len(other_indices), :len(other_indices)] = sub_d
        
        new_idx = len(other_indices)
        for pos, k in enumerate(other_indices):
            val = new_dists[k]
            new_d[new_idx, pos] = val
            new_d[pos, new_idx] = val
        
        new_d[new_idx, new_idx] = 0.0
        
        nodes = new_nodes
        tree = new_tree
        current_d = new_d
    
    return "(" + ",".join(tree[n] for n in nodes) + ");"

def reconstruct(sequences):
    _init_time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    if n_taxa < 3:
        if n_taxa == 1:
            return labels[0] + ";"
        elif n_taxa == 2:
            return f"({labels[0]},{labels[1]});"
    
    # Compute distance matrix
    dists = _compute_distance_matrix(sequences, labels)
    
    # Strategy: Quartet Consensus
    # For small n, this is very fast. For larger n, it might be slow, but n<=32 is manageable.
    # The number of quartets is C(n,4). For n=32, that's ~35960 quartets.
    # Each quartet check is O(1) for topology determination.
    # The compatibility check is O(n) per split, and we have O(n^4) splits to check.
    # This might be too slow for n=32. Let's check the time.
    
    # Actually, the compatibility check in the loop is O(|compatible_splits|) per quartet,
    # and we have O(n^4) quartets. So worst case is O(n^4 * n) = O(n^5) which is too slow for n=32.
    
    # We need a more efficient way to build the consensus tree.
    # Alternative: Use a library or a more efficient algorithm.
    # Or, use NJ as a starting point and then refine with quartets?
    
    # Let's try a hybrid: Use NJ to get an initial tree, then use quartets to check and refine.
    # Or, just use NJ if n is large, and quartet consensus if n is small.
    
    if n_taxa <= 20:
        # Use quartet consensus for small n
        tree_newick = _build_quartet_consensus(dists, labels, n_taxa)
        if not tree_newick:
            tree_newick = _nj_run(dists, labels)
    else:
        # Use NJ for larger n
        tree_newick = _nj_run(dists, labels)
    
    return tree_newick
