# EVOLVE-BLOCK-START
"""Phylogenetic topology recovery: NJ + bootstrap consensus with deadline guard."""
import time
import math
import numpy as np
from collections import Counter

_DEADLINE_FRACTION = 0.88
_TOTAL_BUDGET = 270.0
_START = None


def _now():
    return time.time()


def _init_time():
    global _START
    if _START is None:
        _START = time.time()


def _time_left():
    _init_time()
    return max(0.0, _TOTAL_BUDGET * _DEADLINE_FRACTION - (_now() - _START))


def _jc_distance(a, b):
    """Jukes-Cantor corrected distance between two DNA strings."""
    n = len(a)
    if n == 0:
        return 0.0
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    p = diffs / n
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _nj_newick(labels, dist_matrix):
    """
    Neighbour Joining on a distance matrix.
    labels: list of n taxon names (indices 0..n-1 mapped to names)
    dist_matrix: numpy array of shape (n, n)
    Returns a Newick string (no branch lengths needed for topology, but we include them).
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"

    # We'll work with a list of current nodes and a mapping to subtrees
    nodes = list(range(n))
    subtree = {i: labels[i] for i in range(n)}
    dist = dist_matrix.copy()

    while len(nodes) > 3:
        m = len(nodes)
        # Compute Q-matrix criterion
        # r_i = sum of distances from i to all other current nodes
        r = np.array([sum(dist[i, j] for j in nodes if j != i) for i in nodes])
        # For each pair, compute (m-2)*d(i,j) - r_i - r_j
        best_criterion = float('inf')
        best_i, best_j = -1, -1
        for idx_i, i in enumerate(nodes):
            for idx_j, j in enumerate(nodes):
                if i == j:
                    continue
                criterion = (m - 2) * dist[i, j] - r[idx_i] - r[idx_j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i, best_j = i, j

        i, j = best_i, best_j
        # Compute branch lengths (optional but good for NJ)
        # len(i, new) = 0.5*d(i,j) + (r_i - r_j)/(2*(m-2))
        ri = r[nodes.index(i)]
        rj = r[nodes.index(j)]
        len_i = 0.5 * dist[i, j] + (ri - rj) / (2.0 * (m - 2))
        len_j = 0.5 * dist[i, j] - (ri - rj) / (2.0 * (m - 2))

        new_node = max(nodes) + 1
        new_subtree = f"({subtree[i]}:{len_i:.4f},{subtree[j]}:{len_j:.4f})"

        # Update distance matrix
        for k in nodes:
            if k == i or k == j:
                continue
            dist[new_node, k] = 0.5 * (dist[i, k] + dist[j, k] - dist[i, j])
            dist[k, new_node] = dist[new_node, k]

        # Remove i and j, add new_node
        nodes.remove(i)
        nodes.remove(j)
        nodes.append(new_node)
        subtree[new_node] = new_subtree

    # Final three nodes: join them in a star
    a, b, c = nodes[0], nodes[1], nodes[2]
    # Compute branch lengths for the final triad
    d_ab = dist[a, b]
    d_ac = dist[a, c]
    d_bc = dist[b, c]
    len_a = 0.5 * (d_ab + d_ac - d_bc)
    len_b = 0.5 * (d_ab + d_bc - d_ac)
    len_c = 0.5 * (d_ac + d_bc - d_ab)

    return f"({subtree[a]}:{len_a:.4f},{subtree[b]}:{len_b:.4f},{subtree[c]}:{len_c:.4f})"


def _get_splits(newick, labels_set):
    """
    Extract the set of non-trivial unrooted splits from a Newick string.
    Each split is represented as a frozenset of the smaller side (or sorted tuple)
    to canonicalize.
    """
    # Parse Newick to get all internal splits
    # We'll do a simple recursive parse
    splits = set()
    
    def parse(s, i):
        """Parse Newick, return (subtree_taxa_set, new_index)"""
        if i >= len(s):
            return set(), i
        if s[i] == '(':
            i += 1
            taxa = set()
            while True:
                child_taxa, i = parse(s, i)
                taxa |= child_taxa
                if i >= len(s):
                    break
                if s[i] == ',':
                    i += 1
                elif s[i] == ')':
                    i += 1
                    break
                else:
                    break
            # Skip branch length
            while i < len(s) and s[i] not in (',', ')', ';'):
                i += 1
            return taxa, i
        else:
            # Leaf or label
            j = i
            while j < len(s) and s[j] not in (',', ')', ';', ':'):
                j += 1
            label = s[i:j]
            i = j
            # Skip branch length
            while i < len(s) and s[i] not in (',', ')', ';'):
                i += 1
            return {label}, i
    
    taxa, _ = parse(newick, 0)
    
    # Now we need to find all internal nodes' splits
    # Let's do a different approach: traverse and collect splits
    splits = set()
    
    def collect_splits(s, i):
        """Collect splits from internal nodes."""
        if i >= len(s):
            return set(), i
        if s[i] == '(':
            i += 1
            taxa = set()
            child_splits = set()
            while True:
                child_taxa, child_sp, i = parse_with_splits(s, i)
                taxa |= child_taxa
                child_splits |= child_sp
                if i >= len(s):
                    break
                if s[i] == ',':
                    i += 1
                elif s[i] == ')':
                    i += 1
                    break
                else:
                    break
            # This internal node defines a split: taxa vs (all - taxa)
            # We'll add it later when we have all taxa
            # Skip branch length
            while i < len(s) and s[i] not in (',', ')', ';'):
                i += 1
            return (taxa, child_splits), i
        else:
            j = i
            while j < len(s) and s[j] not in (',', ')', ';', ':'):
                j += 1
            label = s[i:j]
            i = j
            while i < len(s) and s[i] not in (',', ')', ';'):
                i += 1
            return ({label}, set()), i
    
    def parse_with_splits(s, i):
        if i >= len(s):
            return set(), set(), i
        if s[i] == '(':
            i += 1
            taxa = set()
            all_splits = set()
            while True:
                child_taxa, child_splits, i = parse_with_splits(s, i)
                taxa |= child_taxa
                all_splits |= child_splits
                if i >= len(s):
                    break
                if s[i] == ',':
                    i += 1
                elif s[i] == ')':
                    i += 1
                    break
                else:
                    break
            # This node's split
            all_splits.add(frozenset(taxa))
            # Skip branch length
            while i < len(s) and s[i] not in (',', ')', ';'):
                i += 1
            return taxa, all_splits, i
        else:
            j = i
            while j < len(s) and s[j] not in (',', ')', ';', ':'):
                j += 1
            label = s[i:j]
            i = j
            while i < len(s) and s[i] not in (',', ')', ';'):
                i += 1
            return {label}, set(), i
    
    _, all_splits, _ = parse_with_splits(newick, 0)
    
    # Canonicalize: for each split, keep the smaller side (or lexicographically smaller)
    n_taxa = len(labels_set)
    canonical = set()
    for sp in all_splits:
        if len(sp) == 0 or len(sp) == n_taxa:
            continue  # trivial split
        # Canonical form: use the side with fewer taxa, or if equal, the one with smaller min element
        other = labels_set - sp
        if len(sp) < len(other):
            canon = frozenset(sp)
        elif len(other) < len(sp):
            canon = frozenset(other)
        else:
            # Equal size: use the one with smaller minimum element
            if min(sp) < min(other):
                canon = frozenset(sp)
            else:
                canon = frozenset(other)
        canonical.add(canon)
    
    return canonical


def _rf_distance(splits_a, splits_b):
    """Robinson-Foulds distance between two sets of splits."""
    if not splits_a and not splits_b:
        return 0
    intersection = splits_a & splits_b
    return len(splits_a - splits_b) + len(splits_b - splits_a)


def _normalized_rf(splits_a, splits_b):
    """Normalized RF distance (0 to 1)."""
    union = splits_a | splits_b
    if not union:
        return 0.0
    rf = _rf_distance(splits_a, splits_b)
    return rf / len(union)


def reconstruct(sequences):
    _init_time()
    
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    n = len(labels)
    n_sites = len(sequences[labels[0]]) if n > 0 else 0
    
    if n <= 2:
        if n == 1:
            return labels[0] + ";"
        elif n == 2:
            return f"({labels[0]},{labels[1]});"
    
    # Build distance matrix
    def compute_dist_matrix():
        dm = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
                dm[i, j] = d
                dm[j, i] = d
        return dm
    
    # Single NJ first
    dm = compute_dist_matrix()
    single_newick = _nj_newick(labels, dm)
    single_splits = _get_splits(single_newick, set(labels))
    
    # Bootstrap consensus
    # Determine number of bootstrap replicates based on time budget
    # Estimate time for one NJ run
    t0 = _now()
    _ = _nj_newick(labels, dm)  # already done, but let's time a fresh one
    # Actually, let's just do a quick timing
    t1 = _now()
    nj_time = t1 - t0
    
    # Time for distance matrix computation
    t2 = _now()
    _ = compute_dist_matrix()
    t3 = _now()
    dm_time = t3 - t2
    
    per_replicate_time = nj_time + dm_time * 0.5  # bootstrap resampling is cheaper
    
    time_budget = _time_left()
    # Reserve some time for final processing
    time_budget = max(0, time_budget - 5.0)
    
    max_replicates = max(1, int(time_budget / max(per_replicate_time, 0.001)))
    # Cap at reasonable number
    max_replicates = min(max_replicates, 200)
    
    # Bootstrap
    rng = np.random.default_rng(42)
    tree_votes = Counter()
    
    for rep in range(max_replicates):
        if _time_left() < 10.0:
            break
        
        # Resample sites
        boot_indices = rng.integers(0, n_sites, size=n_sites)
        
        # Compute bootstrap distance matrix
        boot_dm = np.zeros((n, n))
        for i in range(n):
            seq_i = sequences[labels[i]]
            for j in range(i + 1, n):
                seq_j = sequences[labels[j]]
                # Count differences at bootstrapped sites
                diffs = sum(1 for idx in boot_indices if seq_i[idx] != seq_j[idx])
                p = diffs / n_sites
                if p >= 0.75:
                    d = 3.0
                else:
                    d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
                boot_dm[i, j] = d
                boot_dm[j, i] = d
        
        boot_newick = _nj_newick(labels, boot_dm)
        boot_splits = _get_splits(boot_newick, set(labels))
        
        # Vote for each split
        for sp in boot_splits:
            tree_votes[sp] += 1
    
    # Build consensus: include splits with frequency > 50%
    if tree_votes:
        consensus_splits = set()
        for sp, count in tree_votes.items():
            if count > max_replicates * 0.5:
                consensus_splits.add(sp)
    else:
        consensus_splits = single_splits
    
    # If consensus is empty or too small, fall back to single NJ
    if len(consensus_splits) < n - 3:
        consensus_splits = single_splits
    
    # Now we need to convert the set of splits back to a Newick string
    # This is a hard problem. Instead, let's find the NJ tree whose splits
    # best match the consensus.
    
    # Alternative: among the bootstrap trees, find the one with highest
    # agreement with consensus splits.
    
    # Let's re-run bootstrap but keep track of full trees
    # Actually, let's just use the NJ tree from the full data as a good approximation,
    # and if we have time, do a quick search.
    
    # For now, return the single NJ tree if consensus doesn't help much
    # Let's try a different approach: use the median tree from bootstrap
    
    # Re-do bootstrap keeping Newick strings
    if max_replicates > 1 and _time_left() > 15.0:
        newick_list = [single_newick]
        for rep in range(max_replicates):
            if _time_left() < 10.0:
                break
            
            boot_indices = rng.integers(0, n_sites, size=n_sites)
            boot_dm = np.zeros((n, n))
            for i in range(n):
                seq_i = sequences[labels[i]]
                for j in range(i + 1, n):
                    seq_j = sequences[labels[j]]
                    diffs = sum(1 for idx in boot_indices if seq_i[idx] != seq_j[idx])
                    p = diffs / n_sites
                    if p >= 0.75:
                        d = 3.0
                    else:
                        d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
                    boot_dm[i, j] = d
                    boot_dm[j, i] = d
            
            boot_newick = _nj_newick(labels, boot_dm)
            newick_list.append(boot_newick)
        
        # Find the tree with minimum average RF to all others
        if len(newick_list) > 1:
            splits_list = [_get_splits(nw, set(labels)) for nw in newick_list]
            best_idx = 0
            best_score = float('inf')
            for i in range(len(splits_list)):
                total_rf = 0
                for j in range(len(splits_list)):
                    if i != j:
                        total_rf += _rf_distance(splits_list[i], splits_list[j])
                if total_rf < best_score:
                    best_score = total_rf
                    best_idx = i
            return newick_list[best_idx] + ";"
    
    return single_newick + ";"

# EVOLVE-BLOCK-END
