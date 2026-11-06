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


def _jc_distance_matrix_binary(bin_matrix, n_sites):
    """
    Compute JC distance matrix from a binary one-hot matrix.
    bin_matrix: shape (n_taxa, n_sites * 4)
    """
    identity_counts = bin_matrix @ bin_matrix.T
    diffs = n_sites - identity_counts
    p = diffs / n_sites
    p = np.clip(p, 0, 0.74)
    jc_dist = -0.75 * np.log(1.0 - 4.0 / 3.0 * p)
    np.fill_diagonal(jc_dist, 0.0)
    return jc_dist


def _sequences_to_binary(sequences, labels):
    """
    Convert list of DNA strings to a binary one-hot matrix.
    Returns numpy array of shape (n, n_sites * 4)
    """
    n = len(labels)
    if n == 0:
        return np.array([])
    
    seq0 = sequences[labels[0]]
    L = len(seq0)
    
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    bin_mat = np.zeros((n, L * 4), dtype=np.float32)
    
    for i, label in enumerate(labels):
        seq = sequences[label]
        for k, char in enumerate(seq):
            idx = char_map.get(char, 0)
            bin_mat[i, k * 4 + idx] = 1.0
            
    return bin_mat


def _nj_newick(labels, dist_matrix):
    """
    Neighbour Joining on a distance matrix.
    labels: list of n taxon names
    dist_matrix: numpy array of shape (n, n)
    Returns a Newick string.
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"

    max_nodes = 2 * n
    dist = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    dist[:n, :n] = dist_matrix
    
    nodes = list(range(n))
    subtree = {i: labels[i] for i in range(n)}
    next_node_id = n

    while len(nodes) > 3:
        m = len(nodes)
        dist_sub = dist[np.ix_(nodes, nodes)]
        r = dist_sub.sum(axis=1)
        
        Q = (m - 2) * dist_sub - r[:, None] - r[None, :]
        
        flat_idx = np.argmin(Q)
        i_idx = flat_idx // m
        j_idx = flat_idx % m
        
        if i_idx == j_idx:
             Q[flat_idx] = np.inf
             flat_idx = np.argmin(Q)
             i_idx = flat_idx // m
             j_idx = flat_idx % m
             
        i = nodes[i_idx]
        j = nodes[j_idx]
        
        ri = r[i_idx]
        rj = r[j_idx]
        d_ij = dist[i, j]
        
        len_i = 0.5 * d_ij + (ri - rj) / (2.0 * (m - 2))
        len_j = 0.5 * d_ij - (ri - rj) / (2.0 * (m - 2))
        
        if len_i < 0: len_i = 0.0
        if len_j < 0: len_j = 0.0

        new_node = next_node_id
        next_node_id += 1
        
        new_subtree = f"({subtree[i]}:{len_i:.4f},{subtree[j]}:{len_j:.4f})"
        
        for k in nodes:
            if k == i or k == j:
                continue
            dist[new_node, k] = 0.5 * (dist[i, k] + dist[j, k] - dist[i, j])
            dist[k, new_node] = dist[new_node, k]
            
        nodes.remove(i)
        nodes.remove(j)
        nodes.append(new_node)
        subtree[new_node] = new_subtree

    a, b, c = nodes[0], nodes[1], nodes[2]
    d_ab = dist[a, b]
    d_ac = dist[a, c]
    d_bc = dist[b, c]
    
    len_a = 0.5 * (d_ab + d_ac - d_bc)
    len_b = 0.5 * (d_ab + d_bc - d_ac)
    len_c = 0.5 * (d_ac + d_bc - d_ab)
    
    if len_a < 0: len_a = 0.0
    if len_b < 0: len_b = 0.0
    if len_c < 0: len_c = 0.0

    return f"({subtree[a]}:{len_a:.4f},{subtree[b]}:{len_b:.4f},{subtree[c]}:{len_c:.4f})"


def _get_splits(newick, labels_set):
    """
    Extract the set of non-trivial unrooted splits from a Newick string.
    """
    splits = set()
    n_taxa = len(labels_set)
    
    def parse_with_splits(s, i):
        if i >= len(s):
            return set(), set(), i
        if s[i] == '(':
            i += 1
            taxa = set()
            all_splits = set()
            while True:
                if i >= len(s):
                    break
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
            all_splits.add(frozenset(taxa))
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
    
    canonical = set()
    for sp in all_splits:
        if len(sp) == 0 or len(sp) == n_taxa:
            continue
        other = labels_set - sp
        if len(sp) < len(other):
            canon = frozenset(sp)
        elif len(other) < len(sp):
            canon = frozenset(other)
        else:
            if min(sp) < min(other):
                canon = frozenset(sp)
            else:
                canon = frozenset(other)
        canonical.add(canon)
        
    return canonical


def reconstruct(sequences):
    _init_time()
    
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    n = len(labels)
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
        
    n_sites = len(sequences[labels[0]])
    
    bin_mat = _sequences_to_binary(sequences, labels)
    
    dm = _jc_distance_matrix_binary(bin_mat, n_sites)
    
    single_newick = _nj_newick(labels, dm)
    single_splits = _get_splits(single_newick, set(labels))
    
    t0 = _now()
    _ = _nj_newick(labels, dm)
    nj_time = _now() - t0
    
    rng = np.random.default_rng(42)
    
    bin_mat_3d = bin_mat.reshape(n, n_sites, 4)
    
    time_budget = _time_left()
    time_budget = max(0, time_budget - 5.0)
    
    t2 = _now()
    idx = rng.integers(0, n_sites, size=n_sites)
    boot_mat_3d = bin_mat_3d[:, idx, :]
    boot_mat_flat = boot_mat_3d.reshape(n, n_sites * 4)
    boot_dm = _jc_distance_matrix_binary(boot_mat_flat, n_sites)
    boot_newick = _nj_newick(labels, boot_dm)
    t3 = _now()
    
    per_replicate_time = t3 - t2
    
    max_replicates = int(time_budget / max(per_replicate_time, 0.001))
    max_replicates = min(max_replicates, 100)
    max_replicates = max(1, max_replicates)
    
    splits_counter = Counter()
    tree_scores = []
    
    splits_counter.update(single_splits)
    tree_scores.append((single_newick, single_splits))
    
    for rep in range(max_replicates):
        if _time_left() < 10.0:
            break
            
        idx = rng.integers(0, n_sites, size=n_sites)
        boot_mat_3d = bin_mat_3d[:, idx, :]
        boot_mat_flat = boot_mat_3d.reshape(n, n_sites * 4)
        boot_dm = _jc_distance_matrix_binary(boot_mat_flat, n_sites)
        boot_newick = _nj_newick(labels, boot_dm)
        boot_splits = _get_splits(boot_newick, set(labels))
        
        splits_counter.update(boot_splits)
        tree_scores.append((boot_newick, boot_splits))
        
    best_tree = single_newick
    best_score = -1
    
    for newick, splits in tree_scores:
        score = 0
        for sp in splits:
            score += splits_counter.get(sp, 0)
        if score > best_score:
            best_score = score
            best_tree = newick
            
    return best_tree + ";"

# EVOLVE-BLOCK-END
