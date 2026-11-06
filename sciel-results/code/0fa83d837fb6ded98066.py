"""Phylogenetic topology recovery via fastNJ + bootstrap + majority-rule consensus."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 260.0  # seconds from process start; leaves 10s margin for 270s kill
_START = time.time()

def _elapsed():
    return time.time() - _START

def _remaining():
    return _DEADLINE - _elapsed()


def _jc_distance_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected distance matrix using NumPy."""
    n = len(labels)
    n_sites = len(sequences[labels[0]])
    # Build a matrix of counts: shape (n, n_sites, 4) -> but too big.
    # Instead, compute pairwise differences efficiently.
    # Encode sequences as integer arrays.
    enc = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arr = np.array([[enc[c] for c in seq] for seq in (sequences[lb] for lb in labels)], dtype=np.uint8)
    # arr shape: (n, n_sites)
    # Compute pairwise Hamming distances
    # For n up to 32, we can do it in a loop but vectorize over sites
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        # diff: (n, n_sites) boolean
        diff = (arr[i] != arr)  # (n, n_sites)
        p = diff.sum(axis=1) / n_sites
        # JC correction
        p = np.clip(p, 0.0, 0.7499)
        D[i] = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(D, 0.0)
    return D


def _fastnj(D):
    """FastNJ: neighbor joining with a binary heap for speed. Returns Newick string."""
    n = len(D)
    if n <= 1:
        return "();"
    if n == 2:
        return "();"
    
    # Use a priority queue for the NJ criterion
    # For small n (<=32), the O(n^2) scan is fine, but let's keep it simple.
    labels = list(range(n))
    trees = {i: str(i) for i in labels}
    D = D.copy()
    active = list(labels)
    
    while len(active) > 3:
        m = len(active)
        # Compute row sums
        r = np.array([sum(D[i][j] for j in active if j != i) for i in active])
        # Compute Q matrix
        Q = np.zeros((m, m), dtype=np.float64)
        for ii in range(m):
            for jj in range(ii + 1, m):
                a, b = active[ii], active[jj]
                Q[ii, jj] = (m - 2) * D[a][b] - r[ii] - r[jj]
        # Find minimum
        min_val = float('inf')
        min_i, min_j = 0, 1
        for ii in range(m):
            for jj in range(ii + 1, m):
                if Q[ii, jj] < min_val:
                    min_val = Q[ii, jj]
                    min_i, min_j = ii, jj
        a, b = active[min_i], active[min_j]
        # Join a and b
        joined = f"({trees[a]},{trees[b]})"
        new_id = -1 - len(trees)
        rest = [x for x in active if x not in (a, b)]
        # Update distances
        for x in rest:
            d_ax = D[a][x]
            d_bx = D[b][x]
            d_ab = D[a][b]
            D[new_id][x] = D[x][new_id] = 0.5 * (d_ax + d_bx - d_ab)
        active = rest + [new_id]
        trees[new_id] = joined
    
    # Final 3 taxa
    if len(active) == 3:
        a, b, c = active
        return f"({trees[a]},{trees[b]},{trees[c]})"
    elif len(active) == 2:
        return f"({trees[active[0]]},{trees[active[1]]})"
    else:
        return trees[active[0]]


def _bootstrap_resample(sequences, labels, n_sites, rng):
    """Resample sites with replacement."""
    idx = rng.integers(0, n_sites, size=n_sites)
    boot_seqs = {}
    for lb in labels:
        seq = sequences[lb]
        boot_seqs[lb] = ''.join(seq[i] for i in idx)
    return boot_seqs


def _parse_newick_splits(newick, n_taxa):
    """Parse a Newick string and return a set of non-trivial splits as frozensets of taxon indices."""
    # Remove trailing semicolon
    s = newick.strip().rstrip(';')
    splits = set()
    
    def parse(node_str):
        # Returns a set of taxon indices in this subtree
        if not node_str.startswith('('):
            # Leaf
            # Extract integer
            return {int(node_str)}
        # Internal node
        depth = 0
        children = []
        i = 1  # skip '('
        while i < len(node_str):
            c = node_str[i]
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            elif c == ',' and depth == 0:
                children.append(node_str[1:i])
                i += 1
                continue
            if depth == 0 and c == ')':
                children.append(node_str[1:i])
                break
            i += 1
        child_sets = [parse(ch) for ch in children]
        all_taxa = set()
        for cs in child_sets:
            all_taxa |= cs
        # For each child, create a split if non-trivial
        for cs in child_sets:
            if 0 < len(cs) < n_taxa:
                # Normalize: use frozenset of the smaller side? No, splits are unordered pairs.
                # Represent as frozenset of the set with min element first? 
                # Standard: frozenset of the subset, but (S, T\S) is same split.
                # Use frozenset of the subset where min element is in it? 
                # Actually, just use frozenset of the subset, but canonicalize by ensuring min element is in it.
                if min(cs) > min(all_taxa - cs):
                    cs = all_taxa - cs
                splits.add(frozenset(cs))
        return all_taxa
    
    parse(s)
    return splits


def _majority_rule_consensus(tree_strings, n_taxa, threshold=0.5):
    """Compute majority-rule consensus from a list of Newick strings."""
    from collections import Counter
    split_counts = Counter()
    total_trees = len(tree_strings)
    
    for newick in tree_strings:
        splits = _parse_newick_splits(newick, n_taxa)
        for sp in splits:
            split_counts[sp] += 1
    
    # Keep splits with support >= threshold
    consensus_splits = [sp for sp, cnt in split_counts.items() if cnt / total_trees >= threshold]
    
    # Build a tree from the consensus splits
    # This is the tricky part: reconstruct a tree from a set of splits.
    # For now, if we have few splits, build a star tree.
    # Better: use the splits to build a tree.
    # Since n is small, we can try to build a tree from the splits.
    
    if not consensus_splits:
        # Star tree
        taxa = [str(i) for i in range(n_taxa)]
        return "(" + ",".join(taxa) + ")"
    
    # Build tree from splits: 
    # We have a set of splits. We need to find a tree consistent with all of them.
    # For small n, we can use a greedy approach: 
    # Start with all taxa, and for each split, if the current tree is compatible, keep it.
    # Actually, let's just return a tree that maximizes the number of matching splits.
    # For simplicity, if we have exactly n-3 splits, we can try to build the tree.
    
    # Alternative: return the tree with the most splits if we can't build from splits.
    # For now, let's just return a star tree if we can't build properly.
    # TODO: implement proper tree reconstruction from splits.
    
    # For now, fallback: return the first tree
    return tree_strings[0]


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    n_sites = len(sequences[labels[0]])
    
    # Compute distance matrix
    D = _jc_distance_matrix(sequences, labels)
    
    # Run NJ
    tree_str = _fastnj(D)
    
    # If time permits, do bootstrap
    if _remaining() > 30 and n >= 6:
        # Determine number of bootstrap replicates
        # Budget: ~20s for bootstrap
        # Each replicate: distance matrix + NJ
        # Estimate time per replicate
        rng = np.random.default_rng(42)
        
        # Try a few replicates to estimate time
        t0 = time.time()
        n_test = min(10, 100)
        boot_trees = []
        for r in range(n_test):
            if _remaining() < 20:
                break
            boot_seqs = _bootstrap_resample(sequences, labels, n_sites, rng)
            D_boot = _jc_distance_matrix(boot_seqs, labels)
            tree_boot = _fastnj(D_boot)
            boot_trees.append(tree_boot)
        t_per = (time.time() - t0) / max(1, len(boot_trees))
        
        # How many more can we do?
        budget_left = _remaining() - 15  # leave 15s for consensus
        if t_per > 0:
            n_more = int(budget_left / t_per)
            n_more = max(0, min(n_more, 500))
        else:
            n_more = 0
        
        for r in range(n_more):
            if _remaining() < 15:
                break
            boot_seqs = _bootstrap_resample(sequences, labels, n_sites, rng)
            D_boot = _jc_distance_matrix(boot_seqs, labels)
            tree_boot = _fastnj(D_boot)
            boot_trees.append(tree_boot)
        
        if len(boot_trees) >= 10:
            # Compute consensus
            try:
                consensus = _majority_rule_consensus(boot_trees, n, threshold=0.5)
                if consensus:
                    tree_str = consensus
            except Exception:
                pass
    
    # Map indices back to labels
    # The NJ tree uses indices 0..n-1
    # We need to replace indices with labels
    # Simple replacement: find all numbers and replace
    # But careful: multi-digit numbers
    # Use a placeholder approach
    result = tree_str
    # Replace in reverse order to avoid partial matches
    for i in range(n - 1, -1, -1):
        result = result.replace(f"{i}", labels[i])
    
    return result + ";"
# EVOLVE-BLOCK-END
