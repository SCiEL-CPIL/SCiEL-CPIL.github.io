"""Improved phylogenetic topology recovery: NJ + RaxML-style NNI polish."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE_SECONDS = 260.0
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _time_left():
    return _DEADLINE_SECONDS - _elapsed()


def _jc_corrected(p):
    """Jukes-Cantor distance correction."""
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Compute JC-corrected pairwise distance matrix using numpy for speed."""
    n = len(labels)
    # Convert sequences to numeric arrays: A=0, C=1, G=2, T=3
    seq_dict = {}
    for lab in labels:
        s = sequences[lab]
        arr = np.empty(len(s), dtype=np.uint8)
        # Vectorized conversion
        arr[:] = 0
        arr[np.frombuffer(s.encode(), dtype=np.uint8) == ord('C')] = 1
        arr[np.frombuffer(s.encode(), dtype=np.uint8) == ord('G')] = 2
        arr[np.frombuffer(s.encode(), dtype=np.uint8) == ord('T')] = 3
        seq_dict[lab] = arr

    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            ai = seq_dict[labels[i]]
            aj = seq_dict[labels[j]]
            p = np.count_nonzero(ai != aj) / len(ai)
            d = _jc_corrected(p)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _newick_from_splits(splits, n):
    """
    Reconstruct a Newick string from a set of splits.
    Each split is a frozenset of taxon indices (subset of {0,...,n-1}).
    We use a greedy construction: repeatedly find a split that separates
    the current set into two parts, build subtrees recursively.
    """
    all_taxa = frozenset(range(n))
    # Filter out trivial splits (size 1 or n-1)
    non_trivial = [s for s in splits if 1 < len(s) < n - 1]
    # Also include all singletons implicitly; the algorithm below handles them.
    
    def build(taxa_set):
        if len(taxa_set) == 1:
            return f"t{list(taxa_set)[0]}"
        if len(taxa_set) == 2:
            a, b = sorted(taxa_set)
            return f"(t{a},t{b})"
        
        # Find a split that divides taxa_set into two non-empty proper subsets
        best_split = None
        best_balance = -1
        for s in non_trivial:
            left = s & taxa_set
            right = taxa_set - left
            if len(left) >= 1 and len(right) >= 1:
                balance = min(len(left), len(right))
                if balance > best_balance:
                    best_balance = balance
                    best_split = (left, right)
        
        if best_split is None:
            # Star polytomy fallback
            taxa_list = sorted(taxa_set)
            return "(" + ",".join(f"t{t}" for t in taxa_list) + ")"
        
        left, right = best_split
        left_nw = build(left)
        right_nw = build(right)
        return f"({left_nw},{right_nw})"
    
    return build(all_taxa) + ";"


def _extract_splits_from_newick(newick, n):
    """Extract all non-trivial splits from a Newick string."""
    # Parse the Newick into a tree structure
    # For simplicity, we'll parse recursively
    splits = set()
    
    def parse(s, pos):
        """Parse Newick starting at position pos. Returns (node, new_pos)."""
        if s[pos] == '(':
            pos += 1
            children = []
            while True:
                child, pos = parse(s, pos)
                children.append(child)
                if pos < len(s) and s[pos] == ',':
                    pos += 1
                elif pos < len(s) and s[pos] == ')':
                    pos += 1
                    break
            # Extract taxa from children
            taxa = set()
            for child in children:
                taxa |= _get_taxa(child)
            if 1 < len(taxa) < n - 1:
                splits.add(frozenset(taxa))
            return (children, taxa), pos
        else:
            # Taxon name
            end = pos
            while end < len(s) and s[end] not in ',):':
                end += 1
            name = s[pos:end]
            # Parse branch length if present
            if end < len(s) and s[end] == ':':
                end += 1
                while end < len(s) and s[end] not in ',):':
                    end += 1
            taxon_idx = int(name[1:])
            return (None, {taxon_idx}), end
    
    _parse, _ = parse(newick, 0)
    return splits


def _get_taxa(node):
    """Get all taxon indices in a parsed node."""
    if node is None:
        return set()
    if isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], set):
        return node[1]
    return set()


def _nni_polish(sequences, labels, dist_matrix, newick_str, time_budget=30.0):
    """
    NNI polish on the tree represented by newick_str.
    We work with a split-based representation for simplicity.
    """
    n = len(labels)
    if n < 4:
        return newick_str
    
    # Get initial splits
    current_splits = _extract_splits_from_newick(newick_str, n)
    
    # Precompute site counts for NNI gain computation
    # Convert sequences to numeric
    seq_arrs = []
    for lab in labels:
        s = sequences[lab]
        arr = np.empty(len(s), dtype=np.uint8)
        arr[:] = 0
        arr[np.frombuffer(s.encode(), dtype=np.uint8) == ord('C')] = 1
        arr[np.frombuffer(s.encode(), dtype=np.uint8) == ord('G')] = 2
        arr[np.frombuffer(s.encode(), dtype=np.uint8) == ord('T')] = 3
        seq_arrs.append(arr)
    seq_matrix = np.array(seq_arrs)  # shape (n, L)
    L = len(sequences[labels[0]])
    
    # Log-likelihood of a split under JC model
    # For a split S, the likelihood contribution is based on the number of
    # taxa on each side and the observed pattern frequencies.
    # Simplified: use the distance-based NJ criterion as a surrogate.
    
    def split_likelihood(s):
        """Approximate likelihood contribution of a split under JC."""
        left = s
        right = frozenset(range(n)) - s
        if len(left) < 2 or len(right) < 2:
            return 0.0
        # Sum of distances within left, within right, and between
        # Use the fact that NNI improves if it reduces the total weighted distance
        # This is a heuristic surrogate
        idx_left = list(left)
        idx_right = list(right)
        # Average distance within each side
        if len(idx_left) >= 2:
            d_left = np.mean(dist_matrix[np.ix_(idx_left, idx_left)])
        else:
            d_left = 0.0
        if len(idx_right) >= 2:
            d_right = np.mean(dist_matrix[np.ix_(idx_right, idx_right)])
        else:
            d_right = 0.0
        d_between = np.mean(dist_matrix[np.ix_(idx_left, idx_right)])
        # We want to minimize total distance
        return -(d_left + d_right + 2 * d_between)
    
    current_ll = sum(split_likelihood(s) for s in current_splits)
    
    best_newick = newick_str
    improved = True
    start = time.time()
    
    while improved and (time.time() - start) < time_budget:
        improved = False
        # Try NNI moves on each split
        for s in list(current_splits):
            left = s
            right = frozenset(range(n)) - s
            if len(left) < 2 or len(right) < 2:
                continue
            # NNI: try moving one taxon from left to right or vice versa
            # For each taxon in left, try moving it to right
            for t in left:
                new_left = left - {t}
                new_right = right | {t}
                if len(new_left) < 2 or len(new_right) < 2:
                    continue
                new_s = frozenset(new_left)
                # Remove old split, add new split
                new_splits = set(current_splits)
                new_splits.discard(s)
                new_splits.add(new_s)
                new_ll = sum(split_likelihood(ns) for ns in new_splits)
                if new_ll > current_ll + 1e-10:
                    current_splits = new_splits
                    current_ll = new_ll
                    improved = True
                    best_newick = _newick_from_splits(current_splits, n)
                    break
            if improved:
                break
    
    if improved:
        return _newick_from_splits(current_splits, n)
    return best_newick


def reconstruct(sequences):
    """
    Reconstruct unrooted tree topology from sequences.
    Uses NJ with JC correction, followed by NNI polish.
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 3:
        if n == 2:
            return f"({labels[0]},{labels[1]});"
        elif n == 1:
            return f"{labels[0]};"
        else:
            return f"({','.join(labels)});"
    
    # Compute distance matrix
    dist = _compute_distance_matrix(sequences, labels)
    
    # Neighbor Joining
    nodes = list(range(n))  # indices into labels
    trees = {i: f"t{i}" for i in range(n)}
    # dist is n x n, we'll maintain a dynamic version
    d = dist.copy()
    
    next_id = n
    while len(nodes) > 3:
        if _time_left() < 5.0:
            break
        m = len(nodes)
        # Compute row sums
        total = np.zeros(m)
        for i in range(m):
            for j in range(m):
                if i != j:
                    total[i] += d[nodes[i], nodes[j]]
        
        best_criterion = float('inf')
        best_i = best_j = -1
        for i in range(m):
            for j in range(i + 1, m):
                criterion = (m - 2) * d[nodes[i], nodes[j]] - total[i] - total[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i, best_j = i, j
        
        a, b = nodes[best_i], nodes[best_j]
        joined = f"({trees[a]},{trees[b]})"
        new_node = next_id
        next_id += 1
        
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            d[new_node, x] = d[x, new_node] = 0.5 * (d[a, x] + d[b, x] - d[a, b])
        
        nodes = rest + [new_node]
        trees[new_node] = joined
    
    # Build final Newick
    if len(nodes) == 3:
        result = "(" + ",".join(trees[x] for x in nodes) + ");"
    else:
        result = "(" + ",".join(trees[x] for x in nodes) + ");"
    
    # NNI polish if time permits
    if _time_left() > 15.0 and n >= 4:
        time_budget = min(_time_left() - 10.0, 40.0)
        if time_budget > 5.0:
            result = _nni_polish(sequences, labels, dist, result, time_budget)
    
    # Final safety check
    if _time_left() < 5.0:
        return result
    
    return result
# EVOLVE-BLOCK-END
