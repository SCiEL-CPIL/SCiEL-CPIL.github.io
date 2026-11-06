"""Neighbour Joining with Jukes-Cantor correction, plus bootstrap consensus refinement."""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START
_START = None
_DEADLINE = 265.0


def _now():
    return time.time()


def _init_time():
    global _START
    if _START is None:
        _START = time.time()


def _time_left():
    _init_time()
    return max(0.0, _DEADLINE - (time.time() - _START))


def _seqs_to_array(sequences):
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    seqs = [sequences[l] for l in labels]
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arr = np.array([[mapping[c] for c in s] for s in seqs], dtype=np.int8)
    return labels, arr


def _jc_matrix(arr):
    """Build pairwise JC distance matrix from integer array."""
    n, L = arr.shape
    if n == 0 or L == 0:
        return np.zeros((n, n), dtype=np.float64)
    # Vectorized computation of pairwise differences
    diff = (arr[:, None, :] != arr[None, :, :])
    counts = diff.sum(axis=2)
    p = counts / L
    p_safe = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1 - (4.0 / 3.0) * p_safe)
    np.fill_diagonal(d, 0)
    return d


def _nj_tree(dist, labels):
    """Neighbour Joining on a distance matrix. Returns Newick string."""
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    all_leaves = set(labels)
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    
    # Use a dictionary of dictionaries for distance storage
    d = {}
    for i in range(n):
        d[i] = {}
        for j in range(n):
            d[i][j] = dist[i][j]
    
    while len(nodes) > 3:
        if _time_left() < 5.0:
            # Fallback: star tree
            return "(" + ",".join(trees[i] for i in nodes) + ");"
        
        n_curr = len(nodes)
        # Compute row sums
        total = {}
        for a in nodes:
            total[a] = sum(d[a][b] for b in nodes if b != a)
        
        # Find pair with minimum Q criterion
        best_criterion = float('inf')
        best_a, best_b = -1, -1
        
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                criterion = (n_curr - 2) * d[a][b] - total[a] - total[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a, best_b = a, b
        
        a, b = best_a, best_b
        
        # Create new node
        new_id = -1 - len(nodes)
        trees[new_id] = f"({trees[a]},{trees[b]})"
        
        # Update distances to new node
        d[new_id] = {}
        for x in nodes:
            if x == a or x == b:
                continue
            d[new_id][x] = 0.5 * (d[a][x] + d[b][x] - d[a][b])
            d[x][new_id] = d[new_id][x]
        
        # Remove old nodes, add new
        nodes.remove(a)
        nodes.remove(b)
        nodes.append(new_id)
    
    # Build final tree from remaining 3 nodes
    final_tree = "(" + ",".join(trees[x] for x in nodes) + ");"
    return final_tree


def _parse_newick_splits(newick):
    """Parse Newick string and extract all non-trivial splits as frozensets.
    Returns a set of frozensets, each representing one side of a split.
    """
    newick = newick.rstrip(';')
    
    # Tokenize
    tokens = []
    i = 0
    while i < len(newick):
        if newick[i] in '(),':
            tokens.append(newick[i])
            i += 1
        else:
            j = i
            while j < len(newick) and newick[j] not in '(),:':
                j += 1
            tokens.append(newick[i:j])
            i = j
    
    # Parse tree recursively
    pos = [0]
    
    def parse():
        if tokens[pos[0]] == '(':
            pos[0] += 1
            left = parse()
            # Skip comma
            if pos[0] < len(tokens) and tokens[pos[0]] == ',':
                pos[0] += 1
            right = parse()
            # Skip closing paren
            if pos[0] < len(tokens) and tokens[pos[0]] == ')':
                pos[0] += 1
            # Check for label
            label = ''
            if pos[0] < len(tokens) and tokens[pos[0]] not in (',', ')', ';'):
                label = tokens[pos[0]]
                pos[0] += 1
            return ('internal', left, right, label)
        else:
            label = tokens[pos[0]]
            pos[0] += 1
            return ('leaf', label)
    
    root = parse()
    
    # Extract splits
    all_leaves = set()
    def collect_leaves(node):
        if node[0] == 'leaf':
            all_leaves.add(node[1])
        else:
            collect_leaves(node[1])
            collect_leaves(node[2])
    collect_leaves(root)
    
    splits = set()
    
    def get_leaves(node):
        if node[0] == 'leaf':
            return {node[1]}
        else:
            return get_leaves(node[1]) | get_leaves(node[2])
    
    def extract_splits(node):
        if node[0] == 'leaf':
            return
        left = node[1]
        right = node[2]
        
        left_leaves = get_leaves(left)
        right_leaves = get_leaves(right)
        
        # For this internal node, the split is (left_leaves, right_leaves)
        # Store the smaller side for canonical representation
        if len(left_leaves) <= len(right_leaves):
            split = frozenset(left_leaves)
        else:
            split = frozenset(right_leaves)
        
        # Only add non-trivial splits (size > 1 and < n-1)
        if 1 < len(split) < len(all_leaves) - 1:
            splits.add(split)
        
        extract_splits(left)
        extract_splits(right)
    
    extract_splits(root)
    return splits


def _rf_distance(s1, s2):
    """Compute Robinson-Foulds distance between two sets of splits."""
    return len(s1.symmetric_difference(s2))


def reconstruct(sequences):
    """Reconstruct unrooted tree topology from DNA sequences using NJ with bootstrap consensus."""
    _init_time()
    
    labels, arr = _seqs_to_array(sequences)
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Build full distance matrix
    d_full = _jc_matrix(arr)
    
    # Run NJ on full data
    tree_full = _nj_tree(d_full, labels)
    
    # If not enough time for bootstrap, return full tree
    if _time_left() < 15.0:
        return tree_full
    
    L = arr.shape[1]
    
    # Run bootstrap replicates
    candidates = []
    candidates.append(tree_full)
    
    max_bootstraps = 50
    num_bootstraps = 0
    
    while num_bootstraps < max_bootstraps and _time_left() > 10.0:
        # Resample sites with replacement
        indices = np.random.randint(0, L, L)
        arr_boot = arr[:, indices]
        
        # Build bootstrap distance matrix
        d_boot = _jc_matrix(arr_boot)
        
        # Run NJ on bootstrap data
        tree_boot = _nj_tree(d_boot, labels)
        candidates.append(tree_boot)
        num_bootstraps += 1
    
    if len(candidates) <= 1:
        return candidates[0]
    
    # Find consensus tree by minimizing average RF distance to all candidates
    # First, compute splits for all candidates
    splits_list = []
    for tree in candidates:
        splits = _parse_newick_splits(tree)
        splits_list.append(splits)
    
    # Find candidate with minimum average RF distance to others
    best_idx = 0
    best_score = float('inf')
    
    for i in range(len(candidates)):
        score = 0
        for j in range(len(candidates)):
            if i != j:
                score += _rf_distance(splits_list[i], splits_list[j])
        score /= (len(candidates) - 1)
        if score < best_score:
            best_score = score
            best_idx = i
    
    return candidates[best_idx]
# EVOLVE-BLOCK-END
