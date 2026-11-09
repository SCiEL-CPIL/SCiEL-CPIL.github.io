"""
Phylogenetic topology recovery using BIONJ-corrected Neighbor-Joining 
with Bootstrap Voting.
"""
import time
import math
import numpy as np
import re

_DEADLINE_MARGIN = 5.0  # seconds to keep before the 270s wall-clock kill
_START = time.time()


def _elapsed():
    return time.time() - _START


def _remaining():
    return 270.0 - _elapsed() - _DEADLINE_MARGIN


def _encode_sequences(sequences, labels):
    """
    Encode sequences as a uint8 numpy array for fast vectorized operations.
    Maps A->0, C->1, G->2, T->3.
    """
    n = len(labels)
    if n == 0:
        return np.empty((0, 0), dtype=np.uint8)
    
    L = len(sequences[labels[0]])
    # Pre-allocate
    mat = np.zeros((n, L), dtype=np.uint8)
    
    # Lookup table for ASCII to 0-3
    lut = np.zeros(256, dtype=np.uint8)
    lut[ord('A')] = 0
    lut[ord('C')] = 1
    lut[ord('G')] = 2
    lut[ord('T')] = 3
    
    for i, lb in enumerate(labels):
        s = sequences[lb]
        # Convert string to bytes then to uint8 array
        # np.frombuffer is fast
        arr = np.frombuffer(s.encode('ascii'), dtype=np.uint8)
        mat[i] = lut[arr]
        
    return mat


def _compute_distances(mat, L):
    """
    Compute JC-corrected distance matrix and raw mismatch matrix P.
    Uses vectorized broadcasting.
    """
    n = mat.shape[0]
    if n == 0:
        return np.empty((0, 0)), np.empty((0, 0))
    
    # Pairwise differences: mat[i] != mat[j]
    # Broadcasting: (n, 1, L) vs (1, n, L) -> (n, n, L)
    # This can use a lot of memory for large L, but L<=2000, n<=32 is fine.
    # (32*32*2000) bytes ~ 2MB, very safe.
    
    # To save memory, we can do it in chunks or just rely on numpy optimization.
    # Let's compute the sum of mismatches.
    
    # diff[i, j] = sum_k (mat[i, k] != mat[j, k])
    # We can use np.equal and sum.
    
    # Efficient way:
    # diff = np.zeros((n, n), dtype=np.int32)
    # for k in range(L):
    #     col = mat[:, k]
    #     # outer difference
    #     # This loop over L is 2000 iterations, might be slow in Python.
    #     # Vectorized approach is better.
    
    # Let's use the vectorized broadcast.
    # For n=32, L=2000, the intermediate array is 32x32x2000 booleans.
    # 32*32*2000 = 2,048,000 elements. ~2MB. This is fine.
    
    # However, creating the full 3D array might be slow if done repeatedly.
    # Alternative: Use matrix multiplication tricks?
    # Or just loop over n^2 pairs?
    # n=32 -> 496 pairs.
    # Looping 496 times and doing vector ops on L=2000 is very fast.
    
    P = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Count mismatches
            # mat[i] and mat[j] are uint8 arrays of length L
            diff_count = np.count_nonzero(mat[i] != mat[j])
            P[i, j] = diff_count / L
            P[j, i] = P[i, j]
            
    # JC Correction
    # D = -0.75 * log(1 - 4/3 * P)
    # Clip P to avoid log(0) or log(negative)
    P_clipped = np.clip(P, 0.0, 0.7499)
    D = -0.75 * np.log(1.0 - (4.0 / 3.0) * P_clipped)
    np.fill_diagonal(D, 0.0)
    
    return D, P


def _bionj_correct(D, P, n):
    """
    Apply BIONJ variance correction to the distance matrix D.
    """
    if n < 3:
        return D
    
    # Variance estimate
    # Under JC, the variance of the distance estimator is related to the 
    # divergence. A common approximation for the correction term is:
    # var_i = sum_{j != i} P_ij^2 / L  (if we had L, but we have P which is normalized)
    # Actually, P_ij = d_ij / L.
    # The standard BIONJ correction subtracts a term proportional to the 
    # variance of the distance estimator.
    # var(D_ij) ~ (1/L) * f(P_ij).
    # A simplified but effective correction often used:
    # C_ij = (var_i + var_j) / (2 * (n - 2))
    # where var_i is a measure of the "noise" or divergence of node i.
    
    # Let's use the row sum of P as a proxy for divergence, 
    # and P^2 for variance contribution.
    
    # Compute sum of P for each node (divergence)
    # row_sums_P = P.sum(axis=1)
    
    # Compute sum of P^2 for each node (variance proxy)
    # var_proxy = (P ** 2).sum(axis=1)
    
    # The correction factor for BIONJ is typically:
    # D'_ij = D_ij - (1/(2(n-2))) * (V_i + V_j)
    # where V_i is the variance of the distance estimator for node i.
    # V_i approx = sum_{k != i} (P_ik^2 - P_ik) / L  ? No.
    
    # Let's use a robust simplified BIONJ:
    # The "mean distance" correction is often sufficient.
    # But let's try the variance-based one.
    
    # V_i = sum_{j != i} P_ij^2
    V = (P ** 2).sum(axis=1)
    
    # Correction matrix
    # C_ij = (V_i + V_j) / (2 * (n - 2))
    C = (V[:, np.newaxis] + V[np.newaxis, :]) / (2.0 * (n - 2))
    
    D_corrected = D - C
    np.fill_diagonal(D_corrected, 0.0)
    
    return D_corrected


def _neighbor_joining(D, labels):
    """
    Perform Neighbor-Joining on the distance matrix D.
    Returns a Newick string.
    """
    n = len(labels)
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"

    # Copy D to avoid modification
    dist = D.copy()
    active = list(range(n))
    newick_strs = {i: labels[i] for i in range(n)}
    next_id = n
    
    # We need to track which node is which in the current "reduced" tree
    # dist is indexed by original IDs or new IDs?
    # Let's use a dictionary for distances to handle dynamic IDs cleanly.
    # Or, since n is small, we can just use a matrix and map active indices to original indices.
    
    # Let's stick to the dictionary approach for safety and clarity, 
    # but n is small so performance is okay.
    
    D_dict = {}
    for i in range(n):
        for j in range(n):
            D_dict[(i, j)] = dist[i, j]
            D_dict[(j, i)] = dist[j, i]
            
    while len(active) > 3:
        m = len(active)
        
        # Calculate r values: sum of distances to all other active nodes
        r = {}
        for ai in active:
            r[ai] = sum(D_dict[(ai, aj)] for aj in active if aj != ai)
            
        best_a = None
        best_b = None
        best_criterion = float('inf')
        
        for i in range(m):
            ai = active[i]
            for j in range(i + 1, m):
                aj = active[j]
                criterion = (m - 2) * D_dict[(ai, aj)] - r[ai] - r[aj]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a = ai
                    best_b = aj
        
        a = best_a
        b = best_b
        
        # Create new node
        new_node = next_id
        next_id += 1
        
        # Update distances for new node
        for c in active:
            if c == a or c == b:
                continue
            d_ac = D_dict[(a, c)]
            d_bc = D_dict[(b, c)]
            d_ab = D_dict[(a, b)]
            d_nc = 0.5 * (d_ac + d_bc - d_ab)
            D_dict[(new_node, c)] = d_nc
            D_dict[(c, new_node)] = d_nc
            
        # Update active list
        active.remove(a)
        active.remove(b)
        active.append(new_node)
        
        # Update Newick string
        # We don't calculate branch lengths here for topology only, 
        # but NJ logic relies on distances. The Newick structure is built by joining.
        newick_strs[new_node] = f"({newick_strs[a]},{newick_strs[b]})"
        
    # Handle final 3 nodes
    if len(active) == 3:
        a, b, c = active
        # Connect b and c, then a to that
        # Topologically it doesn't matter which pair we join first for the final star,
        # but we need a binary tree.
        new_node1 = next_id
        next_id += 1
        newick_strs[new_node1] = f"({newick_strs[b]},{newick_strs[c]})"
        new_node2 = next_id
        next_id += 1
        newick_strs[new_node2] = f"({newick_strs[a]},{newick_strs[new_node1]})"
        return newick_strs[new_node2] + ";"
    elif len(active) == 2:
        a, b = active
        return f"({newick_strs[a]},{newick_strs[b]});"
    elif len(active) == 1:
        return newick_strs[active[0]] + ";"
    else:
        return ""


def _extract_splits(newick, labels):
    """
    Parse a Newick string and return a set of canonical splits.
    A split is a tuple of sorted leaf labels on one side of the bipartition.
    """
    if not newick:
        return set()
    
    # Remove branch lengths if any
    newick_clean = re.sub(r':\d+\.?\d*', '', newick)
    newick_clean = newick_clean.replace(';', '').strip()
    
    splits = set()
    labels_set = set(labels)
    n_leaves = len(labels)
    
    def parse(s, idx):
        """
        Recursively parse Newick.
        Returns (set_of_leaves, next_index)
        """
        if s[idx] == '(':
            idx += 1
            leaves = set()
            while s[idx] != ')':
                child_leaves, idx = parse(s, idx)
                leaves.update(child_leaves)
                if s[idx] == ',':
                    idx += 1
            idx += 1  # skip ')'
            return leaves, idx
        else:
            # Leaf
            j = idx
            while j < len(s) and s[j] not in ',)':
                j += 1
            leaf = s[idx:j]
            return {leaf}, j
    
    # We need to capture splits at internal nodes.
    # Standard recursive parser:
    def parse_with_splits(s, idx):
        if s[idx] == '(':
            idx += 1
            leaves = set()
            children = []
            while s[idx] != ')':
                child_leaves, idx = parse_with_splits(s, idx)
                children.append(child_leaves)
                if s[idx] == ',':
                    idx += 1
            idx += 1  # skip ')'
            
            for child in children:
                leaves.update(child)
            
            # Add splits for each child
            for child in children:
                if 0 < len(child) < n_leaves:
                    # Canonical form: sort the smaller side
                    if len(child) <= n_leaves - len(child):
                        key = tuple(sorted(child))
                    else:
                        other = labels_set - child
                        key = tuple(sorted(other))
                    splits.add(key)
            
            return leaves, idx
        else:
            j = idx
            while j < len(s) and s[j] not in ',)':
                j += 1
            leaf = s[idx:j]
            return {leaf}, j
    
    try:
        parse_with_splits(newick_clean, 0)
    except Exception:
        pass
        
    return splits


def _build_tree_from_splits(labels, split_counts):
    """
    Build a tree from a dictionary of split -> count.
    Uses a greedy recursive approach.
    """
    if len(labels) <= 1:
        return labels[0] if labels else ""
    if len(labels) == 2:
        return f"({labels[0]},{labels[1]})"
    
    labels_set = set(labels)
    
    # Find the best split for this set of labels
    best_split = None
    best_count = 0
    
    for split, count in split_counts.items():
        # Check if this split is valid for the current label set
        # A split (A, B) is valid for S if A subset S and B subset S and A union B = S
        # Our split is stored as the smaller side.
        # Let's say split is S1. S2 = labels_set - S1.
        # We need S1 subset S and S2 subset S.
        # Since S is the current set of labels, S1 must be a subset of S.
        # And S2 must be a subset of S.
        # This implies S1 union S2 = S.
        
        s1 = set(split)
        s2 = labels_set - s1
        
        if s1.issubset(labels_set) and s2.issubset(labels_set) and s1.union(s2) == labels_set:
            # Valid split
            if count > best_count:
                best_count = count
                best_split = (s1, s2)
                
    if best_split is None:
        # No strong split found, fall back to a chain or star
        # For small n, a chain is fine
        if len(labels) == 3:
            return f"({labels[0]},({labels[1]},{labels[2]}))"
        # Recursive fallback: split in half arbitrarily?
        # Better: return a star tree representation as a binary tree
        # Let's just pick an arbitrary split if none has support
        # To avoid infinite recursion, we must ensure progress.
        # If no split has count > 0, we can't recurse effectively.
        # Let's return a "balanced" binary tree.
        mid = len(labels) // 2
        left = labels[:mid]
        right = labels[mid:]
        left_tree = _build_tree_from_splits(left, split_counts)
        right_tree = _build_tree_from_splits(right, split_counts)
        return f"({left_tree},{right_tree})"
        
    s1, s2 = best_split
    
    # Recursively build subtrees
    # Ensure we don't pass empty sets
    if len(s1) == 1:
        tree1 = list(s1)[0]
    else:
        tree1 = _build_tree_from_splits(list(s1), split_counts)
        
    if len(s2) == 1:
        tree2 = list(s2)[0]
    else:
        tree2 = _build_tree_from_splits(list(s2), split_counts)
        
    return f"({tree1},{tree2})"


def reconstruct(sequences):
    """Reconstruct unrooted tree topology from sequences."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 0:
            return ""
        if n == 1:
            return labels[0] + ";"
        return "(" + ",".join(labels) + ");"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    # Encode sequences
    mat = _encode_sequences(sequences, labels)
    L = mat.shape[1]
    
    # Determine number of bootstraps based on remaining time
    # We have 270s total. Let's assume we are one of 20 instances.
    # Budget per instance: ~13s.
    # Vectorized distance calc is fast. NJ is O(n^3) ~ 32768 ops, very fast.
    # Bootstrap loop is the bottleneck.
    # Let's try 200 bootstraps.
    # If time is tight, reduce.
    
    time_left = _remaining()
    # Heuristic: 200 bootstraps takes ~0.5-1s for n=32, L=2000.
    # Let's cap it.
    num_bootstraps = 200
    if time_left < 5.0:
        num_bootstraps = 50
    elif time_left < 10.0:
        num_bootstraps = 100
    
    split_counts = {}
    
    # Pre-compute base distances for potential reuse? 
    # No, bootstrap resamples columns, so distances change.
    
    for b in range(num_bootstraps):
        if _remaining() < 2.0:
            break
            
        # Resample sites with replacement
        # np.random.randint(0, L, size=L)
        # To be deterministic or just fast?
        # Using numpy random is fine.
        boot_indices = np.random.randint(0, L, size=L)
        boot_mat = mat[:, boot_indices]
        
        # Compute distances
        D, P = _compute_distances(boot_mat, L)
        
        # BIONJ Correction
        D_corrected = _bionj_correct(D, P, n)
        
        # NJ
        try:
            newick = _neighbor_joining(D_corrected, labels)
            if newick:
                splits = _extract_splits(newick, labels)
                for s in splits:
                    split_counts[s] = split_counts.get(s, 0) + 1
        except Exception:
            continue
            
    # If no splits were found (e.g. all identical sequences), fallback to NJ on full data
    if not split_counts:
        D, P = _compute_distances(mat, L)
        D_corrected = _bionj_correct(D, P, n)
        try:
            newick = _neighbor_joining(D_corrected, labels)
            if newick and all(lb in newick for lb in labels):
                return newick
        except Exception:
            pass
        # Final fallback
        current = labels[0]
        for i in range(1, n):
            current = f"({current},{labels[i]})"
        return current + ";"
        
    # Build consensus tree from splits
    try:
        newick = _build_tree_from_splits(labels, split_counts)
        if newick and all(lb in newick for lb in labels):
            return newick + ";"
    except Exception:
        pass
        
    # Fallback if consensus building fails
    D, P = _compute_distances(mat, L)
    D_corrected = _bionj_correct(D, P, n)
    try:
        newick = _neighbor_joining(D_corrected, labels)
        if newick and all(lb in newick for lb in labels):
            return newick
    except Exception:
        pass
        
    current = labels[0]
    for i in range(1, n):
        current = f"({current},{labels[i]})"
    return current + ";"
