"""Phylogenetic topology recovery via BIONJ-corrected Neighbor-Joining with bootstrap voting."""
import time
import math
import numpy as np
import re

_DEADLINE_FRACTION = 0.90
_start_time = None


def _encode_sequences(sequences, labels):
    """Encode sequences as uint8 array for fast operations."""
    enc = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([[enc[c] for c in s] for s in (sequences[lab] for lab in labels)], dtype=np.uint8)
    return seqs


def _jc_distance_matrix(seqs):
    """Compute JC-corrected distance matrix from encoded sequences."""
    n = seqs.shape[0]
    L = seqs.shape[1]
    
    # Pairwise differences using broadcasting
    diff = (seqs[:, np.newaxis, :] != seqs[np.newaxis, :, :])
    pmat = diff.sum(axis=2).astype(np.float64) / L
    np.fill_diagonal(pmat, 0.0)
    
    # JC correction
    pmat_clipped = np.clip(pmat, 0, 0.7499)
    dmat = -0.75 * np.log(1.0 - 4.0 * pmat_clipped / 3.0)
    np.fill_diagonal(dmat, 0.0)
    return dmat, pmat


def _bionj_correct(dmat, pmat, n, L):
    """Apply BIONJ variance correction to distance matrix."""
    # BIONJ correction: d_ij_corrected = d_ij - (sigma_i^2 + sigma_j^2) / (2*(n-2))
    # where sigma_i^2 = sum_j (p_ij - p_i_bar)^2 / L approximately
    # More standard: sigma_i^2 = (sum_j p_ij^2 - (sum_j p_ij)^2/(n-1)) / L
    
    # Compute row sums and row sum of squares
    row_sums = pmat.sum(axis=1)
    row_sq_sums = (pmat ** 2).sum(axis=1)
    
    # Variance estimate for each node
    # sigma_i^2 = (row_sq_sums[i] - row_sums[i]^2 / (n-1)) / L
    var = (row_sq_sums - row_sums ** 2 / max(n - 1, 1)) / max(L, 1)
    var = np.clip(var, 0, None)
    
    # Correction matrix
    correction = (var[:, np.newaxis] + var[np.newaxis, :]) / (2.0 * max(n - 2, 1))
    dmat_corrected = dmat - correction
    np.fill_diagonal(dmat_corrected, 0.0)
    
    return dmat_corrected


def _neighbor_joining(dmat, labels):
    """
    Perform Neighbor-Joining to reconstruct a tree.
    Returns a Newick string.
    """
    n = len(labels)
    if n <= 1:
        return labels[0] if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Use a dictionary-based NJ for clarity
    # Map labels to indices for faster array operations
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    idx_to_label = {i: lab for i, lab in enumerate(labels)}
    
    # Active nodes: start with all labels
    active = list(range(n))
    # Distance matrix stored as dict of dicts or as a 2D array with dynamic updates
    # For efficiency, use a dict of dicts
    dist = {}
    for i in range(n):
        for j in range(n):
            dist[(i, j)] = dmat[i, j]
    
    tree_nodes = {}  # node_id -> (child1, child2, len1, len2)
    next_node_id = n  # Start internal node IDs from n
    
    while len(active) > 2:
        m = len(active)
        
        # Compute r_i = sum of distances from i to all other active nodes
        r = np.zeros(m)
        for idx_i, i in enumerate(active):
            for idx_j, j in enumerate(active):
                if i != j:
                    r[idx_i] += dist[(i, j)]
        
        # Compute Q matrix and find minimum
        min_q = float('inf')
        i_node = None
        j_node = None
        
        for idx_i, i in enumerate(active):
            for idx_j, j in enumerate(active):
                if i == j:
                    continue
                q = (m - 2) * dist[(i, j)] - r[idx_i] - r[idx_j]
                if q < min_q:
                    min_q = q
                    i_node = i
                    j_node = j
        
        if i_node is None:
            break
        
        d_ij = dist[(i_node, j_node)]
        # Find indices in active
        idx_i = active.index(i_node)
        idx_j = active.index(j_node)
        
        li = 0.5 * d_ij + (r[idx_i] - r[idx_j]) / (2 * (m - 2))
        lj = 0.5 * d_ij - (r[idx_i] - r[idx_j]) / (2 * (m - 2))
        
        w_id = next_node_id
        next_node_id += 1
        
        tree_nodes[w_id] = (i_node, j_node, li, lj)
        
        # Update distances for w
        for k in active:
            if k == i_node or k == j_node:
                continue
            d_wk = 0.5 * (dist[(i_node, k)] + dist[(j_node, k)] - d_ij)
            dist[(w_id, k)] = d_wk
            dist[(k, w_id)] = d_wk
        
        # Remove i, j and add w
        active.remove(i_node)
        active.remove(j_node)
        active.append(w_id)
    
    # Connect the last two
    if len(active) == 2:
        u, v = active
        d_uv = dist.get((u, v), 0.0)
        w_id = next_node_id
        next_node_id += 1
        tree_nodes[w_id] = (u, v, d_uv / 2.0, d_uv / 2.0)
        active = [w_id]
    
    # Build Newick string
    leaf_set = set(range(n))  # Leaf node IDs are 0..n-1
    
    def build_newick(node):
        if node in leaf_set:
            return labels[node]
        if node not in tree_nodes:
            return labels[node % n]  # Fallback
            
        child1, child2, len1, len2 = tree_nodes[node]
        newick1 = build_newick(child1)
        newick2 = build_newick(child2)
        return f"({newick1}:{len1:.4f},{newick2}:{len2:.4f})"
        
    root = active[0]
    return build_newick(root)


def _star_tree(labels):
    """Fallback: star tree."""
    if len(labels) <= 1:
        return labels[0] if labels else ""
    if len(labels) == 2:
        return f"({labels[0]},{labels[1]})"
    return "(" + ",".join(labels) + ")"


def _extract_splits(newick, labels):
    """Parse a Newick string into a set of unrooted splits (as frozensets)."""
    # Remove branch lengths and semicolon
    newick_clean = re.sub(r':[0-9.eE+-]+', '', newick)
    newick_clean = newick_clean.replace(';', '').strip()
    
    splits = set()
    labels_set = set(labels)
    
    def parse(s, idx):
        if s[idx] != '(':
            # Leaf
            j = idx
            while j < len(s) and s[j] not in ',)':
                j += 1
            leaf = s[idx:j]
            return {leaf}, j
        
        # Internal node
        children = []
        idx += 1
        while s[idx] != ')':
            child, idx = parse(s, idx)
            children.append(child)
            if s[idx] == ',':
                idx += 1
        idx += 1  # skip ')'
        
        combined = set()
        for child in children:
            combined.update(child)
        
        # For each child, add the split
        for child in children:
            if 0 < len(child) < len(labels):
                # Normalize: use the smaller side
                if len(child) <= len(labels) - len(child):
                    splits.add(frozenset(child))
                else:
                    other = labels_set - child
                    splits.add(frozenset(other))
        
        return combined, idx
    
    parse(newick_clean, 0)
    return splits


def _topology_key(splits):
    """Create a hashable key from a set of splits."""
    return frozenset(splits)


def reconstruct(sequences):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Time check
    elapsed = time.time() - _start_time
    remaining = _DEADLINE_FRACTION * 270.0 - elapsed
    if remaining < 5.0:
        return _star_tree(labels)
    
    # Encode sequences
    seqs = _encode_sequences(sequences, labels)
    L = seqs.shape[1]
    
    # 1. Compute Distance Matrix
    dmat, pmat = _jc_distance_matrix(seqs)
    
    # 2. Apply BIONJ correction
    dmat_corrected = _bionj_correct(dmat, pmat, n, L)
    
    # 3. Run NJ on corrected distances
    try:
        tree = _neighbor_joining(dmat_corrected, labels)
    except Exception:
        try:
            tree = _neighbor_joining(dmat, labels)
        except Exception:
            tree = _star_tree(labels)
    
    # 4. Bootstrap voting if time permits
    elapsed = time.time() - _start_time
    remaining = _DEADLINE_FRACTION * 270.0 - elapsed
    
    # For small trees, bootstrap is less helpful; skip if n < 6
    if n >= 6 and remaining > 1.0:
        # Estimate time per bootstrap iteration
        # Distance matrix: O(n^2 * L), NJ: O(n^3)
        # For n=32, L=2000: ~2-5ms per iteration
        # Budget: use up to 40% of remaining time for bootstrap
        boot_budget = remaining * 0.4
        if boot_budget > 0.2:
            # Estimate number of bootstraps
            # Rough: 3ms per bootstrap for small, 10ms for large
            est_time_per_boot = 0.003 * (n / 10.0) + 0.005 * (L / 1000.0)
            num_bootstraps = int(boot_budget / est_time_per_boot)
            num_bootstraps = max(20, min(num_bootstraps, 200))
            
            # Run bootstrap
            topology_counts = {}
            topology_newick = {}
            
            for b in range(num_bootstraps):
                # Time check
                elapsed = time.time() - _start_time
                if elapsed > _DEADLINE_FRACTION * 270.0 - 0.5:
                    break
                
                # Resample sites with replacement
                boot_indices = np.random.randint(0, L, size=L)
                boot_seqs = seqs[:, boot_indices]
                
                # Compute distance matrix
                dmat_boot, pmat_boot = _jc_distance_matrix(boot_seqs)
                
                # Apply BIONJ correction
                dmat_corrected_boot = _bionj_correct(dmat_boot, pmat_boot, n, L)
                
                # Run NJ
                try:
                    tree_boot = _neighbor_joining(dmat_corrected_boot, labels)
                except Exception:
                    continue
                
                # Extract splits and create key
                splits = _extract_splits(tree_boot, labels)
                key = _topology_key(splits)
                
                if key not in topology_counts:
                    topology_counts[key] = 0
                    topology_newick[key] = tree_boot
                topology_counts[key] += 1
            
            # Find the most common topology
            if topology_counts:
                best_key = max(topology_counts, key=topology_counts.get)
                # Only use the bootstrapped tree if it has reasonable support
                # or if it's different from the base tree (might be better)
                base_splits = _extract_splits(tree, labels)
                base_key = _topology_key(base_splits)
                
                # Use the most common topology if it has >50% support
                # or if it's the same as base
                if topology_counts[best_key] > num_bootstraps * 0.5 or best_key == base_key:
                    tree = topology_newick[best_key]
    
    return tree
