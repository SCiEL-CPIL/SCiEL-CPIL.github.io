"""Phylogenetic topology recovery via Fast Distance-based Quartet (FDQ) agglomerative reconstruction."""
import time
import math
import numpy as np

_DEADLINE_FRACTION = 0.92
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
    
    # Pairwise differences
    diff = (seqs[:, np.newaxis, :] != seqs[np.newaxis, :, :])
    pmat = diff.sum(axis=2).astype(np.float64) / L
    np.fill_diagonal(pmat, 0.0)
    
    # JC correction
    pmat_clipped = np.clip(pmat, 0, 0.7499)
    dmat = -0.75 * np.log(1.0 - 4.0 * pmat_clipped / 3.0)
    np.fill_diagonal(dmat, 0.0)
    return dmat, pmat


def _bionj_correct(dmat, pmat, n):
    """Apply BIONJ variance correction to distance matrix."""
    var = (pmat ** 2).sum(axis=1) / max(pmat.shape[1], 1)
    correction = (var[:, np.newaxis] + var[np.newaxis, :]) / (2.0 * max(n - 2, 1))
    dmat_corrected = dmat - correction
    np.fill_diagonal(dmat_corrected, 0.0)
    return dmat_corrected


def _quartet_score_matrix(dmat, n):
    """
    Compute quartet scores for all pairs of taxa.
    
    For each pair (i, j), compute the total quartet support score:
    sum over all other pairs (k, l) of the quartet score for (i, j, k, l).
    
    The quartet score for (a, b, c, d) is:
    (d_ac + d_bd) - (d_ab + d_cd)
    
    A positive value supports the topology (a|b)(c|d) vs (a|c)(b|d), etc.
    For pair (i,j), we want to know which other taxa are "close" to i and j
    in a way that supports grouping them together.
    
    We compute for each pair (i,j):
    S(i,j) = sum_{k,l != i,j, k<l} [ (d_ik + d_jl) - (d_ij + d_kl) ]
    
    This is the total quartet support for the split that groups i,j together
    against k,l being on the other side.
    """
    n = len(dmat)
    # For each pair (i,j), compute the quartet score sum
    # S(i,j) = sum_{k != i,j} sum_{l != i,j,k} [ (d_ik + d_jl) - (d_ij + d_kl) ] / 2
    # Simplify: S(i,j) = sum_k (d_ik + sum_l d_jl - d_ij - sum_l d_kl) / 2 for k != i,j
    # = sum_k (d_ik - d_kl_sum_part) ... let's just compute directly
    
    # Efficient computation:
    # For pair (i,j):
    # S(i,j) = 0.5 * sum_{k != i,j} [ d_ik - d_kj + (sum_{l != i,j,k} d_jl) - (sum_{l != i,j,k} d_kl) ]
    # This is getting complex. Let's use a simpler formulation.
    
    # The standard quartet score for pair (i,j) in the context of all other taxa:
    # For each other pair (k,l) with k,l != i,j:
    #   score = (d_ik + d_jl) - (d_ij + d_kl)
    #   This supports (i|k)(j|l) vs (i|j)(k|l)
    #   If score < 0, it supports (i,j) being together vs (k,l)
    
    # Total support for (i,j) grouping:
    # S(i,j) = sum_{k,l: k<l, k,l != i,j} [ (d_ij + d_kl) - (d_ik + d_jl) ]
    #        = sum_{k,l} d_ij + sum_{k,l} d_kl - sum_{k,l} d_ik - sum_{k,l} d_jl
    
    # Let m = n - 2 (number of other taxa)
    # Number of pairs (k,l) with k<l from m taxa: m*(m-1)/2
    # S(i,j) = m*(m-1)/2 * d_ij + sum_{k<l, k,l != i,j} d_kl - sum_{k != i,j} sum_{l != i,j,k} d_ik / 2 - sum_{k != i,j} sum_{l != i,j,k} d_jl / 2
    #        = m*(m-1)/2 * d_ij + sum_{k<l, k,l != i,j} d_kl - (m-1)/2 * sum_{k != i,j} d_ik - (m-1)/2 * sum_{k != i,j} d_jl
    
    # Precompute row sums
    row_sums = dmat.sum(axis=1)
    
    # sum_{k<l, k,l != i,j} d_kl for all pairs (i,j)
    # total_sum_all = sum_{a<b} d_ab
    # For given (i,j), sum_{k<l, k,l != i,j} d_kl = total_sum_all - sum_{k != i,j} d_ik - sum_{k != i,j} d_jk + d_ij
    # Wait, that's not quite right. Let me think again.
    # sum_{k<l, k,l not in {i,j}} d_kl = (sum over all pairs (a,b) with a<b) d_ab 
    #   - (sum over pairs involving i or j)
    # Pairs involving i: (i,k) for k != i, i.e., d_ik for k != i
    # Pairs involving j: (j,k) for k != j, i.e., d_jk for k != j
    # But (i,j) is counted in both, so:
    # sum_{k<l, k,l not in {i,j}} d_kl = total_sum - sum_{k != i} d_ik - sum_{k != j} d_jk + d_ij
    # = total_sum - (row_sums[i] - d_ij) - (row_sums[j] - d_ij) + d_ij
    # = total_sum - row_sums[i] - row_sums[j] + 3*d_ij
    
    total_sum = dmat[np.triu_indices(n, k=1)].sum()
    
    m = n - 2
    num_pairs = m * (m - 1) / 2.0
    
    # S(i,j) = num_pairs * d_ij + (total_sum - row_sums[i] - row_sums[j] + 3*d_ij) 
    #          - (m-1)/2 * (row_sums[i] - d_ij) - (m-1)/2 * (row_sums[j] - d_ij)
    #        = num_pairs * d_ij + total_sum - row_sums[i] - row_sums[j] + 3*d_ij
    #          - (m-1)/2 * row_sums[i] + (m-1)/2 * d_ij - (m-1)/2 * row_sums[j] + (m-1)/2 * d_ij
    #        = (num_pairs + 3 + (m-1)) * d_ij + total_sum - row_sums[i]*(1 + (m-1)/2) - row_sums[j]*(1 + (m-1)/2)
    #        = (num_pairs + m + 2) * d_ij + total_sum - (m+1)/2 * (row_sums[i] + row_sums[j])
    
    # num_pairs + m + 2 = m*(m-1)/2 + m + 2 = (m^2 - m + 2m + 4)/2 = (m^2 + m + 4)/2
    
    coeff_d = (m * (m - 1) / 2.0 + m + 2)
    coeff_rs = (m + 1) / 2.0
    
    # S(i,j) = coeff_d * d_ij + total_sum - coeff_rs * (row_sums[i] + row_sums[j])
    # For the agglomerative step, we want to find the pair (i,j) that maximizes S(i,j)
    # (most supported quartet grouping)
    
    # Vectorized computation
    rs_sum = row_sums[:, np.newaxis] + row_sums[np.newaxis, :]
    S = coeff_d * dmat + total_sum - coeff_rs * rs_sum
    np.fill_diagonal(S, -np.inf)
    
    return S


def _fdq_reconstruct(dmat, labels):
    """
    Fast Distance-based Quartet (FDQ) agglomerative reconstruction.
    
    Repeatedly merge the pair of taxa/clusters with the highest quartet score.
    This is similar to Neighbor-Joining but uses quartet scores instead of the Q criterion.
    """
    n = len(labels)
    if n <= 1:
        return labels[0] if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    leaf_set = set(labels)
    
    # Work with a copy of the distance matrix
    dmat_work = dmat.copy()
    
    # Track which original taxa each node represents
    # node_leaves[node_id] = set of original taxon indices
    node_leaves = {i: {i} for i in range(n)}
    node_labels = {i: labels[i] for i in range(n)}
    
    active = list(range(n))
    tree_nodes = {}
    next_node_id = n
    
    while len(active) > 2:
        m = len(active)
        if m < 3:
            break
            
        # Compute quartet score matrix for active nodes
        # Extract submatrix
        idx = np.array(active)
        d_sub = dmat_work[np.ix_(idx, idx)]
        
        # Compute quartet scores
        S = _quartet_score_matrix(d_sub, m)
        
        # Find the pair with maximum quartet score
        # Convert to active indices
        i_local, j_local = np.unravel_index(np.argmax(S), S.shape)
        
        # Ensure i_local < j_local for consistency
        if i_local > j_local:
            i_local, j_local = j_local, i_local
        
        i = active[i_local]
        j = active[j_local]
        
        # Compute branch lengths using NJ-style formula
        # r_k = sum of distances from k to all other active nodes
        r = d_sub.sum(axis=1)
        
        d_ij = d_sub[i_local, j_local]
        li = 0.5 * d_ij + (r[i_local] - r[j_local]) / (2.0 * (m - 2))
        lj = 0.5 * d_ij - (r[i_local] - r[j_local]) / (2.0 * (m - 2))
        
        # Clamp branch lengths to be non-negative
        li = max(li, 0.0)
        lj = max(lj, 0.0)
        
        w = next_node_id
        next_node_id += 1
        
        tree_nodes[w] = (i, j, li, lj)
        
        # Update distances for new node w
        for k in active:
            if k == i or k == j:
                continue
            d_wk = 0.5 * (dmat_work[i, k] + dmat_work[j, k] - d_ij)
            dmat_work[w, k] = d_wk
            dmat_work[k, w] = d_wk
        
        # Update node_leaves and node_labels
        node_leaves[w] = node_leaves[i] | node_leaves[j]
        # For labels, we just need to track the structure; the Newick builder will handle it
        node_labels[w] = f"w{w}"
        
        # Remove i, j and add w
        active.remove(i)
        active.remove(j)
        active.append(w)
        
    # Connect the last two
    if len(active) == 2:
        u, v = active
        d_uv = dmat_work[u, v]
        w = next_node_id
        next_node_id += 1
        tree_nodes[w] = (u, v, d_uv / 2.0, d_uv / 2.0)
        node_leaves[w] = node_leaves[u] | node_leaves[v]
        node_labels[w] = f"w{w}"
        active = [w]
    
    # Build Newick string
    def build_newick(node):
        if node in range(n):
            return labels[node]
        if node not in tree_nodes:
            return labels[0]  # Fallback
            
        child1, child2, len1, len2 = tree_nodes[node]
        newick1 = build_newick(child1)
        newick2 = build_newick(child2)
        return f"({newick1}:{len1:.4f},{newick2}:{len2:.4f})"
        
    root = active[0]
    return build_newick(root)


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
        
    leaf_set = set(labels)
    
    dist = {}
    for i in range(n):
        for j in range(n):
            dist[(labels[i], labels[j])] = dmat[i, j]
            
    active_nodes = list(labels)
    tree_nodes = {}
    next_node_id = 0
    
    while len(active_nodes) > 2:
        m = len(active_nodes)
        
        r = {}
        for i in active_nodes:
            r[i] = sum(dist[(i, k)] for k in active_nodes if k != i)
            
        min_q = float('inf')
        i_node = None
        j_node = None
        
        for i in active_nodes:
            for j in active_nodes:
                if i == j:
                    continue
                q = (m - 2) * dist[(i, j)] - r[i] - r[j]
                if q < min_q:
                    min_q = q
                    i_node = i
                    j_node = j
                    
        if i_node is None:
            break
            
        d_ij = dist[(i_node, j_node)]
        li = 0.5 * d_ij + (r[i_node] - r[j_node]) / (2 * (m - 2))
        lj = 0.5 * d_ij - (r[i_node] - r[j_node]) / (2 * (m - 2))
        
        w_name = f"w{next_node_id}"
        next_node_id += 1
        
        tree_nodes[w_name] = (i_node, j_node, li, lj)
        
        for k in active_nodes:
            if k == i_node or k == j_node:
                continue
            d_wk = 0.5 * (dist[(i_node, k)] + dist[(j_node, k)] - d_ij)
            dist[(w_name, k)] = d_wk
            dist[(k, w_name)] = d_wk
            
        active_nodes.remove(i_node)
        active_nodes.remove(j_node)
        active_nodes.append(w_name)
        
    if len(active_nodes) == 2:
        u, v = active_nodes
        d_uv = dist.get((u, v), 0.0)
        w_name = f"w{next_node_id}"
        next_node_id += 1
        tree_nodes[w_name] = (u, v, d_uv / 2.0, d_uv / 2.0)
        active_nodes = [w_name]
        
    def build_newick(node):
        if node in leaf_set:
            return node
        if node not in tree_nodes:
            return node
            
        child1, child2, len1, len2 = tree_nodes[node]
        newick1 = build_newick(child1)
        newick2 = build_newick(child2)
        return f"({newick1}:{len1:.4f},{newick2}:{len2:.4f})"
        
    root = active_nodes[0]
    return build_newick(root)


def _star_tree(labels):
    """Fallback: star tree."""
    if len(labels) <= 1:
        return labels[0] if labels else ""
    if len(labels) == 2:
        return f"({labels[0]},{labels[1]})"
    return "(" + ",".join(labels) + ")"


def _parse_newick_splits(newick, labels):
    """Parse a Newick string into a set of unrooted splits."""
    import re
    newick_clean = re.sub(r':[0-9.]+', '', newick)
    newick_clean = newick_clean.replace(';', '').strip()
    
    splits = set()
    
    def parse(s, idx):
        if s[idx] != '(':
            j = idx
            while j < len(s) and s[j] not in ',)':
                j += 1
            leaf = s[idx:j]
            return {leaf}, j
        
        children = []
        idx += 1
        while s[idx] != ')':
            child, idx = parse(s, idx)
            children.append(child)
            if s[idx] == ',':
                idx += 1
        idx += 1
        
        combined = set()
        for child in children:
            combined.update(child)
        
        for child in children:
            if 0 < len(child) < len(labels):
                if len(child) <= len(labels) - len(child):
                    splits.add(tuple(sorted(child)))
                else:
                    other = labels_set - child
                    splits.add(tuple(sorted(other)))
        
        return combined, idx
    
    labels_set = set(labels)
    parse(newick_clean, 0)
    
    return splits


def _nni_refine(dmat, labels, newick):
    """
    NNI refinement using Fitch parsimony scoring.
    Attempts to improve the tree topology by finding better NNI moves.
    """
    n = len(labels)
    if n < 5:
        return newick
    
    # Parse the current tree to get its splits
    current_splits = _parse_newick_splits(newick, labels)
    
    # For NNI, we need to work with the tree structure
    # This is a simplified version that tries a few random NNI-like moves
    # by re-running FDQ with perturbed distances
    
    best_newick = newick
    best_score = _fitch_cost(dmat, newick, labels)
    
    # Try a few iterations of local search
    max_iters = 5
    for _ in range(max_iters):
        improved = False
        # Perturb the distance matrix slightly and re-run FDQ
        # This is a heuristic; a proper NNI would examine specific moves
        dmat_perturbed = dmat + np.random.normal(0, 0.01, dmat.shape)
        np.fill_diagonal(dmat_perturbed, 0)
        
        try:
            candidate = _fdq_reconstruct(dmat_perturbed, labels)
        except Exception:
            continue
        
        cand_splits = _parse_newick_splits(candidate, labels)
        if cand_splits == current_splits:
            continue
        
        cand_score = _fitch_cost(dmat, candidate, labels)
        if cand_score < best_score:
            best_score = cand_score
            best_newick = candidate
            current_splits = cand_splits
            improved = True
        
        if not improved:
            break
    
    return best_newick


def _fitch_cost(dmat, newick, labels):
    """
    Compute a cost for a tree topology based on the distance matrix.
    Uses a simplified Fitch-like parsimony score.
    """
    splits = _parse_newick_splits(newick, labels)
    if not splits:
        return float('inf')
    
    # For each split, compute the cost as the sum of distances across the split
    # minus the sum of distances within each side
    cost = 0.0
    n = len(labels)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    
    for split in splits:
        split_set = set(label_to_idx[lab] for lab in split)
        complement = set(range(n)) - split_set
        
        # Sum of distances across the split
        cross_dist = 0.0
        for i in split_set:
            for j in complement:
                cross_dist += dmat[i, j]
        
        # Sum of distances within each side
        within_cost = 0.0
        for i in split_set:
            for j in split_set:
                if i < j:
                    within_cost += dmat[i, j]
        for i in complement:
            for j in complement:
                if i < j:
                    within_cost += dmat[i, j]
        
        # The split is good if cross_dist is small relative to within costs
        # This is a rough heuristic
        cost += cross_dist / max(within_cost, 1e-10)
    
    return cost


def _bootstrap_vote(seqs, labels, dmat_base, pmat_base, num_bootstraps):
    """
    Perform bootstrap resampling and vote on topologies.
    """
    n = len(labels)
    L = seqs.shape[1]
    
    topology_counts = {}
    topology_newick = {}
    
    for b in range(num_bootstraps):
        boot_indices = np.random.randint(0, L, size=L)
        boot_seqs = seqs[:, boot_indices]
        
        dmat_boot, pmat_boot = _jc_distance_matrix(boot_seqs)
        dmat_corrected = _bionj_correct(dmat_boot, pmat_boot, n)
        
        try:
            tree = _fdq_reconstruct(dmat_corrected, labels)
        except Exception:
            continue
        
        splits = _parse_newick_splits(tree, labels)
        key = frozenset(splits)
        
        if key not in topology_counts:
            topology_counts[key] = 0
            topology_newick[key] = tree
        topology_counts[key] += 1
        
        elapsed = time.time() - _start_time
        if elapsed > _DEADLINE_FRACTION * 270.0:
            break
    
    if not topology_counts:
        return None
    
    best_key = max(topology_counts, key=topology_counts.get)
    return topology_newick[best_key]


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
    
    # 1. Compute Distance Matrix
    dmat, pmat = _jc_distance_matrix(seqs)
    
    # 2. Apply BIONJ correction
    dmat_corrected = _bionj_correct(dmat, pmat, n)
    
    # 3. Run FDQ on corrected distances
    try:
        tree = _fdq_reconstruct(dmat_corrected, labels)
    except Exception:
        try:
            tree = _neighbor_joining(dmat_corrected, labels)
        except Exception:
            tree = _star_tree(labels)
    
    # 4. Bootstrap voting if time permits
    elapsed = time.time() - _start_time
    remaining = _DEADLINE_FRACTION * 270.0 - elapsed
    
    boot_time_budget = remaining * 0.6
    if boot_time_budget > 0.05:
        per_boot_time = 0.003 * (n / 16.0) ** 1.5
        num_bootstraps = int(boot_time_budget / per_boot_time)
        num_bootstraps = max(10, min(num_bootstraps, 200))
        
        boot_tree = _bootstrap_vote(seqs, labels, dmat, pmat, num_bootstraps)
        if boot_tree is not None:
            tree = boot_tree
    
    return tree
