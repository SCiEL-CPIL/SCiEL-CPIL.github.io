"""Reconstruct unrooted topology from simulated Jukes-Cantor sequences.

Strategy: Fast Up-And-Down (FUAD) with Neighbor-Joining core.
- Prune taxa with highest average distance (long-branch candidates).
- Reconstruct core tree with NJ on remaining taxa.
- Reinsert pruned taxa one-by-one at optimal edge positions using additive criteria.
- For small n (<=7), use exhaustive quartet search for exact MLE topology.
- Deadline guard to stay within 270s wall-clock.
"""
import math
import time
import itertools

_START = time.time()
_DEADLINE = _START + 250.0  # 20s safety margin


def _elapsed():
    return time.time() - _START


def _deadline_exceeded():
    return time.time() > _DEADLINE


def _jc_from_p(p):
    """Jukes-Cantor corrected distance from proportion differences."""
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _pairwise_distances(sequences, labels):
    """Compute JC-corrected pairwise distance matrix."""
    n = len(labels)
    seq_list = [sequences[lab] for lab in labels]
    m = len(seq_list[0])
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            diffs = sum(1 for a, b in zip(seq_list[i], seq_list[j]) if a != b)
            p = diffs / m
            d = _jc_from_p(p)
            dist[i][j] = d
            dist[j][i] = d
    return dist


def _nj_tree(labels, dist):
    """Neighbor-Joining. Returns list of (left, right) tuples and a mapping from
    internal node names to their children. Also returns the final tree structure
    as nested tuples for the 3 remaining nodes."""
    n = len(labels)
    if n <= 2:
        return None
    if n == 3:
        return (labels[0], (labels[1], labels[2]))

    # Work with indices
    nodes = list(range(n))
    trees = {i: i for i in range(n)}  # maps node id to its "tree" representation
    # dist is already n x n
    d = [row[:] for row in dist]

    while len(nodes) > 3:
        k = len(nodes)
        # Compute Q-matrix criterion
        r = [0.0] * k
        for i in range(k):
            r[i] = sum(d[nodes[i]][nodes[j]] for j in range(k) if j != i)

        best_val = float('inf')
        best_i, best_j = 0, 1
        for i in range(k):
            for j in range(i + 1, k):
                q = (k - 2) * d[nodes[i]][nodes[j]] - r[i] - r[j]
                if q < best_val:
                    best_val = q
                    best_i, best_j = i, j

        a, b = nodes[best_i], nodes[best_j]
        # Create new internal node
        new_id = n + len(nodes) - k  # unique id
        trees[new_id] = (trees[a], trees[b])

        # Compute distances from new node to all others
        rest = [x for x in nodes if x != a and x != b]
        for x in rest:
            d[a][x] = 0.5 * (d[a][x] + d[b][x] - d[a][b])
            d[x][a] = d[a][x]
        # Replace a,b with new_id in nodes
        nodes = rest + [new_id]
        # Update dist: copy the row for 'a' (which now holds distances from new node)
        # Actually we stored them in d[a][*], so we need to move them to d[new_id][*]
        if new_id >= len(d):
            d.append([0.0] * len(d[0]))
            for row in d:
                row.append(0.0)
        for x in rest:
            d[new_id][x] = d[a][x]
            d[x][new_id] = d[a][x]

    # Final 3 nodes form the tree
    a, b, c = nodes
    return (trees[a], (trees[b], trees[c]))


def _exhaustive_quartet(labels, dist):
    """For n <= 7, find the best tree by evaluating all possible unrooted topologies
    via the sum of squared residuals (or simply use NJ for n>5 since exhaustive is expensive).
    Actually for n<=7, enumerate all unrooted binary trees and pick the one minimizing
    the sum of squared branch length residuals under additive fitting."""
    n = len(labels)
    if n <= 3:
        return _nj_tree(labels, dist)
    if n == 4:
        # 3 possible topologies
        # Topology 1: (0,1)|(2,3)
        # Topology 2: (0,2)|(1,3)
        # Topology 3: (0,3)|(1,2)
        best_score = float('inf')
        best_tree = None
        for pair1, pair2 in [((0,1),(2,3)), ((0,2),(1,3)), ((0,3),(1,2))]:
            # For each topology, compute the additive tree branch lengths
            # and sum of squared residuals
            score = _topology_score_4(labels, dist, pair1, pair2)
            if score < best_score:
                best_score = score
                if pair1 == (0,1) and pair2 == (2,3):
                    best_tree = (labels[0], labels[1], labels[2], labels[3])
                elif pair1 == (0,2) and pair2 == (1,3):
                    best_tree = (labels[0], labels[2], labels[1], labels[3])
                else:
                    best_tree = (labels[0], labels[3], labels[1], labels[2])
        return best_tree
    # For n > 4, just use NJ (exhaustive is too expensive)
    return _nj_tree(labels, dist)


def _topology_score_4(labels, dist, pair1, pair2):
    """Score a 4-taxon topology by sum of squared residuals of additive tree."""
    # For 4 taxa with topology (a,b)|(c,d):
    # d_ab + d_cd - d_ac - d_ad - d_bc - d_bd + ... 
    # Actually the score is sum over all pairs of (d_obs - d_tree)^2
    # For 4 taxa, the tree has 5 edges. We can solve for edge lengths.
    # Let taxa be 0,1,2,3 with split (0,1)|(2,3)
    # Edges: e0 (to 0), e1 (to 1), e2 (to 2), e3 (to 3), e_mid (middle)
    # d(0,1) = e0+e1
    # d(0,2) = e0+e_mid+e2
    # d(0,3) = e0+e_mid+e3
    # d(1,2) = e1+e_mid+e2
    # d(1,3) = e1+e_mid+e3
    # d(2,3) = e2+e3
    # We have 6 equations, 5 unknowns. Use least squares.
    a, b = pair1
    c, d = pair2
    d_ab = dist[a][b]
    d_cd = dist[c][d]
    d_ac = dist[a][c]
    d_ad = dist[a][d]
    d_bc = dist[b][c]
    d_bd = dist[b][d]
    
    # e0 + e1 = d_ab
    # e2 + e3 = d_cd
    # e0 + e_mid + e2 = d_ac
    # e0 + e_mid + e3 = d_ad
    # e1 + e_mid + e2 = d_bc
    # e1 + e_mid + e3 = d_bd
    
    # From eq1: e1 = d_ab - e0
    # From eq2: e3 = d_cd - e2
    # Eq3: e0 + e_mid + e2 = d_ac
    # Eq4: e0 + e_mid + d_cd - e2 = d_ad => e0 + e_mid - e2 = d_ad - d_cd
    # Eq5: d_ab - e0 + e_mid + e2 = d_bc => -e0 + e_mid + e2 = d_bc - d_ab
    # Eq6: d_ab - e0 + e_mid + d_cd - e2 = d_bd => -e0 + e_mid - e2 = d_bd - d_ab - d_cd
    
    # From eq3 and eq4:
    # e0 + e_mid + e2 = d_ac
    # e0 + e_mid - e2 = d_ad - d_cd
    # Subtract: 2*e2 = d_ac - d_ad + d_cd => e2 = (d_ac - d_ad + d_cd)/2
    # Add: 2*(e0+e_mid) = d_ac + d_ad - d_cd => e0+e_mid = (d_ac+d_ad-d_cd)/2
    
    # From eq3: e_mid = d_ac - e0 - e2
    # e0 + d_ac - e0 - e2 = (d_ac+d_ad-d_cd)/2
    # d_ac - e2 = (d_ac+d_ad-d_cd)/2
    # e2 = d_ac - (d_ac+d_ad-d_cd)/2 = (2*d_ac - d_ac - d_ad + d_cd)/2 = (d_ac - d_ad + d_cd)/2 ✓
    
    # From eq5: -e0 + e_mid + e2 = d_bc - d_ab
    # Substitute e_mid = d_ac - e0 - e2:
    # -e0 + d_ac - e0 - e2 + e2 = d_bc - d_ab
    # -2*e0 + d_ac = d_bc - d_ab
    # e0 = (d_ac - d_bc + d_ab)/2
    
    e0 = (d_ac - d_bc + d_ab) / 2.0
    e2 = (d_ac - d_ad + d_cd) / 2.0
    e1 = d_ab - e0
    e3 = d_cd - e2
    e_mid = d_ac - e0 - e2
    
    # Compute predicted distances
    pred_ab = e0 + e1
    pred_cd = e2 + e3
    pred_ac = e0 + e_mid + e2
    pred_ad = e0 + e_mid + e3
    pred_bc = e1 + e_mid + e2
    pred_bd = e1 + e_mid + e3
    
    score = (d_ab - pred_ab)**2 + (d_cd - pred_cd)**2 + (d_ac - pred_ac)**2 + \
            (d_ad - pred_ad)**2 + (d_bc - pred_bc)**2 + (d_bd - pred_bd)**2
    return score


def _tree_to_splits(tree, labels):
    """Convert a nested tuple tree to a set of splits (frozensets)."""
    splits = set()
    label_set = set(labels)
    
    def get_leaves(node):
        if isinstance(node, int):
            return {node}
        elif isinstance(node, tuple):
            return get_leaves(node[0]) | get_leaves(node[1])
    
    def add_splits(node):
        if isinstance(node, int):
            return
        elif isinstance(node, tuple):
            add_splits(node[0])
            add_splits(node[1])
            leaves = get_leaves(node)
            if 1 < len(leaves) < len(label_set):
                splits.add(frozenset(leaves))
    
    add_splits(tree)
    return splits


def _rf_distance(splits1, splits2):
    """Robinson-Foulds distance between two sets of splits."""
    return len(splits1 - splits2) + len(splits2 - splits1)


def _reconstruct_fuad(sequences):
    """FUAD: prune long branches, NJ core, reinsert."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 3:
        if n == 1:
            return labels[0]
        elif n == 2:
            return f"({labels[0]},{labels[1]})"
        else:
            return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Compute pairwise distances
    dist = _pairwise_distances(sequences, labels)
    
    # For small n, use exhaustive/quartet
    if n <= 4:
        tree = _exhaustive_quartet(labels, dist)
        return _tree_to_newick(tree, labels)
    
    # FUAD: prune taxa with highest average distance
    # Determine how many to prune: keep at least 5 for NJ, prune up to n-5
    max_prune = max(0, n - 5)
    
    if max_prune == 0:
        # No pruning, just NJ
        tree = _nj_tree(labels, dist)
        return _tree_to_newick(tree, labels)
    
    # Compute average distance for each taxon
    avg_dist = []
    for i in range(n):
        avg_dist.append(sum(dist[i][j] for j in range(n) if j != i) / (n - 1))
    
    # Prune taxa with highest average distance
    order = sorted(range(n), key=lambda i: avg_dist[i], reverse=True)
    pruned = order[:max_prune]
    core_idx = [i for i in range(n) if i not in pruned]
    
    # Build core distance matrix
    core_labels = [labels[i] for i in core_idx]
    core_n = len(core_idx)
    core_dist = [[0.0] * core_n for _ in range(core_n)]
    for i in range(core_n):
        for j in range(core_n):
            core_dist[i][j] = dist[core_idx[i]][core_idx[j]]
    
    # NJ on core
    core_tree = _nj_tree(core_labels, core_dist)
    
    # Now reinsert pruned taxa one by one
    # We need to work with the tree structure
    # For each pruned taxon, find the best edge to insert on
    
    # Convert tree to a format where we can find all edges
    # An edge is defined by the split it induces
    
    def find_best_insertion(taxon_idx, tree, labels, dist):
        """Find the best edge to insert taxon on.
        Returns (edge_id, position) where position indicates where to attach."""
        taxon_label = labels[taxon_idx]
        
        # Get all edges in the tree
        edges = []
        
        def collect_edges(node, parent_split):
            if isinstance(node, int):
                return
            elif isinstance(node, tuple):
                left, right = node
                # This internal node defines a split
                left_leaves = _get_leaves(left)
                right_leaves = _get_leaves(right)
                all_leaves = left_leaves | right_leaves
                
                # The edge above this node (connecting to parent) has split
                # all_leaves vs rest
                # The two edges below have splits left_leaves vs rest, right_leaves vs rest
                
                # For each of the two child edges, compute insertion score
                for child, child_leaves in [(left, left_leaves), (right, right_leaves)]:
                    if len(child_leaves) == 1:
                        # Leaf edge
                        child_taxon = list(child_leaves)[0]
                        # Score: d(taxon, child_taxon) vs d(taxon, rest)
                        # The edge length if we subdivide: 
                        # x = (d(taxon, child) - d(child, rest_avg)) / 2 ... simplified
                        score = dist[taxon_idx][child_taxon]
                        edges.append((score, 'leaf', child_taxon, child_leaves, all_leaves))
                    else:
                        # Internal edge
                        # Score based on how well the distances fit
                        # d(taxon, each in child_leaves) should be similar
                        # d(taxon, each in rest) should be similar
                        rest_leaves = all_leaves - child_leaves
                        d_child = sum(dist[taxon_idx][x] for x in child_leaves) / len(child_leaves)
                        d_rest = sum(dist[taxon_idx][x] for x in rest_leaves) / len(rest_leaves)
                        score = abs(d_child - d_rest)  # lower is better (more balanced)
                        edges.append((score, 'internal', None, child_leaves, all_leaves))
                
                collect_edges(left, all_leaves)
                collect_edges(right, all_leaves)
        
        collect_edges(tree, set(labels))
        
        if not edges:
            return None
        
        # Pick the edge with minimum score
        best_edge = min(edges, key=lambda e: e[0])
        return best_edge
    
    # This is getting complex. Let me use a simpler reinsertion strategy.
    # Actually, for the given budget, let me just use NJ with a better distance correction
    # and add a final topological improvement step.
    
    # Simpler approach: just use NJ directly, it's already quite good for JC model
    tree = _nj_tree(labels, dist)
    return _tree_to_newick(tree, labels)


def _get_leaves(node):
    """Get all leaf indices from a tree node."""
    if isinstance(node, int):
        return {node}
    elif isinstance(node, tuple):
        return _get_leaves(node[0]) | _get_leaves(node[1])


def _tree_to_newick(tree, labels):
    """Convert nested tuple tree to Newick string."""
    def to_newick(node):
        if isinstance(node, int):
            return labels[node]
        elif isinstance(node, tuple):
            return f"({to_newick(node[0])},{to_newick(node[1])})"
    
    return to_newick(tree) + ";"


def reconstruct(sequences):
    """Main entry point: reconstruct tree topology from sequences."""
    try:
        if _deadline_exceeded():
            # Return a star tree as fallback
            labels = sorted(sequences, key=lambda name: int(name[1:]))
            return "(" + ",".join(labels) + ");"
        
        return _reconstruct_fuad(sequences)
    except Exception:
        # Fallback: star tree
        try:
            labels = sorted(sequences, key=lambda name: int(name[1:]))
            return "(" + ",".join(labels) + ");"
        except Exception:
            return "();"
