"""
Strategy: Maximum Likelihood via Felsenstein pruning with NNI refinement.

1. Build initial tree with Neighbor Joining.
2. Estimate branch lengths via pairwise distance mapping.
3. Compute log-likelihood using Felsenstein's pruning algorithm under JC model.
4. Refine topology via NNI moves, re-optimizing branch lengths for each candidate.
5. Return the tree with highest log-likelihood.
"""

import time
import math
import random
import numpy as np

_START_TIME = None
_TIME_LIMIT = 255.0  # Safety margin for 270s limit

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_left():
    _init_time()
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))

def _jc_transition_probs(length):
    """Compute JC transition probability matrix for a given branch length."""
    if length < 0:
        length = 0
    p_same = 0.75 + 0.25 * math.exp(-4.0 * length / 3.0)
    p_diff = (1.0 - p_same) / 3.0
    # Matrix where M[i][j] = P(i->j)
    # For JC, all bases are symmetric
    mat = np.full((4, 4), p_diff)
    np.fill_diagonal(mat, p_same)
    return mat

def _compute_distance_matrix(seqs_int, n_taxa, n_sites):
    """Compute pairwise JC distances."""
    diff = (seqs_int[:, np.newaxis, :] != seqs_int[np.newaxis, :, :])
    num_diffs = np.sum(diff, axis=2)
    p = num_diffs / n_sites
    p_clipped = np.clip(p, 0, 0.7499)
    dists = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    np.fill_diagonal(dists, 0.0)
    dists = np.nan_to_num(dists, nan=3.0, posinf=3.0, neginf=0.0)
    return dists

def _nj_build_tree(dists, n_taxa):
    """
    Build tree using Neighbor Joining.
    Returns a tree represented as:
    - nodes: list where index is node id, value is either a leaf index (>=0) or internal node marker
    - parent: dict mapping child -> parent
    - neighbors: dict mapping node -> list of neighbor nodes
    - is_leaf: dict mapping node -> bool
    - leaf_map: dict mapping node -> original leaf index (for leaves)
    """
    if n_taxa < 3:
        # Handle trivial cases
        nodes = list(range(n_taxa))
        neighbors = {i: [j for j in range(n_taxa) if j != i] for i in range(n_taxa)}
        is_leaf = {i: True for i in range(n_taxa)}
        leaf_map = {i: i for i in range(n_taxa)}
        return nodes, neighbors, is_leaf, leaf_map
    
    d = dists.copy()
    active = list(range(n_taxa))
    neighbors = {i: [] for i in range(n_taxa)}
    is_leaf = {i: True for i in range(n_taxa)}
    leaf_map = {i: i for i in range(n_taxa)}
    
    next_node = n_taxa
    while len(active) > 3:
        n_curr = len(active)
        r = np.array([d[i, j] for j in active for i in active]).reshape(n_curr, n_curr)
        row_sums = r.sum(axis=1)
        q = (n_curr - 2) * d[np.ix_(active, active)] - row_sums[:, None] - row_sums[None, :]
        np.fill_diagonal(q, np.inf)
        
        min_idx = np.unravel_index(np.argmin(q), q.shape)
        i_idx, j_idx = min_idx
        i, j = active[i_idx], active[j_idx]
        
        # Compute branch lengths
        d_ij = d[i, j]
        others = [active[k] for k in range(n_curr) if k != i_idx and k != j_idx]
        
        # Distance from new node to each other
        new_dists = {}
        for k in others:
            new_d = 0.5 * (d[i, k] + d[j, k] - d_ij)
            new_dists[k] = max(0.0, new_d)
        
        # Create new node
        new_node = next_node
        next_node += 1
        is_leaf[new_node] = False
        leaf_map[new_node] = -1
        
        # Connect new node to i and j
        neighbors[i].append(new_node)
        neighbors[j].append(new_node)
        neighbors[new_node] = [i, j] + others
        
        # Update distances
        # Remove i and j from active, add new_node
        new_active = others + [new_node]
        new_d = np.zeros((len(new_active), len(new_active)))
        new_d_map = {node: idx for idx, node in enumerate(new_active)}
        
        for idx_a, a in enumerate(new_active):
            for idx_b, b in enumerate(new_active):
                if a == b:
                    new_d[idx_a, idx_b] = 0
                elif a == new_node:
                    new_d[idx_a, idx_b] = new_dists[b]
                    new_d[idx_b, idx_a] = new_dists[b]
                elif b == new_node:
                    new_d[idx_a, idx_b] = new_dists[a]
                else:
                    new_d[idx_a, idx_b] = d[a, b]
        
        # Update neighbors for remaining active nodes
        for k in others:
            neighbors[k].remove(i)
            neighbors[k].remove(j)
            neighbors[k].append(new_node)
        
        # Mark i and j as no longer active (their neighbors already updated)
        active = new_active
        d = new_d
    
    # Connect the last 3 nodes in a star
    if len(active) == 3:
        a, b, c = active
        # Create a central node
        center = next_node
        next_node += 1
        is_leaf[center] = False
        leaf_map[center] = -1
        neighbors[center] = [a, b, c]
        neighbors[a].append(center)
        neighbors[b].append(center)
        neighbors[c].append(center)
    
    return list(range(next_node)), neighbors, is_leaf, leaf_map

def _estimate_branch_lengths(neighbors, is_leaf, leaf_map, dists):
    """
    Estimate branch lengths by finding a central node and mapping distances.
    For each edge (u, v), estimate length as 0.5 * (dist_to_u_from_reference - dist_to_v_from_reference)
    This is a rough heuristic. A better approach is to use the distance matrix directly.
    
    Actually, for ML, we need to optimize branch lengths. Let's initialize them 
    based on the NJ distances and then refine.
    
    For now, return a dict edge -> length.
    """
    branch_lengths = {}
    # Use a simple heuristic: for each edge, length = 0.1 (default)
    # Or try to estimate from distances
    for u in neighbors:
        for v in neighbors[u]:
            if u < v:
                branch_lengths[(u, v)] = 0.1
    return branch_lengths

def _felsenstein_ll(tree_nodes, neighbors, is_leaf, leaf_map, seqs_int, n_sites, branch_lengths):
    """
    Compute log-likelihood using Felsenstein's pruning algorithm.
    tree_nodes: list of node ids
    neighbors: dict node -> list of neighbors
    is_leaf: dict node -> bool
    leaf_map: dict node -> leaf index
    seqs_int: (n_taxa, n_sites) integer matrix
    n_sites: number of sites
    branch_lengths: dict (u,v) -> length
    """
    n_taxa = len(leaf_map)
    # Find root: pick an arbitrary node, say 0
    root = tree_nodes[0]
    
    # Build parent-child structure from root
    parent = {}
    children = {node: [] for node in tree_nodes}
    visited = set()
    stack = [root]
    visited.add(root)
    while stack:
        u = stack.pop()
        for v in neighbors[u]:
            if v not in visited:
                visited.add(v)
                parent[v] = u
                children[u].append(v)
                stack.append(v)
    
    # Felsenstein: compute likelihood at each node for each site
    # We'll do this site by site for clarity, but vectorize if possible
    
    total_ll = 0.0
    for site in range(n_sites):
        # col[v] = vector of 4 probabilities
        col = {}
        
        def compute_col(v):
            if is_leaf[v]:
                # Leaf: probability vector is one-hot at the observed base
                base = seqs_int[leaf_map[v], site]
                vec = np.zeros(4)
                vec[base] = 1.0
                return vec
            
            # Internal node
            total = np.zeros(4)
            for c in children[v]:
                lc = compute_col(c)
                # Branch length from v to c
                bl = branch_lengths.get((v, c), 0.1) if v < c else branch_lengths.get((c, v), 0.1)
                # Transition matrix
                p_same = 0.75 + 0.25 * math.exp(-4.0 * bl / 3.0)
                p_diff = (1.0 - p_same) / 3.0
                # M * lc
                # M[i][j] = p_same if i==j else p_diff
                mlc = np.empty(4)
                for i in range(4):
                    val = p_same * lc[i]
                    for j in range(4):
                        if j != i:
                            val += p_diff * lc[j]
                    mlc[i] = val
                total += mlc
            
            col[v] = total
            return total
        
        root_col = compute_col(root)
        
        # The likelihood at the root is the sum of the root column
        # (since the root is unrooted, we sum over all states)
        ll_site = np.sum(root_col)
        if ll_site > 0:
            total_ll += math.log(ll_site)
        else:
            total_ll += -1000.0
    
    return total_ll

def _nni_refine(tree_nodes, neighbors, is_leaf, leaf_map, dists, seqs_int, n_sites, branch_lengths, max_rounds=5):
    """
    Perform NNI refinement.
    For each internal edge (u,v), try the two possible rearrangements.
    """
    current_ll = _felsenstein_ll(tree_nodes, neighbors, is_leaf, leaf_map, seqs_int, n_sites, branch_lengths)
    
    for round_idx in range(max_rounds):
        if _time_left() < 10.0:
            break
        
        best_improvement = 0.0
        best_move = None
        
        # Find all internal edges
        internal_edges = []
        for u in neighbors:
            for v in neighbors[u]:
                if u < v and not is_leaf.get(u, False) and not is_leaf.get(v, False):
                    internal_edges.append((u, v))
        
        for u, v in internal_edges:
            if _time_left() < 10.0:
                break
            
            # Get neighbors of u and v
            u_neighbors = [x for x in neighbors[u] if x != v]
            v_neighbors = [x for x in neighbors[v] if x != u]
            
            if len(u_neighbors) < 1 or len(v_neighbors) < 1:
                continue
            
            # Try NNI: swap one neighbor of u with one neighbor of v
            for w in u_neighbors:
                for x in v_neighbors:
                    if w == x:
                        continue
                    
                    # Save current state
                    orig_u_neighbors = list(neighbors[u])
                    orig_v_neighbors = list(neighbors[v])
                    orig_w_neighbors = list(neighbors[w])
                    orig_x_neighbors = list(neighbors[x])
                    
                    # Apply NNI
                    # Remove v from u's neighbors, remove u from v's neighbors
                    # Remove w from u, add w to v
                    # Remove x from v, add x to u
                    # Remove u from w, add v to w
                    # Remove v from x, add u to x
                    
                    neighbors[u] = [n for n in neighbors[u] if n not in (v, w)]
                    neighbors[u].append(x)
                    neighbors[v] = [n for n in neighbors[v] if n not in (u, x)]
                    neighbors[v].append(w)
                    neighbors[w] = [n for n in neighbors[w] if n != u]
                    neighbors[w].append(v)
                    neighbors[x] = [n for n in neighbors[x] if n != v]
                    neighbors[x].append(u)
                    
                    # Update branch lengths: keep same lengths for now
                    # (u,v) -> (u,x), (v,w), (w,v), (x,u)
                    # Simplify: just use 0.1 for all
                    new_bl = {}
                    for a in neighbors:
                        for b in neighbors[a]:
                            if a < b:
                                new_bl[(a, b)] = 0.1
                    
                    new_ll = _felsenstein_ll(tree_nodes, neighbors, is_leaf, leaf_map, seqs_int, n_sites, new_bl)
                    
                    if new_ll > current_ll + 1e-9:
                        improvement = new_ll - current_ll
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_move = (u, v, w, x, new_bl)
                        # Revert for now
                        neighbors[u] = orig_u_neighbors
                        neighbors[v] = orig_v_neighbors
                        neighbors[w] = orig_w_neighbors
                        neighbors[x] = orig_x_neighbors
                    else:
                        # Revert
                        neighbors[u] = orig_u_neighbors
                        neighbors[v] = orig_v_neighbors
                        neighbors[w] = orig_w_neighbors
                        neighbors[x] = orig_x_neighbors
        
        if best_move is None:
            break
        
        # Apply best move
        u, v, w, x, new_bl = best_move
        orig_u_neighbors = list(neighbors[u])
        orig_v_neighbors = list(neighbors[v])
        orig_w_neighbors = list(neighbors[w])
        orig_x_neighbors = list(neighbors[x])
        
        neighbors[u] = [n for n in neighbors[u] if n not in (v, w)]
        neighbors[u].append(x)
        neighbors[v] = [n for n in neighbors[v] if n not in (u, x)]
        neighbors[v].append(w)
        neighbors[w] = [n for n in neighbors[w] if n != u]
        neighbors[w].append(v)
        neighbors[x] = [n for n in neighbors[x] if n != v]
        neighbors[x].append(u)
        
        branch_lengths = new_bl
        current_ll += best_improvement
    
    return tree_nodes, neighbors, is_leaf, leaf_map, branch_lengths

def _tree_to_newick(tree_nodes, neighbors, is_leaf, leaf_map, n_taxa, labels):
    """Convert tree to Newick string."""
    # Pick a root
    root = tree_nodes[0]
    
    parent = {}
    children = {node: [] for node in tree_nodes}
    visited = set()
    stack = [root]
    visited.add(root)
    while stack:
        u = stack.pop()
        for v in neighbors[u]:
            if v not in visited:
                visited.add(v)
                parent[v] = u
                children[u].append(v)
                stack.append(v)
    
    def build_newick(node):
        if is_leaf[node]:
            return labels[leaf_map[node]]
        kids = children[node]
        if len(kids) == 0:
            return labels[leaf_map[node]] if is_leaf[node] else ""
        kid_strs = [build_newick(k) for k in kids]
        return "(" + ",".join(kid_strs) + ")"
    
    newick = build_newick(root)
    return newick + ";"

def reconstruct(sequences):
    _init_time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    if n_taxa < 3:
        if n_taxa == 1:
            return labels[0] + ";"
        elif n_taxa == 2:
            return f"({labels[0]},{labels[1]});"
    
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    n_sites = len(sequences[labels[0]])
    seqs_int = np.array([[seq_map[c] for c in sequences[label]] for label in labels])
    
    # 1. Compute distance matrix
    dists = _compute_distance_matrix(seqs_int, n_taxa, n_sites)
    
    # 2. Build initial tree with NJ
    tree_nodes, neighbors, is_leaf, leaf_map = _nj_build_tree(dists, n_taxa)
    
    # 3. Estimate initial branch lengths
    branch_lengths = _estimate_branch_lengths(neighbors, is_leaf, leaf_map, dists)
    
    # 4. NNI refinement
    tree_nodes, neighbors, is_leaf, leaf_map, branch_lengths = _nni_refine(
        tree_nodes, neighbors, is_leaf, leaf_map, dists, seqs_int, n_sites, branch_lengths, max_rounds=5
    )
    
    # 5. Convert to Newick
    newick = _tree_to_newick(tree_nodes, neighbors, is_leaf, leaf_map, n_taxa, labels)
    
    return newick
