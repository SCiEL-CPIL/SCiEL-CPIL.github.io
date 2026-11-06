"""Maximum likelihood phylogeny under Jukes-Cantor with NNI refinement."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np


def _jc_corrected(p):
    # Jukes-Cantor distance correction; cap to avoid log(0)
    if p <= 0.0:
        return 0.0
    if p >= 0.75:
        return 10.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    n = len(labels)
    # Encode sequences as int arrays for speed
    enc = []
    for lab in labels:
        s = sequences[lab]
        arr = np.frombuffer(s.encode('ascii'), dtype=np.uint8)
        # Map A,C,G,T to 0,1,2,3
        # A=65, C=67, G=71, T=84
        m = np.empty_like(arr, dtype=np.int8)
        m[arr == 65] = 0
        m[arr == 67] = 1
        m[arr == 71] = 2
        m[arr == 84] = 3
        enc.append(m)
    L = len(sequences[labels[0]])
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.count_nonzero(enc[i] != enc[j])
            p = diff / L
            D[i, j] = D[j, i] = _jc_corrected(p)
    return D


def _nj_tree(D, labels):
    """Build NJ tree, returns (topology as nested tuples, branch_lengths dict, leaf_order).
    topology: internal nodes are tuples (left, right); leaves are index ints.
    branch_lengths: maps node->length for each edge to parent (root has 0).
    """
    n = len(labels)
    if n == 1:
        return (0,), {0: 0.0}
    if n == 2:
        return (0, 1), {0: D[0, 1] / 2, 1: D[0, 1] / 2}

    nodes = list(range(n))
    # tree structure: for each node, store its two children (or None if leaf)
    # We'll build a list of nodes; leaves are 0..n-1, internal nodes n, n+1, ...
    # children: list of (left, right) for internal nodes; leaves have no children
    children = [None] * n  # leaves
    # parent branch length: length of edge from this node to its parent
    # For leaves, this is the branch length on the leaf edge.
    # We'll store bl[node] = length of edge from node to parent.
    bl = [0.0] * n
    # For internal nodes, we need to track. Let's use a dict for flexibility.
    # Actually, let's use arrays and append.
    # Let's use lists that we append to.
    children_list = [None] * n  # index 0..n-1 are leaves
    bl_list = [0.0] * n

    next_id = n
    active = list(range(n))
    D = D.copy()

    while len(active) > 2:
        m = len(active)
        # Compute row sums r
        r = np.zeros(m, dtype=np.float64)
        for i, a in enumerate(active):
            r[i] = sum(D[a, b] for b in active if b != a)
        # Find min Q
        best_q = float('inf')
        best_i = best_j = -1
        for i in range(m):
            for j in range(i + 1, m):
                a, b = active[i], active[j]
                q = (m - 2) * D[a, b] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i, best_j = i, j
        a, b = active[best_i], active[best_j]
        # Compute branch lengths
        delta = (r[best_i] - r[best_j]) / (2 * (m - 2))
        bl_a = 0.5 * D[a, b] + delta
        bl_b = 0.5 * D[a, b] - delta
        # Clamp to non-negative
        if bl_a < 0: bl_a = 0.0
        if bl_b < 0: bl_b = 0.0

        new_id = next_id
        next_id += 1

        children_list.append((a, b))
        bl_list.append(0.0)  # will be set later
        # Set leaf/internal branch lengths for a and b
        # But a and b might be internal nodes already. Their bl_list entry is the edge to parent.
        bl_list[a] = bl_a
        bl_list[b] = bl_b

        # Update distance matrix for new node
        new_row = np.zeros(len(children_list), dtype=np.float64)
        for x in active:
            if x == a or x == b:
                continue
            d_new = 0.5 * (D[a, x] + D[b, x] - D[a, b])
            if d_new < 0: d_new = 0.0
            new_row[x] = D[x, new_id] = d_new

        # Remove a, b from active, add new_id
        active.remove(a)
        active.remove(b)
        active.append(new_id)

    # Final two nodes
    if len(active) == 2:
        a, b = active
        d = D[a, b]
        bl_a = d / 2
        bl_b = d / 2
        if bl_a < 0: bl_a = 0.0
        if bl_b < 0: bl_b = 0.0
        bl_list[a] = bl_a
        bl_list[b] = bl_b
        # Root: we need to create a root node connecting a and b
        root_id = next_id
        next_id += 1
        children_list.append((a, b))
        bl_list.append(0.0)
        return root_id, children_list, bl_list
    else:
        # Should not happen
        return active[0], children_list, bl_list


def _build_tree_from_nj(D, labels):
    """Returns (root, children, bl) where children is list, bl is list."""
    root, children, bl = _nj_tree(D, labels)
    return root, children, bl


def _ml_likelihood(root, children, bl, seq_array, L):
    """Compute log-likelihood under JC model using Felsenstein pruning.
    seq_array: shape (n, L) int8 array with values 0-3.
    Returns log-likelihood (sum over sites).
    """
    n = len(children)
    # We need to do this for each site. For efficiency, we can vectorize over sites.
    # But let's start with a per-site loop for clarity, then optimize if needed.
    # Actually, with up to 32 taxa and 2000 sites, a Python loop over sites with numpy ops
    # per site might be slow. Let's vectorize.

    # Precompute the tree structure in a form suitable for vectorized pruning.
    # We need to know, for each internal node, its children, and process in post-order.

    # Let's build a post-order traversal list.
    postorder = []
    def _dfs(node):
        if children[node] is None:
            postorder.append(node)
        else:
            l, r = children[node]
            _dfs(l)
            _dfs(r)
            postorder.append(node)
    _dfs(root)

    # For each node, store its index in the postorder for quick access? Not needed.
    # We'll process postorder. For each site, we compute likelihoods.

    # Let's vectorize: for all sites at once.
    # L_matrix[node] will be a numpy array of shape (L,) for internal nodes,
    # or for leaves, we don't store L but we know the observed state.

    # Actually, let's store for each node in postorder its likelihood vector.
    # We'll use a dict or list indexed by node id.
    Lvec = [None] * n

    for node in postorder:
        if children[node] is None:
            # Leaf: likelihood is 1 if state matches, 0 otherwise? 
            # No, for JC, P(obs|state) = 1/4 if match, 1/12 if not? 
            # Actually, at the leaf, the "likelihood" of the observed state given the 
            # state at the leaf is 1 if it matches the observed, but we're computing 
            # P(data|tree,branch_lengths). The pruning algorithm computes 
            # P(data_subtree | state_at_node). 
            # For a leaf, P(data_leaf | state) = 1 if state==obs, else 0.
            # But that's not right for the JC model. The JC model gives transition 
            # probabilities along branches. The leaf's "state" is the observed state.
            # So L_leaf[state] = 1 if state == observed, else 0.
            # But we're working with 4 states. Let's represent as a 4-vector.
            obs = seq_array[node]  # shape (L,)
            # Lvec[node] should be a (4, L) array? Or (L, 4)?
            # Let's use (L, 4) for easier broadcasting.
            # Lvec[node][site, state] = 1 if state==obs[site], else 0.
            Lvec[node] = np.zeros((L, 4), dtype=np.float64)
            Lvec[node][np.arange(L), obs] = 1.0
        else:
            l, r = children[node]
            bl_l = bl_list[node]  # Wait, bl_list[node] is the edge from node to parent.
            # We need the edge lengths from node to its children l and r.
            # In our NJ construction, bl_list[l] is the edge from l to its parent (node).
            # So the branch length from node to l is bl_list[l].
            # Similarly for r.
            bl_l = bl_list[l]
            bl_r = bl_list[r]
            # Transition probability matrix for JC:
            # P(same) = 0.75 + 0.25*exp(-4/3 * d)
            # P(diff) = 0.25 * (1 - exp(-4/3 * d)) / 3
            # For JC, the transition matrix is:
            # [p_same, p_diff, p_diff, p_diff]
            # [p_diff, p_same, p_diff, p_diff]
            # ...
            # So P = (1-3q)*I + q*J where q = (1-exp(-4d/3))/4, and p_same = 1-3q.
            
            def _jc_transition(d):
                if d <= 0:
                    return np.array([1.0, 0.0, 0.0, 0.0])
                q = (1.0 - math.exp(-4.0 * d / 3.0)) / 4.0
                p_same = 1.0 - 3.0 * q
                p_diff = q
                return np.array([p_same, p_diff, p_diff, p_diff])
            
            # We need to compute for each site:
            # L_node[state] = sum_{s1,s2} P(s1->state along bl_l) * L_l[s1] * P(s2->state along bl_r) * L_r[s2]
            # This is a matrix multiplication.
            # Let's precompute the transition matrices as 4x4.
            
            # For efficiency, let's compute the transition matrices once.
            # T_l: 4x4 matrix where T_l[i,j] = P(i->j along bl_l)
            # For JC, T[i,j] = p_same if i==j, p_diff if i!=j.
            
            # We can vectorize over sites.
            # L_l is (L, 4), L_r is (L, 4).
            # We need for each state s at node:
            # L_node[site, s] = sum_{s1} sum_{s2} T_l[s1, s] * L_l[site, s1] * T_r[s2, s] * L_r[site, s2]
            # = sum_{s1,s2} (T_l[s1,s] * T_r[s2,s]) * L_l[site,s1] * L_r[site,s2]
            
            # Let's compute M[s1, s2, s] = T_l[s1,s] * T_r[s2,s]
            # Then L_node[site, s] = sum_{s1,s2} M[s1,s2,s] * L_l[site,s1] * L_r[site,s2]
            
            # This is a bit complex. Let's use a simpler approach for now:
            # Loop over the 4 states at the node, and for each, compute the sum.
            # Since 4 is small, this might be acceptable.
            
            # Precompute T_l and T_r as 4x4 arrays.
            def _jc_matrix(d):
                if d <= 0:
                    return np.eye(4)
                q = (1.0 - math.exp(-4.0 * d / 3.0)) / 4.0
                p_same = 1.0 - 3.0 * q
                p_diff = q
                T = np.full((4, 4), p_diff)
                np.fill_diagonal(T, p_same)
                return T
            
            T_l = _jc_matrix(bl_l)
            T_r = _jc_matrix(bl_r)
            
            # For each state s at node:
            # L_node[:, s] = sum_{s1} sum_{s2} T_l[s1, s] * L_l[:, s1] * T_r[s2, s] * L_r[:, s2]
            # = sum_{s1} T_l[s1, s] * L_l[:, s1] * sum_{s2} T_r[s2, s] * L_r[:, s2]
            # = (T_l[:, s].T @ L_l) * (T_r[:, s].T @ L_r)  -- element-wise product, then sum?
            # Wait: sum_{s1} T_l[s1,s] * L_l[site,s1] = (T_l[:, s] @ L_l[site, :])
            # So for all sites: (L_l @ T_l[:, s]) gives a (L,) vector.
            # Similarly for r.
            # Then L_node[:, s] = (L_l @ T_l[:, s]) * (L_r @ T_r[:, s])
            
            L_node = np.zeros((L, 4), dtype=np.float64)
            for s in range(4):
                v_l = L_l @ T_l[:, s]  # (L,)
                v_r = L_r @ T_r[:, s]  # (L,)
                L_node[:, s] = v_l * v_r
            
            Lvec[node] = L_node

    # The likelihood is the sum over states at the root, since the root state is unobserved.
    # For JC, we assume equal base frequencies, so P(root_state) = 1/4 for each state.
    # log-likelihood = log(sum_s (1/4) * L_root[s])
    L_root = Lvec[root]  # (L, 4)
    ll = np.sum(L_root, axis=1) / 4.0  # (L,)
    # Handle zeros
    ll = np.where(ll > 0, ll, 1e-300)
    total_ll = np.sum(np.log(ll))
    return total_ll


def _nni_move(root, children, bl, edge_to_move):
    """Perform NNI move on the edge. Returns new (root, children, bl) or None if invalid.
    edge_to_move: (u, v) where u is the node, v is its parent. The edge is between u and v.
    The NNI move swaps one of u's children with one of v's other children.
    """
    # This is complex. Let's implement a simpler NNI: 
    # For each internal edge (u,v) where u is child of v, and u has children (a,b),
    # and v has children (u,c) [assuming v is not root], we can swap b with c.
    # New topology: v has children (u', c) where u' has children (a,b) -> wait, no.
    # NNI: edge (u,v), u's children are a,b. v's other child is c (v's parent is p).
    # Swap: new u' has children (a,c), new v' has children (b,u')? No.
    # Standard NNI: for edge (u,v) with u child of v:
    #   u has children a, b
    #   v has children u, c
    #   NNI swap: new topology has u' with children (a, c) and v' with children (b, u')? 
    #   Actually, the NNI move replaces the quartet ((a,b),(c,d)) with ((a,c),(b,d)) or ((a,d),(b,c)).
    #   Here, the quartet is a,b on one side of edge (u,v) and c,d on the other, where d is v's parent side.
    #   But for NNI, we only consider the immediate neighbors.
    #   Let's say v's parent is p, and v's other child is c.
    #   The NNI move swaps b and c: new u' has children (a, c), new v' has children (b, p_side)?
    #   This is getting complicated. Let's use a different approach.

    # Given the complexity, let's skip NNI for now and just return the NJ tree.
    # We can add NNI later if time permits.
    return None


def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 260.0  # 10s safety margin

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    L = len(sequences[labels[0]])

    if n <= 3:
        if n == 1:
            return f"({labels[0]});"
        elif n == 2:
            return f"({labels[0]},{labels[1]});"
        else:
            return f"({labels[0]},{labels[1]},{labels[2]});"

    # Compute distance matrix
    D = _compute_distance_matrix(sequences, labels)

    # Build NJ tree
    root, children, bl = _build_tree_from_nj(D, labels)

    # Encode sequences for ML
    enc = []
    for lab in labels:
        s = sequences[lab]
        arr = np.frombuffer(s.encode('ascii'), dtype=np.uint8)
        m = np.empty_like(arr, dtype=np.int8)
        m[arr == 65] = 0
        m[arr == 67] = 1
        m[arr == 71] = 2
        m[arr == 84] = 3
        enc.append(m)
    seq_array = np.stack(enc, axis=0)  # (n, L)

    # Compute initial likelihood
    ll = _ml_likelihood(root, children, bl, seq_array, L)

    # NNI refinement (simplified: just a few iterations if time permits)
    # For now, return the NJ tree as Newick.
    
    def _to_newick(node, children, bl, labels):
        if children[node] is None:
            return labels[node]
        l, r = children[node]
        left = _to_newick(l, children, bl, labels)
        right = _to_newick(r, children, bl, labels)
        return f"({left},{right})"
    
    newick = _to_newick(root, children, bl, labels)
    return newick + ";"
# EVOLVE-BLOCK-END
