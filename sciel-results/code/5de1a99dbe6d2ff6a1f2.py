import time
import math
import numpy as np

_START_TIME = None
_TIME_LIMIT = 245.0  # Safety margin for 270s limit

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_left():
    _init_time()
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))

def _seqs_to_array(sequences, labels):
    """Convert sequences to integer matrix (0=A, 1=C, 2=G, 3=T)."""
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    n = len(labels)
    L = len(sequences[labels[0]])
    mat = np.zeros((n, L), dtype=np.int8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j, c in enumerate(seq):
            mat[i, j] = seq_map[c]
    return mat

def _jc_matrix(seqs_int):
    """Compute JC-corrected distance matrix."""
    n, L = seqs_int.shape
    # Count mismatches for each pair
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p = np.sum(seqs_int[i] != seqs_int[j]) / L
            p = min(max(p, 0.0), 0.7499)
            d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dists[i, j] = d
            dists[j, i] = d
    return dists

def _nj(dists):
    """
    Standard Neighbor Joining.
    Returns a list of nodes represented as tuples (left, right) for internal nodes
    or a leaf label (string) for leaves.
    We'll use a dictionary-based tree representation for easier manipulation.
    """
    n = len(dists)
    if n < 3:
        if n == 1:
            return 0
        elif n == 2:
            return (0, 1)
    
    # Active nodes: initially 0..n-1
    # We'll track the tree structure using a node ID scheme.
    # Each node is either a leaf (int < n) or an internal node (int >= n).
    # For internal nodes, we store their children in a dict.
    
    nodes = {}  # node_id -> (left_child, right_child) for internal
    # Active set
    active = list(range(n))
    d = dists.copy()
    next_id = n
    
    while len(active) > 2:
        k = len(active)
        # r_i = sum of distances from i to all others
        r = np.array([np.sum(d[active.index(i), active.index(j)] if i != j else 0.0 for j in active) for i in active])
        
        # Q matrix
        q = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                if i == j:
                    q[i, j] = np.inf
                else:
                    ai = active[i]
                    aj = active[j]
                    q[i, j] = (k - 2) * d[ai, aj] - r[i] - r[j]
        
        # Find min
        min_val = np.inf
        min_i, min_j = 0, 1
        for i in range(k):
            for j in range(i + 1, k):
                if q[i, j] < min_val:
                    min_val = q[i, j]
                    min_i, min_j = i, j
        
        ai = active[min_i]
        aj = active[min_j]
        
        # Create new internal node
        new_id = next_id
        next_id += 1
        nodes[new_id] = (ai, aj)
        
        # Remove ai, aj from active, add new_id
        new_active = [x for x in active if x != ai and x != aj]
        new_active.append(new_id)
        
        # Update distance matrix
        # d(new_id, x) = 0.5 * (d(ai, x) + d(aj, x) - d(ai, aj))
        # We need to maintain the distance matrix for active nodes.
        # Let's rebuild it.
        
        new_d = np.zeros((len(new_active), len(new_active)))
        for i_idx, ni in enumerate(new_active):
            for j_idx, nj in enumerate(new_active):
                if i_idx == j_idx:
                    new_d[i_idx, j_idx] = 0.0
                elif i_idx < j_idx:
                    # Determine the original indices
                    # This is getting complicated with the index mapping.
                    # Let's use a dictionary for distances.
                    pass
        
        # Alternative: use a 2D array indexed by node_id, but sparse?
        # Or just keep a dict of distances between active nodes.
        
        # Let's switch to a dict-based distance storage for simplicity and correctness.
        # This might be slower but n is small.
        
        # Actually, let's just use the array but carefully map indices.
        # The issue is that `d` is indexed by original node IDs, but `active` changes.
        # Let's maintain `d` as a dict: d[(u, v)] for u < v.
        
        # Restart with dict-based approach for clarity.
        break
    
    # Let's implement a clean version using dicts for both tree and distances.
    
    # Tree: node_id -> (left, right)
    # Distances: (u, v) -> dist, where u < v
    
    tree = {}
    active = list(range(n))
    dist = {}
    for i in range(n):
        for j in range(i + 1, n):
            dist[(i, j)] = dists[i, j]
            dist[(j, i)] = dists[i, j]
    
    next_id = n
    
    def get_dist(u, v):
        return dist.get((u, v), dist.get((v, u), 0.0))
    
    while len(active) > 2:
        k = len(active)
        # r_i
        r = {}
        for i in active:
            r[i] = sum(get_dist(i, j) for j in active if j != i)
        
        # Q matrix
        q = {}
        for i in active:
            for j in active:
                if i != j:
                    q[(i, j)] = (k - 2) * get_dist(i, j) - r[i] - r[j]
        
        # Find min
        min_val = np.inf
        min_pair = (active[0], active[1])
        for (i, j), val in q.items():
            if val < min_val:
                min_val = val
                min_pair = (i, j)
        
        ai, aj = min_pair
        d_ij = get_dist(ai, aj)
        
        # New node
        new_id = next_id
        next_id += 1
        tree[new_id] = (ai, aj)
        
        # Update distances to new node
        for x in active:
            if x != ai and x != aj:
                d_new = 0.5 * (get_dist(ai, x) + get_dist(aj, x) - d_ij)
                d_new = max(0.0, d_new)
                dist[(new_id, x)] = d_new
                dist[(x, new_id)] = d_new
        
        # Remove ai, aj from active and dist
        active.remove(ai)
        active.remove(aj)
        active.append(new_id)
        
        for x in active:
            if x != new_id:
                # Remove old distances involving ai, aj
                dist.pop((ai, x), None)
                dist.pop((x, ai), None)
                dist.pop((aj, x), None)
                dist.pop((x, aj), None)
        dist.pop((ai, aj), None)
        dist.pop((aj, ai), None)
    
    if len(active) == 2:
        a, b = active
        root_id = next_id
        next_id += 1
        tree[root_id] = (a, b)
        return root_id, tree
    
    # This shouldn't happen for n >= 3
    return active[0] if len(active) == 1 else active[0], tree

def _tree_to_newick(root_id, tree, labels):
    """Convert tree to Newick string."""
    def build(node):
        if node in tree:
            left, right = tree[node]
            return f"({build(left)},{build(right)})"
        else:
            return labels[node]
    
    return build(root_id)

def _parse_newick(newick, labels):
    """
    Parse Newick into a tree structure.
    Returns a dict node -> (left, right) and root_id.
    Leaf nodes are integers (indices into labels).
    Internal nodes are negative integers or integers >= len(labels).
    """
    # This is a simplified parser for the specific format we generate.
    # We'll assume the Newick is valid and matches our format.
    # Actually, it's easier to keep the tree structure in memory and not round-trip through Newick.
    # But for NNI, we need to manipulate the tree.
    
    # Let's assume we keep the tree in memory as `tree` dict and `root_id`.
    # So we don't need this parser if we keep the structure.
    
    pass

def _ll_jc(tree, root_id, seqs_int):
    """
    Compute the log-likelihood of the sequences under JC model, given the tree topology.
    We assume equal branch lengths or optimize them?
    For simplicity, let's first use fixed branch lengths from NJ or just 1.0.
    Actually, to do proper ML, we should optimize branch lengths.
    But for a fast approximation, let's use the Felsenstein pruning algorithm with JC.
    
    For JC, the transition probability matrix is:
    P(t) = [[1/4 + 3/4 * exp(-4t/3), ...], ...]
    Let a = 1/4 + 3/4 * exp(-4t/3)
    Let b = 1/4 - 1/4 * exp(-4t/3)
    P = [[a, b, b, b], [b, a, b, b], [b, b, a, b], [b, b, b, a]]
    
    The likelihood calculation involves summing over all possible nucleotides at internal nodes.
    For JC, due to symmetry, we can simplify.
    Let L_v(c) be the likelihood of the subtree at node v given that v has nucleotide c.
    For a leaf v with observed nucleotide o:
    L_v(c) = 1 if c == o, else 0.
    For an internal node v with children u, w:
    L_v(c) = sum_u sum_w P(c->u, t_vu) * L_u(u) * P(c->w, t_vw) * L_w(w)
    Wait, the probability is P(u->c, t) or P(c->u, t)? It's symmetric for JC.
    L_v(c) = (sum_u P(c->u, t_vu) * L_u(u)) * (sum_w P(c->w, t_vw) * L_w(w))
    
    Due to symmetry of JC, L_v(c) might be the same for all c?
    No, because the leaf data breaks symmetry.
    But we can exploit the fact that P(c->u) depends only on whether c==u or not.
    So L_u can be summarized by two values: L_u(same) and L_u(diff)?
    No, L_u is a vector of 4.
    
    For n=32 and L=2000, doing this naively might be slow.
    But let's try.
    
    We need branch lengths. Let's use the ones from NJ or just set them to 1.0 for now to get a topology.
    Actually, if we don't optimize branch lengths, the likelihood will favor certain topologies incorrectly.
    We should at least optimize branch lengths for each NNI move.
    
    This is getting complex. Let's simplify:
    1. Get NJ tree.
    2. For each NNI move, compute the likelihood with branch lengths optimized (or just fixed at 1.0).
    3. Accept if better.
    
    Given the time limit, let's try a simpler approach:
    - Use NJ tree.
    - Perform a few rounds of NNI, using a fast likelihood approximation.
    - If time is tight, just return NJ.
    
    For now, let's implement a basic NNI with likelihood using fixed branch lengths (e.g., 1.0).
    This is a heuristic, but it might improve topology.
    """
    n, L = seqs_int.shape
    # We need to root the tree for the pruning algorithm.
    # Let's root at an arbitrary internal node or leaf.
    # We'll use a recursive DFS.
    
    # First, we need to know the branch lengths.
    # Let's assume all branch lengths are 1.0 for now.
    # This is a placeholder for a more sophisticated approach.
    
    # For JC, P_same = 0.25 + 0.75 * exp(-4/3)
    # P_diff = 0.25 - 0.25 * exp(-4/3)
    t = 1.0
    exp_term = math.exp(-4.0 * t / 3.0)
    p_same = 0.25 + 0.75 * exp_term
    p_diff = 0.25 - 0.25 * exp_term
    
    # We'll compute the likelihood using dynamic programming.
    # We need to map leaves to their indices in seqs_int.
    # The tree nodes are integers. Leaves are 0..n-1.
    
    def calc_likelihood(node, parent=None):
        """
        Returns a list of 4 likelihoods for the subtree rooted at `node`
        given that `node` has A, C, G, T respectively.
        """
        if node in tree:
            # Internal node
            left, right = tree[node]
            # Get likelihoods from children
            ll_left = calc_likelihood(left, node)
            ll_right = calc_likelihood(right, node)
            
            # For each possible nucleotide c at `node`, compute the likelihood.
            ll = []
            for c in range(4):
                # Sum over left child nucleotides
                sum_left = 0.0
                for u in range(4):
                    p = p_same if u == c else p_diff
                    sum_left += p * ll_left[u]
                
                # Sum over right child nucleotides
                sum_right = 0.0
                for w in range(4):
                    p = p_same if w == c else p_diff
                    sum_right += p * ll_right[w]
                
                ll.append(sum_left * sum_right)
            
            return ll
        else:
            # Leaf node
            # node is the index in labels, which is the index in seqs_int
            observed = seqs_int[node]
            ll = []
            for c in range(4):
                # The likelihood of the leaf data given that the node has nucleotide c
                # is the product over all sites of P(c -> observed_site)
                # For JC, this is the same for all sites.
                # Let's compute the log-likelihood for this leaf given c.
                # But we need to multiply over sites.
                # This is O(L) per leaf, which is fine.
                
                # Actually, we can precompute the likelihood for each leaf for each possible nucleotide.
                # Let's do that for efficiency.
                pass
            
            # For now, let's just return a placeholder.
            # This function is incomplete.
            return [1.0, 1.0, 1.0, 1.0]
    
    # This is getting too complex to implement correctly in a short time.
    # Let's fall back to returning the NJ tree without refinement.
    # This is safer and will definitely finish in time.
    
    return 0.0  # Placeholder

def reconstruct(sequences):
    _init_time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    if n_taxa < 3:
        if n_taxa == 1:
            return labels[0] + ";"
        elif n_taxa == 2:
            return f"({labels[0]},{labels[1]});"
            
    # Convert sequences to integer matrix
    seqs_int = _seqs_to_array(sequences, labels)
    
    # Compute distance matrix
    dists = _jc_matrix(seqs_int)
    
    # Run NJ
    root_id, tree = _nj(dists)
    
    # Convert to Newick
    newick = _tree_to_newick(root_id, tree, labels)
    
    return newick + ";"

# EVOLVE-BLOCK-END
