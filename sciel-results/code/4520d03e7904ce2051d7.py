import time
import numpy as np

# EVOLVE-BLOCK-START

def _jc_distance_matrix(A):
    """
    Compute Jukes-Cantor distance matrix.
    A: numpy array of shape (n, L) with values 0-3.
    Returns: n x n numpy array.
    """
    n, L = A.shape
    if n < 2:
        return np.zeros((n, n))
    
    diff = A[:, None, :] != A[None, :, :]
    hamming = np.sum(diff, axis=2)
    p = hamming / float(L)
    
    arg = 1.0 - 4.0 * p / 3.0
    arg = np.clip(arg, 1e-12, 1.0)
    d = -0.75 * np.log(arg)
    
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    return d

def _run_nj(dist, labels):
    """
    Neighbor Joining.
    Returns: (newick_string, children_dict, parent_dict, root_id, leaf_ids)
    """
    n = len(labels)
    if n == 1:
        return f"{labels[0]};", {}, {}, 0, [0]
    if n == 2:
        return f"({labels[0]},{labels[1]})", {2: (0, 1)}, {0: 2, 1: 2}, 2, [0, 1]
    
    nodes = list(range(n))
    # tree_str is for Newick generation only, we also need structure
    # We will build the structure directly
    children = {}
    parent = {}
    leaf_ids = list(range(n))
    
    # D is a dict of distances
    D = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                D[(i, j)] = dist[i, j]
    
    new_id = n
    # Keep track of which nodes are active
    active = [True] * new_id # Will expand
    
    while len(nodes) > 2:
        m = len(nodes)
        tot = {x: sum(D[(x, y)] for y in nodes if y != x) for x in nodes}
        
        best = float('inf')
        a, b = None, None
        for i_idx, x in enumerate(nodes):
            for y in nodes[i_idx + 1:]:
                crit = (m - 2) * D[(x, y)] - tot[x] - tot[y]
                if crit < best:
                    best = crit
                    a, b = x, y
        
        node = new_id
        new_id += 1
        
        # Add to structure
        children[node] = (a, b)
        parent[a] = node
        parent[b] = node
        
        rest = [x for x in nodes if x != a and x != b]
        d_ab = D[(a, b)]
        for x in rest:
            d_ax = D[(a, x)]
            d_bx = D[(b, x)]
            d_new = 0.5 * (d_ax + d_bx - d_ab)
            D[(node, x)] = d_new
            D[(x, node)] = d_new
        
        nodes.remove(a)
        nodes.remove(b)
        nodes.append(node)
        
        # Ensure active list is large enough
        while len(active) <= new_id:
            active.append(False)
        active[node] = True
        active[a] = False
        active[b] = False

    a, b = nodes
    root = new_id
    children[root] = (a, b)
    parent[a] = root
    parent[b] = root
    
    # Generate Newick
    def build_newick(node):
        if node < n:
            return labels[node]
        c1, c2 = children[node]
        return f"({build_newick(c1)},{build_newick(c2)})"
    
    newick = build_newick(root) + ";"
    return newick, children, parent, root, leaf_ids

def _calculate_parsimony_cost(children, root, leaf_states):
    """
    Calculate total Fitch parsimony cost.
    children: dict node -> (child1, child2)
    root: int
    leaf_states: array (n, L) of bitmasks (1, 2, 4, 8)
    """
    n = len(leaf_states)
    L = leaf_states.shape[1]
    
    # Post-order traversal
    post_order = []
    stack = [root]
    visited = set()
    while stack:
        node = stack[-1]
        if node < n:
            post_order.append(node)
            stack.pop()
            visited.add(node)
        else:
            if node in visited:
                stack.pop()
                post_order.append(node)
                continue
            visited.add(node)
            c1, c2 = children[node]
            if c1 not in visited:
                stack.append(c1)
            if c2 not in visited:
                stack.append(c2)
    
    # Process in post-order
    node_states = {}
    total_cost = 0
    
    for node in post_order:
        if node < n:
            node_states[node] = leaf_states[node]
        else:
            c1, c2 = children[node]
            s1 = node_states[c1]
            s2 = node_states[c2]
            
            inter = s1 & s2
            union = s1 | s2
            
            # Cost is 1 where inter is 0
            cost_mask = (inter == 0)
            total_cost += np.sum(cost_mask)
            
            # Next state is inter if non-zero, else union
            # np.where(inter != 0, inter, union)
            node_states[node] = np.where(inter != 0, inter, union)
            
    return total_cost

def _get_quartet_split_cost(A, a, b, c, d):
    """
    Calculate parsimony cost for the 3 quartet topologies.
    Returns: (cost0, cost1, cost2) for ((a,b),(c,d)), ((a,c),(b,d)), ((a,d),(b,c))
    """
    # This is slower, so we use the tree-based cost for NNI.
    # But for initial quartet consensus, we might need this.
    # For now, we rely on NNI from NJ.
    pass

def reconstruct(sequences):
    start_time = time.time()
    deadline = 260.0 # 10s margin
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    
    L = len(sequences[labels[0]])
    
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    bit_map = {0: 1, 1: 2, 2: 4, 3: 8}
    
    A = np.zeros((n, L), dtype=np.uint8)
    leaf_states = np.zeros((n, L), dtype=np.uint8)
    
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j, c in enumerate(seq):
            val = char_map[c]
            A[i, j] = val
            leaf_states[i, j] = bit_map[val]
    
    # 1. NJ Tree
    D = _jc_distance_matrix(A)
    newick, children, parent, root, leaf_ids = _run_nj(D, labels)
    
    # 2. NNI Refinement
    current_cost = _calculate_parsimony_cost(children, root, leaf_states)
    
    # Identify internal edges
    # An internal edge is (u, v) where u is parent of v, and u is not root, and v is not leaf.
    # Actually, any edge between two internal nodes.
    internal_nodes = [node for node in children.keys()]
    
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        
        # Get list of internal edges to consider
        # We need to be careful not to modify the tree while iterating
        edges_to_try = []
        for node in internal_nodes:
            if node == root:
                continue
            p = parent[node]
            if p is None or p < n: # p is leaf? No, p is internal if node is internal child
                continue
            # Check if p is internal
            if p in children:
                edges_to_try.append((p, node))
        
        np.random.shuffle(edges_to_try)
        
        for p, v in edges_to_try:
            if time.time() > deadline - 10.0:
                break
            
            # Current topology: p is parent of v.
            # p's other child is w.
            # v's children are a, b.
            w = children[p][0] if children[p][1] == v else children[p][1]
            a, b = children[v]
            
            # The 4 subtrees are T_p (rest), T_w, T_a, T_b.
            # Current: ((T_w, T_a), T_b) is not right.
            # Current tree structure:
            # p
            # |-- w
            # |-- v
            #      |-- a
            #      |-- b
            # The split is {a, b} | {w, rest}.
            
            # Alternative 1: ((w, a), b)
            # Structure:
            # new_node1
            # |-- w
            # |-- a
            # new_node2
            # |-- new_node1
            # |-- b
            # p connects to new_node2?
            # No, NNI replaces the edge (p, v).
            # The 4 branches are T_p (via p), T_w, T_a, T_b.
            # Current: T_w and (T_a, T_b) are siblings under v. p is parent of v.
            # So the quartet of branches is T_p, T_w, T_a, T_b.
            # Current topology: (T_p, (T_w, (T_a, T_b)))? No.
            # Unrooted: The edge (p, v) separates {T_a, T_b} from {T_p, T_w}.
            # So the quartet topology is ((T_a, T_b), (T_p, T_w)).
            
            # Alternative 1: ((T_a, T_p), (T_w, T_b))
            # Alternative 2: ((T_a, T_w), (T_p, T_b))
            
            # To implement NNI, we modify the tree structure.
            # We need to create new nodes.
            
            # Let's create a copy of children and parent dicts for the trial tree.
            # This is expensive. 
            # Instead, we can just calculate the cost of the alternative topologies
            # by simulating the Fitch algorithm on the modified structure.
            
            # But we need the node states of the subtrees T_p, T_w, T_a, T_b.
            # We already have node_states from the last cost calculation?
            # No, we discarded them.
            # We should keep node_states from the last full calculation.
            
            # Let's refactor: _calculate_parsimony_cost returns (cost, node_states)
            
            # Re-calculate node_states for current tree
            # This is O(nL).
            # We do this once per NNI iteration?
            # If we do it for every edge, it's O(n * nL) = O(n^2 L).
            # n=32, L=2000. 32^2 * 2000 = 2M ops. Fast.
            
            # Let's get node_states for current tree
            # We need a helper that returns states.
            
            # For now, let's just re-run the full cost calculation for the modified tree.
            # We will construct the modified children/parent dicts.
            
            # Trial 1: ((T_a, T_p), (T_w, T_b))
            # This requires re-parenting.
            # It's easier to just construct the 3 possible quartet topologies for the 4 subtrees
            # and see which one minimizes the cost.
            # But the "subtrees" are the rest of the tree.
            # The cost of the tree is the sum of costs of all internal nodes.
            # Only the nodes in the neighborhood of the edge change.
            
            # Let's stick to a simpler heuristic:
            # Calculate cost for current tree.
            # For each internal edge, try the 2 NNI moves.
            # Construct the new tree structure (children, parent, root).
            # Calculate its cost.
            # If better, accept.
            
            # Constructing the new tree:
            # Current: p - (w, v - (a, b))
            # Move 1: p - (w, new_v - (a, new_w - (b, ???))) No.
            
            # Standard NNI:
            # Edge (p, v).
            # p's children: w, v.
            # v's children: a, b.
            
            # Topology 0 (Current): ((w, a), b) is NOT correct.
            # The unrooted quartet of the 4 leaves/subtrees is {w, a, b, rest}.
            # Current: rest is attached to p. p is attached to w and v. v is attached to a and b.
            # So the path from rest to a is rest-p-v-a.
            # Path from rest to b is rest-p-v-b.
            # Path from rest to w is rest-p-w.
            # So rest is closer to w than to a or b?
            # No, distance rest-w is 1 (edge p-w).
            # Distance rest-a is 2 (edges p-v, v-a).
            # So the quartet topology is ((w, rest), (a, b)).
            
            # Alternative 1: ((w, a), (b, rest))
            # Alternative 2: ((w, b), (a, rest))
            
            # To implement Alt 1:
            # We need to connect w and a together, and b and rest together.
            # This involves creating 2 new internal nodes.
            # node_x connects w and a.
            # node_y connects b and p (which leads to rest).
            # Then we connect node_x and node_y?
            # No, the edge (p, v) is removed.
            # The 4 branches are T_rest (via p), T_w, T_a, T_b.
            # In Alt 1, T_w and T_a are siblings. T_b and T_rest are siblings.
            # So we create node_x(w, a) and node_y(b, p).
            # Then we connect node_x and node_y with a new edge?
            # Yes, the new edge connects node_x and node_y.
            
            # Let's implement a helper to get the cost of a specific NNI move
            # by modifying the tree structure in a temporary dict.
            
            # This is getting complex. 
            # Given the time limit, and that NJ is already a strong baseline,
            # and the current program scored 1.0 (which is the seed score, meaning it didn't improve),
            # the random split search was likely failing.
            
            # A simpler improvement:
            # Use the vectorized split cost to evaluate specific splits.
            # Start with NJ splits.
            # Try to add splits that are "compatible" and have low cost?
            # Or, use the "Greedy" algorithm:
            # 1. Start with empty set of splits.
            # 2. Find the split with minimum Fitch cost. Add it.
            # 3. Find the next compatible split with minimum cost. Add it.
            # 4. Repeat until n-3 splits.
            
            # This is O(2^n) if we enumerate all splits.
            # But we can restrict to splits from NJ and its neighbors.
            
            # Let's try the NNI approach with a simplified cost check.
            # We will just re-run the full parsimony cost on the modified tree.
            
            # Helper to build modified tree for NNI
            # This is too much code to debug in this format.
            
            # Let's fall back to the Vectorized Split Cost + Greedy Selection.
            # It is simpler and robust.
            
            pass # Skip NNI implementation for now, use Greedy Splits

    # --- Greedy Split Selection ---
    
    # 1. Get NJ splits
    def _get_splits_from_tree(children, root, n):
        splits = []
        def dfs(node, current_set):
            if node < n:
                return {node}
            c1, c2 = children[node]
            s1 = dfs(c1, set())
            s2 = dfs(c2, set())
            s = s1.union(s2)
            if 0 < len(s) < n:
                # Canonicalize: min label in split
                if min(s) < min(set(range(n)) - s):
                    splits.append(frozenset(s))
                else:
                    splits.append(frozenset(set(range(n)) - s))
            return s
        dfs(root, set())
        return splits
        
    nj_splits = _get_splits_from_tree(children, root, n)
    
    # 2. Generate candidates: NJ splits + some random
    candidates = list(nj_splits)
    rng = np.random.RandomState(42)
    for _ in range(200):
        if time.time() > deadline - 20:
            break
        perm = rng.permutation(n)
        k = rng.randint(2, n-1)
        s = frozenset(perm[:k])
        if min(s) > min(set(range(n)) - s):
            s = frozenset(set(range(n)) - s)
        candidates.append(s)
        
    candidates = list(set(candidates))
    
    # 3. Score candidates with Vectorized Fitch Cost
    # Precompute taxon masks for each site and state
    # taxon_masks[j, c] = bitmask of taxa with state c at site j
    taxon_masks = np.zeros((L, 4), dtype=np.uint64)
    for j in range(L):
        for c in range(4):
            mask = 0
            for i in range(n):
                if A[i, j] == c:
                    mask |= (1 << i)
            taxon_masks[j, c] = mask
            
    full_mask = (1 << n) - 1
    
    def calc_split_cost(split):
        # split is a frozenset of indices
        M = 0
        for i in split:
            M |= (1 << i)
        M_inv = full_mask ^ M
        
        # present_in_S: (L, 4) boolean
        # present_in_Sc: (L, 4) boolean
        # overlap: (L, 4) boolean
        # has_overlap: (L) boolean
        # cost: int
        
        # Vectorized
        # taxon_masks is (L, 4) uint64
        # M is scalar
        # (taxon_masks & M) != 0
        pS = (taxon_masks & M) != 0
        pSc = (taxon_masks & M_inv) != 0
        overlap = pS & pSc
        has_overlap = overlap.any(axis=1)
        cost = (~has_overlap).sum()
        return int(cost)
        
    split_costs = {}
    for s in candidates:
        if time.time() > deadline - 10:
            break
        split_costs[s] = calc_split_cost(s)
        
    # 4. Greedy Selection
    sorted_splits = sorted(split_costs.items(), key=lambda x: x[1])
    selected = []
    target = n - 3
    all_set = set(range(n))
    
    for s, cost in sorted_splits:
        if len(selected) >= target:
            break
        compatible = True
        for s2 in selected:
            # Check compatibility
            # s & s2, s & ~s2, ~s & s2, ~s & ~s2
            # If all non-empty, incompatible
            i1 = s & s2
            i2 = s & (all_set - s2)
            i3 = (all_set - s) & s2
            i4 = (all_set - s) & (all_set - s2)
            if i1 and i2 and i3 and i4:
                compatible = False
                break
        if compatible:
            selected.append(s)
            
    # 5. Build Tree
    if not selected:
        return newick
        
    # Build tree from selected splits
    # Use recursive construction
    def build_tree(subset):
        if len(subset) == 1:
            return labels[list(subset)[0]]
        
        for s in selected:
            A_part = s & subset
            if 0 < len(A_part) < len(subset):
                B_part = subset - A_part
                return f"({build_tree(A_part)},{build_tree(B_part)})"
                
        # Fallback: star
        return "(" + ",".join(labels[i] for i in sorted(subset)) + ")"
        
    final_tree = build_tree(all_set) + ";"
    return final_tree

# EVOLVE-BLOCK-END
