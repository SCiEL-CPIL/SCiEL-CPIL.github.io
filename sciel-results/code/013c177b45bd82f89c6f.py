<<<<<<< SEARCH
def _nni_polish(newick, D, labels):
    """
    Perform NNI polish to improve topology based on distance fit.
    """
    nodes, adj, leaves = _parse_newick(newick)
    n = len(nodes)
    
    if n <= 3:
        return newick
    
    # Map leaf labels to node IDs
    label_to_node = {leaf: node for node, leaf in leaves.items()}
    
    # Map node IDs to labels for output
    node_to_label = {node: leaf for leaf, node in label_to_node.items()}
    
    # We need to compute the cost of the tree.
    # Cost = Sum of squared differences between D[i][j] and path_length(i, j)
    # To compute path lengths, we need branch lengths.
    # We can estimate branch lengths using least squares for the current topology.
    # This is O(N^2) per topology evaluation.
    
    def compute_branch_lengths(adj, D, label_to_node):
        """
        Estimate branch lengths using least squares.
        Returns dict: (u, v) -> length
        """
        # Build system of equations
        # For each pair of leaves (i, j): sum of branch lengths on path = D[i][j]
        # This is a linear system A x = b
        # A is (N_pairs x N_edges), b is (N_pairs x 1)
        # N_pairs ~ N^2/2, N_edges ~ N-3
        # For N=32, N_pairs ~ 500, N_edges ~ 29.
        # Solving 500x29 system is fast.
        
        edges = []
        edge_idx = {}
        for u in adj:
            for v, _ in adj[u]:
                if u < v:
                    idx = len(edges)
                    edges.append((u, v))
                    edge_idx[(u, v)] = idx
                    edge_idx[(v, u)] = idx
        
        num_edges = len(edges)
        if num_edges == 0:
            return {}
        
        # Get leaf indices
        leaf_nodes = list(label_to_node.values())
        num_leaves = len(leaf_nodes)
        
        # Build A and b
        # We will use numpy least squares
        # A[i, k] = 1 if edge k is on path between leaf i and leaf j, else 0
        # b[i] = D[leaf_i, leaf_j]
        
        # To save memory, we can build the system iteratively or use sparse matrices.
        # But N is small.
        
        # Let's build the full matrix
        num_pairs = num_leaves * (num_leaves - 1) // 2
        A = np.zeros((num_pairs, num_edges))
        b = np.zeros(num_pairs)
        
        pair_idx = 0
        for i in range(num_leaves):
            for j in range(i + 1, num_leaves):
                li = leaf_nodes[i]
                lj = leaf_nodes[j]
                
                # Find path between li and lj in adj
                # BFS
                visited = {li}
                path = []
                queue = [(li, [])]
                found = False
                while queue:
                    node, p = queue.pop(0)
                    if node == lj:
                        found = True
                        path = p
                        break
                    for neighbor, _ in adj[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, p + [(node, neighbor)]))
                
                if found:
                    # path is list of (u, v)
                    for u, v in path:
                        e_idx = edge_idx[(u, v)]
                        A[pair_idx, e_idx] = 1.0
                    
                    # Get distance
                    # D is indexed by labels, but we have node IDs.
                    # We need to map node IDs back to labels to index D.
                    # Wait, D was computed for labels.
                    # label_to_node maps label -> node.
                    # We need node -> label.
                    # We have node_to_label.
                    li_label = node_to_label[li]
                    lj_label = node_to_label[lj]
                    
                    # Get index in D
                    # D is n x n where n is number of labels.
                    # labels is sorted list of labels.
                    # We need the index of li_label in labels.
                    # This is inefficient.
                    # Let's pre-compute a map from label to index in D.
                    pass
        
        # This approach is getting complicated and slow due to repeated BFS and mapping.
        # Alternative: Use the NJ distance matrix D directly with node indices?
        # No, D is defined for leaves.
        
        # Let's simplify:
        # Instead of least squares, use a heuristic:
        # For each edge, length = 0.5 * (dist_to_rest - dist_between_ends)?
        # This is the NJ formula, which is valid for the *correct* topology.
        # For an incorrect topology, it might give negative lengths.
        # We can clamp to 0.
        
        # Let's try a simpler cost: Sum of branch lengths (minimize total length).
        # We need branch lengths.
        # We can compute branch lengths using the formula:
        # l(u, v) = 0.5 * (sum_{x in S_u} D(u, x) + sum_{x in S_v} D(v, x) - sum_{x in S_u, y in S_v} D(x, y) / |S_u||S_v| ... )
        # This is complex.
        
        # Let's use a different cost: **Quadratic Assignment Problem (QAP)**
        # Minimize sum_{i<j} D[i][j] * (1 if i,j separated by edge e else 0) ... no.
        
        # Given the time constraints and the fact that NJ is already 1.0 in inspiration,
        # maybe NNI is overkill and risky to implement correctly in this short timeframe.
        # The "structural change" could be **using a more robust distance metric** or **multiple NJ runs with different random perturbations** (if we had randomness, but we don't).
        
        # Let's stick to the **simplest possible NNI** that uses the **existing distance matrix** to estimate branch lengths locally.
        
        # For an edge (u, v) in the tree:
        # Let S_u be the set of leaves in the subtree rooted at u (when edge is cut).
        # Let S_v be the set of leaves in the subtree rooted at v.
        # The length of the edge can be approximated by:
        # l = 0.5 * (avg_dist(S_u, S_v) - avg_dist(S_u, S_u) - avg_dist(S_v, S_v)) ? No.
        
        # Actually, the NJ formula for the new node distance is:
        # d(new, x) = 0.5 * (d(i, x) + d(j, x) - d(i, j))
        # This suggests that the "length" of the branch connecting the joined pair to the rest is related to these terms.
        
        # Let's try this:
        # For each NNI move, calculate the **Sum of Pairwise Distances** implied by the tree using the **additive** property.
        # If the tree is additive, sum of branch lengths on path = D[i][j].
        # If not, there's error.
        
        # To avoid the complexity of full least squares, I will **skip the NNI polish** and rely on the **robust NJ** implementation.
        # The inspiration 1 scored 1.0 with simple NJ. The current program scored 0.74 with complex NJ+Bootstrap.
        # The difference is likely the Bootstrap Consensus step degrading the result.
        # Therefore, the "structural change" is **removing the Bootstrap Consensus** and **using a clean, optimized NJ**.
        
        return newick

    # Since I decided to skip NNI, this function is a no-op.
    # But I should provide a fallback.
    return newick
=======
def _fitch_score(adj, leaves, label_to_node):
    """
    Compute Fitch parsimony score for a tree topology.
    Returns the number of steps (lower is better).
    """
    # Build a rooted version of the tree for Fitch
    # Root at an arbitrary leaf
    leaf_nodes = [node for node, label in leaves.items()]
    if not leaf_nodes:
        return 0
    
    # Root at first leaf
    root = leaf_nodes[0]
    
    # Build parent/children structure via DFS from root
    parent = {root: None}
    children = {root: []}
    visited = {root}
    stack = [root]
    while stack:
        u = stack.pop()
        for v, _ in adj[u]:
            if v not in visited:
                visited.add(v)
                parent[v] = u
                children[v] = []
                children[u].append(v)
                stack.append(v)
    
    # Fitch algorithm: bottom-up
    # For each leaf: state = {the character at that leaf}
    # For each internal node: intersection of children's states, or union if empty
    # Count steps when union is used
    
    # We need character data at each leaf
    # Actually, Fitch score doesn't need actual characters - it just needs the tree topology
    # and we count the minimum number of changes.
    # But we need to know the characters! Let me reconsider.
    # 
    # Actually for scoring a tree against data, we need the sequences.
    # Let me redesign: pass sequences and labels.
    
    # This function as-is can't work without data. Let me create a separate function.
    return 0  # Placeholder, will be replaced


def _fitch_parsimony(adj, leaves, label_to_node, sequences, labels):
    """
    Compute Fitch parsimony score for a tree topology given sequence data.
    """
    leaf_nodes = [node for node, label in leaves.items()]
    if not leaf_nodes:
        return 0
    
    root = leaf_nodes[0]
    
    parent = {root: None}
    children = {root: []}
    visited = {root}
    stack = [root]
    while stack:
        u = stack.pop()
        for v, _ in adj[u]:
            if v not in visited:
                visited.add(v)
                parent[v] = u
                children[v] = []
                children[u].append(v)
                stack.append(v)
    
    # Get post-order traversal
    postorder = []
    stack2 = [(root, False)]
    while stack2:
        node, processed = stack2.pop()
        if processed:
            postorder.append(node)
        else:
            stack2.append((node, True))
            for child in children[node]:
                stack2.append((child, False))
    
    # Fitch algorithm
    states = {}
    steps = 0
    
    for node in postorder:
        if node in leaves:
            # Leaf: state is the set of characters at this position
            # We need to do this per site, so let's accumulate
            # Actually, Fitch works per site. Let me restructure.
            pass
    
    # Better approach: do Fitch per site
    L = len(sequences[labels[0]]) if labels else 0
    if L == 0:
        return 0
    
    total_steps = 0
    
    for site in range(L):
        # Character at each leaf for this site
        leaf_char = {}
        for node, label in leaves.items():
            leaf_char[node] = set([sequences[label][site]])
        
        # Process in post-order
        node_states = {}
        site_steps = 0
        
        for node in postorder:
            if node in leaves:
                node_states[node] = leaf_char[node]
            else:
                child_states = [node_states[child] for child in children[node]]
                # Intersection
                if child_states:
                    inter = child_states[0].copy()
                    for cs in child_states[1:]:
                        inter &= cs
                    if inter:
                        node_states[node] = inter
                    else:
                        # Union
                        union = set()
                        for cs in child_states:
                            union |= cs
                        node_states[node] = union
                        site_steps += 1
        
        total_steps += site_steps
    
    return total_steps


def _nni_polish(newick, D, labels, sequences):
    """
    Perform NNI polish to improve topology using parsimony score.
    """
    nodes, adj, leaves = _parse_newick(newick)
    n = len(nodes)
    num_leaves = len(leaves)
    
    if num_leaves <= 4:
        return newick
    
    label_to_node = {leaf: node for node, leaf in leaves.items()}
    
    # Compute initial parsimony score
    current_score = _fitch_parsimony(adj, leaves, label_to_node, sequences, labels)
    
    # For NNI, we need to find all internal edges and try swapping
    # An NNI move: for an internal edge (u,v), the four neighbors are a,b,c,d
    # where u connects to a and v, v connects to b and c (and u connects to d)
    # Wait, let me think about this more carefully.
    # 
    # In an unrooted tree, an internal edge connects two internal nodes u and v.
    # u has 3 neighbors: v, x, y
    # v has 3 neighbors: u, z, w
    # The NNI move removes edge (u,v) and adds one of:
    #   (x,z) and (y,w)  [already have (x,y) and (z,w) implicitly]
    #   (x,w) and (y,z)
    # So we swap the connections.
    
    # Find all internal edges
    internal_edges = []
    for u in adj:
        for v, _ in adj[u]:
            if u < v and u not in leaves and v not in leaves:
                internal_edges.append((u, v))
    
    improved = True
    max_rounds = 50
    round_num = 0
    
    while improved and round_num < max_rounds:
        round_num += 1
        improved = False
        
        if _time_left() < 30:
            break
        
        for u, v in internal_edges:
            if _time_left() < 15:
                break
            
            # Get neighbors of u and v (excluding each other)
            u_neighbors = [x for x, _ in adj[u] if x != v]
            v_neighbors = [x for x, _ in adj[v] if x != u]
            
            if len(u_neighbors) != 2 or len(v_neighbors) != 2:
                continue
            
            a, b = u_neighbors
            c, d = v_neighbors
            
            # Current configuration: u connected to a, b, v; v connected to c, d, u
            # Alternative 1: u connected to a, b, c; v connected to d, u, ... 
            # Actually: remove (u,v), add (u,c) and (v,b)
            # Or: remove (u,v), add (u,d) and (v,b)
            
            # Try swap 1: connect u-c and v-b (instead of u-v, a-b on u, c-d on v)
            # New: u connects to a, c; v connects to b, d; and u-v removed
            # Wait, that's not right. Let me think again.
            
            # Original: 
            #   u -- a
            #   u -- b  
            #   u -- v
            #   v -- c
            #   v -- d
            
            # NNI alternative 1:
            #   u -- a
            #   u -- c
            #   v -- b
            #   v -- d
            #   (u,v) removed
            
            # NNI alternative 2:
            #   u -- a
            #   u -- d
            #   v -- b
            #   v -- c
            #   (u,v) removed
            
            # Try alternative 1
            # Remove (u,v), (u,b), (v,c)
            # Add (u,c), (v,b)
            
            # Make a copy of adj
            adj_copy = {k: list(v2) for k, v2 in adj.items()}
            
            # Remove edges
            adj_copy[u] = [(x, l) for x, l in adj_copy[u] if x not in (v, b)]
            adj_copy[v] = [(x, l) for x, l in adj_copy[v] if x not in (u, c)]
            adj_copy[b] = [(x, l) for x, l in adj_copy[b] if x != u]
            adj_copy[c] = [(x, l) for x, l in adj_copy[c] if x != v]
            
            # Add edges
            adj_copy[u].append((c, 0.0))
            adj_copy[c].append((u, 0.0))
            adj_copy[v].append((b, 0.0))
            adj_copy[b].append((v, 0.0))
            
            new_score = _fitch_parsimony(adj_copy, leaves, label_to_node, sequences, labels)
            
            if new_score < current_score:
                # Accept
                adj = adj_copy
                current_score = new_score
                improved = True
                break  # Restart scanning
            
            # Try alternative 2
            adj_copy2 = {k: list(v2) for k, v2 in adj.items()}
            
            adj_copy2[u] = [(x, l) for x, l in adj_copy2[u] if x not in (v, b)]
            adj_copy2[v] = [(x, l) for x, l in adj_copy2[v] if x not in (u, d)]
            adj_copy2[b] = [(x, l) for x, l in adj_copy2[b] if x != u]
            adj_copy2[d] = [(x, l) for x, l in adj_copy2[d] if x != v]
            
            adj_copy2[u].append((d, 0.0))
            adj_copy2[d].append((u, 0.0))
            adj_copy2[v].append((b, 0.0))
            adj_copy2[b].append((v, 0.0))
            
            new_score2 = _fitch_parsimony(adj_copy2, leaves, label_to_node, sequences, labels)
            
            if new_score2 < current_score:
                adj = adj_copy2
                current_score = new_score2
                improved = True
                break
    
    # Convert back to Newick
    return _tree_to_newick(adj, leaves, label_to_node)


def _tree_to_newick(adj, leaves, label_to_node):
    """Convert adjacency list back to Newick string."""
    # Root at an arbitrary leaf
    leaf_nodes = list(leaves.keys())
    if not leaf_nodes:
        return ";"
    
    root = leaf_nodes[0]
    
    # Build rooted tree via DFS
    parent = {root: None}
    children = {root: []}
    visited = {root}
    stack = [root]
    while stack:
        u = stack.pop()
        for v, _ in adj[u]:
            if v not in visited:
                visited.add(v)
                parent[v] = u
                children[v] = []
                children[u].append(v)
                stack.append(v)
    
    def to_newick(node):
        if node in leaves:
            return leaves[node]
        if not children[node]:
            return f"({leaves.get(node, '?')})"
        parts = [to_newick(child) for child in children[node]]
        return "(" + ",".join(parts) + ")"
    
    result = to_newick(root)
    return result + ";"
>>>>>>> REPLACE

