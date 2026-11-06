import numpy as np
import time
import math

def _fitch_score(tree_edges, sequences, labels):
    """
    Calculate the Fitch parsimony score for a given tree topology.
    tree_edges: list of (u, v) tuples representing the unrooted tree.
    sequences: dict of label -> string
    labels: list of labels
    """
    n_taxa = len(labels)
    n_sites = len(sequences[labels[0]])
    
    # Build adjacency list
    adj = {l: [] for l in labels}
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # We need to root the tree arbitrarily to do post-order traversal
    # Pick the first label as root
    root = labels[0]
    
    # Build parent pointers and order
    parent = {root: None}
    order = []
    stack = [root]
    visited = {root}
    
    while stack:
        node = stack.pop()
        order.append(node)
        for neighbor in adj[node]:
            if neighbor not in visited:
                parent[neighbor] = node
                visited.add(neighbor)
                stack.append(neighbor)
        
    # Reverse order for post-order (children before parents)
    order.reverse()
    
    # Initialize sets for each node and site
    # sets[node][site] = 4-bit mask
    # 0: A, 1: C, 2: G, 3: T
    # Mask bits: 1<<0 for A, 1<<1 for C, etc.
    
    # Map char to bit
    char_to_bit = {'A': 1, 'C': 2, 'G': 4, 'T': 8}
    
    # Pre-compute site masks for taxa
    # taxa_masks[label][site] = mask
    taxa_masks = np.zeros((n_taxa, n_sites), dtype=np.int8)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    
    for i, l in enumerate(labels):
        seq = sequences[l]
        for j, c in enumerate(seq):
            taxa_masks[i, j] = char_to_bit[c]
    
    # sets for each node
    # nodes are indexed 0 to n_taxa-1
    # We will store the mask for each site for each node
    # node_masks[node_idx][site] = mask
    node_masks = np.zeros((n_taxa, n_sites), dtype=np.int8)
    
    # Set leaves
    for i, l in enumerate(labels):
        node_masks[i] = taxa_masks[i]
    
    # score
    score = 0
    
    # Process nodes in post-order
    for node in order:
        idx = label_to_idx[node]
        
        # Find children
        children = [n for n in adj[node] if parent.get(n) == node]
        
        if not children:
            # Leaf, already set
            continue
            
        # Intersection of children's masks
        # Start with all bits set (15)
        current_mask = np.full(n_sites, 15, dtype=np.int8)
        
        for child in children:
            c_idx = label_to_idx[child]
            # current_mask = current_mask & node_masks[c_idx]
            current_mask &= node_masks[c_idx]
            
        # Where current_mask is 0, we need to union
        # union_mask = OR of all children's masks
        union_mask = np.zeros(n_sites, dtype=np.int8)
        for child in children:
            c_idx = label_to_idx[child]
            union_mask |= node_masks[c_idx]
            
        # Update node mask
        # If intersection is 0, use union, else intersection
        # np.where(current_mask == 0, union_mask, current_mask)
        node_masks[idx] = np.where(current_mask == 0, union_mask, current_mask)
        
        # Score: number of sites where intersection was empty
        # This corresponds to a change
        score += np.sum(current_mask == 0)
        
    return score

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 170.0 # Leave 10s margin for 180s limit
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    n_sites = len(sequences[labels[0]])
    
    if n <= 3:
        # Trivial tree
        if n == 3:
            return f"({labels[0]},{labels[1]},{labels[2]});"
        elif n == 2:
            return f"({labels[0]},{labels[1]});"
        else:
            return f"({labels[0]});"

    # 1. Compute Distance Matrix
    # Convert sequences to numpy arrays of integers
    # 0:A, 1:C, 2:G, 3:T
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_matrix = np.zeros((n, n_sites), dtype=np.int8)
    for i, l in enumerate(labels):
        seq = sequences[l]
        for j, c in enumerate(seq):
            seq_matrix[i, j] = char_map[c]
    
    # Pairwise differences
    # diff_matrix[i, j] = number of differences
    # Use broadcasting
    # diff = (seq_matrix[:, None, :] != seq_matrix[None, :, :]).sum(axis=2)
    # This creates a n x n x L array, which might be large.
    # n=32, L=2000 -> 32*32*2000 = 2M ints. 2MB. Fine.
    
    diff = (seq_matrix[:, None, :] != seq_matrix[None, :, :]).sum(axis=2)
    
    # JC Distance
    # p = diff / L
    # d = -0.75 * log(1 - 4/3 * p)
    p = diff.astype(float) / n_sites
    # Handle p >= 0.75
    p_safe = np.clip(p, 0, 0.7499)
    dist_matrix = -0.75 * np.log(1 - 4.0/3.0 * p_safe)
    np.fill_diagonal(dist_matrix, 0)
    
    # 2. Neighbor Joining
    # We will use a standard NJ implementation
    # Nodes: list of current node indices
    # We need to track the tree structure
    
    # For NJ, we work with a matrix.
    # Let's use a library-free NJ that returns an adjacency list.
    
    # Initialize
    nodes = list(range(n))
    # dist is the current distance matrix
    # We need to update dist as we join nodes.
    
    # Tree edges
    edges = []
    
    # We need a way to represent the "super-taxon"
    # Let's use a dictionary to map current node names to their internal structure
    # Or just build the Newick string directly? No, we need the topology for SPR.
    # Let's build an adjacency list for the unrooted tree.
    
    # To make NJ easier to implement correctly with topology tracking:
    # We maintain a list of "active" nodes.
    # When we join a and b, we create a new node c.
    # We add edges (a, c) and (b, c) to the final tree? No, (a, b) is not an edge.
    # The edge is between the new node c and the rest? No.
    # In NJ, when joining a and b, we create a new node.
    # The edges added to the tree are (a, new_node) and (b, new_node).
    # And the distance from new_node to other nodes x is updated.
    
    # Let's implement NJ returning a list of edges (u, v) where u, v are original labels or internal nodes.
    # To simplify SPR later, it's better to have a clean tree structure.
    
    # Alternative: Use a library? No, standard lib only.
    # I will implement NJ carefully.
    
    # State:
    # active_nodes: list of node IDs (integers)
    # node_to_label: dict mapping node ID to original label or internal name
    # dist: numpy array of distances between active nodes
    # tree_edges: list of (u, v) pairs
    
    active_nodes = list(range(n))
    node_label = {i: labels[i] for i in range(n)}
    current_dist = dist_matrix.copy()
    tree_edges = []
    
    next_node_id = n
    
    while len(active_nodes) > 2:
        k = len(active_nodes)
        
        # Calculate Q matrix
        # Q[i, j] = (k-2) * d[i, j] - r[i] - r[j]
        # where r[i] = sum_j d[i, j]
        
        # r is the row sum of current_dist
        r = current_dist.sum(axis=1)
        
        # Q matrix
        # We only need to find the min Q[i, j] for i != j
        # Q = (k-2) * current_dist - r[:, None] - r[None, :]
        
        # Mask diagonal and already processed?
        # Since we remove nodes, we just work on the submatrix of active nodes.
        
        # Map active nodes to indices in current_dist?
        # It's easier to keep current_dist as a full n x n matrix and just ignore inactive nodes?
        # No, we need to update distances to the new node.
        
        # Let's restructure to keep a list of active node indices and a dense matrix for them.
        
        # Re-initializing NJ loop with better structure:
        pass # The above logic is getting messy. Let's use a simpler approach for NJ.
        
    # Simpler NJ Implementation:
    # 1. Create a list of nodes.
    # 2. While > 2 nodes:
    #    a. Compute Q matrix for all pairs of active nodes.
    #    b. Find min Q.
    #    c. Let a, b be the pair.
    #    d. Create new node c.
    #    e. Add edges (a, c) and (b, c) to tree_edges.
    #    f. Update distances from c to all other active nodes x:
    #       d(c, x) = 0.5 * (d(a, x) + d(b, x) - d(a, b))
    #    g. Remove a and b from active nodes, add c.
    #    h. Update distance matrix.
    
    # Let's restart the NJ part with this clear logic.
    
    active_nodes = list(range(n))
    node_labels = {i: labels[i] for i in range(n)}
    dist = dist_matrix.copy()
    edges = []
    next_id = n
    
    while len(active_nodes) > 2:
        k = len(active_nodes)
        
        # Compute row sums
        # We need to sum distances to OTHER active nodes
        # dist is n x n. We only care about active_nodes.
        
        # Create a submatrix of active nodes
        # idxs = np.array(active_nodes)
        # sub_dist = dist[np.ix_(idxs, idxs)]
        
        # To avoid copying, we can compute r on the fly or use the submatrix.
        # For n=32, copying is cheap.
        
        idxs = np.array(active_nodes)
        sub_dist = dist[np.ix_(idxs, idxs)]
        
        r = sub_dist.sum(axis=1)
        
        # Q matrix
        # Q[i, j] = (k-2) * sub_dist[i, j] - r[i] - r[j]
        # We want min Q for i != j
        
        # Mask diagonal
        np.fill_diagonal(sub_dist, np.inf)
        
        Q = (k - 2) * sub_dist - r[:, None] - r[None, :]
        
        # Find min
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = min_idx
        
        a = idxs[i]
        b = idxs[j]
        
        # Create new node
        c = next_id
        next_id += 1
        
        # Add edges
        edges.append((a, c))
        edges.append((b, c))
        
        # Update distances
        # d(c, x) = 0.5 * (d(a, x) + d(b, x) - d(a, b))
        # for all x in active_nodes except a, b
        
        # Update dist matrix
        # First, compute distances to c
        # dist[a, c] and dist[b, c] are not needed for the tree, but we need d(c, x)
        
        # d_ab = dist[a, b]
        d_ab = dist[a, b]
        
        # For each other active node x
        for x in active_nodes:
            if x != a and x != b:
                dist[c, x] = 0.5 * (dist[a, x] + dist[b, x] - d_ab)
                dist[x, c] = dist[c, x]
        
        # Update active nodes
        active_nodes.remove(a)
        active_nodes.remove(b)
        active_nodes.append(c)
        
        # Update node label (for final Newick, we might need it, but we build Newick from edges)
        node_labels[c] = f"n{c}"
        
    # Final 2 nodes
    a, b = active_nodes
    edges.append((a, b))
    
    # Now we have a list of edges (u, v) where u, v are node IDs.
    # We need to map node IDs back to original labels for the leaves.
    # Internal nodes are just IDs.
    
    # Convert edges to use labels for leaves
    # For SPR, we need to work with this tree structure.
    
    # 3. Parsimony Refinement (SPR)
    
    # We need to convert the edge list to a structure that allows Fitch scoring and SPR moves.
    # The edge list represents an unrooted tree.
    # For Fitch, we need to root it.
    # For SPR, we need to identify subtrees.
    
    # Let's build an adjacency list from the edges.
    # Nodes are 0..n-1 (leaves) and n..next_id-1 (internal).
    
    adj = {i: [] for i in range(next_id)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Function to get a Newick string from the current tree (edges)
    def get_newick(edges, labels, node_labels):
        # Build adjacency
        adj = {i: [] for i in range(next_id)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Root at node 0 (leaf)
        root = 0
        parent = {root: None}
        order = []
        stack = [root]
        visited = {root}
        
        while stack:
            node = stack.pop()
            order.append(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    parent[neighbor] = node
                    visited.add(neighbor)
                    stack.append(neighbor)
                    
        # Post-order traversal to build Newick
        def build_newick(node):
            children = [n for n in adj[node] if parent.get(n) == node]
            if not children:
                # Leaf
                return node_labels[node]
            else:
                child_strs = [build_newick(c) for c in children]
                return "(" + ",".join(child_strs) + ")"
        
        return build_newick(root) + ";"
        
    # Initial Score
    current_edges = edges[:]
    current_score = _fitch_score(current_edges, sequences, labels)
    
    # SPR Loop
    # We will try to improve the score.
    # Time check
    max_iterations = 20 # Safety limit
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        if time.time() > deadline - 10.0:
            break
            
        best_move = None
        best_score = current_score
        
        # For each edge, try pruning and reattaching
        # To avoid modifying the list while iterating, make a copy
        edges_copy = current_edges[:]
        
        for u, v in edges_copy:
            # Prune u from v
            # The subtree containing u (not v) is the one to move.
            # We need to find all nodes in that subtree.
            
            # Find subtree rooted at u (away from v)
            subtree_nodes = set()
            stack = [u]
            visited = {v} # Don't go back to v
            while stack:
                node = stack.pop()
                subtree_nodes.add(node)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
                        
            # The edge to remove is (u, v).
            # The subtree is subtree_nodes.
            # The rest of the tree is everything else.
            
            # Reattach the subtree to any other edge in the rest of the tree.
            # Valid edges are those not incident to the subtree (except the one we just removed? No, we removed (u,v)).
            # We can reattach to any edge (x, y) in the rest of the tree, where x, y are not in subtree_nodes.
            # Actually, we can reattach to any node in the rest of the tree, effectively creating a new edge.
            # In unrooted SPR, we prune a subtree and reattach it to any other edge.
            
            # Get edges in the rest of the tree
            rest_edges = []
            for x, y in current_edges:
                if x not in subtree_nodes and y not in subtree_nodes:
                    rest_edges.append((x, y))
                
            # Try reattaching to each rest edge
            for x, y in rest_edges:
                # New edges:
                # Remove (u, v)
                # Add (u, x) and (v, y)? No.
                # If we reattach the subtree (rooted at u) to the edge (x, y),
                # we remove (x, y), add (x, u) and (y, u)? No, that changes the structure.
                # Standard SPR:
                # Prune subtree S at edge (u, v).
                # Reattach S to edge (x, y) by removing (x, y) and adding (x, u) and (y, u)?
                # No, we add a new node? No, unrooted SPR doesn't add nodes.
                # We replace edge (x, y) with (x, u) and (y, u)? That would make u have degree 3?
                # u was degree 1 in S (connected to v). v is in Rest.
                # If we connect u to x and y, u has degree 3.
                # But x and y lose their connection.
                # This is valid if we treat it as:
                # Remove (u, v) and (x, y).
                # Add (u, x) and (u, y).
                # This changes the topology.
                
                # Let's verify this is a valid SPR move.
                # Yes, this is one way to define it.
                
                # Check if this move is trivial (i.e., reattaching to the same place)
                # If x == v or y == v, it's the same edge.
                # If x == u or y == u, it's impossible (u is in S).
                
                if x == v or y == v:
                    continue
                
                # Construct new edges
                new_edges = []
                for e in current_edges:
                    if e == (u, v) or e == (v, u):
                        continue
                    if e == (x, y) or e == (y, x):
                        continue
                    new_edges.append(e)
                
                new_edges.append((u, x))
                new_edges.append((u, y))
                
                # Calculate score
                # To speed up, we can check if the move is likely to be good?
                # No, just calculate.
                score = _fitch_score(new_edges, sequences, labels)
                
                if score < best_score:
                    best_score = score
                    best_move = new_edges[:]
                    
        if best_move is not None:
            current_edges = best_move
            current_score = best_score
            improved = True
            
    # Final Newick
    # We need to map internal node IDs to labels?
    # The get_newick function uses node_labels.
    # For internal nodes, node_labels[c] = f"n{c}".
    # This is fine.
    
    return get_newick(current_edges, labels, node_labels)
