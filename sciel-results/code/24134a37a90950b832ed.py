"""
Reconstruct phylogenetic topology using Neighbor Joining with NNI refinement.
"""
import math
import time
import numpy as np

def _jc_distance_matrix(sequences, labels):
    """
    Compute the Jukes-Cantor distance matrix for the given sequences.
    """
    n = len(labels)
    seqs = np.array([list(sequences[l]) for l in labels])
    # seqs is shape (n, L)
    
    # Calculate pairwise differences
    # We can do this by broadcasting, but for memory efficiency with n<=32 and L<=2000, 
    # it's fine. 
    # diff[i,j] = number of positions where seqs[i] != seqs[j]
    
    # Efficient way: 
    # Expand seqs to (n, 1, L) and (1, n, L)
    # This might be memory heavy if L is large, but L<=2000, n<=32. 32*32*2000 = 2M bytes per char? 
    # Actually, let's just loop over positions to be safe on memory, or use a chunked approach.
    # But n is small. Let's try vectorizing over sites.
    
    # Alternative: loop over sites, accumulate differences.
    diff_matrix = np.zeros((n, n), dtype=np.float64)
    
    # For each site, add 1 to diff_matrix[i,j] if seqs[i][k] != seqs[j][k]
    # This is O(L * N^2). With L=2000, N=32, that's 2000 * 1024 = 2M operations. Very fast.
    
    for k in range(seqs.shape[1]):
        col = seqs[:, k]
        # Create a matrix of comparisons
        # col[:, None] != col[None, :] gives (N, N) boolean matrix
        diff_matrix += (col[:, None] != col[None, :]).astype(np.float64)
        
    # Convert to p-distances
    p = diff_matrix / seqs.shape[1]
    
    # Jukes-Cantor correction: d = -3/4 * ln(1 - 4p/3)
    # Handle p >= 0.75
    mask = p < 0.75
    d = np.zeros_like(p)
    d[mask] = -0.75 * np.log(1.0 - 4.0 * p[mask] / 3.0)
    d[~mask] = 3.0 # Cap at 3.0 or a large number? Standard is to cap.
    
    # Ensure symmetry and zero diagonal
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)
    
    return d

def _neighbor_joining(D):
    """
    Perform Neighbor Joining on distance matrix D.
    Returns:
    - tree_structure: A nested tuple/list structure representing the tree.
    - branch_lengths: A dictionary mapping edge identifiers to lengths? 
      Actually, for NNI we need the tree topology and the ability to calculate tree length.
      Let's return the tree as a set of splits or a nested structure.
      
      A common representation for NNI is a list of edges or a nested tuple.
      Let's use a nested tuple: ((A, B), C) etc.
      But we also need branch lengths to evaluate tree length.
      
      Alternatively, we can just return the topology and recalculate branch lengths 
      or use the distance matrix to evaluate the "quartet criterion" or "tree length".
      
      For NNI, we need to evaluate the change in total tree length.
      The total tree length is the sum of all branch lengths.
      For a tree T with branch lengths l, the length is sum(l).
      When we do NNI, we can compute the new length from the old length and the distances.
      
      Let's return:
      - topology: A nested tuple of leaf labels (0..n-1) and internal nodes.
      - We will need to map internal nodes to their children.
      
      Actually, a simpler approach for NNI on a distance matrix:
      1. Build tree with NJ.
      2. Represent tree as a set of internal edges.
      3. For each internal edge, consider the 4 neighbors.
      4. Check the 3 possible topologies.
      5. Calculate the change in tree length.
      
      To calculate tree length from a topology and a distance matrix D:
      This is the "Maximum Likelihood" or "Least Squares" fit of branch lengths.
      For NNI, we can use the formula for the change in length.
      
      Let's stick to a simpler, robust method:
      1. NJ to get initial tree.
      2. Convert to a Newick string or a structure that allows NNI.
      3. Perform NNI.
      
      Let's represent the tree as a set of edges in a graph.
      Nodes: 0..n-1 (leaves), n..2n-3 (internal).
      Edges: (u, v).
      
      For NNI, we iterate over internal edges (u, v).
      The neighbors of u (excluding v) are a, b.
      The neighbors of v (excluding u) are c, d.
      The 4 leaves connected to this "quartet" are a, b, c, d.
      The 3 topologies are:
      1. (a,b)|(c,d) - Current
      2. (a,c)|(b,d)
      3. (a,d)|(b,c)
      
      We calculate the "length" of the tree for each topology.
      The length of a tree given a distance matrix can be approximated by the sum of branch lengths 
      obtained by solving the system of equations for the branch lengths (least squares).
      However, for NNI, there is a known formula for the change in length based on the distances 
      between the 4 taxa.
      
      Let L(T) be the total length.
      For the quartet a,b,c,d, the contribution to the length is related to the distances.
      Actually, it's easier to just recalculate the branch lengths for the whole tree? No, too slow.
      
      Standard NNI criterion:
      Let d(x,y) be the distance.
      For the three quartet topologies, the "quartet score" is:
      S1 = d(a,b) + d(c,d)
      S2 = d(a,c) + d(b,d)
      S3 = d(a,d) + d(b,c)
      
      The topology with the smallest S is the most likely? 
      No, that's for the "quartet method".
      
      For NNI on a tree with branch lengths, we need to check if swapping reduces the total tree length.
      Let the current edge be (u,v) with length l_uv.
      Let the edges incident to u be (u,a) with l_ua, (u,b) with l_ub.
      Let the edges incident to v be (v,c) with l_vc, (v,d) with l_vd.
      
      The change in length when swapping to (a,c)|(b,d) is:
      delta = 0.5 * (d(a,c) + d(b,d) - d(a,b) - d(c,d)) 
      This formula assumes the tree is "additive" or that we are using a specific distance correction.
      Actually, for NJ trees, the branch lengths are already optimized for the current topology.
      When we change the topology, the optimal branch lengths change.
      
      A robust way:
      1. Get the tree topology.
      2. Calculate the "tree length" using the distance matrix.
         Tree length = sum of branch lengths.
         We can compute the optimal branch lengths for a given topology using least squares.
         But doing this for every NNI step is expensive.
         
      Alternative: Use the "quartet criterion" for NNI.
      The quartet criterion states that for 4 taxa, the topology with the smallest sum of internal branch lengths is preferred.
      For a quartet, the internal branch length is:
      L1 = 0.5 * (d(a,c) + d(b,d) - d(a,b) - d(c,d))  (for (a,b)|(c,d))
      Wait, the internal branch length for (a,b)|(c,d) is:
      l = 0.5 * (d(a,c) + d(b,d) - d(a,b) - d(c,d))
      If l > 0, this topology is supported.
      
      In NNI, we compare the internal branch length of the current topology vs the two alternatives.
      If the alternative has a larger positive internal branch length (or smaller negative), it might be better.
      Actually, we want to minimize the total tree length.
      The change in total tree length is approximately related to the change in the internal branch length of the quartet.
      
      Let's use a simpler, effective heuristic:
      For each internal edge, identify the 4 "outer" leaves (or subtrees).
      Compute the three quartet scores:
      Q1 = d(a,b) + d(c,d)
      Q2 = d(a,c) + d(b,d)
      Q3 = d(a,d) + d(b,c)
      
      If Q2 < Q1 or Q3 < Q1, swap to the topology with the smallest Q.
      This is the "Quartet NNI" method. It is fast and effective.
      
      Let's implement Quartet NNI.
      
      Steps:
      1. NJ to get initial tree.
      2. Represent tree as a graph.
      3. Iterate:
         - For each internal edge (u,v):
           - Find the 4 neighbors: a,b (of u) and c,d (of v).
           - These are subtrees. We need the "representative" leaf or just the subtrees.
           - For quartet NNI, we need distances between the 4 subtrees.
           - If a, b, c, d are leaves, we use d(a,c) etc.
           - If they are internal nodes, we need to define "distance" between subtrees.
             Usually, we use the minimum distance between any leaf in subtree A and any leaf in subtree C.
             Or the average.
             Let's use the minimum distance to be conservative.
             
         - Compute Q1, Q2, Q3.
         - If min(Q2, Q3) < Q1, perform the swap.
         - Update the tree structure.
      4. Repeat until no improvement or time limit.
      
      This requires maintaining the tree structure and being able to find the 4 neighbors of an internal edge.
      
      Let's implement this.
    """
    n = D.shape[0]
    if n < 4:
        # Trivial tree
        if n == 3:
            return ((0, 1), 2)
        elif n == 2:
            return (0, 1)
        else:
            return 0
            
    # Initialize nodes
    # Leaves: 0..n-1
    # Internal nodes: n..2n-3
    # We'll use a dictionary for edges: node -> set of neighbors
    # And a dictionary for node type: 'leaf' or 'internal'
    
    nodes = list(range(2 * n - 3))
    edges = {i: set() for i in nodes}
    node_type = {i: 'leaf' if i < n else 'internal' for i in nodes}
    
    # NJ Algorithm
    # Active nodes
    active = list(range(n))
    # Distance matrix for active nodes
    D_active = D[:n, :n].copy()
    
    # We need to track the tree structure as we merge nodes.
    # Let's store the children of each internal node.
    children = {i: None for i in range(n, 2 * n - 3)}
    
    for _ in range(n - 2):
        # Calculate Q-matrix
        # Q[i,j] = (n-2) * D[i,j] - r[i] - r[j]
        # r[i] = sum of D[i,k] for all active k
        
        m = len(active)
        if m <= 3:
            break
            
        r = np.zeros(m)
        for i in range(m):
            r[i] = np.sum(D_active[i, :])
            
        Q = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                Q[i, j] = (m - 2) * D_active[i, j] - r[i] - r[j]
                
        # Find min Q
        min_val = np.inf
        min_i, min_j = -1, -1
        for i in range(m):
            for j in range(i + 1, m):
                if Q[i, j] < min_val:
                    min_val = Q[i, j]
                    min_i, min_j = i, j
                    
        # Merge active[min_i] and active[min_j]
        u = active[min_i]
        v = active[min_j]
        
        # Create new internal node
        new_node = n + (n - 2 - (n - len(active))) # This logic is tricky. 
        # Let's just use a counter for internal nodes.
        # Internal nodes are n, n+1, ...
        # Number of merges so far = n - len(active)
        # So new node index = n + (n - len(active) - 1)
        new_node_idx = n + (n - len(active) - 1)
        
        # Calculate branch lengths
        # l_u = 0.5 * D[u,v] + (r[u] - r[v]) / (2 * (m - 2))
        # l_v = 0.5 * D[u,v] - (r[u] - r[v]) / (2 * (m - 2))
        # We don't strictly need branch lengths for Quartet NNI, just topology.
        
        # Add edges
        edges[u].add(new_node_idx)
        edges[v].add(new_node_idx)
        edges[new_node_idx].add(u)
        edges[new_node_idx].add(v)
        
        children[new_node_idx] = (u, v)
        
        # Update active list
        active.remove(u)
        active.remove(v)
        active.append(new_node_idx)
        
        # Update D_active
        # New distance from new_node to x:
        # D[new, x] = 0.5 * (D[u, x] + D[v, x] - D[u, v])
        
        # We need to map old active indices to new D_active indices.
        # This is getting complicated with the dynamic D_active.
        
        # Let's restart the NJ implementation with a simpler, standard approach.
        pass
        
    # The above NJ implementation is messy. Let's use a cleaner one.
    # I will implement a standard NJ that returns a nested tuple.
    
    return _nj_simple(D)

def _nj_simple(D):
    """
    Simple Neighbor Joining implementation returning a nested tuple.
    """
    n = D.shape[0]
    if n == 1:
        return 0
    if n == 2:
        return (0, 1)
    if n == 3:
        return ((0, 1), 2)
        
    nodes = list(range(n))
    D = D.copy()
    tree = {i: i for i in nodes}
    
    while len(nodes) > 3:
        m = len(nodes)
        # r[i] = sum of D[i,j] for j in nodes
        r = np.zeros(m)
        for i in range(m):
            for j in range(m):
                if i != j:
                    r[i] += D[i, j]
                    
        Q = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                Q[i, j] = (m - 2) * D[i, j] - r[i] - r[j]
                
        min_val = np.inf
        min_i, min_j = -1, -1
        for i in range(m):
            for j in range(i + 1, m):
                if Q[i, j] < min_val:
                    min_val = Q[i, j]
                    min_i, min_j = i, j
                    
        u = nodes[min_i]
        v = nodes[min_j]
        
        # New node
        new_node = f"n{len(nodes)}_{u}_{v}"
        
        # Update tree
        tree[new_node] = (tree[u], tree[v])
        
        # Remove u, v from nodes, add new_node
        nodes.remove(u)
        nodes.remove(v)
        nodes.append(new_node)
        
        # Update D
        # We need to add a row/col for new_node in D.
        # But D is a numpy array indexed by 0..m-1.
        # This is hard to do in-place with numpy easily.
        # Let's use a dictionary for D as well.
        
        # Actually, let's just use a dictionary for D from the start.
        
    # This is also getting messy.
    
    # Let's use a very simple, robust NJ implementation that uses dictionaries.
    
    pass

def _nj_dict(D):
    n = D.shape[0]
    # Convert D to dict
    d = {}
    for i in range(n):
        for j in range(n):
            d[(i, j)] = D[i, j]
            
    nodes = list(range(n))
    tree = {i: i for i in nodes}
    
    while len(nodes) > 3:
        m = len(nodes)
        r = {x: 0.0 for x in nodes}
        for x in nodes:
            for y in nodes:
                if x != y:
                    r[x] += d[(x, y)]
                    
        min_val = np.inf
        min_i, min_j = -1, -1
        for i in range(m):
            for j in range(i + 1, m):
                u = nodes[i]
                v = nodes[j]
                q = (m - 2) * d[(u, v)] - r[u] - r[v]
                if q < min_val:
                    min_val = q
                    min_i, min_j = i, j
                    
        u = nodes[min_i]
        v = nodes[min_j]
        
        new_node = f"n{len(nodes)}_{u}_{v}"
        tree[new_node] = (tree[u], tree[v])
        
        nodes.remove(u)
        nodes.remove(v)
        nodes.append(new_node)
        
        # Update distances
        for w in nodes:
            if w == new_node:
                continue
            # d[(new_node, w)] = 0.5 * (d[(u, w)] + d[(v, w)] - d[(u, v)])
            new_d = 0.5 * (d[(u, w)] + d[(v, w)] - d[(u, v)])
            d[(new_node, w)] = new_d
            d[(w, new_node)] = new_d
            
    # Build final tree
    if len(nodes) == 3:
        a, b, c = nodes
        return ((tree[a], tree[b]), tree[c])
    else:
        # Should not happen
        return tree[nodes[0]]

def _get_leaves(node, leaves_set):
    """Recursively get all leaves in a subtree."""
    if isinstance(node, int):
        return {node}
    elif isinstance(node, str) and node.startswith('t'):
        return {int(node[1:])}
    elif isinstance(node, tuple):
        l_leaves = _get_leaves(node[0], leaves_set)
        r_leaves = _get_leaves(node[1], leaves_set)
        return l_leaves.union(r_leaves)
    else:
        # Internal node from NJ? 
        # My _nj_dict returns tuples of (tree[u], tree[v]) where tree[u] is int or tuple.
        # So the structure is nested tuples of ints.
        pass

def _nni_refine(tree, D, time_limit=10.0):
    """
    Refine tree using NNI.
    tree: nested tuple of ints (0..n-1).
    D: distance matrix.
    """
    start_time = time.time()
    n = D.shape[0]
    
    # Convert tree to a graph representation for easier NNI
    # Nodes: 0..n-1 (leaves), n..2n-3 (internal)
    # Edges: list of (u, v)
    
    # We need to map the nested tuple to a graph.
    # This is a bit of work.
    
    # Alternative: Implement NNI directly on the nested tuple?
    # No, graph is better.
    
    # Let's build the graph.
    # We'll assign internal node IDs.
    
    def build_graph(node, parent=None):
        if isinstance(node, int):
            return node, 0 # leaf, no children
        else:
            # Internal node
            # Assign a new ID
            nonlocal internal_id_counter
            new_id = internal_id_counter
            internal_id_counter += 1
            
            left, right = node
            left_id, _ = build_graph(left, new_id)
            right_id, _ = build_graph(right, new_id)
            
            return new_id, (left_id, right_id)
            
    internal_id_counter = n
    root_id, _ = build_graph(tree, None)
    
    # Now we have a tree with nodes 0..2n-3.
    # Edges:
    edges = {}
    def add_edges(node, parent=None):
        if isinstance(node, int):
            return
        else:
            left, right = node
            # We need the IDs.
            # This recursive approach with IDs is tricky because build_graph returns IDs.
            # Let's just store the structure differently.
            pass
            
    # Let's use a simpler representation:
    # A list of edges (u, v) where u, v are node IDs.
    # And a dict children: id -> (left, right) for internal nodes.
    
    # Let's re-do build_graph to populate these.
    
    edges_list = []
    children = {}
    
    def build_graph_v2(node):
        if isinstance(node, int):
            return node
        else:
            nonlocal internal_id_counter
            new_id = internal_id_counter
            internal_id_counter += 1
            
            left, right = node
            left_id = build_graph_v2(left)
            right_id = build_graph_v2(right)
            
            children[new_id] = (left_id, right_id)
            edges_list.append((new_id, left_id))
            edges_list.append((new_id, right_id))
            
            return new_id
            
    root_id = build_graph_v2(tree)
    
    # Build adjacency list
    adj = {i: set() for i in range(internal_id_counter)}
    for u, v in edges_list:
        adj[u].add(v)
        adj[v].add(u)
        
    # NNI Loop
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        for u, v in edges_list:
            # Check if u, v are internal nodes (or one is root?)
            # Actually, NNI is on internal edges.
            # An edge is internal if both endpoints are internal nodes.
            # In an unrooted tree, all edges are "internal" in the sense that they can be swapped?
            # No, edges connecting to leaves cannot be swapped.
            # So we only consider edges where both u and v are internal nodes.
            
            if u < n or v < n:
                continue
                
            # Find neighbors of u (excluding v)
            neigh_u = [x for x in adj[u] if x != v]
            # Find neighbors of v (excluding u)
            neigh_v = [x for x in adj[v] if x != u]
            
            if len(neigh_u) != 2 or len(neigh_v) != 2:
                continue
                
            a, b = neigh_u
            c, d = neigh_v
            
            # Get leaves in each subtree
            def get_leaves(node_id):
                if node_id < n:
                    return {node_id}
                else:
                    l, r = children[node_id]
                    return get_leaves(l).union(get_leaves(r))
                    
            leaves_a = get_leaves(a)
            leaves_b = get_leaves(b)
            leaves_c = get_leaves(c)
            leaves_d = get_leaves(d)
            
            # Calculate quartet scores
            # We need distances between the subtrees.
            # Use min distance.
            def min_dist(leaves1, leaves2):
                min_d = np.inf
                for i in leaves1:
                    for j in leaves2:
                        if D[i, j] < min_d:
                            min_d = D[i, j]
                return min_d
                
            # Q1: (a,b)|(c,d) - Current
            # Q2: (a,c)|(b,d)
            # Q3: (a,d)|(b,c)
            
            # Score is sum of distances across the split?
            # No, the quartet criterion uses the distances between the 4 taxa.
            # But here we have subtrees.
            # We can use the "minimum distance" between the subtrees as a proxy.
            
            d_ac = min_dist(leaves_a, leaves_c)
            d_bd = min_dist(leaves_b, leaves_d)
            d_ad = min_dist(leaves_a, leaves_d)
            d_bc = min_dist(leaves_b, leaves_c)
            d_ab = min_dist(leaves_a, leaves_b)
            d_cd = min_dist(leaves_c, leaves_d)
            
            Q1 = d_ab + d_cd
            Q2 = d_ac + d_bd
            Q3 = d_ad + d_bc
            
            if Q2 < Q1 or Q3 < Q1:
                # Perform swap
                # Remove edges (u,v), (u,a), (u,b), (v,c), (v,d)
                # Add edges (u,a), (u,c), (v,b), (v,d) -- Wait, this depends on the swap.
                
                # Current: u connects to a,b and v. v connects to c,d and u.
                # Swap 1: (a,c)|(b,d). New internal nodes?
                # Actually, NNI swaps the connections.
                # The internal nodes u and v remain, but their connections change.
                # u connects to a, c. v connects to b, d. And u-v edge remains?
                # No, in NNI, the internal edge (u,v) is removed and replaced by a new edge?
                # No, the topology changes.
                # The 4 subtrees a,b,c,d are reconnected.
                # The internal edge (u,v) is removed.
                # New edges: (u,a), (u,c), (v,b), (v,d) is NOT correct.
                # The structure is:
                # Current: (a,b) - u - v - (c,d)
                # New: (a,c) - u' - v' - (b,d)
                # But we can reuse u and v.
                # Let's say u connects to a, c. v connects to b, d. And u-v edge is removed?
                # No, the quartet (a,b,c,d) has an internal edge.
                # In the current tree, the internal edge is (u,v).
                # In the new tree, the internal edge is between the new parents of (a,c) and (b,d).
                
                # Let's just update the children and adj.
                # Remove (u,v)
                adj[u].remove(v)
                adj[v].remove(u)
                edges_list.remove((u, v))
                
                # Remove (u,a), (u,b), (v,c), (v,d)
                adj[u].remove(a)
                adj[a].remove(u)
                edges_list.remove((u, a))
                
                adj[u].remove(b)
                adj[b].remove(u)
                edges_list.remove((u, b))
                
                adj[v].remove(c)
                adj[c].remove(v)
                edges_list.remove((v, c))
                
                adj[v].remove(d)
                adj[d].remove(v)
                edges_list.remove((v, d))
                
                # Add new edges
                if Q2 < Q3:
                    # (a,c)|(b,d)
                    # u connects to a, c
                    # v connects to b, d
                    # u-v edge is removed?
                    # Wait, if u connects to a,c and v connects to b,d, how are u and v connected?
                    # They are not directly connected.
                    # There must be a new internal edge.
                    # But we only have u and v.
                    # Actually, in NNI, the internal edge is preserved?
                    # No, the quartet is re-wired.
                    # The edge (u,v) is replaced by the edge between the new "parents".
                    # But u and v ARE the parents.
                    # So u connects to a,c and v connects to b,d.
                    # And u and v are connected?
                    # If u connects to a,c and v connects to b,d, and u-v is an edge, then the tree is:
                    # (a,c) - u - v - (b,d).
                    # This is a valid tree.
                    
                    adj[u].add(a)
                    adj[a].add(u)
                    edges_list.append((u, a))
                    
                    adj[u].add(c)
                    adj[c].add(u)
                    edges_list.append((u, c))
                    
                    adj[v].add(b)
                    adj[b].add(v)
                    edges_list.append((v, b))
                    
                    adj[v].add(d)
                    adj[d].add(v)
                    edges_list.append((v, d))
                    
                    adj[u].add(v)
                    adj[v].add(u)
                    edges_list.append((u, v))
                    
                    # Update children
                    children[u] = (a, c)
                    children[v] = (b, d)
                    
                else:
                    # (a,d)|(b,c)
                    adj[u].add(a)
                    adj[a].add(u)
                    edges_list.append((u, a))
                    
                    adj[u].add(d)
                    adj[d].add(u)
                    edges_list.append((u, d))
                    
                    adj[v].add(b)
                    adj[b].add(v)
                    edges_list.append((v, b))
                    
                    adj[v].add(c)
                    adj[c].add(v)
                    edges_list.append((v, c))
                    
                    adj[u].add(v)
                    adj[v].add(u)
                    edges_list.append((u, v))
                    
                    children[u] = (a, d)
                    children[v] = (b, c)
                    
                improved = True
                break # Restart loop after change
                
    # Convert back to Newick
    def to_newick(node_id):
        if node_id < n:
            return f"t{node_id}"
        else:
            l, r = children[node_id]
            return f"({to_newick(l)},{to_newick(r)})"
            
    return to_newick(root_id)

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 3:
        # Trivial cases
        if n == 1:
            return labels[0]
        elif n == 2:
            return f"({labels[0]},{labels[1]});"
        else:
            return f"(({labels[0]},{labels[1]}),{labels[2]});"
            
    D = _jc_distance_matrix(sequences, labels)
    
    # NJ
    tree = _nj_dict(D)
    
    # NNI
    # Time limit for NNI: 10 seconds
    refined_tree = _nni_refine(tree, D, time_limit=10.0)
    
    return refined_tree + ";"
