"""
Phylogenetic topology recovery using Quartet Consensus and NNI refinement.
"""
import math
import time
import itertools
import numpy as np
from collections import defaultdict

def _jc_distance_matrix(sequences, labels):
    """Compute JC distance matrix using NumPy."""
    n = len(labels)
    # Map labels to indices
    label_to_idx = {l: i for i, l in enumerate(labels)}
    
    # Convert sequences to numeric matrix
    # A=0, C=1, G=2, T=3
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([[base_map[base] for base in seq] for seq in sequences.values()])
    # Ensure order matches labels
    seqs = seqs[[label_to_idx[l] for l in labels]]
    
    # Compute pairwise differences
    # seqs is (n, L)
    # diff[i, j, k] is 1 if seqs[i,k] != seqs[j,k]
    # We can do this with broadcasting
    # (n, 1, L) vs (n, L, 1) -> (n, n, L)
    # This might be memory heavy for n=32, L=2000 -> 32*32*2000 = 2M ints. OK.
    diff = (seqs[:, np.newaxis, :] != seqs[np.newaxis, :, :])
    
    # Sum differences
    diffs = diff.sum(axis=2)
    
    # Compute JC distance
    p = diffs.astype(float) / seqs.shape[1]
    # Avoid log(0) or log(negative)
    p = np.clip(p, 0, 0.7499)
    dist = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    
    # Set diagonal to 0
    np.fill_diagonal(dist, 0.0)
    
    return dist

def _precompute_best_quartets(dist, labels):
    """
    For every 4-taxon subset, determine the best split based on distance sums.
    Returns a dictionary: (t0, t1, t2, t3) -> (split_id, score)
    split_id: 0 for 01|23, 1 for 02|13, 2 for 03|12 (indices in the tuple)
    """
    n = len(labels)
    best_quartets = {}
    
    # Iterate over all 4-taxon subsets
    for combo in itertools.combinations(range(n), 4):
        i, j, k, l = combo
        # Distances
        dij = dist[i, j]
        dkl = dist[k, l]
        dik = dist[i, k]
        djl = dist[j, l]
        dil = dist[i, l]
        djk = dist[j, k]
        
        # Splits:
        # 0: ij|kl -> dij + dkl
        # 1: ik|jl -> dik + djl
        # 2: il|jk -> dil + djk
        
        s0 = dij + dkl
        s1 = dik + djl
        s2 = dil + djk
        
        scores = [s0, s1, s2]
        min_score = min(scores)
        # If tie, pick the first one
        best_split = scores.index(min_score)
        
        best_quartets[combo] = best_split
        
    return best_quartets

def _build_nj_tree(labels, dist):
    """Build NJ tree. Returns adjacency list."""
    n = len(labels)
    if n <= 3:
        # Star tree
        adj = {i: set() for i in range(n)}
        for i in range(n):
            for j in range(n):
                if i != j:
                    adj[i].add(j)
        return adj, list(range(n))
        
    # Nodes are initially leaves 0..n-1
    # We will add internal nodes n, n+1, ...
    nodes = list(range(n))
    trees = {i: i for i in range(n)} # Not needed for adjacency, just for tracking?
    # We need to maintain the distance matrix for active nodes
    # active_dist is a dict of dicts or a 2D array that we shrink
    
    # Use a list of active node indices
    active_nodes = list(range(n))
    # dist_matrix is the original dist
    # We need to update distances for new nodes
    
    # To make it easier, we'll keep a dictionary of distances between active nodes
    # active_dist[a][b]
    active_dist = {a: {b: dist[a, b] for b in active_nodes if b != a} for a in active_nodes}
    
    internal_node_count = n
    
    while len(active_nodes) > 3:
        m = len(active_nodes)
        # Compute Q matrix
        # r[a] = sum(dist[a][b] for b in active_nodes)
        r = {a: sum(active_dist[a][b] for b in active_nodes if b != a) for a in active_nodes}
        
        best_criterion = float('inf')
        best_pair = None
        
        for i, a in enumerate(active_nodes):
            for b in active_nodes[i+1:]:
                criterion = (m - 2) * active_dist[a][b] - r[a] - r[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_pair = (a, b)
                    
        a, b = best_pair
        
        # Create new internal node
        new_node = internal_node_count
        internal_node_count += 1
        
        # Calculate distances from new_node to all other active nodes
        for x in active_nodes:
            if x != a and x != b:
                d_new_x = 0.5 * (active_dist[a][x] + active_dist[b][x] - active_dist[a][b])
                active_dist[new_node][x] = d_new_x
                active_dist[x][new_node] = d_new_x
                
        # Remove a and b, add new_node
        active_nodes.remove(a)
        active_nodes.remove(b)
        active_nodes.append(new_node)
        
        # Clean up active_dist
        for x in active_nodes:
            del active_dist[x][a]
            del active_dist[x][b]
        del active_dist[a]
        del active_dist[b]
        
    # Now active_nodes has 3 nodes
    # Connect them in a star
    adj = {i: set() for i in range(internal_node_count)}
    
    # We need to reconstruct the tree structure.
    # The NJ algorithm above only gives the final 3 nodes.
    # We need to track the children.
    
    # Let's restart NJ with proper tracking.
    pass

def _build_nj_tree_tracked(labels, dist):
    """Build NJ tree and return adjacency list."""
    n = len(labels)
    if n <= 3:
        adj = {i: set() for i in range(n)}
        for i in range(n):
            for j in range(n):
                if i != j:
                    adj[i].add(j)
        return adj
        
    active_nodes = list(range(n))
    # dist is a numpy array, we will use a dict for active distances
    # d[a][b]
    d = {a: {b: dist[a, b] for b in active_nodes if b != a} for a in active_nodes}
    
    # children[u] = set of v
    children = {i: set() for i in range(n)}
    # parent[v] = u (for internal nodes)
    parent = {}
    
    internal_id = n
    
    while len(active_nodes) > 3:
        m = len(active_nodes)
        r = {a: sum(d[a][b] for b in active_nodes if b != a) for a in active_nodes}
        
        best_criterion = float('inf')
        best_pair = None
        
        for i, a in enumerate(active_nodes):
            for b in active_nodes[i+1:]:
                criterion = (m - 2) * d[a][b] - r[a] - r[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_pair = (a, b)
                    
        a, b = best_pair
        
        new_node = internal_id
        internal_id += 1
        
        # Add new node to d
        d[new_node] = {}
        for x in active_nodes:
            if x != a and x != b:
                val = 0.5 * (d[a][x] + d[b][x] - d[a][b])
                d[new_node][x] = val
                d[x][new_node] = val
                
        # Update children
        children[new_node] = {a, b}
        parent[a] = new_node
        parent[b] = new_node
        
        # Remove a, b from active_nodes and d
        active_nodes.remove(a)
        active_nodes.remove(b)
        active_nodes.append(new_node)
        
        for x in active_nodes:
            del d[x][a]
            del d[x][b]
        del d[a]
        del d[b]
        
    # Final 3 nodes
    nodes = active_nodes
    # Connect them in a star
    # The root is the last internal node created? No, just connect the 3.
    # In an unrooted tree, the last 3 nodes form a triangle? No, a star with 3 leaves.
    # One of them is the "root" of this final cluster.
    # Actually, just connect them all to each other? No, binary tree.
    # The last merge created a node connected to 2. The 3rd node is connected to that node?
    # No, when 3 nodes remain, they form a star.
    # We need to add edges between them.
    # Let's say we pick nodes[0] as the center? No, just connect them in a triangle?
    # Unrooted binary tree with 3 leaves is just a star.
    # But internal nodes must have degree 3.
    # If we have 3 active nodes, and we stop, we need to connect them.
    # If all 3 are leaves, we connect them to a new internal node?
    # Or if some are internal, we connect them.
    # Standard NJ: when 3 nodes remain, connect them to a central node?
    # No, the last step of NJ merges 2 nodes into 1.
    # So the loop runs until 3 nodes are left.
    # The 3 nodes are connected in a triangle? No.
    # The tree is built by the merges.
    # The final 3 nodes are connected to each other?
    # If we have nodes A, B, C.
    # A and B are connected to an internal node X. C is connected to X?
    # Actually, the last merge was (A, B) -> X.
    # Then we have X and C.
    # Wait, the loop condition is `> 3`.
    # So it stops when 3 nodes are left.
    # These 3 nodes are connected to each other?
    # No, they are the roots of the 3 subtrees.
    # We need to connect them.
    # We can add a new internal node R and connect R to all 3.
    # But R would have degree 3.
    # The 3 nodes already have their other connections.
    # So this forms a valid unrooted tree.
    
    final_root = internal_id
    internal_id += 1
    children[final_root] = set(nodes)
    for node in nodes:
        parent[node] = final_root
        
    # Build adjacency list
    adj = {i: set() for i in range(internal_id)}
    for u, v_set in children.items():
        for v in v_set:
            adj[u].add(v)
            adj[v].add(u)
            
    return adj

def _get_quartet_topology(adj, a, b, c, d):
    """
    Determine which of the 3 splits the tree induces for taxa a,b,c,d.
    Returns 0 for ab|cd, 1 for ac|bd, 2 for ad|bc.
    """
    # Root the tree at an arbitrary node, say a
    # Then find LCA of (b,c), (b,d), (c,d)
    # Or simpler:
    # Check if ab|cd is a split.
    # A split S|T is in the tree if removing the edge corresponding to S|T separates S and T.
    
    # Helper: check if split {a,b}|{c,d} is in the tree
    def has_split(s1, s2):
        # BFS from s1
        visited = set(s1)
        queue = list(s1)
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if v not in visited:
                    if v in s2:
                        return False # Connected, so not separated?
                        # Wait, if we can reach s2 from s1 without crossing the cut, they are connected.
                        # We want to know if the path between any x in s1 and y in s2 crosses the "central" edge.
                        # Actually, if the tree contains the split, then s1 and s2 are separated by an edge.
                        # So if we remove that edge, s1 and s2 are in different components.
                        # How to find the edge?
                        # The edge is the one on the path between any x in s1 and y in s2 that is "between" the sets.
                        # Simpler: 
                        # The tree contains split s1|s2 if and only if the path between any x in s1 and y in s2 
                        # passes through the same internal edge.
                        # And that edge separates s1 and s2.
                        
                        # Let's use a different check:
                        # For 4 taxa, the topology is determined by the 3 paths.
                        # If d(a,b) + d(c,d) < d(a,c) + d(b,d) AND d(a,b) + d(c,d) < d(a,d) + d(b,c), then ab|cd is the split.
                        # But we don't have distances in the tree, we have the tree structure.
                        # We can compute path lengths (number of edges) in the tree.
                        # Let L(x,y) be the number of edges in the path x-y.
                        # If L(a,b) + L(c,d) < L(a,c) + L(b,d) and L(a,b) + L(c,d) < L(a,d) + L(b,c), then ab|cd is the induced split.
                        # This works for unrooted trees!
                        
                        return True # Placeholder
                    visited.add(v)
                    queue.append(v)
        return True # If we didn't find s2, they are separated?
        
    # Let's use the path length heuristic.
    # Compute path lengths between all pairs in {a,b,c,d}
    def path_len(u, v):
        # BFS
        if u == v: return 0
        visited = {u}
        queue = [(u, 0)]
        while queue:
            node, dist = queue.pop(0)
            for nb in adj[node]:
                if nb not in visited:
                    if nb == v:
                        return dist + 1
                    visited.add(nb)
                    queue.append((nb, dist + 1))
        return -1 # Should not happen
        
    lab = path_len(a, b)
    lcd = path_len(c, d)
    lac = path_len(a, c)
    lbd = path_len(b, d)
    lad = path_len(a, d)
    lbc = path_len(b, c)
    
    s0 = lab + lcd
    s1 = lac + lbd
    s2 = lad + lbc
    
    scores = [s0, s1, s2]
    min_score = min(scores)
    return scores.index(min_score)

def _calculate_quartet_score(adj, best_quartets, labels):
    """Calculate the total quartet score of the tree."""
    score = 0
    n = len(labels)
    for combo in itertools.combinations(range(n), 4):
        i, j, k, l = combo
        a, b, c, d = labels[i], labels[j], labels[k], labels[l]
        # Map labels to indices for adj?
        # My adj uses indices 0..n-1 for leaves.
        # So a, b, c, d are labels. I need indices.
        # Let's assume labels are t0..tn-1.
        # idx = int(label[1:])
        
        ia = int(a[1:])
        ib = int(b[1:])
        ic = int(c[1:])
        id_ = int(d[1:])
        
        tree_split = _get_quartet_topology(adj, ia, ib, ic, id_)
        best_split = best_quartets[combo]
        
        if tree_split == best_split:
            score += 1
    return score

def _nni_refine(adj, best_quartets, labels, time_limit):
    """Refine tree using NNI and Quartet Score."""
    n = len(labels)
    # Precompute label to index
    label_to_idx = {l: int(l[1:]) for l in labels}
    
    # Current score
    current_score = _calculate_quartet_score(adj, best_quartets, labels)
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        # Find all internal edges
        # Internal nodes are those with degree 3 (in unrooted binary tree)
        # Leaves have degree 1
        internal_nodes = [u for u in adj if len(adj[u]) == 3]
        
        for u in internal_nodes:
            # For each neighbor v of u
            for v in list(adj[u]):
                if len(adj[v]) != 3:
                    continue
                # u and v are internal, connected
                # Get other neighbors
                u_neighbors = [x for x in adj[u] if x != v]
                v_neighbors = [x for x in adj[v] if x != u]
                
                if len(u_neighbors) != 2 or len(v_neighbors) != 2:
                    continue
                
                a, b = u_neighbors
                c, d = v_neighbors
                
                # Try NNI move 1: u-c, v-a (and u-b, v-d? No)
                # Original: u-a, u-b, u-v, v-c, v-d
                # Move 1: Remove u-v, u-b, v-c. Add u-c, v-b.
                # Wait, standard NNI:
                # Type 1: u connected to a, c. v connected to b, d.
                # Edges: u-a, u-c, u-v? No, u-v is removed.
                # So u is connected to a, c. v is connected to b, d.
                # And we need to connect u and v? No, NNI removes the edge u-v.
                # So the tree splits into two components?
                # No, NNI creates a new tree.
                # The new edges are:
                # Option 1: u-c and v-a? No.
                # Let's trace:
                # Before: (a, b) - u - v - (c, d)
                # After NNI 1: (a, c) - u' - v' - (b, d) ??
                # Actually, the nodes u and v are kept.
                # Remove u-v.
                # Add u-c and v-a?
                # Then u is connected to a, b, c. v is connected to a, c, d.
                # This creates a cycle a-u-c-v-a.
                
                # Correct NNI:
                # Remove u-v.
                # Add u-c and v-b?
                # u: a, b, c. v: b, c, d. Cycle b-u-c-v-b.
                
                # I am struggling with the NNI definition on unrooted trees.
                # Let's use the property:
                # The NNI move changes the split for the quartet {a,b,c,d}.
                # The new tree has split a,c|b,d or a,d|b,c.
                # To implement:
                # 1. Remove edges u-v, u-b, v-c.
                # 2. Add edges u-c, v-b.
                # Check degrees:
                # u: was a,b,v. Remove b,v. Add c. -> a,c. Degree 2.
                # v: was u,c,d. Remove u,c. Add b. -> b,d. Degree 2.
                # We need to connect u and v? No.
                # We need to add ONE edge between the two components?
                # No, NNI is a local move.
                # The standard NNI on an internal edge (u,v):
                # The four subtrees attached to u and v are A, B (to u) and C, D (to v).
                # The new tree connects A to C and B to D (or A to D and B to C).
                # This means we remove the edge u-v.
                # And we remove the edges u-B and v-C.
                # And we add edges u-C and v-B.
                # Wait, if we remove u-v, u-B, v-C, and add u-C, v-B:
                # u is connected to A, C.
                # v is connected to B, D.
                # Are u and v connected? No.
                # So the tree falls apart?
                # No, A, B, C, D are subtrees.
                # u is the root of subtree A+B. v is root of C+D.
                # If we disconnect u and v, we have two trees.
                # We need to reconnect them.
                # The new connections are:
                # u is connected to A and C.
                # v is connected to B and D.
                # And u is connected to v? No.
                # The new edge is between the "roots" of the new combinations?
                
                # Let's look at a reference.
                # NNI:
                # Before: ...-A-u-B-... and ...-C-v-D-... with u-v.
                # After: ...-A-u-C-... and ...-B-v-D-... with u-v? No.
                # After: ...-A-u-C-v-B-D... ?
                
                # Actually, the simplest way to implement NNI in code is to modify the adjacency list directly.
                # Remove u-v.
                # Remove u-b and v-c.
                # Add u-c and v-b.
                # This results in:
                # u: neighbors a, c
                # v: neighbors b, d
                # u and v are not connected.
                # This is a forest.
                # We need to add an edge.
                # The edge should be between u and v?
                # If we add u-v back, we are back to start.
                
                # I think I am missing the fact that u and v are internal nodes.
                # In a binary tree, internal nodes have degree 3.
                # If we remove u-v, u has degree 2, v has degree 2.
                # We add u-c and v-b.
                # u has degree 3 (a, c, ...?) No, u was connected to a, b, v.
                # Remove b, v. Add c. u is connected to a, c. Degree 2.
                # We need one more neighbor for u.
                # And one more for v.
                # The missing connection is between u and v?
                # If we add u-v, we have u-a, u-c, u-v. v-b, v-d, v-u.
                # This is a valid tree!
                # So the move is:
                # 1. Remove u-v.
                # 2. Remove u-b and v-c.
                # 3. Add u-c and v-b.
                # 4. Add u-v.
                # Wait, that's just swapping b and c?
                # No, originally u-b, v-c. Now u-c, v-b.
                # And u-v is kept.
                # This is a valid NNI move!
                
                # Let's verify the split.
                # Before: u connects a,b. v connects c,d.
                # After: u connects a,c. v connects b,d.
                # The split for {a,b,c,d} changes from ab|cd to ac|bd.
                
                # So the implementation is:
                # adj[u].remove(b)
                # adj[v].remove(c)
                # adj[u].add(c)
                # adj[v].add(b)
                # (u-v remains)
                
                # Let's try this.
                
                # Apply Move 1
                adj[u].remove(b)
                adj[v].remove(c)
                adj[u].add(c)
                adj[v].add(b)
                
                new_score = _calculate_quartet_score(adj, best_quartets, labels)
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                else:
                    # Revert
                    adj[u].remove(c)
                    adj[v].remove(b)
                    adj[u].add(b)
                    adj[v].add(c)
                    
                # Try Move 2 (a,d | b,c)
                # Remove u-b, v-d. Add u-d, v-b.
                adj[u].remove(b)
                adj[v].remove(d)
                adj[u].add(d)
                adj[v].add(b)
                
                new_score2 = _calculate_quartet_score(adj, best_quartets, labels)
                
                if new_score2 > current_score:
                    current_score = new_score2
                    improved = True
                else:
                    # Revert
                    adj[u].remove(d)
                    adj[v].remove(b)
                    adj[u].add(b)
                    adj[v].add(d)

    return adj

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 250
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 3:
        if n == 1:
            return labels[0] + ";"
        elif n == 2:
            return "(" + labels[0] + "," + labels[1] + ");"
        else:
            return "(" + ",".join(labels) + ");"
    
    # Step 1: Distance Matrix
    dist = _jc_distance_matrix(sequences, labels)
    
    # Step 2: Precompute Best Quartets
    best_quartets = _precompute_best_quartets(dist, labels)
    
    # Step 3: NJ Tree
    adj = _build_nj_tree_tracked(labels, dist)
    
    # Step 4: NNI Refinement
    if time.time() < deadline - 10:
        adj = _nni_refine(adj, best_quartets, labels, deadline - 10)
        
    # Step 5: Convert to Newick
    # Root at 0 (or any internal node)
    # Find an internal node
    internal_nodes = [u for u in adj if len(adj[u]) == 3]
    if not internal_nodes:
        root = 0
    else:
        root = internal_nodes[0]
        
    def to_newick(node, parent):
        neighbors = [x for x in adj[node] if x != parent]
        if node < n: # Leaf
            return labels[node]
        if not neighbors:
            return labels[node] # Should not happen for internal
        children_strs = [to_newick(nb, node) for nb in neighbors]
        return "(" + ",".join(children_strs) + ")"
        
    newick = to_newick(root, None) + ";"
    return newick
