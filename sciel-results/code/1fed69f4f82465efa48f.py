import numpy as np
import time
import sys

def _jc_distance_matrix(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    seqs = np.array([list(sequences[label]) for label in labels], dtype='U1')
    
    # Efficient pairwise mismatch count
    # Using broadcasting: (n, 1, L) vs (1, n, L)
    mismatches = (seqs[:, None, :] != seqs[None, :, :]).sum(axis=2)
    p = mismatches / seqs.shape[1]
    
    d = np.zeros((n, n))
    mask = p < 0.75
    d[mask] = -0.75 * np.log(1.0 - 4.0 * p[mask] / 3.0)
    d[~mask] = 10.0  # Large penalty for saturated distances
    np.fill_diagonal(d, 0.0)
    
    return d, labels

def _neighbor_joining(dist, labels):
    n = len(labels)
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    
    # Initialize distance dictionary
    dist_dict = {}
    for i in range(n):
        for j in range(i+1, n):
            dist_dict[(i, j)] = dist[i, j]
            dist_dict[(j, i)] = dist[i, j]
    
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        
        # Build m x m distance matrix for active nodes
        d_active = np.zeros((m, m))
        for i_idx, ai in enumerate(active):
            for j_idx, aj in enumerate(active):
                if i_idx != j_idx:
                    d_active[i_idx, j_idx] = dist_dict[(ai, aj)]
        
        # Compute r vector: sum of distances from each node
        r = d_active.sum(axis=1)
        
        # Compute Q matrix: Q[i,j] = (m-2)*d[i,j] - r[i] - r[j]
        Q = (m - 2) * d_active - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        # Find minimum Q
        min_flat = np.argmin(Q)
        i_idx, j_idx = np.unravel_index(min_flat, Q.shape)
        if i_idx > j_idx:
            i_idx, j_idx = j_idx, i_idx
        
        best_i = active[i_idx]
        best_j = active[j_idx]
        
        # Create new node
        new_id = next_id
        next_id += 1
        
        tree_i = trees[best_i]
        tree_j = trees[best_j]
        new_tree = f"({tree_i},{tree_j})"
        
        # Compute new distances to other active nodes
        for x in active:
            if x not in (best_i, best_j):
                d_new_x = 0.5 * (dist_dict[(best_i, x)] + dist_dict[(best_j, x)] - dist_dict[(best_i, best_j)])
                dist_dict[(new_id, x)] = d_new_x
                dist_dict[(x, new_id)] = d_new_x
        
        # Remove best_i, best_j from active
        active = [x for x in active if x not in (best_i, best_j)]
        active.append(new_id)
        
        trees[new_id] = new_tree
    
    # Final 3 nodes
    if len(active) == 3:
        a, b, c = active
        return f"({trees[a]},({trees[b]},{trees[c]}));"
    else:
        return "(" + ",".join(trees[x] for x in active) + ");"

def _parse_newick_to_splits(newick, n_taxa):
    """Parse Newick string and return set of non-trivial splits."""
    newick = newick.strip()
    if newick.endswith(';'):
        newick = newick[:-1]
    
    # Parse the tree structure
    taxa_set = set()
    # First extract all taxa labels
    import re
    taxa_list = re.findall(r't\d+', newick)
    for t in taxa_list:
        taxa_set.add(t)
    
    if len(taxa_set) != n_taxa:
        return set()
    
    # Build adjacency list from Newick
    # We need to parse the tree properly
    splits = set()
    
    # Recursive parser
    def parse(node_str):
        """Return (set_of_taxa_in_subtree, children_list)"""
        node_str = node_str.strip()
        if not node_str:
            return set(), []
        
        if node_str.startswith('('):
            # Internal node
            # Find matching close paren
            depth = 0
            i = 0
            while i < len(node_str):
                if node_str[i] == '(':
                    depth += 1
                elif node_str[i] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            
            inner = node_str[1:i]
            # Split inner by commas at depth 0
            children = []
            depth = 0
            start = 0
            for j in range(len(inner)):
                if inner[j] == '(':
                    depth += 1
                elif inner[j] == ')':
                    depth -= 1
                elif inner[j] == ',' and depth == 0:
                    children.append(inner[start:j])
                    start = j + 1
            children.append(inner[start:])
            
            all_taxa = set()
            child_taxa_list = []
            for c in children:
                ct, _ = parse(c)
                all_taxa |= ct
                child_taxa_list.append(ct)
            
            return all_taxa, child_taxa_list
        else:
            # Leaf
            # Extract taxon label
            label = node_str.split('(')[0].split(')')[0].strip()
            # Remove branch length if present
            if ':' in label:
                label = label.split(':')[0].strip()
            return {label}, []
    
    all_taxa, children = parse(newick)
    
    # Now extract splits: for each internal edge, the split is (taxa on one side, taxa on other)
    # In an unrooted tree, each internal edge corresponds to a split
    # We can find splits by looking at each internal node's children
    # For each internal node with children c1, c2, ..., ck, the edge to each child ci gives a split
    # But we need to be careful about the root
    
    # Alternative: for each edge in the tree, compute the split
    # An unrooted tree with n taxa has n-3 internal edges
    # Each internal edge splits taxa into two sets of size >= 2
    
    # Let's use a different approach: 
    # For each subset S of taxa with 2 <= |S| <= n-2, check if S is a split
    # But that's exponential. Instead, use the tree structure.
    
    # Build the tree as a graph
    # Assign IDs to nodes
    node_id = 0
    nodes = {}  # id -> (type, data)
    
    def assign_ids(node_str, parent_id):
        nonlocal node_id
        node_str = node_str.strip()
        if not node_str:
            return None
        
        if node_str.startswith('('):
            # Internal node
            my_id = node_id
            node_id += 1
            nodes[my_id] = ('internal', [])
            
            depth = 0
            i = 0
            while i < len(node_str):
                if node_str[i] == '(':
                    depth += 1
                elif node_str[i] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            
            inner = node_str[1:i]
            children = []
            depth = 0
            start = 0
            for j in range(len(inner)):
                if inner[j] == '(':
                    depth += 1
                elif inner[j] == ')':
                    depth -= 1
                elif inner[j] == ',' and depth == 0:
                    children.append(inner[start:j])
                    start = j + 1
            children.append(inner[start:])
            
            for c in children:
                child_id = assign_ids(c, my_id)
                if child_id is not None:
                    nodes[my_id][1].append(child_id)
            
            return my_id
        else:
            # Leaf
            label = node_str.split('(')[0].split(')')[0].strip()
            if ':' in label:
                label = label.split(':')[0].strip()
            my_id = node_id
            node_id += 1
            nodes[my_id] = ('leaf', label)
            return my_id
    
    root_id = assign_ids(newick, None)
    
    # Build adjacency list
    adj = {i: [] for i in range(node_id)}
    for nid, (ntype, data) in nodes.items():
        if ntype == 'internal':
            for child in data:
                adj[nid].append(child)
                adj[child].append(nid)
    
    # For each internal edge (u, v), compute the split
    # Root the tree at an arbitrary internal node
    # Then for each edge, one side is the subtree below
    
    # Find all internal nodes
    internal_nodes = [nid for nid, (ntype, _) in nodes.items() if ntype == 'internal']
    
    if len(internal_nodes) == 0:
        return set()
    
    # Root at the first internal node
    root = internal_nodes[0]
    
    # Do BFS/DFS to compute subtrees
    subtree_taxa = {}
    visited = set()
    stack = [(root, None)]
    while stack:
        node, parent = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        
        if nodes[node][0] == 'leaf':
            subtree_taxa[node] = {nodes[node][1]}
        else:
            all_taxa = set()
            for child in adj[node]:
                if child != parent:
                    stack.append((child, node))
            # After processing children
            for child in adj[node]:
                if child != parent and child in subtree_taxa:
                    all_taxa |= subtree_taxa[child]
            subtree_taxa[node] = all_taxa
    
    # For each internal edge, compute the split
    # An edge (u, v) where both u and v are internal (or one is internal)
    # If we remove edge (u, v), the two components are:
    # - If v is child of u: subtree_taxa[v] and total_taxa - subtree_taxa[v]
    # - We only want splits where both sides have size >= 2
    
    total_taxa = set()
    for nid, (ntype, data) in nodes.items():
        if ntype == 'leaf':
            total_taxa.add(data)
    
    for u, v in adj.items():
        for w in v:
            if w == u:
                continue
            # Edge (u, w)
            # Determine which is parent in rooted tree
            # This is getting complex. Let me use a simpler approach.
    
    # Simpler: for each internal node except root, the edge to its parent gives a split
    # where one side is subtree_taxa[node]
    splits = set()
    for nid in range(node_id):
        if nodes[nid][0] == 'internal' and nid != root:
            # Find parent
            # In the rooted tree, the parent is the neighbor closer to root
            # This is still complex without explicit parent pointers
            pass
    
    # Let me use a different, cleaner approach
    # Parse the tree and for each internal node, consider each child
    # The edge between the internal node and its child defines a split
    # One side: all taxa in the child's subtree
    # Other side: all taxa not in the child's subtree
    # Only add to splits if both sides have size >= 2
    
    # Re-parse with explicit parent tracking
    def parse_with_parent(node_str, parent_id, adj_list, taxa_map):
        node_str = node_str.strip()
        if not node_str:
            return None
        
        if node_str.startswith('('):
            my_id = len(adj_list)
            adj_list.append([])
            if parent_id is not None:
                adj_list[parent_id].append(my_id)
                adj_list[my_id].append(parent_id)
            
            depth = 0
            i = 0
            while i < len(node_str):
                if node_str[i] == '(':
                    depth += 1
                elif node_str[i] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            
            inner = node_str[1:i]
            children = []
            depth = 0
            start = 0
            for j in range(len(inner)):
                if inner[j] == '(':
                    depth += 1
                elif inner[j] == ')':
                    depth -= 1
                elif inner[j] == ',' and depth == 0:
                    children.append(inner[start:j])
                    start = j + 1
            children.append(inner[start:])
            
            for c in children:
                parse_with_parent(c, my_id, adj_list, taxa_map)
            
            return my_id
        else:
            label = node_str.split('(')[0].split(')')[0].strip()
            if ':' in label:
                label = label.split(':')[0].strip()
            my_id = len(adj_list)
            adj_list.append([])
            taxa_map[my_id] = label
            if parent_id is not None:
                adj_list[parent_id].append(my_id)
                adj_list[my_id].append(parent_id)
            return my_id
    
    adj_list = []
    taxa_map = {}
    root = parse_with_parent(newick, None, adj_list, taxa_map)
    
    total_taxa = set(taxa_map.values())
    
    # Compute subtree taxa for each node via DFS
    subtree_taxa = {}
    def compute_subtree(node, parent):
        if node in taxa_map:
            subtree_taxa[node] = {taxa_map[node]}
        else:
            all_t = set()
            for child in adj_list[node]:
                if child != parent:
                    compute_subtree(child, node)
                    all_t |= subtree_taxa[child]
            subtree_taxa[node] = all_t
    
    compute_subtree(root, None)
    
    # For each internal node (except root), the edge to its parent defines a split
    # But we need to handle the unrooted nature
    # For each edge in the tree, if we remove it, we get two components
    # An edge is "internal" if both components have size >= 2
    # We can iterate over all edges
    
    for u in range(len(adj_list)):
        for v in adj_list[u]:
            if u < v:  # Avoid double counting
                # Component 1: subtree of v (if v is "below" u)
                # But we need to know the root orientation
                # Let's just compute both sides
                # Side 1: all taxa reachable from v without going through u
                # Side 2: all taxa reachable from u without going through v
                
                def get_component(start, exclude):
                    visited = set()
                    stack = [start]
                    while stack:
                        node = stack.pop()
                        if node in visited:
                            continue
                        visited.add(node)
                        for neighbor in adj_list[node]:
                            if neighbor != exclude and neighbor not in visited:
                                stack.append(neighbor)
                    taxa = set()
                    for nid in visited:
                        if nid in taxa_map:
                            taxa.add(taxa_map[nid])
                    return taxa
                
                side1 = get_component(v, u)
                side2 = total_taxa - side1
                
                if len(side1) >= 2 and len(side2) >= 2:
                    # Canonical representation: sort the two frozensets
                    s1 = frozenset(side1)
                    s2 = frozenset(side2)
                    splits.add((s1, s2) if s1 < s2 else (s2, s1))
    
    return splits

def _robinson_foulds(tree1_splits, tree2_splits):
    """Compute normalized Robinson-Foulds distance."""
    if not tree1_splits and not tree2_splits:
        return 0.0
    
    t1 = set()
    for s in tree1_splits:
        t1.add(s)
    
    t2 = set()
    for s in tree2_splits:
        t2.add(s)
    
    intersection = t1 & t2
    union = t1 | t2
    
    if not union:
        return 0.0
    
    return 1.0 - len(intersection) / len(union)

def reconstruct(sequences):
    start_time = time.time()
    deadline = 260.0  # Safety margin before 270s kill
    
    d, labels = _jc_distance_matrix(sequences)
    n = len(labels)
    
    # Check time
    if time.time() - start_time > deadline - 5:
        if n <= 2:
            return ",".join(labels) + ";"
        return "(" + ",".join(labels) + ");"
    
    # NJ
    tree_newick = _neighbor_joining(d, labels)
    
    return tree_newick
