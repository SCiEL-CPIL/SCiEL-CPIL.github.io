def _spr_improve(newick, sequences, labels, deadline):
    start_time = time.time()
    remaining_time = deadline - start_time
    
    if remaining_time < 30:
        return newick
    
    nodes, adj, leaf_to_node, node_to_label, root_id = _parse_newick_to_tree(newick, labels)
    n_nodes = len(nodes)
    n_taxa = len(labels)
    
    seq_len = len(sequences[labels[0]])
    enc = _encode_sequences(sequences, labels)
    
    # Optimize branch lengths for initial tree
    bl, current_ll = _optimize_branch_lengths(adj, leaf_to_node, enc, n_taxa, seq_len, max_iter=10, tol=1e-4)
    
    best_newick = newick
    best_ll = current_ll
    best_bl = bl
    
    # Build parent-child from root
    parent = {root_id: None}
    children = {root_id: []}
    visited = {root_id}
    queue = [root_id]
    while queue:
        u = queue.pop(0)
        for v in adj.get(u, set()):
            if v not in visited:
                visited.add(v)
                parent[v] = u
                if v not in children:
                    children[v] = []
                children[u].append(v)
                queue.append(v)
    
    leaf_nodes = set(leaf_to_node.values())
    
    # Get all edges
    edges = []
    for u in adj:
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    
    max_moves = 1000
    moves_done = 0
    
    while time.time() < deadline - 5:
        if moves_done >= max_moves:
            break
            
        improved = False
        for edge_idx, (u, v) in enumerate(edges):
            if time.time() > deadline - 5:
                break
            if moves_done >= max_moves:
                break
            
            if parent.get(v) != u:
                continue
            
            # Get the subtree rooted at v
            subtree_nodes = set()
            stack = [v]
            while stack:
                node = stack.pop()
                if node in subtree_nodes:
                    continue
                subtree_nodes.add(node)
                for c in children.get(node, []):
                    stack.append(c)
            
            # Try regrafting at other edges
            for reg_edge_idx, (ru, rv) in enumerate(edges):
                if reg_edge_idx == edge_idx:
                    continue
                if time.time() > deadline - 5 or moves_done >= max_moves:
                    break
                
                if ru in subtree_nodes or rv in subtree_nodes:
                    continue
                
                new_adj = {k: set(v2) for k, v2 in adj.items()}
                
                new_adj[u].discard(v)
                new_adj[v].discard(u)
                
                if ru in new_adj[v]:
                    continue
                
                new_adj[v].add(ru)
                new_adj[ru].add(v)
                
                n_edges = sum(len(s) for s in new_adj.values()) // 2
                if n_edges != n_nodes - 1:
                    continue
                
                visited_set = {u}
                q = [u]
                while q:
                    node = q.pop(0)
                    for nb in new_adj[node]:
                        if nb not in visited_set:
                            visited_set.add(nb)
                            q.append(nb)
                
                if len(visited_set) != n_nodes:
                    continue
                
                moves_done += 1
                
                # Compute likelihood for new topology
                # Use fewer iterations for BL optimization during search,
                # but enough to get a good estimate.
                new_bl, new_ll = _optimize_branch_lengths(new_adj, leaf_to_node, enc, n_taxa, seq_len, max_iter=5, tol=1e-3)
                
                if new_ll > best_ll:
                    adj = new_adj
                    best_ll = new_ll
                    best_bl = new_bl
                    best_newick = _tree_to_newick(adj, leaf_to_node, node_to_label, root_id)
                    improved = True
                    
                    # Rebuild parent-child and edges
                    parent = {root_id: None}
                    children = {root_id: []}
                    visited = {root_id}
                    queue = [root_id]
                    while queue:
                        uu = queue.pop(0)
                        for vv in adj.get(uu, set()):
                            if vv not in visited:
                                visited.add(vv)
                                parent[vv] = uu
                                if vv not in children:
                                    children[vv] = []
                                children[uu].append(vv)
                                queue.append(vv)
                    
                    edges = []
                    for uu in adj:
                        for vv in adj[uu]:
                            if uu < vv:
                                edges.append((uu, vv))
                    break
        
        if not improved:
            break
    
    return best_newick
