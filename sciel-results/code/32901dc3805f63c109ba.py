<<<<<<< SEARCH
    def local_search(adj, max_time=30):
        """Local search: try edge swaps to improve edge count"""
        adj_list = make_adj_list(adj)
        current_count = count_edges(adj)
        best_adj = [row[:] for row in adj]
        best_count = current_count
        search_deadline = time.time() + max_time
        
        # Get current edges and missing edges
        current_edges = set()
        missing_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    current_edges.add((i, j))
                else:
                    missing_edges.append((i, j))
        
        # Try to add missing edges
        for u, v in missing_edges:
            if time.time() > search_deadline:
                break
            
            # Tentatively add
            adj[u][v] = True
            adj[v][u] = True
            adj_list[u].append(v)
            adj_list[v].append(u)
            
            if not check_q3_with_edge(u, v, adj, adj_list):
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_adj = [row[:] for row in adj]
            else:
                # Revert
                adj[u][v] = False
                adj[v][u] = False
                adj_list[u].remove(v)
                adj_list[v].remove(u)
        
        # Try edge swaps: remove one edge, try to add two
        current_edges = set()
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    current_edges.add((i, j))
        
        missing_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i][j]:
                    missing_edges.append((i, j))
        
        random.shuffle(missing_edges)
        
        for edge_to_remove in list(current_edges):
            if time.time() > search_deadline:
                break
            
            e1, e2 = edge_to_remove
            # Remove the edge
            adj[e1][e2] = False
            adj[e2][e1] = False
            adj_list[e1].remove(e2)
            adj_list[e2].remove(e1)
            current_count -= 1
            
            # Try to add missing edges
            added = 0
            added_edges = []
            for u, v in missing_edges:
                if u == e1 and v == e2:
                    continue
                # Check if this edge was already present (it shouldn't be, but just in case)
                if adj[u][v]:
                    continue
                    
                adj[u][v] = True
                adj[v][u] = True
                adj_list[u].append(v)
                adj_list[v].append(u)
                
                if not check_q3_with_edge(u, v, adj, adj_list):
                    added += 1
                    added_edges.append((u, v))
                    current_count += 1
                else:
                    adj[u][v] = False
                    adj[v][u] = False
                    adj_list[u].remove(v)
                    adj_list[v].remove(u)
                
                if added >= 2:
                    break
            
            # If we added more than we removed, keep the change
            if added > 1:
                if current_count > best_count:
                    best_count = current_count
                    best_adj = [row[:] for row in adj]
            else:
                # Revert the removal
                adj[e1][e2] = True
                adj[e2][e1] = True
                adj_list[e1].append(e2)
                adj_list[e2].append(e1)
                current_count += 1
                # Revert added edges
                for u, v in added_edges:
                    adj[u][v] = False
                    adj[v][u] = False
                    adj_list[u].remove(v)
                    adj_list[v].remove(u)
                    current_count -= 1
        
        return best_adj, best_count
=======
    def local_search(adj, max_time=30):
        """Local search with random kicks to escape local optima"""
        adj_list = make_adj_list(adj)
        current_count = count_edges(adj)
        best_adj = [row[:] for row in adj]
        best_count = current_count
        search_deadline = time.time() + max_time
        
        def try_add_missing_edges(adj, adj_list, current_count):
            """Try to add all possible missing edges"""
            missing = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not adj[i][j]:
                        missing.append((i, j))
            
            for u, v in missing:
                if time.time() > search_deadline:
                    break
                
                adj[u][v] = True
                adj[v][u] = True
                adj_list[u].append(v)
                adj_list[v].append(u)
                
                if not check_q3_with_edge(u, v, adj, adj_list):
                    current_count += 1
                else:
                    adj[u][v] = False
                    adj[v][u] = False
                    adj_list[u].remove(v)
                    adj_list[v].remove(u)
            
            return current_count
        
        # First pass: add all possible missing edges
        current_count = try_add_missing_edges(adj, adj_list, current_count)
        if current_count > best_count:
            best_count = current_count
            best_adj = [row[:] for row in adj]
        
        # Multiple kicks to escape local optima
        num_kicks = 0
        while time.time() < search_deadline - 5:
            # Create a kick: remove k random edges
            current_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i][j]]
            if not current_edges:
                break
            
            kick_size = min(3, len(current_edges))
            random_kicks = random.sample(current_edges, kick_size)
            
            # Remove kick edges
            for u, v in random_kicks:
                adj[u][v] = False
                adj[v][u] = False
                adj_list[u].remove(v)
                adj_list[v].remove(u)
                current_count -= 1
            
            # Try to add back missing edges
            current_count = try_add_missing_edges(adj, adj_list, current_count)
            
            if current_count > best_count:
                best_count = current_count
                best_adj = [row[:] for row in adj]
            else:
                # Revert to best if kick didn't help
                adj = [row[:] for row in best_adj]
                adj_list = make_adj_list(adj)
                current_count = best_count
            
            num_kicks += 1
        
        return best_adj, best_count
>>>>>>> REPLACE

