<<<<<<< SEARCH
    def local_search(adj_bits, missing_edges, deadline):
        """Run local search until deadline or convergence."""
        current_edges = edge_count(adj_bits)
        best_edges = current_edges
        best_adj_bits = adj_bits[:]
        best_missing = missing_edges[:]
        
        all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        
        while time.time() < deadline:
            random.shuffle(missing_edges)
            added_in_pass = 0
            new_missing = []
            for u, v in missing_edges:
                if time.time() > deadline:
                    break
                if can_add(adj_bits, u, v):
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    added_in_pass += 1
                    current_edges += 1
                else:
                    new_missing.append((u, v))
            
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj_bits = adj_bits[:]
                best_missing = new_missing[:]
            
            missing_edges = new_missing
            
            if added_in_pass == 0:
                # Try a kick: remove some random edges
                if current_edges > 5:
                    kick_size = random.randint(3, 7)
                    existing_edges = get_edges_from_bits(adj_bits)
                    random.shuffle(existing_edges)
                    for _ in range(kick_size):
                        if not existing_edges:
                            break
                        u, v = existing_edges.pop()
                        adj_bits[u] &= ~(1 << v)
                        adj_bits[v] &= ~(1 << u)
                        current_edges -= 1
                        missing_edges.append((u, v))
                else:
                    # If very few edges, just break
                    break
        
        return best_edges, best_adj_bits
=======
    def local_search(adj_bits, missing_edges, deadline):
        """Run local search until deadline or convergence."""
        current_edges = edge_count(adj_bits)
        best_edges = current_edges
        best_adj_bits = adj_bits[:]
        
        while time.time() < deadline:
            random.shuffle(missing_edges)
            added_in_pass = 0
            new_missing = []
            for u, v in missing_edges:
                if time.time() > deadline:
                    break
                if can_add(adj_bits, u, v):
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    added_in_pass += 1
                    current_edges += 1
                else:
                    new_missing.append((u, v))
            
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj_bits = adj_bits[:]
            
            missing_edges = new_missing
            
            if added_in_pass == 0:
                # Try a kick: remove some random edges
                if current_edges > 5:
                    kick_size = random.randint(3, 7)
                    existing_edges = get_edges_from_bits(adj_bits)
                    random.shuffle(existing_edges)
                    for _ in range(kick_size):
                        if not existing_edges:
                            break
                        u, v = existing_edges.pop()
                        adj_bits[u] &= ~(1 << v)
                        adj_bits[v] &= ~(1 << u)
                        current_edges -= 1
                        missing_edges.append((u, v))
                else:
                    break
        
        return best_edges, best_adj_bits
>>>>>>> REPLACE

