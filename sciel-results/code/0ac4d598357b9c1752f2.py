    # Local search on the best solution
    if time.time() < deadline - 10:
        adj = best_adj[:]
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try adding any missing edge
            for u, v in all_edges:
                if not (adj[u] >> v) & 1:
                    if is_valid_add(u, v, adj):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        improved = True
                        # Restart scan to find more additions
                        break 
            
            # If no edge could be added, try removing one and re-optimizing
            if not improved:
                # Remove a random edge
                existing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (adj[i] >> j) & 1]
                if existing_edges:
                    u, v = random.choice(existing_edges)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    # Try to add edges again
                    for uu, vv in all_edges:
                        if not (adj[uu] >> vv) & 1:
                            if is_valid_add(uu, vv, adj):
                                adj[uu] |= (1 << vv)
                                adj[vv] |= (1 << uu)
                                improved = True
                                break
                    if count_edges(adj) > best_edges:
                        best_edges = count_edges(adj)
                        best_adj = adj[:]

