        # 2. Local Search
        edge_set = set(map(tuple, current_edges))
        
        while time.time() < deadline:
            random.shuffle(all_edges)
            progress = False
            for u, v in all_edges:
                if (u, v) in edge_set:
                    continue
                
                if creates_c6(u, v, adj):
                    # Try to kick
                    if current_edges:
                        idx = random.randint(0, len(current_edges) - 1)
                        ru, rv = current_edges[idx]
                        adj[ru] ^= (1 << rv)
                        adj[rv] ^= (1 << ru)
                        current_edges.pop(idx)
                        edge_set.remove((ru, rv))
                        progress = True
                    break # Restart loop after kick
                else:
                    # Add
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.append([u, v])
                    edge_set.add((u, v))
                    progress = True
                    if len(current_edges) > best_count:
                        best_count = len(current_edges)
                        best_edges = current_edges[:]
            
            if not progress:
                break # Stuck, restart greedy
