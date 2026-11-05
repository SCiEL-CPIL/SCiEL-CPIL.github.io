    def try_build(n, min_deg_target, time_limit=10.0):
        try_start = time.time()
        adj = [set() for _ in range(n)]
        deg = [0] * n
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        for attempt in range(5):
            if time.time() > try_start + time_limit:
                break

            adj = [set() for _ in range(n)]
            deg = [0] * n
            remaining = list(all_edges)
            random.shuffle(remaining)
            remaining_set = set(remaining)
            
            added_edges = []
            
            # We want to add until no more edges can be added or min_deg is reached
            # But we stop if min_deg is reached to save time?
            # No, we need to ensure min_deg is reached.
            # If we reach min_deg, we can stop.
            
            while len(remaining_set) > 0:
                if time.time() > try_start + time_limit:
                    break
                
                if min(deg) >= min_deg_target:
                    break
                
                min_deg = min(deg)
                min_vertices = [i for i in range(n) if deg[i] == min_deg]
                u = random.choice(min_vertices)
                
                # Try to add an edge incident to u
                success = False
                # Try up to 20 random v's
                for _ in range(20):
                    v = random.randint(0, n - 1)
                    if v == u:
                        continue
                    edge = (min(u, v), max(u, v))
                    if edge not in remaining_set:
                        continue
                    
                    # Check C4
                    valid = True
                    for w in adj[u]:
                        if adj[w] & adj[v]:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Add edge
                    adj[u].add(v)
                    adj[v].add(u)
                    deg[u] += 1
                    deg[v] += 1
                    added_edges.append((u, v))
                    remaining_set.remove(edge)
                    success = True
                    break
                
                if not success:
                    # If we can't add any edge to u, maybe u is "stuck"?
                    # We can just try another u in the next iteration.
                    # But if all min_deg vertices are stuck, we might loop.
                    # To avoid infinite loop, if we fail, we can remove u from consideration?
                    # Or just continue. The random v choice will change.
                    pass

            if min(deg) >= min_deg_target:
                edges = [[u, v] for u, v in added_edges]
                return edges

        return None
