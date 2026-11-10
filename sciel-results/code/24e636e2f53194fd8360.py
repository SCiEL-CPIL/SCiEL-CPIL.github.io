def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_edges = []
    best_count = 0
    
    rng = random.Random(42)
    
    trial = 0
    while time.time() < deadline:
        trial += 1
        order = ALL_PAIRS[:]
        rng.shuffle(order)
        
        adj = [0] * N
        current_edges = set()
        
        for u, v in order:
            if time.time() > deadline - 1.0:
                break
            if creates_q3(u, v, adj) is None:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.add((u, v))
                
        # Refinement
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for u, v in ALL_PAIRS:
                if (u, v) in current_edges:
                    continue
                if creates_q3(u, v, adj) is None:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    improved = True
                    
        # Perturbation if stuck
        if not improved and time.time() < deadline - 5.0:
            # Try a few random removals
            for _ in range(10):
                if time.time() > deadline - 1.0:
                    break
                if not current_edges:
                    break
                # Pick random edge to remove
                rem_edge = rng.choice(list(current_edges))
                u, v = rem_edge
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                current_edges.remove(rem_edge)
                
                old_count = len(current_edges)
                
                # Re-greedy
                imp2 = True
                while imp2 and time.time() < deadline - 1.0:
                    imp2 = False
                    for uu, vv in ALL_PAIRS:
                        if (uu, vv) in current_edges:
                            continue
                        if creates_q3(uu, vv, adj) is None:
                            adj[uu] |= (1 << vv)
                            adj[vv] |= (1 << uu)
                            current_edges.add((uu, vv))
                            imp2 = True
                            
                if len(current_edges) > old_count:
                    # Keep the change, break perturbation loop? Or continue?
                    # Continue to see if we can improve further
                    pass
                else:
                    # Revert
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.add(rem_edge)
                    # If revert, maybe stop perturbing this trial?
                    # Let's stop perturbing this trial to save time
                    break

        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
            
    return [[u, v] for u, v in best_edges]
