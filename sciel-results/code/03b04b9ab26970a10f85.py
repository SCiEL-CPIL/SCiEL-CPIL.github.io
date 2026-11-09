    def find_valid_graph(n, time_limit):
        time_start = time.time()
        
        while time.time() - time_start < time_limit:
            # Initialize random graph
            adj = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i, j] = True
                        adj[j, i] = True
            
            # Repair loop
            steps = 0
            max_steps_per_restart = 10000 # Arbitrary limit
            
            while steps < max_steps_per_restart:
                if time.time() - time_start > time_limit:
                    return None, False
                    
                bad_set, is_k5 = find_one_bad_set(n, adj)
                
                if bad_set is None:
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i, j]:
                                edges.append((i, j))
                    return edges, True
                
                a, b = random.choice([(bad_set[i], bad_set[j]) for i in range(5) for j in range(i+1, 5)])
                adj[a, b] = not adj[a, b]
                adj[b, a] = adj[a, b]
                
                steps += 1
                
            # If we reached max_steps, restart
            continue

        return None, False
