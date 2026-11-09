    def construct():
        # ... precomputation ...
        
        best_edges = []
        best_count = 0
        
        start_time = time.time()
        deadline = start_time + 115.0
        
        # Estimate time for one iteration
        # If precomputation took T_pre, we have 115 - T_pre left.
        
        # Let's just run as many as possible.
        restart = 0
        while time.time() < deadline - 10:
            # Run Greedy
            removed_edges = [False] * num_edges_total
            # ... greedy logic with random tie-breaking ...
            
            # Run Polish
            # ... polish logic ...
            
            # Count edges
            curr_count = sum(1 for e in range(num_edges_total) if not removed_edges[e])
            
            if curr_count > best_count:
                best_count = curr_count
                best_edges = [[inv_edge_map[e][0], inv_edge_map[e][1]] for e in range(num_edges_total) if not removed_edges[e]]
            
            restart += 1
            # Optional: break if we hit a known upper bound or no improvement in last K restarts?
            # For now, just keep going.
            
        return best_edges
    