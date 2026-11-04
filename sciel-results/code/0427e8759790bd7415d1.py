<<<<<<< SEARCH
    def greedy_min_degree_random():
        """Randomized greedy min-degree heuristic for independent set."""
        remaining = set(range(N_S))
        s = set()
        
        while remaining:
            min_deg = float('inf')
            candidates = []
            for v in remaining:
                deg = sum(1 for nb in adj_list[v] if nb in remaining)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [v]
                elif deg == min_deg:
                    candidates.append(v)
            
            # Pick randomly from candidates to ensure diversity
            min_node = candidates[np.random.randint(len(candidates))]
            s.add(min_node)
            to_remove = {min_node} | {nb for nb in adj_list[min_node] if nb in remaining}
            remaining -= to_remove
        
        return s
=======
    def greedy_min_degree_random():
        """Vectorized randomized greedy min-degree heuristic for independent set."""
        remaining_mask = np.ones(N_S, dtype=bool)
        s = set()
        
        while remaining_mask.any():
            # Calculate degrees of remaining nodes
            # adj_bool is (N_S, N_S)
            # degrees[i] = number of neighbors of i that are in remaining
            degrees = np.sum(adj_bool[:, remaining_mask], axis=1)
            
            # Mask out non-remaining nodes to ignore them
            degrees[~remaining_mask] = np.inf
            
            min_deg = np.min(degrees)
            
            # Find all nodes with min_deg
            candidates = np.where(degrees == min_deg)[0]
            
            # Pick randomly
            min_node = int(candidates[np.random.randint(len(candidates))])
            s.add(min_node)
            
            # Remove min_node and its neighbors from remaining
            # Neighbors are where adj_bool[min_node] is True
            to_remove_mask = np.zeros(N_S, dtype=bool)
            to_remove_mask[min_node] = True
            to_remove_mask[adj_bool[min_node]] = True
            
            remaining_mask &= ~to_remove_mask
        
        return s
>>>>>>> REPLACE
