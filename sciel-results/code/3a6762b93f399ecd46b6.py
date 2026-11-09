    def pick_weighted_common(current_edges, adj, n, rng):
        # Calculate weights (common neighbors)
        weights = []
        total_weight = 0
        for u, v in current_edges:
            common = 0
            for w in adj[u]:
                if w != v:
                    e_vw = (v, w) if v < w else (w, v)
                    if e_vw in current_edges:
                        common += 1
            # Add a small base weight to avoid zero division or zero prob for hard edges
            w_val = common + 0.1 
            weights.append((common + 0.1, (u,v)))
            total_weight += common + 0.1
        
        if total_weight == 0:
            return rng.choice(list(current_edges))
        
        # Select based on weights
        r = rng.random() * total_weight
        cumulative = 0
        for w_val, e in weights:
            cumulative += w_val
            if cumulative >= r:
                return e
        return list(current_edges)[-1]
