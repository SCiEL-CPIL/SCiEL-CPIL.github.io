"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    start_time = time.time()
    
    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)
    
    def color_graph_mis(edges, deadline):
        """Color via repeated maximal independent set extraction with local search."""
        if not edges:
            return []
        
        m = len(edges)
        
        # Build adjacency list for the graph
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        # Precompute covers and conflict masks
        covers = []
        for u, v in edges:
            cov = {u, v}
            cov.update(adj.get(u, set()))
            cov.update(adj.get(v, set()))
            covers.append(cov)
        
        conflict_masks = [0] * m
        for i in range(m):
            cov_i = covers[i]
            mask = 0
            for j in range(m):
                if i == j:
                    continue
                u_j, v_j = edges[j]
                if u_j in cov_i or v_j in cov_i:
                    mask |= (1 << j)
            conflict_masks[i] = mask
        
        # Conflict degree for each edge
        conflict_deg = [bin(cm).count('1') for cm in conflict_masks]
        
        def greedy_mis_color(remaining_mask, rng):
            """Extract maximal independent sets greedily to color remaining edges."""
            color = [-1] * m
            color_count = 0
            uncolored = remaining_mask
            
            while uncolored and time.time() < deadline:
                # Find a maximal independent set among uncolored edges
                # Strategy: pick edge with fewest conflicts first (most "compatible")
                candidates = [i for i in range(m) if (uncolored >> i) & 1]
                if not candidates:
                    break
                
                # Sort by conflict degree within uncolored set (ascending)
                def score(i):
                    cm = conflict_masks[i] & uncolored
                    return bin(cm).count('1')
                
                candidates.sort(key=score)
                
                # Greedily build MIS
                mis = []
                mis_mask = 0
                for i in candidates:
                    if not ((uncolored >> i) & 1):
                        continue
                    # Check if i conflicts with any already in MIS
                    if conflict_masks[i] & mis_mask:
                        continue
                    mis.append(i)
                    mis_mask |= (1 << i)
                
                if not mis:
                    # Shouldn't happen, but safety
                    break
                
                # Assign this color
                for i in mis:
                    color[i] = color_count
                
                # Remove from uncolored
                for i in mis:
                    uncolored &= ~(1 << i)
                
                color_count += 1
            
            # Any remaining uncolored (shouldn't happen)
            if uncolored:
                for i in range(m):
                    if (uncolored >> i) & 1:
                        color[i] = color_count
                        color_count += 1
            
            return color, color_count
        
        # Try multiple MIS-based colorings with different random seeds
        best_color = None
        best_count = m + 1
        
        rng = Random(12345)
        
        # First: deterministic MIS (no randomness, just sorted by conflict degree)
        det_color, det_count = greedy_mis_color((1 << m) - 1, deadline)
        if det_count < best_count:
            best_count = det_count
            best_color = det_color[:]
        
        # Then: randomized MIS with multiple restarts
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            
            # Randomize the order of candidates slightly
            def randomized_greedy_mis(remaining_mask, rng_local):
                color = [-1] * m
                color_count = 0
                uncolored = remaining_mask
                
                while uncolored and time.time() < deadline:
                    candidates = [i for i in range(m) if (uncolored >> i) & 1]
                    if not candidates:
                        break
                    
                    # Score with some randomness
                    def score(i):
                        cm = conflict_masks[i] & uncolored
                        base = bin(cm).count('1')
                        # Add small random perturbation
                        return base + rng_local.random() * 0.1
                    
                    candidates.sort(key=score)
                    
                    mis = []
                    mis_mask = 0
                    for i in candidates:
                        if not ((uncolored >> i) & 1):
                            continue
                        if conflict_masks[i] & mis_mask:
                            continue
                        mis.append(i)
                        mis_mask |= (1 << i)
                    
                    if not mis:
                        break
                    
                    for i in mis:
                        color[i] = color_count
                    
                    for i in mis:
                        uncolored &= ~(1 << i)
                    
                    color_count += 1
                
                if uncolored:
                    for i in range(m):
                        if (uncolored >> i) & 1:
                            color[i] = color_count
                            color_count += 1
                
                return color, color_count
            
            color, cnt = randomized_greedy_mis((1 << m) - 1, rng)
            if cnt < best_count:
                best_count = cnt
                best_color = color[:]
        
        # Local search: try to reduce colors by moving edges
        if best_color is not None and best_count > 1:
            color = best_color[:]
            num_colors = best_count
            
            # Build color class masks
            color_masks = [0] * num_colors
            for i in range(m):
                c = color[i]
                if c >= 0:
                    color_masks[c] |= (1 << i)
            
            improved = True
            while improved and time.time() < deadline - 3:
                improved = False
                
                # Try to eliminate the highest color
                top_color = num_colors - 1
                if top_color < 0:
                    break
                
                top_mask = color_masks[top_color]
                if top_mask == 0:
                    num_colors -= 1
                    continue
                
                # Try to move each edge from top_color to another color
                edges_in_top = [i for i in range(m) if (top_mask >> i) & 1]
                
                moved_any = False
                for i in edges_in_top:
                    if not ((top_mask >> i) & 1):
                        continue
                    
                    # Try to move edge i to an existing color
                    for c in range(top_color):
                        if color_masks[c] == 0:
                            continue
                        # Check if i can be added to color c
                        if conflict_masks[i] & color_masks[c]:
                            continue
                        # Also check: removing i from top and adding to c is valid
                        # (removal is always valid, addition checked above)
                        color[i] = c
                        color_masks[top_color] &= ~(1 << i)
                        color_masks[c] |= (1 << i)
                        moved_any = True
                        break
                    
                    if not moved_any and time.time() > deadline - 3:
                        break
                
                if moved_any:
                    # Check if top color is now empty
                    if color_masks[top_color] == 0:
                        num_colors -= 1
                    improved = True
            
            if num_colors < best_count:
                best_count = num_colors
                best_color = color[:]
        
        return best_color
    
    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() - start_time > 130:
            # Fallback to simple greedy
            edges = instance(*spec)
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colour.items() if set(f) & near}
                colour[e] = next(c for c in range(len(edges)) if c not in taken)
            out.append([colour[e] for e in edges])
            continue
            
        edges = instance(*spec)
        deadline = min(start_time + 140, time.time() + 40)
        colors = color_graph_mis(edges, deadline)
        out.append(colors)
        
    return out
# EVOLVE-BLOCK-END
