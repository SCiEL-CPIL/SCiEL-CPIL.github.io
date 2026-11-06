"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    import sys
    sys.setrecursionlimit(10000)
    
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s kill
    
    def instance(n, p, seed):
        rng = Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    
    def min_vertex_cover(n, edges):
        """Find minimum vertex cover using branch and bound."""
        if not edges:
            return set()
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Isolated vertices can be excluded
        active = set(range(n))
        for v in range(n):
            if not adj[v]:
                active.discard(v)
        
        best = set(active)  # worst case: all vertices
        best_size = len(best)
        
        # Sort edges for consistency
        edges_sorted = sorted(edges)
        
        def branch(vertices_left, chosen, edges_remaining):
            nonlocal best, best_size
            if time.time() > DEADLINE:
                return
            if not edges_remaining:
                if len(chosen) < best_size:
                    best_size = len(chosen)
                    best = set(chosen)
                return
            # Prune: even if we pick nothing more, we can't beat best
            if len(chosen) >= best_size:
                return
            
            # Pick an edge
            u, v = edges_remaining[0]
            remaining = edges_remaining[1:]
            
            # Option 1: pick u
            # Remove all edges incident to u
            new_edges_1 = []
            for a, b in remaining:
                if a != u and b != u:
                    new_edges_1.append((a, b))
            # Also remove u from consideration
            new_vertices_1 = vertices_left - {u}
            branch(new_vertices_1, chosen | {u}, new_edges_1)
            
            # Option 2: pick v
            new_edges_2 = []
            for a, b in remaining:
                if a != v and b != v:
                    new_edges_2.append((a, b))
            new_vertices_2 = vertices_left - {v}
            branch(new_vertices_2, chosen | {v}, new_edges_2)
        
        # Better approach: use a more efficient B&B
        # Convert to set-based representation for speed
        edge_set = set(edges)
        # Build adjacency as frozensets
        adj_list = [set() for _ in range(n)]
        for u, v in edges:
            adj_list[u].add(v)
            adj_list[v].add(u)
        
        best_cover = set(range(n))
        best_size = len(best_cover)
        
        # Use a recursive B&B with bitmasks for small graphs
        if n <= 45:
            # Use integer bitmask
            full_mask = (1 << n) - 1
            edge_mask = 0
            for u, v in edges:
                edge_mask |= (1 << (u * n + v))
            
            # Precompute: for each vertex, which edges it covers
            vertex_edges = [0] * n
            for u, v in edges:
                idx = u * n + v
                vertex_edges[u] |= (1 << idx)
                vertex_edges[v] |= (1 << idx)
            
            # Count edges
            num_edges = len(edges)
            
            # B&B with pruning
            # State: (current_mask_of_chosen, uncovered_edges_mask, chosen_count)
            # Use a priority queue or DFS
            
            def bnb(chosen_mask, uncovered, count):
                nonlocal best_cover, best_size
                if time.time() > DEADLINE:
                    return
                if uncovered == 0:
                    if count < best_size:
                        best_size = count
                        # Reconstruct cover
                        cover = set()
                        m = chosen_mask
                        for v in range(n):
                            if m & (1 << v):
                                cover.add(v)
                        best_cover = cover
                    return
                if count >= best_size:
                    return
                # Find an uncovered edge
                # Get lowest set bit
                lsb = uncovered & (-uncovered)
                idx = lsb.bit_length() - 1
                u = idx // n
                v = idx % n
                
                # Option 1: pick u
                new_uncovered_1 = uncovered & ~vertex_edges[u]
                bnb(chosen_mask | (1 << u), new_uncovered_1, count + 1)
                
                # Option 2: pick v
                new_uncovered_2 = uncovered & ~vertex_edges[v]
                bnb(chosen_mask | (1 << v), new_uncovered_2, count + 1)
            
            bnb(0, edge_mask, 0)
            return best_cover
        else:
            # Fallback: greedy
            remaining = set(edges)
            chosen = set()
            while remaining:
                # Find vertex covering most edges
                best_v = None
                best_count = -1
                for v in range(n):
                    count = sum(1 for u, w in remaining if u == v or w == v)
                    if count > best_count:
                        best_count = count
                        best_v = v
                chosen.add(best_v)
                remaining = {(u, w) for u, w in remaining if u != best_v and w != best_v}
            return chosen
    
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > DEADLINE:
            # Return greedy solutions for remaining
            edges = instance(n, p, seed)
            remaining = set(edges)
            chosen = set()
            while remaining:
                best_v = None
                best_count = -1
                for v in range(n):
                    count = sum(1 for u, w in remaining if u == v or w == v)
                    if count > best_count:
                        best_count = count
                        best_v = v
                chosen.add(best_v)
                remaining = {(u, w) for u, w in remaining if u != best_v and w != best_v}
            out.append(sorted(chosen))
        else:
            edges = instance(n, p, seed)
            cover = min_vertex_cover(n, edges)
            out.append(sorted(cover))
    
    return out
# EVOLVE-BLOCK-END
