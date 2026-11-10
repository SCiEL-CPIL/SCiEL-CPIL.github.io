import random
import time
from itertools import combinations

def construct():
    n = 16
    all_edges = list(combinations(range(n), 2))
    best_edges = []
    best_count = 0
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Helper: check if adding edge (u,v) creates a triangle using bitmasks
    # adj is a list of bitmasks
    
    def is_not_3_colorable(adj):
        # Backtracking 3-colorability check with bitmask adjacency
        # adj is list of int bitmasks
        colors = [-1] * n
        
        def backtrack(v):
            if v == n:
                return True
            # Try colors 0, 1, 2
            used = 0
            mask = adj[v]
            while mask:
                lsb = mask & -mask
                u = lsb.bit_length() - 1
                mask ^= lsb
                if colors[u] != -1:
                    used |= 1 << colors[u]
            for c in range(3):
                if not (used & (1 << c)):
                    colors[v] = c
                    if backtrack(v + 1):
                        return True
            colors[v] = -1
            return False
        
        # Ordering: try vertices with highest degree first? 
        # For simplicity, fixed order 0..n-1, but we can permute.
        # Let's use a static ordering that might be better or just 0..n-1.
        # Actually, dynamic ordering is better for search but harder to implement recursively simply.
        # Let's stick to simple recursion but maybe reorder vertices once at start?
        # Or just rely on the fact that n=16 is small.
        
        # Optimization: MRV (Minimum Remaining Values) heuristic
        uncolored = list(range(n))
        
        def backtrack_mrv():
            if not uncolored:
                return True
            
            # Pick vertex with fewest available colors
            best_v = -1
            best_avail = 4
            for v in uncolored:
                used = 0
                mask = adj[v]
                while mask:
                    lsb = mask & -mask
                    u = lsb.bit_length() - 1
                    mask ^= lsb
                    if colors[u] != -1:
                        used |= 1 << colors[u]
                avail = 3 - bin(used & 7).count('1')
                if avail < best_avail:
                    best_avail = avail
                    best_v = v
                    if avail == 0:
                        return False
                    if avail == 1:
                        break
            
            v = best_v
            uncolored.remove(v)
            
            used = 0
            mask = adj[v]
            while mask:
                lsb = mask & -mask
                u = lsb.bit_length() - 1
                mask ^= lsb
                if colors[u] != -1:
                    used |= 1 << colors[u]
            
            for c in range(3):
                if not (used & (1 << c)):
                    colors[v] = c
                    if backtrack_mrv():
                        return True
            colors[v] = -1
            uncolored.append(v)
            return False
        
        return not backtrack_mrv()

    def generate_maximal_tf():
        random.shuffle(all_edges)
        adj = [0] * n
        edges = []
        for u, v in all_edges:
            # Check if adding (u,v) creates a triangle
            # Common neighbors of u and v must be empty
            if (adj[u] & adj[v]) == 0:
                # Add edge
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                edges.append((u, v))
        return edges, adj

    # Try many random maximal triangle-free graphs
    num_trials = 20000
    for i in range(num_trials):
        if time.time() > deadline:
            break
            
        edges, adj = generate_maximal_tf()
        # Check if 4-chromatic (not 3-colorable)
        if is_not_3_colorable(adj):
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = list(edges)
                
    # Local search to add more edges?
    # Since it's maximal triangle-free, we can't add edges without creating triangles.
    # However, we can try "swap" moves: remove an edge, then try to add others.
    # Or, we can try to start from a known good structure (like blow-up of Grotzsch) and optimize.
    
    # Let's also try a specific construction: Blow-up of Grotzsch graph.
    # Grotzsch graph is 11 vertices, 20 edges.
    # If we blow up vertices, we need to maintain triangle-free.
    # This is complex to do generally, so we stick to random search + local improvement if found.
    
    if best_edges:
        # Try to improve the best found graph via local search
        adj = [0] * n
        for u, v in best_edges:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            
        current_edges = list(best_edges)
        
        # Attempt to add edges by removing others?
        # Or just try random restarts of the maximal construction again with bias?
        # The random maximal construction is already "maximal".
        # Different orderings give different maximal graphs.
        # We have already done 20000 trials.
        
        # One more idea: Check if we can add an edge by removing a conflicting one?
        # This is a 2-opt move.
        # If (u,v) is not an edge, but creates triangles with w, we could remove (u,w) and add (u,v)?
        # This is getting complex. The random search over 20000 orders is likely sufficient for n=16.
        
        # Let's do a quick check: is there any pair (u,v) not in edges that can be added?
        # No, it's maximal.
        
        # Let's try to "shrink" and "expand" to escape local optima of the random greedy?
        # No, the random greedy produces maximal graphs. The space of maximal TF graphs is large.
        # 20000 samples should find the densest one with high probability.
        
        pass

    # If we didn't find any (unlikely), return seed
    if not best_edges:
        # Grötzsch graph
        cycle = [(i, (i + 1) % 5) for i in range(5)]
        edges = [list(e) for e in cycle]
        for u, w in cycle:
            edges += [[u, 5 + w], [w, 5 + u]]
        return edges + [[5 + u, 10] for u in range(5)]
        
    return [list(e) for e in best_edges]
