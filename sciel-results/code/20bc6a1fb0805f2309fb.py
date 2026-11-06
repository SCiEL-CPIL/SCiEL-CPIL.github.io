import time
import random
import itertools

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # We'll try multiple random orderings of edges
    best_edges = []  # best valid edge list found
    
    # For efficiency, we maintain adjacency as a dict of dicts
    # adj[u][v] = color
    
    def try_greedy(order, deadline):
        """Greedy: process edges in given order, add if colorable."""
        adj = {}  # adj[u][v] = color
        edges_added = []
        
        for (u, v) in order:
            if time.time() > deadline:
                break
            # Try to find a color for edge (u,v) that doesn't create a non-rainbow 4-cycle
            # A 4-cycle involving edge (u,v): u-v-w-x-u where w,x are other vertices
            # The other 3 edges are (v,w), (w,x), (x,u)
            # We need: for all w,x with w!=x, w!=u,v, x!=u,v:
            #   if (v,w), (w,x), (x,u) all exist, then color(u,v) must differ from all three
            
            forbidden = set()
            for w in range(n):
                if w == u or w == v:
                    continue
                if v not in adj or w not in adj.get(v, {}):
                    continue
                cw_vw = adj[v][w]
                for x in range(n):
                    if x == u or x == v or x == w:
                        continue
                    if w not in adj or x not in adj.get(w, {}):
                        continue
                    if x not in adj or u not in adj.get(x, {}):
                        continue
                    cw_wx = adj[w][x]
                    cw_xu = adj[x][u]
                    # The 4-cycle is u-v-w-x-u
                    # Edges: (u,v)=?, (v,w)=cw_vw, (w,x)=cw_wx, (x,u)=cw_xu
                    # Need all 4 distinct
                    forbidden.add(cw_vw)
                    forbidden.add(cw_wx)
                    forbidden.add(cw_xu)
            
            # Also need to check: the new color must not equal any of the 3 other edges in any 4-cycle
            # We already collected forbidden colors. Pick a color not in forbidden.
            available = [c for c in range(num_colors) if c not in forbidden]
            
            if not available:
                continue  # Can't add this edge
            
            # Pick a color (randomly from available, or strategically)
            # To maximize future flexibility, pick the color that appears least among existing edges?
            # For simplicity, pick randomly
            color = random.choice(available)
            
            # Add the edge
            if u not in adj:
                adj[u] = {}
            if v not in adj:
                adj[v] = {}
            adj[u][v] = color
            adj[v][u] = color
            edges_added.append((u, v, color))
        
        return edges_added
    
    # Run multiple trials with different random orderings
    num_trials = 15
    trial_time_budget = (deadline - start_time) / num_trials
    
    for trial in range(num_trials):
        trial_deadline = start_time + (trial + 1) * trial_time_budget
        if trial_deadline > deadline - 5:
            trial_deadline = deadline - 5
        
        if time.time() > trial_deadline - 1:
            break
        
        # Shuffle edge order
        order = all_pairs[:]
        random.shuffle(order)
        
        result = try_greedy(order, trial_deadline)
        
        if len(result) > len(best_edges):
            best_edges = result
    
    # If we found a good solution, return it
    if best_edges:
        return [[u, v, c] for (u, v, c) in best_edges]
    
    # Fallback: a star (always valid)
    return [[0, v, 0] for v in range(1, 20)]
