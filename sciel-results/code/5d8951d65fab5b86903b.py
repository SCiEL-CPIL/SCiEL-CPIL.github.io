"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    n = 20
    # State: adj[u][v] is color (0-19) or -1 if no edge
    adj = [[-1 for _ in range(n)] for _ in range(n)]
    
    def get_c4s_involving(u, v):
        """
        Yield (w, x) such that u-v-w-x-u is a 4-cycle in the current graph.
        Assumes (u,v) is in the graph.
        """
        for w in range(n):
            if w == u or w == v: continue
            if adj[v][w] == -1: continue
            for x in range(n):
                if x == u or x == v or x == w: continue
                if adj[u][x] == -1: continue
                if adj[w][x] != -1:
                    yield (w, x)

    def is_rainbow(c1, c2, c3, c4):
        return len({c1, c2, c3, c4}) == 4

    def check_change(u, v, new_c):
        """Check if changing color of (u,v) to new_c results in 0 violations for C4s involving (u,v)."""
        for w, x in get_c4s_involving(u, v):
            if not is_rainbow(new_c, adj[v][w], adj[w][x], adj[x][u]):
                return False
        return True

    def check_add(u, v, c):
        """Check if adding (u,v) with color c creates any non-rainbow C4."""
        # Temporarily add to check? Or just simulate.
        # Simulation is safer.
        for w in range(n):
            if w == u or w == v: continue
            if adj[v][w] == -1: continue
            for x in range(n):
                if x == u or x == v or x == w: continue
                if adj[u][x] == -1: continue
                if adj[w][x] != -1:
                    if not is_rainbow(c, adj[v][w], adj[w][x], adj[x][u]):
                        return False
        return True

    start_time = time.time()
    time_limit = 115.0
    
    best_edges = 0
    best_adj = [row[:] for row in adj]
    
    # Strategy: Start with complete graph, repair, then try to improve.
    # We do a few restarts with different random initializations.
    
    for restart in range(10):
        if time.time() - start_time > time_limit:
            break
        
        # 1. Initialize Complete Graph with Random Colors
        adj = [[-1 for _ in range(n)] for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                c = random.randint(0, 19)
                adj[u][v] = c
                adj[v][u] = c
        
        current_edges = n * (n - 1) // 2
        
        # 2. Repair Phase: Remove or Recolor edges to satisfy constraints
        # We iterate until no violations or time runs out.
        # To be efficient, we pick a random edge and check if it's part of a violation.
        # Better: Maintain a list of violations? Too expensive to update.
        # Simpler: Randomly pick an edge, check all C4s involving it. If violation, fix it.
        
        max_iter = 5000
        for _ in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            
            # Pick a random edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v or adj[u][v] == -1:
                continue
            
            # Check if this edge is involved in any non-rainbow C4
            violated = False
            for w, x in get_c4s_involving(u, v):
                if not is_rainbow(adj[u][v], adj[v][w], adj[w][x], adj[x][u]):
                    violated = True
                    break
            
            if not violated:
                continue
            
            # Try to recolor (u, v)
            fixed = False
            # Try all colors
            for c in range(20):
                if c == adj[u][v]:
                    continue
                if check_change(u, v, c):
                    adj[u][v] = c
                    adj[v][u] = c
                    fixed = True
                    break
            
            if not fixed:
                # Try to recolor one of the other edges in the violating cycle?
                # We identified a specific violating (w,x) pair above, but we broke out of the loop.
                # Let's find one violating cycle again to target its other edges.
                target_w, target_x = -1, -1
                for w, x in get_c4s_involving(u, v):
                    if not is_rainbow(adj[u][v], adj[v][w], adj[w][x], adj[x][u]):
                        target_w, target_x = w, x
                        break
                
                if target_w != -1:
                    # Edges in cycle: (u,v), (v,target_w), (target_w,target_x), (target_x,u)
                    edges_to_try = [(v, target_w), (target_w, target_x), (target_x, u)]
                    random.shuffle(edges_to_try)
                    
                    for eu, ev in edges_to_try:
                        if adj[eu][ev] == -1:
                            continue
                        for c in range(20):
                            if c == adj[eu][ev]:
                                continue
                            if check_change(eu, ev, c):
                                adj[eu][ev] = c
                                adj[ev][eu] = c
                                fixed = True
                                break
                        if fixed:
                            break
                
                if not fixed:
                    # Remove an edge from the cycle
                    # Prefer removing (u,v) or a random edge from the cycle
                    # Let's remove (u,v) for simplicity, or random
                    if random.random() < 0.5:
                        adj[u][v] = -1
                        adj[v][u] = -1
                        current_edges -= 1
                    else:
                        # Remove random edge from the violating cycle
                        eu, ev = random.choice([(u,v), (v,target_w), (target_w,target_x), (target_x,u)])
                        adj[eu][ev] = -1
                        adj[ev][eu] = -1
                        current_edges -= 1

        # 3. Improvement Phase: Try to add back edges
        # We have a valid graph now. Try to add removed edges.
        # We can iterate over all missing edges and try to add them.
        # To avoid getting stuck, we can do a few passes.
        
        for _ in range(5):
            if time.time() - start_time > time_limit:
                break
            progress = False
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u][v] != -1:
                        continue
                    # Try to add (u,v)
                    for c in range(20):
                        if check_add(u, v, c):
                            adj[u][v] = c
                            adj[v][u] = c
                            current_edges += 1
                            progress = True
                            break
            if not progress:
                break
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = [row[:] for row in adj]

    coloured_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v] != -1:
                coloured_edges.append([u, v, best_adj[u][v]])
    
    return coloured_edges
# EVOLVE-BLOCK-END
