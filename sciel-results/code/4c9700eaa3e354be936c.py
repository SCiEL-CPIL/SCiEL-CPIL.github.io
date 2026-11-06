import random
import time

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # List of all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    best_graph = None
    best_edge_count = 0
    
    # We will try multiple restarts
    # Each restart:
    # 1. Shuffle edges
    # 2. Greedily add edges with valid colors
    # 3. Track best edge count
    
    max_restarts = 2000 # Adjust based on speed
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        random.shuffle(all_edges)
        
        # Graph representation
        # adjacency list: adj[u] = {v: color}
        adj = {u: {} for u in range(n)}
        edge_count = 0
        
        for u, v in all_edges:
            # Check if we can add edge (u, v)
            # Find all 4-cycles that would be formed by adding (u, v)
            # A 4-cycle is u-v-w-x-u
            # Requires w in adj[v], x in adj[u], and edge (w,x) in graph
            
            forbidden_colors = set()
            impossible = False
            
            # Iterate over neighbors of v
            for w, c_vw in adj[v].items():
                # Iterate over neighbors of u
                for x, c_xu in adj[u].items():
                    # Check if w and x are connected
                    if w == x:
                        continue # Triangle, not 4-cycle
                    if x in adj[w]:
                        c_wx = adj[w][x]
                        # The cycle is u-v-w-x-u
                        # Colors: c(uv), c(vw), c(wx), c(xu)
                        # c(uv) is the one we are choosing.
                        # Existing colors: c_vw, c_wx, c_xu
                        
                        # If existing colors are not distinct, this cycle can never be rainbow
                        if c_vw == c_wx or c_wx == c_xu or c_xu == c_vw:
                            impossible = True
                            break
                        
                        # If distinct, c(uv) must be different from all three
                        forbidden_colors.add(c_vw)
                        forbidden_colors.add(c_wx)
                        forbidden_colors.add(c_xu)
                        
                if impossible:
                    break
            
            if impossible:
                continue
            
            # If forbidden_colors has all 20 colors, we can't add this edge
            if len(forbidden_colors) >= num_colors:
                continue
            
            # Pick a random color from allowed
            allowed = [c for c in range(num_colors) if c not in forbidden_colors]
            chosen_color = random.choice(allowed)
            
            # Add edge
            adj[u][v] = chosen_color
            adj[v][u] = chosen_color
            edge_count += 1
            
        if edge_count > best_edge_count:
            best_edge_count = edge_count
            # Store the graph
            # We need to return [[u, v, c], ...]
            best_graph = []
            for u in range(n):
                for v, c in adj[u].items():
                    if u < v:
                        best_graph.append([u, v, c])
                        
    # If no graph was found (shouldn't happen, at least empty graph is valid), return empty
    if best_graph is None:
        return []
        
    return best_graph

# EVOLVE-BLOCK-START
# (The function above is the implementation)
# EVOLVE-BLOCK-END
