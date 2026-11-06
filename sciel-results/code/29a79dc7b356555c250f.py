import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    target_edges = 65
    
    # Helper to calculate score
    def calc_score(adj, deg):
        max_sum = 0
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                if ai & (1 << j):
                    common = ai & adj[j]
                    # Iterate through common neighbors
                    c = common
                    while c:
                        lsb = c & -c
                        k = lsb.bit_length() - 1
                        s = deg[i] + deg[j] + deg[k]
                        if s > max_sum:
                            max_sum = s
                        c ^= lsb
        return max_sum

    # Helper to get list of edges from adj
    def get_edges(adj):
        edges = []
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                if ai & (1 << j):
                    edges.append((i, j))
        return edges

    # Helper to get list of non-edges
    def get_non_edges(adj):
        non_edges = []
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                if not (ai & (1 << j)):
                    non_edges.append((i, j))
        return non_edges

    # Initialize with K_{6,10} + matching
    # Vertices 0-5 are A, 6-15 are B
    adj = [0] * n
    deg = [0] * n
    
    # Add all A-B edges
    for u in range(6):
        for v in range(6, 16):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
            
    # Add matching in B
    # (6,7), (8,9), (10,11), (12,13), (14,15)
    matches = [(6,7), (8,9), (10,11), (12,13), (14,15)]
    for u, v in matches:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        deg[u] += 1
        deg[v] += 1
        
    current_score = calc_score(adj, deg)
    best_score = current_score
    best_adj = adj[:]
    best_deg = deg[:]
    
    edges = get_edges(adj)
    non_edges = get_non_edges(adj)
    
    # Local Search
    # We will try random swaps
    # To speed up, we can pre-compute the effect of swaps? No, too complex.
    # Just loop.
    
    # Use a random seed for reproducibility? No, let it be random.
    # But we want to beat the baseline.
    
    iterations = 0
    no_improve_count = 0
    max_no_improve = 5000 # Restart or diversify if stuck
    
    while time.time() < deadline:
        # Pick a random existing edge to remove
        if not edges: break
        u, v = random.choice(edges)
        
        # Pick a random non-existing edge to add
        if not non_edges: break
        x, y = random.choice(non_edges)
        
        # Ensure we are not swapping the same edge (impossible since one is in, one is out)
        # But ensure (u,v) != (x,y) and (u,v) != (y,x) etc. 
        # Since edges are stored as (min, max) and non_edges as (min, max), 
        # and u<v, x<y, we just check if (u,v) == (x,y). 
        # Since one is in `edges` and one in `non_edges`, they are disjoint sets.
        
        # Perform Swap
        # Remove (u,v)
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        deg[u] -= 1
        deg[v] -= 1
        
        # Add (x,y)
        adj[x] ^= (1 << y)
        adj[y] ^= (1 << x)
        deg[x] += 1
        deg[y] += 1
        
        new_score = calc_score(adj, deg)
        
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_adj = adj[:]
                best_deg = deg[:]
            no_improve_count = 0
            # Update edge lists
            edges.remove((u, v))
            non_edges.append((u, v))
            
            if (x, y) in non_edges:
                non_edges.remove((x, y))
            else:
                # This shouldn't happen if logic is correct
                pass 
            edges.append((x, y))
            
        elif new_score == current_score:
            # Accept with some probability to explore
            if random.random() < 0.3:
                no_improve_count += 1
                # Update edge lists
                edges.remove((u, v))
                non_edges.append((u, v))
                
                if (x, y) in non_edges:
                    non_edges.remove((x, y))
                edges.append((x, y))
            else:
                # Revert
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                deg[u] += 1
                deg[v] += 1
                
                adj[x] ^= (1 << y)
                adj[y] ^= (1 << x)
                deg[x] -= 1
                deg[y] -= 1
                no_improve_count += 1
        else:
            # Revert
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            deg[u] += 1
            deg[v] += 1
            
            adj[x] ^= (1 << y)
            adj[y] ^= (1 << x)
            deg[x] -= 1
            deg[y] -= 1
            no_improve_count += 1
            
        iterations += 1
        
        # If stuck for a long time, try a more aggressive restart or diversification
        # For now, just keep going. The random walk on equal scores helps.
        
        # Optional: If no improvement for a very long time, we could try to target specific triangles.
        # But simple random search with N=16 is often effective.
        
    # Return the best found
    # We need to return the edges list from best_adj
    final_edges = get_edges(best_adj)
    
    return {'n': 16, 'edges': final_edges}

# EVOLVE-BLOCK-END
