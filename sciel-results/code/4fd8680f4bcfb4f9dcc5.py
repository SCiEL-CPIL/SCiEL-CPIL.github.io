import time
import random
import itertools

# EVOLVE-BLOCK-START

def construct():
    n = 15
    # Vertices 0..14
    # Split into two sets A and B for initial bipartite structure
    # Sizes 7 and 8
    A = list(range(7))
    B = list(range(7, 15))
    
    # Initialize edges with K_{7,8}
    edges = set()
    for u in A:
        for v in B:
            edges.add((u, v))
            
    def get_neighbors(edges):
        neighbors = {i: set() for i in range(n)}
        for u, v in edges:
            neighbors[u].add(v)
            neighbors[v].add(u)
        return neighbors

    def get_degrees(neighbors):
        return {i: len(neighbors[i]) for i in range(n)}

    def is_valid_addition(u, v, neighbors, degrees):
        # Check if adding (u,v) creates a violation
        # 1. Middle edge: s-u-v-t
        nu = neighbors[u]
        nv = neighbors[v]
        
        for s in nu:
            for t in nv:
                if s == t:
                    continue
                if degrees[s] == degrees[t]:
                    return False
        
        # 2. Left endpoint: u-v-w-z
        deg_u_new = degrees[u] + 1
        for w in nv:
            for z in neighbors[w]:
                if z == v or z == u:
                    continue
                if degrees[z] == deg_u_new:
                    return False

        # 3. Right endpoint: z-w-v-u
        deg_v_new = degrees[v] + 1
        for w in nu:
            for z in neighbors[w]:
                if z == u or z == v:
                    continue
                if degrees[z] == deg_v_new:
                    return False

        return True

    best_edges = set(edges)
    best_count = len(best_edges)
    
    start_time = time.time()
    deadline = start_time + 85.0 # 5s margin
    
    while time.time() < deadline:
        # Greedy phase
        neighbors = get_neighbors(edges)
        degrees = get_degrees(neighbors)
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Try to add any valid edge
            all_pairs = list(itertools.combinations(range(n), 2))
            random.shuffle(all_pairs)
            
            for u, v in all_pairs:
                if (u, v) in edges or (v, u) in edges:
                    continue
                if is_valid_addition(u, v, neighbors, degrees):
                    # Add edge
                    edges.add((u, v))
                    # Update neighbors and degrees
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    degrees[u] += 1
                    degrees[v] += 1
                    changed = True
                    if len(edges) > best_count:
                        best_edges = set(edges)
                        best_count = len(edges)
                    break # Restart loop with updated state
            
        # If we are stuck, try a kick
        if not changed and len(edges) > 0:
            # Remove a random edge
            edge_to_remove = random.choice(list(edges))
            edges.remove(edge_to_remove)
            # Rebuild neighbors and degrees
            neighbors = get_neighbors(edges)
            degrees = get_degrees(neighbors)

    # Return best found
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
        
    return result

# EVOLVE-BLOCK-END
