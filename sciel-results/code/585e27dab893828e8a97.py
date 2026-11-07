"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit, 10s buffer
    
    n = 15
    
    def get_degrees(edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return deg
    
    def is_valid(edges):
        deg = get_degrees(edges)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check all paths s-u-v-t where s,u,v,t are distinct
        for u in range(n):
            for v in adj[u]:
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True
    
    # Start with a star graph (14 edges)
    best_edges = [[0, v] for v in range(1, n)]
    best_count = len(best_edges)
    
    # Try adding edges greedily
    all_possible = [(u, v) for u in range(n) for v in range(u+1, n)]
    current = set(tuple(e) for e in best_edges)
    
    # Sort by some heuristic - try edges that connect high-degree vertices
    def edge_heuristic(e):
        u, v = e
        return 0  # neutral for now
    
    random.seed(42)
    random.shuffle(all_possible)
    
    for u, v in all_possible:
        if (u, v) in current or (v, u) in current:
            continue
        if time.time() > deadline:
            break
        
        new_edges = list(current) + [(u, v)]
        # Convert to list of lists for is_valid
        new_edges_list = [[a, b] for a, b in new_edges]
        
        if is_valid(new_edges_list):
            current.add((u, v))
            best_edges = [[a, b] for a, b in current]
            best_count = len(best_edges)
    
    # Try to improve with local search
    for _ in range(100):
        if time.time() > deadline:
            break
        
        # Try swapping: remove one edge, add another
        if not best_edges:
            break
        
        removed = random.choice(best_edges)
        current_removed = set(tuple(e) for e in best_edges) - {tuple(removed), tuple(reversed(removed))}
        
        # Try adding a random edge
        added = None
        for _ in range(20):
            if time.time() > deadline:
                break
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if (u, v) in current_removed or (v, u) in current_removed:
                continue
            new_edges = list(current_removed) + [(u, v)]
            new_edges_list = [[a, b] for a, b in new_edges]
            if is_valid(new_edges_list):
                added = (u, v)
                break
        
        if added is not None:
            new_edges = list(current_removed) + [added]
            new_edges_list = [[a, b] for a, b in new_edges]
            if len(new_edges_list) > best_count and is_valid(new_edges_list):
                best_edges = new_edges_list
                best_count = len(best_edges)
    
    # Ensure proper types
    result = []
    for e in best_edges:
        result.append([int(e[0]), int(e[1])])
    
    return result
# EVOLVE-BLOCK-END
