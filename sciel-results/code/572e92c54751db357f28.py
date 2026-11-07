import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    n = 15
    # Start with K_{7,8}
    # Part A: vertices 0-6 (size 7), Part B: vertices 7-14 (size 8)
    part_a = list(range(7))
    part_b = list(range(7, 15))
    
    edges = set()
    # Add all bipartite edges
    for u in part_a:
        for v in part_b:
            edges.add((u, v))
    
    def get_degree(edges_set):
        deg = [0] * n
        for u, v in edges_set:
            deg[u] += 1
            deg[v] += 1
        return deg
    
    def is_valid(edges_set):
        deg = get_degree(edges_set)
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check every 3-edge path s-u-v-t with s != t
        for u in range(n):
            for v in adj[u]:
                for t in adj[v]:
                    if t == u:
                        continue
                    # s is any neighbor of u except v
                    for s in adj[u]:
                        if s == v or s == t:
                            continue
                        # Path s-u-v-t
                        if deg[s] == deg[t]:
                            return False
        return True
    
    def violation_count(edges_set):
        deg = get_degree(edges_set)
        adj = [set() for _ in range(n)]
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        
        count = 0
        for u in range(n):
            for v in adj[u]:
                for t in adj[v]:
                    if t == u:
                        continue
                    for s in adj[u]:
                        if s == v or s == t:
                            continue
                        if deg[s] == deg[t]:
                            count += 1
        return count
    
    # Greedy: try adding all possible remaining edges
    all_possible = set()
    for i in range(n):
        for j in range(i+1, n):
            all_possible.add((i, j))
    
    remaining = all_possible - edges
    
    # Sort remaining edges by some heuristic - try intra-part edges first
    def edge_priority(e):
        u, v = e
        in_a = u < 7 and v < 7
        in_b = u >= 7 and v >= 7
        if in_a or in_b:
            return 0  # intra-part, likely safer
        return 1
    
    remaining_list = sorted(remaining, key=edge_priority)
    
    current_edges = set(edges)
    
    # Try adding edges greedily
    for e in remaining_list:
        if time.time() > deadline:
            break
        test_edges = current_edges | {e}
        if is_valid(test_edges):
            current_edges = test_edges
    
    # Local search: try to add more edges via random restarts
    best_edges = current_edges
    best_count = len(current_edges)
    
    attempts = 0
    while time.time() < deadline - 5.0:
        attempts += 1
        # Random perturbation: remove a random edge, then try to re-add
        if current_edges:
            edge_to_remove = random.choice(list(current_edges))
            perturbed = current_edges - {edge_to_remove}
            
            # Try to rebuild greedily
            for e in remaining_list:
                if e in perturbed:
                    continue
                test_edges = perturbed | {e}
                if is_valid(test_edges):
                    perturbed = test_edges
            
            if len(perturbed) > best_count:
                best_edges = perturbed
                best_count = len(perturbed)
                current_edges = perturbed
    
    # Convert to list of lists
    result = []
    for u, v in best_edges:
        result.append([u, v])
    
    return result
