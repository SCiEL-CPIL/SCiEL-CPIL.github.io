import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = 110.0
    
    n = 16
    num_edges = n * (n - 1) // 2
    
    edge_index = {}
    index_edge = []
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_index[(u, v)] = idx
            index_edge.append((u, v))
            idx += 1
            
    q3_edges_local = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    
    patterns = []
    seen_patterns = set()
    for perm in itertools.permutations(range(8)):
        edges = set()
        for u, v in q3_edges_local:
            gu, gv = perm[u], perm[v]
            if gu > gv: gu, gv = gv, gu
            edges.add((gu, gv))
        edges_frozen = frozenset(edges)
        if edges_frozen not in seen_patterns:
            seen_patterns.add(edges_frozen)
            patterns.append(edges_frozen)
            
    all_q3s = []
    for vertices in itertools.combinations(range(n), 8):
        local_to_global = {i: vertices[i] for i in range(8)}
        for pattern in patterns:
            global_edges = set()
            for u, v in pattern:
                gu = local_to_global[u]
                gv = local_to_global[v]
                if gu > gv: gu, gv = gv, gu
                global_edges.add(edge_index[(gu, gv)])
            all_q3s.append(frozenset(global_edges))
            
    num_q3s = len(all_q3s)
    
    # Precompute edge_to_q3s
    edge_to_q3s = [[] for _ in range(num_edges)]
    for i, q3 in enumerate(all_q3s):
        for e in q3:
            edge_to_q3s[e].append(i)
            
    best_removed = None
    best_size = num_edges + 1
    
    greedy_deadline = start_time + 40.0
    
    while time.time() < greedy_deadline:
        uncovered = set(range(num_q3s))
        edge_counts = [0] * num_edges
        
        for i in range(num_q3s):
            for e in all_q3s[i]:
                edge_counts[e] += 1
                
        removed = set()
        
        while uncovered:
            max_count = max(edge_counts)
            candidates = [e for e in range(num_edges) if edge_counts[e] == max_count and e not in removed]
            if not candidates:
                break
            e_star = random.choice(candidates)
            
            removed.add(e_star)
            
            new_uncovered = set()
            for i in uncovered:
                q3 = all_q3s[i]
                if e_star in q3:
                    for e in q3:
                        edge_counts[e] -= 1
                else:
                    new_uncovered.add(i)
            uncovered = new_uncovered
            
        size = len(removed)
        if size < best_size:
            best_size = size
            best_removed = set(removed)
            
    if best_removed is not None:
        current_removed = set(best_removed)
        
        # Calculate initial cover counts
        cover_count = [0] * num_q3s
        for e in current_removed:
            for i in edge_to_q3s[e]:
                cover_count[i] += 1
                
        # Remove redundant edges
        changed = True
        while changed and time.time() < deadline - 10.0:
            changed = False
            for e in list(current_removed):
                # Check if e is redundant
                redundant = True
                for i in edge_to_q3s[e]:
                    if cover_count[i] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove e
                    current_removed.remove(e)
                    for i in edge_to_q3s[e]:
                        cover_count[i] -= 1
                    changed = True
                    
        if len(current_removed) < best_size:
            best_size = len(current_removed)
            best_removed = current_removed
            
    result_edges = []
    for i in range(num_edges):
        if i not in best_removed:
            u, v = index_edge[i]
            result_edges.append([u, v])
            
    return result_edges
