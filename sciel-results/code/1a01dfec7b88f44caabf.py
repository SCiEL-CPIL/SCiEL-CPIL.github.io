# EVOLVE-BLOCK-START
import random
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    
    # Bipartite sets
    A = [i for i in range(n) if i % 2 == 0]
    B = [i for i in range(n) if i % 2 == 1]
    
    # Candidate edges: all pairs (a, b) not in base_edges
    base_set = set((min(u, v), max(u, v)) for u, v in base_edges)
    candidates = []
    for a in A:
        for b in B:
            edge = (min(a, b), max(a, b))
            if edge not in base_set:
                candidates.append(edge)
                
    def get_diameter(added_edges):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in base_edges:
            adj[u].append(v)
            adj[v].append(u)
        for u, v in added_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # BFS from each node to find max distance
        max_d = 0
        for i in range(n):
            dist = [-1] * n
            dist[i] = 0
            q = [i]
            for node in q:
                for nei in adj[node]:
                    if dist[nei] == -1:
                        dist[nei] = dist[node] + 1
                        q.append(nei)
                        if dist[nei] > max_d:
                            max_d = dist[nei]
        return max_d

    def is_valid(added_edges):
        return get_diameter(added_edges) <= 4

    # Phase 1: Greedy Initialization
    # Add edges that most reduce diameter
    current_edges = set()
    for _ in range(20): # Limit iterations
        d = get_diameter(list(current_edges))
        if d <= 4:
            break
        best_edge = None
        best_reduction = 0
        # Precompute distances
        dists = [[0]*n for _ in range(n)]
        # Actually, just check improvement in diameter or max dist
        # For simplicity, pick edge that connects farthest nodes?
        # Let's use the same scoring as before but faster
        # Find max dist pair
        max_d = 0
        max_pair = (0, 0)
        for i in range(n):
            for j in range(i+1, n):
                # This is O(N^2) BFS if done naively. 
                # We just computed diameter. Let's just pick a random candidate that reduces diameter?
                # Or pick edge (0, 23) first?
                pass
        
        # Simple Greedy: Pick edge that minimizes new diameter
        # To avoid O(N^2) BFS for every candidate, we just try a few random ones or the "longest" potential
        # For initialization, let's just add edges connecting ends
        # Edge (0, 23) is valid (0 in A, 23 in B)
        if (0, 23) not in current_edges and (0, 23) in base_set or (0, 23) not in base_set:
             # (0,23) is not in base path
             if (0, 23) not in current_edges:
                 current_edges.add((0, 23))
        
        # Add more if needed
        # Let's use a simple loop
        for _ in range(50):
            d = get_diameter(list(current_edges))
            if d <= 4:
                break
            # Pick random candidate
            e = random.choice(candidates)
            if e not in current_edges:
                current_edges.add(e)
                
    # Ensure valid start for SA
    if not is_valid(list(current_edges)):
        # Fallback: Add more random edges until valid
        while not is_valid(list(current_edges)):
            e = random.choice(candidates)
            current_edges.add(e)
            if len(current_edges) > 20: # Safety
                break

    # Phase 2: Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.95
    
    best_edges = set(current_edges)
    best_cost = len(best_edges)
    
    current_diam = get_diameter(list(current_edges))
    current_cost = len(current_edges) + 100 * max(0, current_diam - 4)
    
    start_time = time.time()
    steps = 0
    max_steps = 100000
    
    while steps < max_steps and time.time() - start_time < 45:
        new_edges = set(current_edges)
        move_type = random.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add':
            if candidates:
                e = random.choice(candidates)
                if e not in new_edges:
                    new_edges.add(e)
                else:
                    continue
        elif move_type == 'remove':
            if current_edges:
                e = random.choice(list(current_edges))
                new_edges.remove(e)
            else:
                continue
        else: # swap
            if current_edges and candidates:
                e_remove = random.choice(list(current_edges))
                e_add = random.choice(candidates)
                if e_add not in new_edges:
                    new_edges.remove(e_remove)
                    new_edges.add(e_add)
                else:
                    continue
        
        new_diam = get_diameter(list(new_edges))
        new_cost = len(new_edges) + 100 * max(0, new_diam - 4)
        
        delta = new_cost - current_cost
        
        if delta < 0 or (T > T_min and random.random() < pow(2.718, -delta / T)):
            current_edges = new_edges
            current_diam = new_diam
            current_cost = new_cost
            
            # Update best if valid and smaller
            if current_diam <= 4 and len(current_edges) < len(best_edges):
                best_edges = set(current_edges)
                best_cost = len(best_edges)
                
        T *= cooling
        steps += 1
        
    # Phase 3: Local Search Polish on best_edges
    changed = True
    while changed:
        changed = False
        for e in list(best_edges):
            test_edges = best_edges - {e}
            if is_valid(list(test_edges)):
                best_edges = test_edges
                changed = True
                
    return [list(e) for e in best_edges]
# EVOLVE-BLOCK-END
