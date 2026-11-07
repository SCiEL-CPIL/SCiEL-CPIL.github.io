import time
import random
import sys

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = 88.0  # Leave 2s margin for 90s limit
    
    n = P['n']
    base_edges = P['edges']
    
    # Bipartition
    A = [i for i in range(n) if i % 2 == 0]
    B = [i for i in range(n) if i % 2 == 1]
    
    # Base adjacency
    base_adj = [[False] * n for _ in range(n)]
    for u, v in base_edges:
        base_adj[u][v] = True
        base_adj[v][u] = True
        
    # Candidate edges: (u, v) where u in A, v in B, and not in base
    candidates = []
    cand_set = set()
    for u in A:
        for v in B:
            if not base_adj[u][v]:
                candidates.append((u, v))
                cand_set.add((u, v))
                
    def get_adj(added_edges):
        adj = [row[:] for row in base_adj]
        for u, v in added_edges:
            adj[u][v] = True
            adj[v][u] = True
        return adj

    def compute_diameter(adj):
        max_d = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = [src]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in range(n):
                    if adj[u][v] and dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            for d in dist:
                if d > max_d:
                    max_d = d
        return max_d

    def is_valid(added_edges):
        # Triangle-free is guaranteed by bipartition
        adj = get_adj(added_edges)
        diam = compute_diameter(adj)
        return diam <= 4

    # Initial solution: Hardcoded or Greedy
    # Let's try a greedy approach to get a good start
    # Start with empty set, add edges that reduce diameter most
    current_edges = []
    
    # Heuristic: Connect central nodes to peripheral
    # Or just use the hardcoded one if it's valid
    hardcoded = [[12, 1], [12, 3], [12, 6], [12, 9], [12, 15], [12, 18], [12, 21], [12, 23]]
    if is_valid(hardcoded):
        current_edges = hardcoded
    else:
        # Fallback: Greedy
        # While diameter > 4, add best edge
        for _ in range(20):
            adj = get_adj(current_edges)
            diam = compute_diameter(adj)
            if diam <= 4:
                break
            best_edge = None
            best_diam = diam
            for e in candidates:
                if e in current_edges: continue
                new_edges = current_edges + [e]
                new_adj = get_adj(new_edges)
                new_diam = compute_diameter(new_adj)
                if new_diam < best_diam:
                    best_diam = new_diam
                    best_edge = e
            if best_edge is None:
                break
            current_edges.append(best_edge)
            
    if not is_valid(current_edges):
        # If still invalid, try adding random edges until valid
        while not is_valid(current_edges) and time.time() < deadline - 5:
            e = random.choice(candidates)
            if e not in current_edges:
                current_edges.append(e)
                
    if not is_valid(current_edges):
        return [] # Should not happen

    best_solution = list(current_edges)
    
    # Local Search: Remove edges
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(len(current_edges)):
            if time.time() > deadline - 10: break
            trial = current_edges[:i] + current_edges[i+1:]
            if is_valid(trial):
                current_edges = trial
                improved = True
                break
                
    if len(current_edges) < len(best_solution):
        best_solution = list(current_edges)
        
    # Simulated Annealing
    # Cost function
    def cost(added_edges):
        if not added_edges:
             return 0
        adj = get_adj(added_edges)
        diam = compute_diameter(adj)
        penalty = 0
        if diam > 4:
            penalty = 100 * (diam - 4)
        return len(added_edges) + penalty

    current_cost = cost(current_edges)
    best_cost = cost(best_solution)
    T = 10.0
    T_min = 0.01
    cooling = 0.99
    
    while time.time() < deadline - 2:
        if T < T_min:
            break
            
        # Generate move
        move_type = random.random()
        new_edges = list(current_edges)
        
        if move_type < 0.3:
            # Remove
            if new_edges:
                idx = random.randint(0, len(new_edges) - 1)
                new_edges.pop(idx)
        elif move_type < 0.6:
            # Add
            e = random.choice(candidates)
            if e not in new_edges:
                new_edges.append(e)
        else:
            # Swap
            if new_edges:
                idx = random.randint(0, len(new_edges) - 1)
                old_e = new_edges.pop(idx)
                new_e = random.choice(candidates)
                if new_e not in new_edges:
                    new_edges.append(new_e)
                    
        new_cost = cost(new_edges)
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            current_edges = new_edges
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = list(current_edges)
                
        T *= cooling
        
    # Final local search on best_solution
    if time.time() < deadline - 1:
        trial = list(best_solution)
        improved = True
        while improved:
            improved = False
            for i in range(len(trial)):
                t = trial[:i] + trial[i+1:]
                if is_valid(t):
                    trial = t
                    improved = True
                    break
        if len(trial) < len(best_solution):
            best_solution = trial

    return [[int(u), int(v)] for u, v in best_solution]
