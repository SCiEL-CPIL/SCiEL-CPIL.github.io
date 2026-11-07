import time
import random
from collections import deque
import sys

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set(tuple(sorted(e)) for e in P['edges'])
    
    # Generate all valid bipartite edges (triangle-free guarantee)
    # u < v, u not in base_edges, (u%2 != v%2)
    valid_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_edges:
                continue
            if (u % 2) == (v % 2):
                continue
            valid_edges.append((u, v))
            
    valid_edges_set = set(valid_edges)

    def get_diameter(edge_list):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in P['edges']:
            adj[u].append(v)
            adj[v].append(u)
        for u, v in edge_list:
            adj[u].append(v)
            adj[v].append(u)
            
        max_d = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                node = q.popleft()
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        q.append(nb)
                        if dist[nb] > max_d:
                            max_d = dist[nb]
                            if max_d > 4:
                                return max_d # Early exit
        return max_d

    def is_valid(edge_list):
        # Check for duplicates (shouldn't happen if we use sets, but good for safety)
        if len(edge_list) != len(set(tuple(sorted(e)) for e in edge_list)):
            return False
        # Check no base edges
        for e in edge_list:
            if tuple(sorted(e)) in base_edges:
                return False
        # Check diameter
        if get_diameter(edge_list) > 4:
            return False
        return True

    # Strategy: Local Search / Pruning
    # We want to find the minimum number of edges.
    # We try k from 6 upwards.
    
    deadline = time.time() + 80.0
    best_solution = None
    best_count = 100
    
    # Helper to run a pruning search starting from a specific set
    def prune_search(initial_set):
        current = list(initial_set)
        # Sort for determinism in pruning order? Or random?
        # Random order helps avoid local minima.
        random.shuffle(current)
        
        improved = True
        while improved:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline - 5.0:
                    return current
                # Try removing current[i]
                trial = current[:i] + current[i+1:]
                if is_valid(trial):
                    current = trial
                    improved = True
                    break # Restart scan after modification
        
        return current

    # We will try to find a solution with k edges.
    # We generate random starting sets of size K_start (e.g., 12) and prune.
    # If the result has size <= k, we check if we can find exactly k or lower.
    
    # To be efficient, we just run many random restarts of the pruning process
    # and keep the best (smallest) valid set found.
    
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Create a random starting set
        # To ensure we don't start with an invalid (diameter > 4) set that is hard to fix,
        # we should start with a "dense" set that is likely valid.
        # A set of all edges with distance <= 5 is usually valid for N=24, k=4.
        # Let's just pick a random subset of valid_edges of size 15.
        # If it's invalid, pruning won't work directly (pruning assumes valid start).
        # So we need a valid start.
        
        # Heuristic for valid start:
        # Connect every 3rd vertex to a central hub?
        # Or just take a random set and check. If invalid, skip?
        # Better: Start with a known good structure and perturb.
        
        # Known good structure:
        # Hubs at 6, 12, 18?
        # Edges: (1,6), (7,12), (13,18), (19,23)?
        # Let's just use a random dense set and verify.
        
        # Random dense set:
        # Pick 20 random edges.
        start_set = random.sample(valid_edges, min(20, len(valid_edges)))
        
        # Check if start_set is valid.
        # If not, we can't prune it.
        # We can try to "grow" it? No, that's complex.
        # Instead, we can try to repair it?
        
        # Alternative: Start with a "Greedy Valid" set.
        # Add edges one by one to reduce diameter until <= 4.
        
        def greedy_construct():
            current = []
            # We want to minimize diameter.
            # Sort valid edges by "potential impact"?
            # Just random order.
            edges_copy = valid_edges[:]
            random.shuffle(edges_copy)
            
            # We don't know when to stop, but we can stop when diameter <= 4.
            # However, greedy addition might add too many.
            # We add an edge only if it reduces the current diameter.
            
            current_diam = get_diameter([]) # 23
            for e in edges_copy:
                if time.time() > deadline - 5.0:
                    break
                trial = current + [e]
                new_diam = get_diameter(trial)
                if new_diam < current_diam:
                    current = trial
                    current_diam = new_diam
                    if current_diam <= 4:
                        break
            return current

        # Run greedy, then prune.
        g_set = greedy_construct()
        
        if is_valid(g_set):
            pruned = prune_search(g_set)
            if is_valid(pruned):
                if len(pruned) < best_count:
                    best_count = len(pruned)
                    best_solution = pruned[:]
        else:
            # Greedy didn't reach diameter 4? Unlikely with 24 nodes and many edges.
            # If it didn't, skip.
            pass

    # If we found a solution, return it.
    if best_solution is not None:
        # Ensure types are int
        return [[int(u), int(v)] for u, v in best_solution]
    
    # Fallback: A known valid solution (e.g. 10 edges)
    # We can construct one quickly if search fails.
    # But search should not fail for N=24.
    # Let's provide a hardcoded valid solution just in case.
    # Edges: (1, 10), (11, 20), (2, 12), (14, 23), (3, 13), (15, 22), (4, 14), (16, 21), (5, 15), (17, 20) -- wait, duplicates/parity?
    # (1,10) O-E
    # (11,20) O-E
    # (2,12) E-E Invalid.
    
    # Let's just return the best found. If none, return empty (invalid) or a safe guess.
    # But the task says "Minimize". If we don't find one, we should return the best valid one.
    # If best_solution is None, it means we didn't find any valid set in the time.
    # This is highly unlikely.
    
    return []
# EVOLVE-BLOCK-END
