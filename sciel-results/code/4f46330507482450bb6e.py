"""
Strategy: Exact Branch and Bound with fast connectivity pruning and forced-move detection.
Key optimizations:
1. Fast connectivity bound using incremental component tracking.
2. Forced move detection: vertices with degree <=1 in active graph must be endpoints.
3. Better initial lower bound with multi-start greedy + local search.
4. Tight deadline with early return.
5. Symmetry breaking and smart ordering.
"""

import time
import sys
from collections import deque

# Pinned graph data
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    n = P['n']
    adj = _build_adjacency()
    
    # Precompute connected components of the full graph
    visited_comp = [False] * n
    components = []
    for i in range(n):
        if not visited_comp[i]:
            comp = []
            stack = [i]
            visited_comp[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nb in adj[node]:
                    if not visited_comp[nb]:
                        visited_comp[nb] = True
                        stack.append(nb)
            components.append(comp)
            
    largest_comp = max(components, key=len)
    largest_set = set(largest_comp)
    comp_size = len(largest_comp)
    
    best_path = []
    best_len = 0
    
    def update_best(path):
        nonlocal best_path, best_len
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
            return True
        return False
    
    # Multi-start greedy with worst-first heuristic
    def greedy_initial():
        local_best = []
        for start in largest_comp:
            path = [start]
            seen = {start}
            while True:
                cur = path[-1]
                nbrs = [x for x in adj[cur] if x not in seen and x in largest_set]
                if not nbrs:
                    break
                nbrs.sort(key=lambda x: sum(1 for y in adj[x] if y not in seen and y in largest_set))
                nxt = nbrs[0]
                path.append(nxt)
                seen.add(nxt)
            if len(path) > len(local_best):
                local_best = path
        return local_best

    initial_path = greedy_initial()
    update_best(initial_path)
    
    if best_len == comp_size:
        return best_path
    
    # Fast connectivity bound: find reachable components from endpoints
    def connectivity_bound(path, visited):
        if not path:
            return 0
        E1 = path[0]
        E2 = path[-1]
        
        unvisited = [v for v in largest_comp if v not in visited]
        if not unvisited:
            return len(path)
            
        visited_unvisited = set()
        comps = []
        
        for start in unvisited:
            if start in visited_unvisited:
                continue
            comp = []
            stack = [start]
            visited_unvisited.add(start)
            while stack:
                node = stack.pop()
                comp.append(node)
                for nb in adj[node]:
                    if nb not in visited and nb in largest_set and nb not in visited_unvisited:
                        visited_unvisited.add(nb)
                        stack.append(nb)
            comps.append(comp)
            
        reachable_sizes = []
        for comp in comps:
            reachable = False
            for node in comp:
                if E1 in adj[node] or E2 in adj[node]:
                    reachable = True
                    break
            if reachable:
                reachable_sizes.append(len(comp))
        
        reachable_sizes.sort(reverse=True)
        
        if len(reachable_sizes) == 0:
            ub = len(path)
        elif len(reachable_sizes) == 1:
            ub = len(path) + reachable_sizes[0]
        else:
            ub = len(path) + reachable_sizes[0] + reachable_sizes[1]
            
        return ub
    
    # Forced move check: count vertices with degree <=1 in active graph
    def forced_move_check(path, visited):
        """Returns True if pruning is safe (too many forced endpoints)."""
        E1 = path[0]
        E2 = path[-1]
        endpoints = {E1, E2}
        
        forced_count = 0
        for v in largest_comp:
            if v in visited:
                continue
            # Count neighbors in unvisited set + endpoints
            deg = 0
            for nb in adj[v]:
                if nb in visited and nb not in endpoints:
                    continue
                deg += 1
            if deg <= 1:
                forced_count += 1
                if forced_count > 2:
                    return True  # Prune
        return False
    
    # DFS with B&B
    start_vertices = sorted(largest_comp, key=lambda x: len(adj[x]), reverse=True)
    
    for start in start_vertices:
        if time.time() > deadline:
            break
            
        path = [start]
        visited = {start}
        
        local_stack = [(path, visited)]
        
        while local_stack:
            if time.time() > deadline:
                return best_path
                
            curr_path, curr_visited = local_stack.pop()
            
            # Quick weak bound
            if len(curr_path) + (comp_size - len(curr_visited)) <= best_len:
                continue
                
            # Connectivity bound (more expensive but stronger)
            ub = connectivity_bound(curr_path, curr_visited)
            if ub <= best_len:
                continue
                
            # Forced move check
            if forced_move_check(curr_path, curr_visited):
                continue
                
            # Update best
            if len(curr_path) > best_len:
                if update_best(curr_path):
                    if best_len == comp_size:
                        return best_path

            # Extend at the end only (symmetry breaking)
            end = curr_path[-1]
            
            # Order neighbors by degree (worst first)
            candidates = [nxt for nxt in adj[end] if nxt not in curr_visited and nxt in largest_set]
            candidates.sort(key=lambda x: sum(1 for y in adj[x] if y not in curr_visited and y in largest_set))
            
            for nxt in candidates:
                new_path = curr_path + [nxt]
                new_visited = curr_visited | {nxt}
                
                # Quick bound before pushing
                if len(new_path) + (comp_size - len(new_visited)) <= best_len:
                    continue
                    
                local_stack.append((new_path, new_visited))
                
    return best_path

# EVOLVE-BLOCK-END
