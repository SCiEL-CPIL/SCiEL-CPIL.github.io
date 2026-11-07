"""Packing edge-disjoint cycles on one vertex set."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

DEADLINE = 78  # safety margin before 90s kill

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    n = P['n']
    edges_list = P['edges']
    num_edges = len(edges_list)
    
    # Build edge index mapping
    edge_index = {}
    for i, (u, v) in enumerate(edges_list):
        if u > v:
            u, v = v, u
        edge_index[(u, v)] = i
    
    # Adjacency with edge indices, sorted for determinism
    adj = [set() for _ in range(n)]
    adj_edges = [[] for _ in range(n)]  # list of (neighbor, edge_idx)
    for idx, (u, v) in enumerate(edges_list):
        adj[u].add(v)
        adj[v].add(u)
        adj_edges[u].append((v, idx))
        adj_edges[v].append((u, idx))
    for v in range(n):
        adj_edges[v].sort()
    
    # Check degree constraints for Hamiltonian cycles
    min_degree = min(len(adj[v]) for v in range(n))
    max_possible_cycles = min(10, min_degree // 2)
    max_possible_cycles = min(max_possible_cycles, num_edges // n)
    
    def find_hamiltonian_cycles(k_target, deadline):
        """Find up to k_target edge-disjoint Hamiltonian cycles using DFS with pruning."""
        used_edges = [False] * num_edges
        cycles = []
        
        # Precompute adjacency as bitmasks for faster checks
        adj_mask = [0] * n
        for v in range(n):
            for nb, eidx in adj_edges[v]:
                adj_mask[v] |= (1 << nb)
        
        def check_connectivity(visited_mask, used_edges, start):
            """Check if the subgraph induced by unvisited vertices + start is connected."""
            unvisited = [v for v in range(n) if not (visited_mask & (1 << v))]
            if not unvisited:
                return True
            
            # Build subgraph on unvisited vertices + start
            nodes = unvisited + [start]
            node_set = set(nodes)
            
            # BFS/DFS from start
            visited = set([start])
            queue = deque([start])
            while queue:
                cur = queue.popleft()
                for nb, eidx in adj_edges[cur]:
                    if eidx and used_edges[eidx]:
                        continue
                    if nb in node_set and nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            
            return len(visited) == len(nodes)
        
        def count_usable_edges(v, visited_mask, used_edges, start):
            """Count unused edges from v to unvisited vertices or start."""
            count = 0
            for nb, eidx in adj_edges[v]:
                if used_edges[eidx]:
                    continue
                if nb == start or not (visited_mask & (1 << nb)):
                    count += 1
            return count
        
        def dfs_cycle(start, path, visited_mask, depth):
            """Find a Hamiltonian cycle starting at 'start' avoiding used edges."""
            if time.time() > deadline:
                return None
            
            if depth == n:
                last = path[-1]
                key = (min(last, start), max(last, start))
                if key in edge_index:
                    eidx = edge_index[key]
                    if not used_edges[eidx]:
                        return list(path)
                return None
            
            last = path[-1]
            remaining = n - depth - 1  # vertices left to visit after current
            
            # Get candidate neighbors, sorted by number of usable edges (most constrained first)
            candidates = []
            for neighbor, eidx in adj_edges[last]:
                if visited_mask & (1 << neighbor):
                    continue
                if used_edges[eidx]:
                    continue
                new_mask = visited_mask | (1 << neighbor)
                
                # Pruning 1: Check each unvisited vertex has at least one usable edge
                viable = True
                unvisited = [w for w in range(n) if not (new_mask & (1 << w))]
                for w in unvisited:
                    if count_usable_edges(w, new_mask, used_edges, start) < 1:
                        viable = False
                        break
                
                if not viable:
                    continue
                
                # Pruning 2: Check start vertex has enough usable edges to close the cycle
                # Start needs at least 1 edge to the last vertex in the path (which will be unvisited now)
                # This is handled at the end when depth == n
                
                # Pruning 3: Connectivity check (expensive, only do for larger remaining sets)
                if remaining >= 4:
                    if not check_connectivity(new_mask, used_edges, start):
                        continue
                
                # Score by number of usable edges from neighbor (most constrained first)
                score = count_usable_edges(neighbor, new_mask, used_edges, start)
                candidates.append((score, neighbor, eidx, new_mask))
            
            # Sort by score (fewest usable edges first - most constrained)
            candidates.sort(key=lambda x: x[0])
            
            for score, neighbor, eidx, new_mask in candidates:
                path.append(neighbor)
                used_edges[eidx] = True
                result = dfs_cycle(start, path, new_mask, depth + 1)
                used_edges[eidx] = False
                path.pop()
                if result is not None:
                    return result
            return None
        
        # Try to find cycles one by one
        for c in range(k_target):
            if time.time() > deadline - 3:
                break
            
            # Try different starting vertices if the first fails
            found = None
            for start_v in range(n):
                if time.time() > deadline - 3:
                    break
                found = dfs_cycle(start_v, [start_v], 1 << start_v, 1)
                if found is not None:
                    break
            
            if found is None:
                break
            
            # Mark edges of this cycle as used
            for i in range(n):
                u = found[i]
                v = found[(i + 1) % n]
                key = (min(u, v), max(u, v))
                eidx = edge_index[key]
                used_edges[eidx] = True
            
            cycles.append(found)
        
        return cycles

    # Try to find maximum number of edge-disjoint Hamiltonian cycles
    best_cycles = None
    best_count = 0
    
    for k in range(max_possible_cycles, 0, -1):
        if time.time() > deadline - 2:
            break
        
        cycles = find_hamiltonian_cycles(k, deadline)
        
        if cycles is not None and len(cycles) > best_count:
            best_cycles = cycles
            best_count = len(cycles)
            
            if best_count >= max_possible_cycles:
                break
    
    # Fallback: if we didn't find enough, try to find at least 1
    if best_cycles is None or len(best_cycles) == 0:
        for i, (u, v) in enumerate(edges_list):
            if time.time() > deadline:
                break
            for w in adj[u] & adj[v]:
                if w != u and w != v:
                    best_cycles = [[u, v, w]]
                    best_count = 1
                    break
            if best_cycles:
                break
    
    if best_cycles is None or len(best_cycles) == 0:
        return [[0, 1, 2]]
    
    result = []
    for cycle in best_cycles:
        result.append([int(v) for v in cycle])
    
    return result

# EVOLVE-BLOCK-END
