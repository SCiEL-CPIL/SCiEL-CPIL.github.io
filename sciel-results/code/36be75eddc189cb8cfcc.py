"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency and edge set
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
    
    def get_cycle_edges(cycle):
        es = set()
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            es.add((min(u, v), max(u, v)))
        return es
    
    def find_ham_cycle(used_edges, start=0, time_limit=80):
        """Find a Hamiltonian cycle using backtracking with strong pruning."""
        # Build remaining adjacency
        rem_adj = [set() for _ in range(n)]
        for u in range(n):
            for v in adj[u]:
                e = (min(u, v), max(u, v))
                if e not in used_edges:
                    rem_adj[u].add(v)
        
        # Check minimum degree >= 2
        for u in range(n):
            if len(rem_adj[u]) < 2:
                return None
        
        # Check connectivity of remaining graph
        visited_conn = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited_conn:
                continue
            visited_conn.add(node)
            for nb in rem_adj[node]:
                if nb not in visited_conn:
                    stack.append(nb)
        if len(visited_conn) != n:
            return None
        
        # Precompute edge keys for quick lookup
        edge_key = {}
        for u in range(n):
            for v in rem_adj[u]:
                ek = (min(u, v), max(u, v))
                edge_key[ek] = ek
        
        # Use DFS with strong pruning
        path = [start]
        visited_set = {start}
        edges_used_local = set()
        
        # Precompute neighbor lists sorted by degree for MRV
        sorted_neighbors = {}
        for u in range(n):
            nbs = [(len(rem_adj[v]), v) for v in rem_adj[u]]
            nbs.sort()
            sorted_neighbors[u] = nbs
        
        def dfs(current, path_len):
            if time.time() > time_limit:
                return None
            
            if path_len == n:
                last = path[-1]
                e = (min(last, start), max(last, start))
                if e not in edges_used_local and e in edge_set:
                    return list(path)
                return None
            
            current = path[-1]
            # Try neighbors in MRV order (fewest remaining options first)
            for _, nb in sorted_neighbors[current]:
                if nb in visited_set:
                    continue
                e = (min(current, nb), max(current, nb))
                if e in edges_used_local:
                    continue
                
                # Strong pruning: check if adding this edge makes any unvisited vertex
                # have fewer than 2 available edges to complete the cycle
                feasible = True
                new_visited = visited_set | {nb}
                
                for v in range(n):
                    if v in new_visited:
                        continue
                    # Count available edges from v to vertices not yet in path
                    # (v can connect to unvisited vertices and to start for closing)
                    avail = 0
                    for nb2 in rem_adj[v]:
                        e2 = (min(v, nb2), max(v, nb2))
                        if e2 not in edges_used_local and e2 != e:
                            # nb2 must be unvisited or be start (for closing edge)
                            if nb2 not in visited_set or nb2 == start:
                                avail += 1
                    # v needs at least 2 connections: one to continue path, one more
                    # If v is adjacent to start, it can use that for closing
                    if avail < 2:
                        feasible = False
                        break
                
                if not feasible:
                    continue
                
                # Additional pruning: check that nb still has at least one unvisited neighbor
                # (unless it's the last vertex before returning to start)
                if path_len < n - 1:
                    nb_has_unvisited = False
                    for nb2 in rem_adj[nb]:
                        if nb2 not in visited_set and nb2 != current:
                            e2 = (min(nb, nb2), max(nb, nb2))
                            if e2 not in edges_used_local:
                                nb_has_unvisited = True
                                break
                    if not nb_has_unvisited:
                        # Check if nb can connect to start directly (only if it's last)
                        if path_len != n - 1:
                            continue
                
                # Commit
                path.append(nb)
                visited_set.add(nb)
                edges_used_local.add(e)
                
                result = dfs(nb, path_len + 1)
                
                # Undo
                path.pop()
                visited_set.discard(nb)
                edges_used_local.discard(e)
                
                if result is not None:
                    return result
            
            return None
        
        return dfs(start, 1)
    
    # Main search: greedily find edge-disjoint Hamiltonian cycles
    best_cycles = []
    
    # Try multiple strategies for finding cycles
    strategies = [
        lambda used: find_ham_cycle(used, start=0, time_limit=deadline - 5),
        lambda used: find_ham_cycle(used, start=1, time_limit=deadline - 5),
        lambda used: find_ham_cycle(used, start=10, time_limit=deadline - 5),
    ]
    
    for attempt in range(15):
        if time.time() > deadline - 10:
            break
        
        used_edges = set()
        cycles = []
        
        for cycle_idx in range(10):
            if time.time() > deadline - 5:
                break
            
            # Use different starting vertices for diversity
            start_vertex = cycle_idx % 3
            cycle = find_ham_cycle(used_edges, start=start_vertex, time_limit=deadline - 5)
            if cycle is None:
                break
            
            ce = get_cycle_edges(cycle)
            used_edges |= ce
            cycles.append(cycle)
        
        if len(cycles) > len(best_cycles):
            best_cycles = list(cycles)
        
        if len(best_cycles) >= 3:
            break
    
    # If we found cycles, return them
    if best_cycles:
        # Ensure proper types
        result = []
        for c in best_cycles:
            result.append([int(v) for v in c])
        return result
    
    # Fallback: find any valid cycle
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        return [[int(u), int(v), int(w)]]
    
    return [[0, 1, 2]]

# EVOLVE-BLOCK-END
