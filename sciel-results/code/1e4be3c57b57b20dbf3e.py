import time
import random

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    deadline = time.time() + 55.0  # Safety margin
    n = P['n']
    
    # Build adjacency and edge set
    edge_set = set()
    adj = {i: set() for i in range(n)}
    for u, v in P['edges']:
        ek = (min(u,v), max(u,v))
        edge_set.add(ek)
        adj[u].add(v)
        adj[v].add(u)
    
    def edge_key(u, v):
        return (min(u,v), max(u,v))
        
    def find_hamiltonian_cycle(available_edges, start=0, time_limit=5.0):
        """Find a Hamiltonian cycle using only edges in available_edges.
        Uses backtracking with aggressive pruning."""
        t0 = time.time()
        path = [start]
        visited = {start}
        
        # Precompute degree of each vertex in available_edges
        deg_in_avail = [0] * n
        for u, w in available_edges:
            deg_in_avail[u] += 1
            deg_in_avail[w] += 1
        
        # Precompute neighbor lists sorted by degree for each vertex
        # This helps with the "most constrained first" heuristic
        def get_candidates(last):
            cands = []
            for nb in adj[last]:
                if nb not in visited:
                    if edge_key(last, nb) in available_edges:
                        cands.append(nb)
            if not cands:
                return []
            # Sort by degree in available graph (most constrained first)
            cands.sort(key=lambda x: deg_in_avail[x])
            return cands
        
        def backtrack():
            if time.time() - t0 > time_limit:
                return "timeout"
            
            if len(path) == n:
                last = path[-1]
                if edge_key(last, start) in available_edges:
                    return list(path)
                return None
            
            last = path[-1]
            candidates = get_candidates(last)
            
            if not candidates:
                return None

            for nb in candidates:
                ek = edge_key(last, nb)
                
                # Quick pruning: check if nb still has enough connections
                # nb needs at least 1 other edge in available (besides last-nb)
                # to potentially be part of a cycle
                nb_other_deg = 0
                for u, w in available_edges:
                    if (u == nb or w == nb) and (u,w) != ek:
                        nb_other_deg += 1
                if nb_other_deg < 1:
                    continue
                
                # Check unvisited vertices for degree >= 2 after using edge (last, nb)
                valid = True
                for v in range(n):
                    if v in visited or v == nb:
                        continue
                    # Count degree of v in available_edges, excluding edge (last, nb) if incident
                    deg_v = 0
                    for u, w in available_edges:
                        if u == v or w == v:
                            if (u,w) != ek:
                                deg_v += 1
                    if deg_v < 2:
                        valid = False
                        break
                
                if not valid:
                    continue

                path.append(nb)
                visited.add(nb)
                result = backtrack()
                if result is not None and result != "timeout":
                    return result
                path.pop()
                visited.remove(nb)
                
            return None
        
        result = backtrack()
        if result == "timeout":
            return None
        return result

    cycles = []
    available = set(edge_set)
    
    # Try to find 3 cycles
    max_cycles = 3
    for _ in range(max_cycles):
        if time.time() > deadline:
            break
        
        found = None
        
        # Try different starting vertices
        for start in range(n):
            if time.time() > deadline:
                break
            remaining_time = deadline - time.time()
            if remaining_time < 0.5:
                break
            time_limit = min(remaining_time * 0.4, 8.0)
            found = find_hamiltonian_cycle(available, start=start, time_limit=time_limit)
            if found is not None:
                break
        
        if found is None:
            break
            
        # Remove used edges
        for i in range(len(found)):
            u = found[i]
            v = found[(i+1) % len(found)]
            available.discard(edge_key(u, v))
        cycles.append(found)
        
        # Check if remaining graph can still support a Hamiltonian cycle
        # (min degree >= 2 and connected)
        min_deg = min(sum(1 for u,v in available if u==i or v==i) for i in range(n))
        if min_deg < 2:
            break
        # Quick connectivity check
        if available:
            visited_conn = {0}
            stack = [0]
            while stack:
                node = stack.pop()
                for nb in adj[node]:
                    if nb not in visited_conn and edge_key(node, nb) in available:
                        visited_conn.add(nb)
                        stack.append(nb)
            if len(visited_conn) < n:
                break

    if len(cycles) == 3:
        result = [list(c) for c in cycles]
        # Verify all constraints before returning
        used_edges = set()
        for cyc in result:
            if len(set(cyc)) != n:
                return [list(range(n))]  # Fallback, should not happen
            for i in range(len(cyc)):
                u = cyc[i]
                v = cyc[(i+1) % len(cyc)]
                ek = edge_key(u, v)
                if ek in used_edges:
                    return [list(range(n))]  # Fallback
                used_edges.add(ek)
        return result
    elif cycles:
        return [list(c) for c in cycles]
    else:
        # Fallback: try to find at least 1 cycle
        random.seed(42)
        for _ in range(50):
            found = find_hamiltonian_cycle(set(edge_set), start=random.randint(0, n-1), time_limit=2.0)
            if found:
                return [found]
        return [list(range(n))]
# EVOLVE-BLOCK-END
