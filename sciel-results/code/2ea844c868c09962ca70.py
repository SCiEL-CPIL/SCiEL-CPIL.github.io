import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 55
    
    # Build adjacency list
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for u in adj:
        adj[u].sort()
    
    all_edges_set = set((min(u, v), max(u, v)) for u, v in edges)
    
    def find_ham_cycle(adj_mod, forbidden_edges):
        start = 0
        path = [start]
        visited = {start}
        
        def backtrack():
            if time.time() > deadline:
                return False
            if len(path) == n:
                last = path[-1]
                if start in adj_mod[last] and (min(last, start), max(last, start)) not in forbidden_edges:
                    return True
                return False
            
            last = path[-1]
            for neighbor in adj_mod[last]:
                if neighbor not in visited:
                    edge_key = (min(last, neighbor), max(last, neighbor))
                    if edge_key in forbidden_edges:
                        continue
                    if len(path) == n - 1:
                        if start not in adj_mod[neighbor]:
                            continue
                        if (min(neighbor, start), max(neighbor, start)) in forbidden_edges:
                            continue
                    
                    path.append(neighbor)
                    visited.add(neighbor)
                    if backtrack():
                        return True
                    path.pop()
                    visited.remove(neighbor)
            return False
        
        if backtrack():
            return path
        return None
    
    def get_cycle_edges(cycle):
        es = set()
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i+1) % len(cycle)]
            es.add((min(u, v), max(u, v)))
        return es
    
    def is_valid_ham_cycle(cycle, forbidden_edges):
        if len(cycle) != n or set(cycle) != set(range(n)):
            return False
        for e in get_cycle_edges(cycle):
            if e not in all_edges_set or e in forbidden_edges:
                return False
        return True
    
    def find_diverse_ham_cycles(adj_base, forbidden_edges, count=3):
        """Find up to 'count' distinct Hamiltonian cycles using different neighbor orderings."""
        cycles = []
        seen = set()
        
        orderings = ['asc', 'desc', 'even_first', 'odd_first']
        
        for ord_name in orderings:
            if time.time() > deadline or len(cycles) >= count:
                break
            
            mod_adj = {i: [] for i in range(n)}
            for u in range(n):
                nbrs = list(adj_base[u])
                if ord_name == 'asc':
                    nbrs.sort()
                elif ord_name == 'desc':
                    nbrs.sort(reverse=True)
                elif ord_name == 'even_first':
                    nbrs.sort()
                    even = [nbrs[i] for i in range(0, len(nbrs), 2)]
                    odd = [nbrs[i] for i in range(1, len(nbrs), 2)]
                    nbrs = even + odd
                elif ord_name == 'odd_first':
                    nbrs.sort()
                    odd = [nbrs[i] for i in range(1, len(nbrs), 2)]
                    even = [nbrs[i] for i in range(0, len(nbrs), 2)]
                    nbrs = odd + even
                mod_adj[u] = nbrs
            
            c = find_ham_cycle(mod_adj, forbidden_edges)
            if c and is_valid_ham_cycle(c, forbidden_edges):
                min_idx = c.index(min(c))
                norm1 = c[min_idx:] + c[:min_idx]
                norm2 = [norm1[0]] + norm1[1:][::-1]
                norm = min(norm1, norm2)
                key = tuple(norm)
                if key not in seen:
                    seen.add(key)
                    cycles.append(c)
        
        return cycles
    
    def trace_2regular_cycle(rem_adj):
        """Trace a single cycle in a 2-regular graph. Returns list of vertices or None."""
        if not all(len(nei) == 2 for nei in rem_adj.values()):
            return None
        visited = set()
        curr = 0
        path = []
        while curr not in visited:
            visited.add(curr)
            path.append(curr)
            if len(path) == 1:
                next_node = rem_adj[curr][0]
            else:
                prev_node = path[-2]
                next_node = rem_adj[curr][0] if rem_adj[curr][1] == prev_node else rem_adj[curr][1]
            curr = next_node
        if len(path) == n and 0 in rem_adj[path[-1]]:
            return path
        return None
    
    # Try to find 3 edge-disjoint Hamiltonian cycles
    best_result = []
    
    # Get diverse C1 candidates
    c1_candidates = find_diverse_ham_cycles(adj, set(), count=4)
    
    for c1 in c1_candidates:
        if time.time() > deadline:
            break
        used_1 = get_cycle_edges(c1)
        
        # Get diverse C2 candidates given C1
        c2_candidates = find_diverse_ham_cycles(adj, used_1, count=4)
        
        for c2 in c2_candidates:
            if time.time() > deadline:
                break
            used_2 = get_cycle_edges(c2)
            all_used = used_1 | used_2
            
            # Check if remaining forms a single Hamiltonian cycle
            rem_adj = {i: [] for i in range(n)}
            for u, v in edges:
                key = (min(u, v), max(u, v))
                if key not in all_used:
                    rem_adj[u].append(v)
                    rem_adj[v].append(u)
            
            c3 = trace_2regular_cycle(rem_adj)
            if c3 and is_valid_ham_cycle(c3, all_used):
                result = [list(c1), list(c2), list(c3)]
                # Final validation
                vset = set(range(n))
                seen_edges = set()
                valid = True
                for c in result:
                    if set(c) != vset or len(c) != n:
                        valid = False
                        break
                    for e in get_cycle_edges(c):
                        if e in seen_edges:
                            valid = False
                            break
                        seen_edges.add(e)
                    if not valid:
                        break
                if valid:
                    best_result = result
                    break
        
        if best_result:
            break
    
    # Fallback: try to get at least 2 cycles
    if not best_result:
        c1 = find_ham_cycle(adj, set())
        if c1 and is_valid_ham_cycle(c1, set()):
            used = get_cycle_edges(c1)
            c2 = find_ham_cycle(adj, used)
            if c2 and is_valid_ham_cycle(c2, used):
                best_result = [list(c1), list(c2)]
    
    return [list(map(int, c)) for c in best_result]

# EVOLVE-BLOCK-END
