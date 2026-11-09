import time
import itertools
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 80  # 80s limit with safety margin
    
    # Build adjacency list
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Sort neighbors to make search deterministic and potentially faster
    for u in adj:
        adj[u].sort()
        
    all_vertices = set(range(n))
    
    def find_ham_cycle(adj, forbidden_edges, start=0, neighbor_order=None):
        """
        Finds a Hamiltonian cycle in the graph defined by adj, 
        avoiding edges in forbidden_edges.
        Returns a list of vertices representing the cycle, or None.
        """
        if time.time() > deadline:
            return None
        
        # Use custom neighbor order if provided, else use adj
        def get_neighbors(u):
            if neighbor_order is not None:
                return neighbor_order[u]
            return adj[u]
        
        path = [start]
        visited = {start}
        path_set = {start}
        
        def backtrack():
            if time.time() > deadline:
                return False
                
            if len(path) == n:
                # Check if last node connects to start
                last = path[-1]
                if start in adj[last] and (min(last, start), max(last, start)) not in forbidden_edges:
                    return True
                return False
            
            last = path[-1]
            neighbors = get_neighbors(last)
            
            # Heuristic: sort neighbors by degree in remaining graph (Warnsdorff-like)
            if len(path) > 1 and len(path) < n - 2:
                def degree_score(nb):
                    if nb in visited:
                        return 0
                    edge_key = (min(last, nb), max(last, nb))
                    if edge_key in forbidden_edges:
                        return -1
                    # Count unvisited neighbors of nb (excluding last)
                    count = 0
                    for nn in adj[nb]:
                        if nn not in visited and nn != last:
                            ek = (min(nb, nn), max(nb, nn))
                            if ek not in forbidden_edges:
                                count += 1
                    return count
                neighbors = sorted([nb for nb in neighbors if nb not in visited], key=degree_score)
            
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                # Check if edge is forbidden
                edge_key = (min(last, neighbor), max(last, neighbor))
                if edge_key in forbidden_edges:
                    continue
                # Heuristic: if only one node left, it must connect to start
                if len(path) == n - 1:
                    if start not in adj[neighbor]:
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

    def try_find_three_cycles(neighbor_orders):
        """Try to find 3 edge-disjoint Hamiltonian cycles using given neighbor orders."""
        cycles = []
        used_edges = set()
        
        for order in neighbor_orders:
            c1 = find_ham_cycle(adj, used_edges, start=0, neighbor_order=order)
            if c1 is None:
                continue
                
            temp_used = set(used_edges)
            for i in range(len(c1)):
                u = c1[i]
                v = c1[(i+1)%len(c1)]
                temp_used.add((min(u,v), max(u,v)))
            
            c2 = find_ham_cycle(adj, temp_used, start=0, neighbor_order=order)
            if c2 is None:
                continue
            
            temp_used2 = set(temp_used)
            for i in range(len(c2)):
                u = c2[i]
                v = c2[(i+1)%len(c2)]
                temp_used2.add((min(u,v), max(u,v)))
            
            # Check for C3
            rem_adj = {i: [] for i in range(n)}
            for u, v in edges:
                if (min(u,v), max(u,v)) not in temp_used2:
                    rem_adj[u].append(v)
                    rem_adj[v].append(u)
                    
            if all(len(nei) == 2 for nei in rem_adj.values()):
                visited_rem = set()
                curr = 0
                path_rem = []
                while curr not in visited_rem:
                    visited_rem.add(curr)
                    path_rem.append(curr)
                    if len(path_rem) == 1:
                        next_node = rem_adj[curr][0]
                    else:
                        prev_node = path_rem[-2]
                        next_node = rem_adj[curr][0] if rem_adj[curr][1] == prev_node else rem_adj[curr][1]
                    curr = next_node
                    
                if len(path_rem) == n and 0 in rem_adj[path_rem[-1]]:
                    cycles = [c1, c2, path_rem]
                    break
        
        return cycles

    # Generate different neighbor orderings for restarts
    base_order = {u: adj[u][:] for u in range(n)}
    
    # Try different orderings
    orderings = [base_order]
    
    # Reverse order
    rev_order = {u: adj[u][::-1] for u in range(n)}
    orderings.append(rev_order)
    
    # Randomized orders
    rng = random.Random(42)
    for _ in range(5):
        rand_order = {}
        for u in range(n):
            lst = adj[u][:]
            rng.shuffle(lst)
            rand_order[u] = lst
        orderings.append(rand_order)
    
    # Try to find 3 cycles with different orderings
    cycles = try_find_three_cycles(orderings)
    
    if not cycles:
        # Fallback: just find what we can greedily
        used_edges = set()
        for order in orderings:
            c = find_ham_cycle(adj, used_edges, start=0, neighbor_order=order)
            if c:
                cycles.append(c)
                for i in range(len(c)):
                    u = c[i]
                    v = c[(i+1)%len(c)]
                    used_edges.add((min(u,v), max(u,v)))
                if time.time() > deadline:
                    break
    
    # Convert to lists of ints
    result = [list(map(int, c)) for c in cycles]
    
    # Check validity: all cycles must have same vertex set
    if result:
        vset = set(result[0])
        for c in result:
            if set(c) != vset:
                return []
        # Check edge disjointness
        seen_edges = set()
        for c in result:
            for i in range(len(c)):
                u, v = c[i], c[(i+1)%len(c)]
                key = (min(u,v), max(u,v))
                if key in seen_edges:
                    return []
                seen_edges.add(key)
                
    return result

# EVOLVE-BLOCK-END
