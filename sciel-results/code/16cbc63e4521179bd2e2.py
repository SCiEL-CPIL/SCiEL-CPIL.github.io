import time
import sys

sys.setrecursionlimit(10000)

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for i in range(n):
        adj[i].sort()

    all_edges = set()
    for u, v in edges:
        if u < v:
            all_edges.add((u, v))
        else:
            all_edges.add((v, u))

    def edge_key(u, v):
        return (u, v) if u < v else (v, u)

    def get_remaining_degree(node, used_edges):
        count = 0
        for nb in adj[node]:
            e = edge_key(node, nb)
            if e not in used_edges:
                count += 1
        return count

    def find_hamiltonian_cycle(used_edges, start=0, first_nb=None):
        """
        Finds a Hamiltonian cycle in the graph defined by all_edges \ used_edges.
        If first_nb is provided, the cycle must start 0 -> first_nb.
        """
        if first_nb is not None:
            e_start = edge_key(start, first_nb)
            if e_start in used_edges:
                return None
            initial_path = [start, first_nb]
            initial_used = set(used_edges)
            initial_used.add(e_start)
        else:
            # Try all neighbors
            # We'll implement a loop outside or inside? 
            # For efficiency in the decomposition search, we usually fix first_nb.
            # If first_nb is None, we try all valid first_nbs.
            for nb in adj[start]:
                e = edge_key(start, nb)
                if e in used_edges:
                    continue
                res = find_hamiltonian_cycle(used_edges, start, nb)
                if res:
                    return res
            return None

        def backtrack(path, used):
            if len(path) == n:
                last = path[-1]
                e = edge_key(last, start)
                if e not in used and e in all_edges:
                    return path[:]
                return None
            
            current = path[-1]
            candidates = []
            for nb in adj[current]:
                e = edge_key(current, nb)
                if e in used:
                    continue
                if nb in path:
                    if nb == start and len(path) == n:
                        pass
                    else:
                        continue
                candidates.append(nb)
            
            # Heuristic: sort by remaining degree
            candidates.sort(key=lambda x: get_remaining_degree(x, used))
            
            for nb in candidates:
                e = edge_key(current, nb)
                new_used = set(used)
                new_used.add(e)
                res = backtrack(path + [nb], new_used)
                if res:
                    return res
            return None

        return backtrack(initial_path, initial_used)

    start_time = time.time()
    deadline = 85.0

    # Find C1
    # Fix first edge (0, 1) to break symmetry
    c1 = find_hamiltonian_cycle(set(), 0, 1)
    if not c1:
        # Try other first edges if (0,1) fails? Unlikely.
        c1 = find_hamiltonian_cycle(set(), 0, None)
        if not c1:
            return []
            
    used1 = set()
    for i in range(n):
        u = c1[i]
        v = c1[(i+1)%n]
        used1.add(edge_key(u, v))
        
    # Find C2
    # Try possible first edges for C2
    # Neighbors of 0 not in C1
    neighbors_of_0 = adj[0]
    used_neighbors = set()
    for i in range(n):
        used_neighbors.add(c1[i])
        used_neighbors.add(c1[(i+1)%n])
        
    candidate_first_nbs_c2 = [nb for nb in neighbors_of_0 if edge_key(0, nb) not in used1]
    
    c2 = None
    for fnb in candidate_first_nbs_c2:
        if time.time() > deadline:
            break
        c2 = find_hamiltonian_cycle(used1, 0, fnb)
        if c2:
            break
            
    if not c2:
        # If C2 not found with fixed C1, we might need to backtrack C1.
        # This is complex. Given the graph structure, C1=(0,1) should work.
        # If it doesn't, we return [c1] (score 1) or try random restarts for C1.
        # Let's try a few random C1s if this fails.
        import random
        for _ in range(10):
            if time.time() > deadline:
                break
            # Random first edge
            u, v = random.choice(edges)
            # Ensure 0 is in the cycle? No, HC visits all.
            # We can just pick a random neighbor of 0.
            fnb = random.choice(adj[0])
            c1_try = find_hamiltonian_cycle(set(), 0, fnb)
            if c1_try:
                used1_try = set()
                for i in range(n):
                    u_t = c1_try[i]
                    v_t = c1_try[(i+1)%n]
                    used1_try.add(edge_key(u_t, v_t))
                
                cand_c2 = [nb for nb in neighbors_of_0 if edge_key(0, nb) not in used1_try]
                for fnb2 in cand_c2:
                    if time.time() > deadline:
                        break
                    c2_try = find_hamiltonian_cycle(used1_try, 0, fnb2)
                    if c2_try:
                        used2_try = set(used1_try)
                        for i in range(n):
                            u_t = c2_try[i]
                            v_t = c2_try[(i+1)%n]
                            used2_try.add(edge_key(u_t, v_t))
                        
                        remaining = all_edges - used2_try
                        if len(remaining) == n:
                            rem_adj = [[] for _ in range(n)]
                            for u_t, v_t in remaining:
                                rem_adj[u_t].append(v_t)
                                rem_adj[v_t].append(u_t)
                            
                            valid = all(len(rem_adj[i]) == 2 for i in range(n))
                            if valid:
                                c3_try = [0]
                                prev_t = 0
                                curr_t = rem_adj[0][0]
                                while curr_t != 0:
                                    c3_try.append(curr_t)
                                    next_t = rem_adj[curr_t][0] if rem_adj[curr_t][0] != prev_t else rem_adj[curr_t][1]
                                    prev_t = curr_t
                                    curr_t = next_t
                                
                                if len(c3_try) == n:
                                    return [c1_try, c2_try, c3_try]
        
        return [c1] # Fallback

    used2 = set(used1)
    for i in range(n):
        u = c2[i]
        v = c2[(i+1)%n]
        used2.add(edge_key(u, v))
        
    # Check C3
    remaining = all_edges - used2
    if len(remaining) == n:
        rem_adj = [[] for _ in range(n)]
        for u, v in remaining:
            rem_adj[u].append(v)
            rem_adj[v].append(u)
        
        valid = all(len(rem_adj[i]) == 2 for i in range(n))
        if valid:
            c3 = [0]
            prev = 0
            curr = rem_adj[0][0]
            while curr != 0:
                c3.append(curr)
                next_curr = rem_adj[curr][0] if rem_adj[curr][0] != prev else rem_adj[curr][1]
                prev = curr
                curr = next_curr
            
            if len(c3) == n:
                return [c1, c2, c3]
                
    return [c1, c2] # Fallback if C3 invalid, but should be valid if C1, C2 are HCs in 6-reg graph?
    # Wait, if C1 and C2 are HCs, they use 42 edges. 63-42=21 edges left.
    # If the graph is 3-edge-colorable (which a decomposition implies), the remainder is a 1-factor? No, a cycle.
    # If C1 and C2 are edge disjoint HCs, the remainder is a 2-regular graph (degree 2 at each vertex).
    # So it is a collection of cycles. Since it has 21 edges and 21 vertices, and is 2-regular, it must be a single cycle (HC) if it's connected.
    # If it's disconnected, it's multiple cycles. But the task requires "same vertex set".
    # So if remainder is not a single HC, we fail to get 3.
    # But the knowledge says 3 is possible.

    return [c1]

