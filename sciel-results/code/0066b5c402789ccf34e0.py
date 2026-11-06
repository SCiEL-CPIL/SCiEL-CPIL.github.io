import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def popcount(x):
        return bin(x).count('1')
        
    def has_k4(adj, n):
        # adj is list of bitmasks
        # Count K4s
        k4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    # Count edges in common
                    c = common
                    edges_in_common = 0
                    while c:
                        lsb = c & -c
                        idx = (lsb.bit_length() - 1)
                        edges_in_common += popcount(adj[idx] & common)
                        c ^= lsb
                    k4_count += edges_in_common // 2
        return k4_count

    def has_indep_6(adj, n):
        # Backtracking for independent set of size 6
        sys.setrecursionlimit(10000)
        
        def exists_indep_k(available, k):
            if k == 0:
                return True
            if popcount(available) < k:
                return False
            
            # Heuristic: Pick a vertex with highest degree in the induced subgraph?
            # Or just pick the first one?
            # Let's pick the first available vertex
            # To improve pruning, we can try to pick a vertex that removes many candidates
            # i.e., high degree in the remaining graph.
            
            # Get list of available vertices
            # This might be slow if done every step.
            # Alternative: just iterate
            v = -1
            best_v = -1
            max_deg = -1
            c = available
            while c:
                lsb = c & -c
                idx = (lsb.bit_length() - 1)
                # Degree of idx in available is popcount(adj[idx] & available)
                # But we want to maximize removal, so maximize degree
                deg = popcount(adj[idx] & available)
                if deg > max_deg:
                    max_deg = deg
                    best_v = idx
                c ^= lsb
            
            if best_v == -1:
                return False
                
            # Branch 1: Include best_v
            # New available: remove best_v and its neighbors
            new_avail = available & ~(adj[best_v] | (1 << best_v))
            if exists_indep_k(new_avail, k - 1):
                return True
            
            # Branch 2: Exclude best_v
            new_avail = available & ~(1 << best_v)
            if exists_indep_k(new_avail, k):
                return True
                
            return False
            
        return exists_indep_k((1 << n) - 1, 6)

    def count_violations(adj, n):
        k4s = has_k4(adj, n)
        indep = 1 if has_indep_6(adj, n) else 0
        return k4s, indep

    def make_paley(p):
        # p must be a prime congruent to 1 mod 4
        squares = {x * x % p for x in range(1, p)}
        adj = [0] * p
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    # Start with Paley(29)
    best_n = 29
    best_graph = make_paley(29)
    
    # Try to extend to larger n
    for target_n in range(30, 36):
        if time.time() > deadline - 20:
            break
            
        # Try to add vertices one by one
        adj = best_graph[:]
        n = best_n
        
        for new_v in range(n, target_n):
            if time.time() > deadline - 20:
                break
                
            # Find a valid neighborhood for new_v
            # Constraints:
            # 1. No K4: For any triangle (u, v, w) in existing graph, new_v cannot be connected to all three.
            # 2. No Indep 6: For any independent set of size 5 in existing graph, new_v cannot be non-connected to all five.
            
            # Heuristic: Try to maximize degree or minimize conflicts
            # We will try random neighborhoods and check validity
            
            found = False
            attempts = 0
            while not found and attempts < 500 and time.time() < deadline - 20:
                attempts += 1
                # Random neighborhood
                neigh = 0
                for i in range(n):
                    if random.random() < 0.5:
                        neigh |= (1 << i)
                        
                # Check constraints
                valid = True
                
                # Check K4
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u] & (1 << v):
                            common = adj[u] & adj[v]
                            # If new_v is connected to u, v, and any w in common, we have a K4
                            if (neigh & (1 << u)) and (neigh & (1 << v)):
                                # Check if there is any w in common connected to new_v
                                # Actually, if new_v is connected to u and v, and there is an edge (w, u) and (w, v),
                                # then {u, v, w, new_v} is a K4 if new_v is connected to w.
                                # So we just need to check if neigh & common is non-empty?
                                # No, if neigh & common has any bit set, say w, then u-v-w is a triangle,
                                # and if new_v is connected to u, v, w, it's a K4.
                                if neigh & common:
                                    valid = False
                                    break
                    if not valid:
                        break
                if not valid:
                    continue
                    
                # Check Indep 6
                # If new_v is NOT connected to an independent set of size 5, we have an indep set of size 6.
                # So we need to ensure that for every independent set of size 5, new_v is connected to at least one.
                # This is hard to check directly.
                # Alternative: Check if the new graph has an independent set of size 6.
                # We can do a quick check:
                # Add new_v to adj
                adj_new = adj[:]
                adj_new.append(neigh)
                for i in range(n):
                    if neigh & (1 << i):
                        adj_new[i] |= (1 << n)
                        
                if has_indep_6(adj_new, n + 1):
                    valid = False
                else:
                    # Also check K4 in the new graph just in case
                    if has_k4(adj_new, n + 1) > 0:
                        valid = False
                    else:
                        # Accept
                        adj = adj_new
                        n += 1
                        found = True
                        break
                        
            if not found:
                break
                
        if n > best_n:
            best_n = n
            best_graph = adj
            
        # Local Search to polish
        if time.time() > deadline - 10:
            break
            
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        current_k4s, current_indep = count_violations(best_graph, best_n)
        current_cost = current_k4s + 10 * current_indep
        
        while time.time() < deadline - 5 and T > T_min:
            # Pick random edge to flip
            u = random.randint(0, best_n - 1)
            v = random.randint(0, best_n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            # Flip edge (u, v)
            if best_graph[u] & (1 << v):
                best_graph[u] ^= (1 << v)
                best_graph[v] ^= (1 << u)
            else:
                best_graph[u] |= (1 << v)
                best_graph[v] |= (1 << u)
                
            new_k4s, new_indep = count_violations(best_graph, best_n)
            new_cost = new_k4s + 10 * new_indep
            delta = new_cost - current_cost
            
            if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                current_cost = new_cost
                current_k4s = new_k4s
                current_indep = new_indep
                if current_cost == 0:
                    break
            else:
                # Revert
                if best_graph[u] & (1 << v):
                    best_graph[u] ^= (1 << v)
                    best_graph[v] ^= (1 << u)
                else:
                    best_graph[u] |= (1 << v)
                    best_graph[v] |= (1 << u)
                    
            T *= cooling

    # Convert adj list of bitmasks to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_graph[i] & (1 << j):
                edges.append([i, j])
                
    return {'n': best_n, 'edges': edges}

