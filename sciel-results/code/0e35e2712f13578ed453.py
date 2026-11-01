import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Fallback: The seed program's logic (LUW graph)
    # This ensures we always return a valid graph if the heuristic fails
    def fallback_construct():
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Heuristic Search
    def get_girth(adj, n, max_girth_check=12):
        # Returns the girth (length of shortest cycle)
        # BFS from each vertex
        min_girth = float('inf')
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            while queue:
                u = queue.pop(0)
                if dist[u] >= max_girth_check // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and parent[v] != u:
                        # Found a cycle
                        # Length is dist[u] + dist[v] + 1
                        # But we need to ensure it's the shortest
                        # Actually, BFS finds shortest paths.
                        # If we find a back edge, the cycle length is dist[u] + dist[v] + 1
                        # However, this might not be the *shortest* cycle in the whole graph,
                        # but it is a cycle. We want the minimum over all such findings.
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
        return min_girth

    def find_shortest_cycle(adj, n, max_depth=12):
        # Returns the list of vertices in the shortest cycle
        best_cycle = None
        best_len = float('inf')
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            while queue:
                u = queue.pop(0)
                if dist[u] >= max_depth // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and parent[v] != u:
                        # Cycle found
                        # Reconstruct path
                        cycle = [u, v]
                        curr = u
                        while curr != start:
                            curr = parent[curr]
                            cycle.append(curr)
                        curr = v
                        while curr != start:
                            curr = parent[curr]
                            cycle.append(curr)
                        # Remove duplicates and order
                        # Actually, the cycle is u -> ... -> start -> ... -> v -> u
                        # Let's just calculate length
                        len_u = dist[u]
                        len_v = dist[v]
                        total_len = len_u + len_v + 1
                        if total_len < best_len:
                            best_len = total_len
                            # Reconstruct properly
                            path1 = []
                            curr = u
                            while curr != -1:
                                path1.append(curr)
                                if curr == start: break
                                curr = parent[curr]
                            path1.reverse()
                            
                            path2 = []
                            curr = v
                            while curr != -1:
                                path2.append(curr)
                                if curr == start: break
                                curr = parent[curr]
                            path2.reverse()
                            
                            # Combine
                            cycle = path1 + path2[1:]
                            best_cycle = cycle
        return best_cycle

    def random_4_regular(n):
        # Generate a random 4-regular graph
        # Using configuration model
        if n < 2: return []
        if n == 2: return [(0,1), (0,1)] # Not simple, but k=4 impossible for n=2 simple
        
        # For simple graphs, we need to avoid multi-edges and loops
        # A simpler way: Start with a ring, then random swaps
        adj = {i: set() for i in range(n)}
        # Initial ring
        for i in range(n):
            adj[i].add((i+1)%n)
            adj[(i+1)%n].add(i)
        
        # Need degree 4, currently 2. Add 2 more edges per vertex.
        # Create a list of "open" ends
        # This is tricky to do randomly without collisions.
        
        # Alternative: Random permutation based
        # 1. Create list of vertices repeated 4 times: [0,0,0,0, 1,1,1,1, ...]
        # 2. Shuffle
        # 3. Pair up consecutive elements to form edges
        # 4. Check for loops and multi-edges. If found, reshuffle.
        
        attempts = 0
        while attempts < 1000:
            ends = []
            for i in range(n):
                ends.extend([i]*4)
            random.shuffle(ends)
            
            adj = {i: set() for i in range(n)}
            valid = True
            for i in range(0, 2*n*4, 2):
                u = ends[i]
                v = ends[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            
            if valid:
                return adj
            attempts += 1
        return None

    def try_improve(adj, n, target_girth, time_limit):
        best_adj = [set(nei) for nei in adj.values()]
        best_girth = get_girth(best_adj, n, target_girth + 2)
        
        if best_girth >= target_girth:
            return best_adj, best_girth

        current_adj = best_adj
        current_girth = best_girth
        T = 1.0
        cooling = 0.99
        
        while time.time() < time_limit:
            if current_girth >= target_girth:
                return current_adj, current_girth
            
            # Find a short cycle
            cycle = find_shortest_cycle(current_adj, n, min(current_girth, target_girth))
            if not cycle:
                return current_adj, current_girth
            
            # Pick an edge in the cycle to swap
            if len(cycle) < 2: continue
            edge_idx = random.randint(0, len(cycle)-1)
            u = cycle[edge_idx]
            v = cycle[(edge_idx+1) % len(cycle)]
            
            # Try to find a partner edge (w, x) to swap with (u, v)
            # Swap: remove (u,v) and (w,x), add (u,w) and (v,x)
            # Constraints:
            # 1. (u,w) and (v,x) must not exist
            # 2. u != w, v != x
            # 3. w, x must be in the graph
            # 4. The new edges should not create a cycle shorter than current_girth
            
            # Candidate w, x
            # w must be a neighbor of someone? No, just any vertex.
            # But (w,x) must be an existing edge.
            # Let's pick a random edge (w,x) from the graph
            # And check if swap is valid
            
            # To be efficient, pick w from neighbors of u? No.
            # Let's pick a random edge (w,x)
            all_edges = []
            for i in range(n):
                for j in current_adj[i]:
                    if i < j:
                        all_edges.append((i,j))
            
            if not all_edges: continue
            
            # Try a few random swaps
            for _ in range(10):
                w, x = random.choice(all_edges)
                if (u, v) == (w, x) or (u, v) == (x, w):
                    continue
                if u in (w, x) or v in (w, x):
                    continue
                
                # Check if (u,w) and (v,x) are valid
                if w in current_adj[u] or x in current_adj[v]:
                    continue
                if x in current_adj[u] or w in current_adj[v]:
                    # Try the other orientation: (u,x) and (v,w)
                    if w in current_adj[v] or x in current_adj[u]:
                        continue
                    # Swap (u,v) and (w,x) with (u,x) and (v,w)
                    new_adj = [set(nei) for nei in current_adj]
                    new_adj[u].remove(v)
                    new_adj[v].remove(u)
                    new_adj[w].remove(x)
                    new_adj[x].remove(w)
                    
                    new_adj[u].add(x)
                    new_adj[x].add(u)
                    new_adj[v].add(w)
                    new_adj[w].add(v)
                    
                    # Check girth
                    new_girth = get_girth(new_adj, n, target_girth + 2)
                    
                    if new_girth > current_girth:
                        current_adj = new_adj
                        current_girth = new_girth
                        if current_girth > best_girth:
                            best_adj = current_adj
                            best_girth = current_girth
                        break
                    elif new_girth == current_girth:
                        # Accept with prob
                        if random.random() < 0.1:
                            current_adj = new_adj
                            break
                    else:
                        # Worse, accept with prob exp(-delta/T)
                        delta = current_girth - new_girth
                        if random.random() < (2.718 ** (-delta / T)):
                            current_adj = new_adj
                            break
            T *= cooling
        
        return best_adj, best_girth

    # Main logic
    # Try to find a graph with n close to the Moore bound
    moore_bound = 2 * (1 + 3 + 9 + 27 + 81) # for g=10, k=4
    # Actually 2 * sum_{i=0}^{4} 3^i = 2 * 121 = 242
    start_n = 243
    max_n = 350
    
    best_result = None
    best_n = float('inf')
    
    for n in range(start_n, max_n + 1):
        if time.time() > deadline - 10:
            break
            
        # Try a few restarts for this n
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            adj = random_4_regular(n)
            if not adj:
                continue
                
            improved_adj, girth = try_improve(adj, n, g, deadline - 5)
            
            if girth >= g:
                # Valid graph found
                edges = []
                for u in range(n):
                    for v in improved_adj[u]:
                        if u < v:
                            edges.append((u, v))
                if n < best_n:
                    best_n = n
                    best_result = (n, edges)
                # If we found one, we can try smaller n? No, we are iterating up.
                # But we started from low. So this is the best so far.
                # We can break if we found one with n close to bound?
                # Let's continue to see if we can find a smaller one?
                # We are iterating n from 243 upwards.
                # So the first valid n is the smallest.
                # We can return immediately?
                # It's safer to try a few more restarts to ensure it's not a fluke?
                # No, if it's valid, it's valid.
                # But maybe a different random seed finds a valid graph for a *smaller* n?
                # No, we are going 243, 244, ...
                # So if we find one at 243, that's the best possible (lower bound).
                # If we find one at 250, we should stop and return it?
                # Yes, because we are looking for minimum n.
                return best_result
        
    if best_result:
        return best_result
    else:
        # Fallback to seed
        return fallback_construct()

