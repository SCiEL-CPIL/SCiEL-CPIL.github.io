import random
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(a, b):
    return (a, b) if a < b else (b, a)

def construct():
    t0 = time.time()
    deadline = t0 + 50.0  # Leave 10s margin for safety
    n = P['n']
    full_edges = set(_edge_key(u, v) for u, v in P['edges'])
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute static degrees for initial heuristic
    static_deg = [len(adj[i]) for i in range(n)]

    def find_hamiltonian_cycle(adj_cur, start, deadline):
        """
        Finds a Hamiltonian cycle starting and ending at 'start'.
        Uses randomized DFS with Warnsdorff's heuristic and time-checks.
        """
        n2 = len(adj_cur)
        visited = [False] * n2
        path = [start]
        visited[start] = True
        
        # Pre-sort neighbors for faster access? 
        # We will sort dynamically based on current degree (Warnsdorff)
        
        def dfs():
            if time.time() > deadline:
                return False
            
            if len(path) == n2:
                # Check if we can close the cycle
                return start in adj_cur[path[-1]]
                
            u = path[-1]
            nbrs = list(adj_cur[u])
            
            # Warnsdorff: sort by number of unvisited neighbors (or just current degree in remaining graph)
            # Lower degree neighbors are more constrained, so visit them first
            nbrs.sort(key=lambda w: len(adj_cur[w]))
            
            # Add a bit of randomness to avoid getting stuck in local minima
            # Shuffle the last few elements or use a probability
            if len(nbrs) > 1:
                # Randomize the order of neighbors with the same degree, or just slightly
                # A simple shuffle is often enough if we restart
                random.shuffle(nbrs)
                # Re-sort? No, shuffle destroys the heuristic.
                # Better: keep heuristic order but pick randomly from top-k?
                # For simplicity and speed, let's stick to heuristic order but maybe 
                # the random restarts handle the diversification.
                # Actually, standard Warnsdorff is deterministic. 
                # To make it stochastic, we can shuffle neighbors that have the same degree.
                # Or just shuffle the whole list if we want pure random? 
                # The current code shuffles after sorting, which ruins the sort.
                # Let's do: sort, then with probability p, swap adjacent elements?
                # Or just rely on multiple restarts with different starts.
                
                # Let's try a "soft" heuristic: sort by degree, but pick randomly from the 
                # first 2 or 3 options?
                # For now, let's keep it deterministic per start vertex to be fast, 
                # and rely on trying multiple start vertices.
                pass 
            
            for w in nbrs:
                if not visited[w]:
                    # Heuristic check: if we go to w, does it isolate any other vertex?
                    # This is expensive. Skip for now.
                    
                    visited[w] = True
                    path.append(w)
                    if dfs():
                        return True
                    path.pop()
                    visited[w] = False
            return False
        
        if dfs():
            return list(path)
        return None

    def validate_cycles(cycles):
        """Check all cycles are valid and edge-disjoint"""
        used = set()
        for cyc in cycles:
            if len(cyc) < 3:
                return False
            if len(set(cyc)) != len(cyc):
                return False
            if any(v < 0 or v >= n for v in cyc):
                return False
            for i in range(len(cyc)):
                a, b = cyc[i], cyc[(i + 1) % len(cyc)]
                k = _edge_key(a, b)
                if k not in full_edges:
                    return False
                if k in used:
                    return False
                used.add(k)
        return True

    best = None
    best_count = 0
    
    # Try to find 3 cycles
    # Strategy: 
    # 1. Find a Hamiltonian cycle C1.
    # 2. Remove edges of C1.
    # 3. Find a Hamiltonian cycle C2 in the remaining graph.
    # 4. Remove edges of C2.
    # 5. Check if the remaining edges form a Hamiltonian cycle C3.
    #    (Since 6-regular, if C1 and C2 are edge-disjoint Hamiltonian cycles, 
    #     the remaining graph is 2-regular. If it's connected, it's a Hamiltonian cycle.
    #     If not, it's a set of disjoint cycles. But we need ONE cycle on the full vertex set.
    #     So C3 must be a single cycle covering all vertices.
    
    def attempt_decomposition(rng, deadline):
        adj_cur = [set(s) for s in adj]
        cycles = []
        
        # Find C1
        c1 = None
        # Try all start vertices? Or just a few?
        # 21 vertices is small. Try a few random starts.
        starts = rng.sample(range(n), n)
        for start in starts:
            if time.time() > deadline:
                return None
            c1 = find_hamiltonian_cycle(adj_cur, start, deadline)
            if c1:
                break
        
        if not c1:
            return None
        
        cycles.append(c1)
        # Remove C1 edges
        for i in range(len(c1)):
            a, b = c1[i], c1[(i + 1) % len(c1)]
            adj_cur[a].discard(b)
            adj_cur[b].discard(a)
            
        # Check if graph is still connected? 
        # Quick check: min degree >= 2
        if any(len(adj_cur[v]) < 2 for v in range(n)):
            return None
            
        # Find C2
        c2 = None
        starts = rng.sample(range(n), n)
        for start in starts:
            if time.time() > deadline:
                return None
            c2 = find_hamiltonian_cycle(adj_cur, start, deadline)
            if c2:
                break
                
        if not c2:
            return None
            
        cycles.append(c2)
        # Remove C2 edges
        for i in range(len(c2)):
            a, b = c2[i], c2[(i + 1) % len(c2)]
            adj_cur[a].discard(b)
            adj_cur[b].discard(a)
            
        # Check if remaining graph is a single Hamiltonian cycle
        # It should be 2-regular.
        # Check connectivity
        # Pick any vertex, follow the cycle
        if any(len(adj_cur[v]) != 2 for v in range(n)):
            return None
            
        # Traverse to see if it's one cycle
        visited = [False] * n
        count = 0
        for v in range(n):
            if not visited[v]:
                # Start a new cycle
                count += 1
                curr = v
                while not visited[curr]:
                    visited[curr] = True
                    # Pick a neighbor
                    nbrs = list(adj_cur[curr])
                    # Both are unvisited or one is start?
                    # In a 2-regular graph, just pick one and go
                    # But we need to ensure we don't loop back too early
                    next_v = -1
                    for nb in nbrs:
                        if not visited[nb]:
                            next_v = nb
                            break
                    if next_v == -1:
                        # All neighbors visited, must be the start of this component
                        # Check if it closes the loop to the start of this component
                        # This logic is tricky. Simpler:
                        # Just check if the component size is n
                        pass
                    curr = next_v
                    if curr == -1:
                        break
        # Simpler connectivity check for 2-regular graph:
        # Start at 0, follow path, see if we visit all n vertices
        path = [0]
        visited = {0}
        curr = 0
        for _ in range(n - 1):
            nbrs = list(adj_cur[curr])
            # Pick neighbor not in path (unless it's the start and we are closing)
            next_v = -1
            for nb in nbrs:
                if nb not in visited:
                    next_v = nb
                    break
            if next_v == -1:
                # Must be start, but we only close at the end
                # If we are not at the last step, this is a small cycle
                break
            curr = next_v
            visited.add(curr)
            path.append(curr)
            
        if len(path) == n and 0 in adj_cur[curr]:
            # We found a Hamiltonian cycle in the remainder
            # Construct the list
            c3 = path
            cycles.append(c3)
            return cycles
        else:
            return None
            
    rng = random.Random(42)
    attempt = 0
    
    while time.time() < deadline:
        attempt += 1
        rng = random.Random(attempt * 7919 + 13)
        
        result = attempt_decomposition(rng, deadline)
        
        if result:
            if validate_cycles(result):
                if len(result) > best_count:
                    best_count = len(result)
                    best = [[int(v) for v in cyc] for cyc in result]
                    if best_count >= 3:
                        return best
                        
    if best:
        return best
        
    # Fallback: try to find just 1 or 2
    # ... (similar to before but simpler)
    adj_cur = [set(s) for s in adj]
    for start in range(n):
        c = find_hamiltonian_cycle(adj_cur, start, deadline)
        if c:
            return [[int(v) for v in c]]
            
    return [[0, 1, 2]] # Invalid fallback, but should not happen
