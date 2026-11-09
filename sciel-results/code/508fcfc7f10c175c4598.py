import time
import random

N = 40
DEADLINE_SECONDS = 85

def construct():
    t0 = time.time()
    random.seed(42)
    
    def make_edge_set_from_list(edge_list):
        edge_set = set()
        adj = {i: set() for i in range(N)}
        for a, b in edge_list:
            u, v = min(a, b), max(a, b)
            if u == v:
                continue
            edge_set.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
        return edge_set, adj
    
    def validate_cycles(cycles, edge_set):
        for L in range(3, N + 1):
            if L not in cycles:
                return False
            cyc = cycles[L]
            if len(cyc) != L:
                return False
            if len(set(cyc)) != L:
                return False
            for i in range(L):
                u = cyc[i]
                v = cyc[(i + 1) % L]
                if (min(u, v), max(u, v)) not in edge_set:
                    return False
        return True
    
    def find_cycle_of_length(L, adj, edge_set, max_time=0.5):
        """Find a simple cycle of length L using randomized DFS with pruning."""
        if L < 3 or L > N:
            return None
        
        start_time = time.time()
        
        # Try multiple random starts and orders
        for attempt in range(50):
            if time.time() - start_time > max_time:
                break
            
            # Pick a random start vertex with sufficient degree
            candidates = [v for v in range(N) if len(adj[v]) >= 2]
            if not candidates:
                return None
            
            start = random.choice(candidates)
            visited = set()
            path = []
            found = [None]
            
            def dfs(u, depth):
                if found[0] is not None:
                    return True
                if time.time() - start_time > max_time:
                    return True
                
                if depth == L:
                    if start in adj[u]:
                        found[0] = path[:]
                        return True
                    return False
                
                visited.add(u)
                path.append(u)
                
                # Get neighbors, shuffle for randomization
                neighbors = list(adj[u])
                random.shuffle(neighbors)
                
                for v in neighbors:
                    if v == start and depth == L - 1:
                        found[0] = path[:]
                        break
                    if v not in visited:
                        # Pruning: check if enough unvisited neighbors remain
                        remaining = L - depth - 1
                        if remaining > 0:
                            unvisited_count = sum(1 for x in adj[u] if x not in visited and x != start)
                            if unvisited_count < remaining:
                                continue
                        
                        if dfs(v, depth + 1):
                            break
                
                path.pop()
                visited.remove(u)
                return found[0] is not None
            
            dfs(start, 0)
            if found[0] is not None:
                return found[0]
        
        return None
    
    # Strategy: Build a sparse base graph and iteratively add edges while finding cycles
    # Start with just a Hamiltonian cycle (40 edges)
    base_edges = [(i, (i + 1) % N) for i in range(N)]
    
    # We need cycles of all lengths 3-40
    # The Hamiltonian cycle gives us length 40
    # We need to add chords to enable shorter cycles
    
    # Key insight: A chord (i, j) creates a cycle of length |i-j|+1 and N-|i-j|+1
    # So we need chords that cover all lengths 3-39
    
    # Let's use a systematic approach: add chords from vertex 0 to create small cycles
    # Chord (0, k) creates cycle of length k+1 (going 0-1-2-...-k-0) and 41-k (going 0-39-38-...-k-0)
    
    # For length L, we need a chord that creates a cycle of that length
    # If we have chord (0, L-1), we get cycle of length L
    
    # But we can be smarter: use chords between non-adjacent vertices on the rim
    # Chord (a, b) where a < b creates cycles of length (b-a)+1 and N-(b-a)+1
    
    # To minimize edges, we want each chord to cover as many new cycle lengths as possible
    
    # Let's try: use chords (0, k) for strategic k values
    # This gives cycles of length k+1 and 41-k
    
    # We need all lengths 3-39 (40 already covered by rim)
    # Lengths 3-39 need to be covered
    
    # If we add chord (0, k), we cover length k+1 and 41-k
    # So chord (0, 2) covers length 3 and 39
    # Chord (0, 3) covers length 4 and 38
    # Chord (0, 4) covers length 5 and 37
    # ...
    # Chord (0, 19) covers length 20 and 21
    
    # So chords (0, 2) through (0, 19) cover all lengths 3-39!
    # That's 18 chords + 40 rim edges = 58 edges total
    
    # But maybe we can do better by using chords between other vertices
    
    # Alternative: use a "wheel" structure or other sparse graph
    
    # Let's try a different approach: start with rim + minimal chords, then greedily remove
    
    # Initial construction: rim + chords (0, k) for k in 2..19
    initial_edges = list(base_edges)
    for k in range(2, N - 1):
        initial_edges.append((0, k))
    
    edge_set, adj = make_edge_set_from_list(initial_edges)
    
    # Generate initial cycles
    cycles = {}
    for L in range(3, N + 1):
        if L == N:
            cycles[L] = list(range(N))
        else:
            # Use chord (0, L-1) to create cycle of length L
            # Cycle: 0, 1, 2, ..., L-1, 0
            cycles[L] = list(range(L))
    
    # Validate
    if not validate_cycles(cycles, edge_set):
        # Fallback to complete graph
        all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        edge_set, adj = make_edge_set_from_list(all_edges)
        for L in range(3, N + 1):
            cycles[L] = list(range(L))
        return {'edges': list(edge_set), 'cycles': [cycles[L] for L in range(3, N + 1)]}
    
    # Greedy edge removal with better heuristics
    # Sort edges by "criticality": edges used in fewer cycles first
    def edge_criticality(e, cycles, edge_set):
        """Count how many cycles use this edge."""
        count = 0
        for L in range(3, N + 1):
            cyc = cycles[L]
            for i in range(L):
                u = cyc[i]
                v = cyc[(i + 1) % L]
                if (min(u, v), max(u, v)) == e:
                    count += 1
                    break
        return count
    
    # Iterative removal: try to remove each edge, if successful keep it removed
    # Repeat until no more edges can be removed
    progress = True
    while progress:
        if time.time() - t0 > DEADLINE_SECONDS - 5:
            break
        
        progress = False
        # Get current edges sorted by criticality (lowest first)
        current_edges = list(edge_set)
        current_edges.sort(key=lambda e: edge_criticality(e, cycles, edge_set))
        
        for e in current_edges:
            if time.time() - t0 > DEADLINE_SECONDS - 5:
                break
            
            u, v = e
            if e not in edge_set:
                continue
            
            # Temporarily remove edge
            edge_set.remove(e)
            adj[u].remove(v)
            adj[v].remove(u)
            
            # Check if all cycles still exist or can be found
            all_valid = True
            new_cycles = {}
            
            for L in range(3, N + 1):
                old_cyc = cycles[L]
                
                # Check if old cycle still valid
                valid = True
                for i in range(L):
                    cu = old_cyc[i]
                    cv = old_cyc[(i + 1) % L]
                    if (min(cu, cv), max(cu, cv)) not in edge_set:
                        valid = False
                        break
                
                if valid:
                    new_cycles[L] = old_cyc
                else:
                    # Need to find new cycle
                    new_cyc = find_cycle_of_length(L, adj, edge_set, max_time=0.3)
                    if new_cyc is None:
                        all_valid = False
                        break
                    new_cycles[L] = new_cyc
            
            if all_valid:
                cycles = new_cycles
                progress = True
            else:
                # Restore edge
                edge_set.add(e)
                adj[u].add(v)
                adj[v].add(u)
    
    # Final validation
    if not validate_cycles(cycles, edge_set):
        # Reset to initial valid state
        edge_set, adj = make_edge_set_from_list(initial_edges)
        for L in range(3, N + 1):
            if L == N:
                cycles[L] = list(range(N))
            else:
                cycles[L] = list(range(L))
    
    final_edges = list(edge_set)
    return {'edges': final_edges, 'cycles': [cycles[L] for L in range(3, N + 1)]}
