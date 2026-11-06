import sys
import time

N = 40

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit to be safe

    # Base edges: Hamiltonian cycle
    base_edges = [(i, (i + 1) % N) for i in range(N)]
    
    # Candidate chord sets
    # We want to minimize total edges, so try fewer chords first.
    # But we need to ensure pancyclicity.
    # Let's try a set of candidates ordered by number of chords (ascending).
    
    candidates = []
    
    # Candidate 1: 8 chords of step 5
    # Edges: (i, i+5) for i in 0..7
    chords_8a = [(i, i + 5) for i in range(8)]
    candidates.append(chords_8a)
    
    # Candidate 2: 8 chords of step 10
    chords_8b = [(i, i + 10) for i in range(8)]
    candidates.append(chords_8b)
    
    # Candidate 3: 9 chords of step 10
    chords_9 = [(i, i + 10) for i in range(9)]
    candidates.append(chords_9)
    
    # Candidate 4: 10 chords of step 10
    chords_10 = [(i, i + 10) for i in range(10)]
    candidates.append(chords_10)
    
    # Candidate 5: 10 chords of step 5 (redundant but maybe better distribution?)
    chords_10b = [(i, i + 5) for i in range(10)]
    candidates.append(chords_10b)

    # Fallback: 15 chords to be safe
    chords_15 = [(i, i + 10) for i in range(15)]
    candidates.append(chords_15)

    # Helper to check if a graph is pancyclic and return cycles
    def check_pancyclic(edges):
        if time.time() > deadline:
            return None
        
        # Build adjacency list
        adj = {i: set() for i in range(N)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        cycles = []
        
        for L in range(3, N + 1):
            if time.time() > deadline:
                return None
                
            # Find a cycle of length L
            cycle = None
            
            # Try starting from each vertex
            for start in range(N):
                if time.time() > deadline:
                    return None
                # DFS to find a simple path of length L-1 from start to a neighbor
                # Path: start -> v1 -> ... -> v(L-1) -> start
                # We need a path of length L-1 ending at a neighbor of start
                
                # Optimization: Fix the path to start at 0?
                # Since the graph is not necessarily vertex transitive, we must check all starts.
                # But for these symmetric-ish constructions, checking a few starts might suffice.
                # Let's just do a bounded search.
                
                # Use a simple recursive DFS with pruning
                # State: current_node, path_list, visited_set
                # Target: len(path) == L and last node is connected to start
                
                found = False
                # To avoid deep recursion issues or slow performance, we can use a heuristic
                # But N=40 is small.
                
                # We'll use a stack-based DFS
                stack = [(start, [start], {start})]
                while stack:
                    node, path, visited = stack.pop()
                    if len(path) == L:
                        # Check if last node is connected to start
                        if start in adj[node]:
                            cycle = path
                            found = True
                            break
                        else:
                            continue
                    else:
                        # Extend path
                        # Order neighbors to prioritize unvisited
                        for neighbor in adj[node]:
                            if neighbor not in visited:
                                # Pruning: if we can't reach L-1 more steps, skip?
                                # Simple check: if len(path) + (N - len(visited)) < L: break? No, that's for covering all.
                                # Just standard DFS.
                                stack.append((neighbor, path + [neighbor], visited | {neighbor}))
                
                if found:
                    break
            
            if not cycle:
                return None
            cycles.append(cycle)
            
        return cycles

    best_result = None
    min_edges = float('inf')
    
    for chords in candidates:
        if time.time() > deadline:
            break
            
        # Combine base edges and chords
        # Ensure no duplicates
        edge_set = set()
        for u, v in base_edges:
            if u > v: u, v = v, u
            edge_set.add((u, v))
            
        for u, v in chords:
            if u > v: u, v = v, u
            edge_set.add((u, v))
            
        edges = list(edge_set)
        
        if len(edges) >= min_edges:
            continue
            
        cycles = check_pancyclic(edges)
        
        if cycles is not None:
            # Valid construction
            best_result = {
                'edges': edges,
                'cycles': cycles
            }
            min_edges = len(edges)
            # If we found a very good one (e.g., 48 edges), we can stop?
            # Theoretical min is 40 (impossible for pancyclic).
            # 48 is score 5.0. 49 is 4.44. 50 is 4.0.
            # Let's continue to see if we can find fewer, but 8 chords is likely the limit for this simple structure.
            # If we find one with <= 49 edges, it's a good score.
            if min_edges <= 49:
                # We found a score of at least 4.44. 
                # Is it worth searching for 48? Yes, we are already iterating.
                # But if time is running out, we should return.
                if time.time() > deadline - 5:
                    break

    # If no candidate worked, use the seed (fan graph)
    if best_result is None:
        # Fallback to seed
        edges = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
        # Need to find cycles for this graph too?
        # The seed program provided cycles as list(range(k)).
        # Let's verify if list(range(k)) is a valid cycle in the seed graph.
        # Seed graph: Star 0-1..39 plus path 1-2-...-39.
        # Cycle of length 3: 0-1-2-0? Edge (2,0) exists? Yes, 0 is connected to all.
        # Cycle of length k: 0-1-2-...-(k-1)-0.
        # Edges: (0,1), (1,2), ..., (k-2, k-1), (k-1, 0).
        # All these edges exist in the seed graph.
        # So the seed cycles are valid.
        cycles = [list(range(k)) for k in range(3, N + 1)]
        best_result = {
            'edges': edges,
            'cycles': cycles
        }

    return best_result
