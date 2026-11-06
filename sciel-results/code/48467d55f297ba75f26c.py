import time

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    edges = set()
    for i in range(N - 1):
        edges.add((i, i + 1))
    
    all_chords = []
    for u in range(N):
        for v in range(u + 1, N):
            if v - u > 1:
                all_chords.append((u, v))
    
    covered_lengths = set()
    
    def get_cycle_lengths(edge_set):
        adj = {i: [] for i in range(N)}
        for u, v in edge_set:
            adj[u].append(v)
            adj[v].append(u)
        
        lengths = set()
        
        def dfs(start, current, visited, path):
            for neighbor in adj[current]:
                if neighbor == start and len(path) >= 2:
                    lengths.add(len(path) + 1)
                elif neighbor > start and neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(start, neighbor, visited, path)
                    path.pop()
                    visited.remove(neighbor)
        
        for start in range(N):
            visited = {start}
            dfs(start, start, visited, [start])
            
        return lengths

    target_lengths = set(range(3, N + 1))
    
    while len(covered_lengths & target_lengths) < len(target_lengths):
        if time.time() > deadline:
            break
            
        best_chord = None
        best_new_lengths = set()
        
        # Sort chords by potential? Just iterate.
        for chord in all_chords:
            if chord in edges:
                continue
                
            if time.time() > deadline:
                break
                
            new_edges = edges | {chord}
            new_lengths = get_cycle_lengths(new_edges)
            new_covered = new_lengths - covered_lengths
            
            if len(new_covered) > len(best_new_lengths):
                best_new_lengths = new_covered
                best_chord = chord
        
        if best_chord is None:
            break
            
        edges.add(best_chord)
        covered_lengths = get_cycle_lengths(edges)
        
        if len(best_new_lengths) == 0:
            break

    final_lengths = get_cycle_lengths(edges)
    
    cycles = []
    # Build adj for final graph
    adj = {i: [] for i in range(N)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for k in range(3, N + 1):
        found_cycle = None
        if k in final_lengths:
            def dfs_find(start, current, visited, path):
                nonlocal found_cycle
                if found_cycle:
                    return
                for neighbor in adj[current]:
                    if neighbor == start and len(path) >= 2:
                        if len(path) + 1 == k:
                            found_cycle = path + [start]
                            return
                    elif neighbor > start and neighbor not in visited:
                        visited.add(neighbor)
                        path.append(neighbor)
                        dfs_find(start, neighbor, visited, path)
                        path.pop()
                        visited.remove(neighbor)
                        
            for start in range(N):
                if found_cycle:
                    break
                visited = {start}
                dfs_find(start, start, visited, [start])
                
        if found_cycle:
            cycles.append(found_cycle)
        else:
            # Fallback
            cycles.append(list(range(k)))

    return {'edges': list(edges), 'cycles': cycles}
