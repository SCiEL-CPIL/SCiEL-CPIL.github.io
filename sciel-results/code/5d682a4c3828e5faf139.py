import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Precompute full mask for potential subset checks if needed, 
    # but backtracking is likely sufficient.
    
    best_partition = None
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    # Try k from 1 to n
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
            
        # Backtracking state
        # parts: list of dicts or tuples (vertices, can_be_clique, can_be_ind)
        # We store parts as lists of vertices, and maintain flags.
        
        def backtrack(remaining, k_limit, parts):
            # parts is a list of [set_of_vertices, is_clique, is_ind]
            if not remaining:
                # Found a solution
                nonlocal best_partition
                best_partition = [list(p[0]) for p in parts]
                return True
            
            if time.time() > deadline:
                return False
                
            # Heuristic: pick vertex with highest degree in remaining subgraph?
            # Or just the first one.
            v = remaining[0]
            rest = remaining[1:]
            
            # Try adding v to existing parts
            for i in range(len(parts)):
                part_verts, is_cliq, is_ind = parts[i]
                
                # Check if v can be added
                new_is_cliq = is_cliq
                new_is_ind = is_ind
                
                if new_is_cliq:
                    # v must be connected to all in part_verts
                    for u in part_verts:
                        if u not in adj[v]:
                            new_is_cliq = False
                            break
                            
                if new_is_ind:
                    # v must not be connected to any in part_verts
                    for u in part_verts:
                        if u in adj[v]:
                            new_is_ind = False
                            break
                            
                if new_is_cliq or new_is_ind:
                    # Add v
                    part_verts.add(v)
                    parts[i] = [part_verts, new_is_cliq, new_is_ind]
                    
                    if backtrack(rest, k_limit, parts):
                        return True
                        
                    # Undo
                    part_verts.remove(v)
                    parts[i] = [part_verts, is_cliq, is_ind]
                    
            # Try adding v to a new part
            if len(parts) < k_limit:
                # Create new part
                new_part = [{v}, True, True]
                parts.append(new_part)
                
                if backtrack(rest, k_limit, parts):
                    return True
                    
                # Undo
                parts.pop()
                
            return False

        initial_parts = []
        remaining = list(range(n))
        
        if backtrack(remaining, k, initial_parts):
            if best_partition:
                # Ensure types are correct
                # The problem requires lists of ints
                # best_partition is already list of lists of ints
                return best_partition
        
    # Fallback: Greedy from seed
    es = {tuple(e) for e in edges}
    parts = []
    for v in range(n):
        for part in parts:
            trial = part + [v]
            hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
            if all(hits) or not any(hits):
                part.append(v)
                break
        else:
            parts.append([v])
    return parts
