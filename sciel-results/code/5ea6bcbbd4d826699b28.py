import time

def construct():
    P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
    
    n = P['n']
    edges = set()
    for e in P['edges']:
        edges.add((e[0], e[1]))
        edges.add((e[1], e[0]))
        
    # Adjacency matrix for fast lookup
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True

    # Precompute neighbors for each vertex
    neighbors = [set() for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if adj[u][v]:
                neighbors[u].add(v)

    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin

    # We search for minimum k
    # k=1 is possible only if graph is clique or independent set
    # We just iterate k
    
    for k in range(1, n + 1):
        # Backtracking search
        # parts: list of lists of vertices
        # part_types: list of 'C', 'I', or 'U'
        # We assign vertices 0 to n-1
        # Symmetry breaking: vertex 0 must be in part 0
        
        parts = [[] for _ in range(k)]
        part_types = ['U'] * k
        # Initialize part 0 with vertex 0
        parts[0].append(0)
        # part 0 type is 'U' (size 1)
        
        # We can order vertices to prune faster? 
        # Standard order 0..n-1 is fine for n=14.
        
        def backtrack(idx):
            if time.time() > deadline:
                return False, None
            
            if idx == n:
                # All vertices assigned
                # Construct result
                result = []
                for p in parts:
                    if p:
                        result.append(list(p))
                return True, result

            v = idx
            # Try to place v in each part
            # To break symmetry among empty parts or parts with same content?
            # Simple symmetry breaking: 
            # If part i is empty, we can only place v in the first empty part? 
            # Not quite, because parts might have different types.
            # But if two parts are identical in content and type, we can skip duplicates.
            
            seen_states = set()
            
            for i in range(k):
                # Pruning: if this part state was already tried for this vertex, skip
                # State can be (len(parts[i]), part_types[i], frozenset(parts[i]))
                # This is expensive to compute. 
                # Simpler: if parts[i] is empty, and we've already tried placing v in an empty part, skip.
                # But empty parts are indistinguishable.
                
                if not parts[i]:
                    if i > 0 and not parts[i-1]:
                        continue # Only place in first empty part
                    # Actually, if parts[i] is empty, it's 'U'.
                    # If we placed v in an empty part j < i, then parts[j] is no longer empty.
                    # So this logic needs care.
                    
                # Check if v can be added to part i
                valid = True
                new_type = part_types[i]
                
                if part_types[i] == 'C':
                    # Must be connected to all in parts[i]
                    for u in parts[i]:
                        if not adj[v][u]:
                            valid = False
                            break
                    if valid:
                        new_type = 'C'
                elif part_types[i] == 'I':
                    # Must NOT be connected to any in parts[i]
                    for u in parts[i]:
                        if adj[v][u]:
                            valid = False
                            break
                    if valid:
                        new_type = 'I'
                else: # 'U'
                    if len(parts[i]) == 0:
                        new_type = 'U'
                    elif len(parts[i]) == 1:
                        u = parts[i][0]
                        if adj[v][u]:
                            new_type = 'C'
                        else:
                            new_type = 'I'
                    else:
                        # Should not happen if we maintain types correctly
                        valid = False

                if not valid:
                    continue
                
                # Symmetry breaking for identical parts
                # If part i is empty, and part i-1 is empty, skip i.
                # More generally, if part i has same content and type as part j < i, skip.
                # For small n, this might not be needed, but let's add a simple check.
                # We can skip if i > 0 and parts[i] == parts[i-1] and part_types[i] == part_types[i-1]
                # This only works if we process parts in order and they are sorted?
                # Let's just use a set of seen (frozenset(part), type) for this vertex placement step.
                
                key = (frozenset(parts[i]), new_type)
                if key in seen_states:
                    continue
                seen_states.add(key)

                # Assign
                parts[i].append(v)
                part_types[i] = new_type
                
                success, res = backtrack(idx + 1)
                
                if success:
                    return True, res
                    
                # Unassign
                parts[i].pop()
                part_types[i] = 'U' if len(parts[i]) < 2 else part_types[i] # Restore type? 
                # Wait, if it was 'C' or 'I', it stays 'C' or 'I' if we pop back to size >= 2?
                # No, if we pop, the size decreases.
                # If size becomes 1, type should be 'U'.
                # If size becomes 0, type should be 'U'.
                # If size remains >= 2, type is unchanged.
                
                if len(parts[i]) <= 1:
                    part_types[i] = 'U'
                # If len > 1, part_types[i] is already correct (C or I)

            return False, None

        success, res = backtrack(1) # Start from vertex 1, since 0 is fixed in part 0
        
        if success:
            # Verify result
            # Check all parts are clique or independent
            for part in res:
                is_clique = True
                is_indep = True
                for i in range(len(part)):
                    for j in range(i+1, len(part)):
                        if adj[part[i]][part[j]]:
                            is_indep = False
                        else:
                            is_clique = False
                if not (is_clique or is_indep):
                    raise ValueError("Invalid partition found")
            
            # Ensure types are correct Python int/list
            return [list(map(int, part)) for part in res]

    # If no solution found (should not happen as k=n is always valid)
    return [list(map(int, [i])) for i in range(n)]
