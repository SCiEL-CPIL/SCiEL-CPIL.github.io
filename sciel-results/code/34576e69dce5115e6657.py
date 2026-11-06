import random

def construct():
    # Target n=47
    n = 47
    min_deg_required = n - 39  # 8
    
    # We will try to construct such a graph using a greedy approach
    # that prioritizes increasing the degree of the lowest-degree vertices.
    
    # To handle randomness and potential dead ends, we try multiple seeds/strategies
    for trial in range(500):
        random.seed(trial)
        
        adj = [set() for _ in range(n)]
        
        # Precompute all potential edges
        # We will iterate through potential edges in a randomized order,
        # but we will specifically check if adding an edge helps the min-degree vertex.
        
        # A more effective greedy:
        # 1. Start with empty graph.
        # 2. While True:
        #    a. Find vertex u with min degree.
        #    b. Find a valid neighbor v for u (non-adjacent, no C4 created).
        #    c. If found, add edge.
        #    d. If not found, pick next min-degree vertex.
        #    e. If all vertices are saturated (can't add any edge), break.
        
        # To avoid infinite loops or getting stuck in local minima where 
        # some vertices are stuck but others could be helped by different choices,
        # we can add some randomness to the choice of v.
        
        max_steps = 5000
        steps = 0
        while steps < max_steps:
            steps += 1
            # Find vertex with minimum degree
            min_deg = min(len(adj[i]) for i in range(n))
            
            # If we have reached the required min degree for all, we can stop?
            # No, we just need to ensure we don't get stuck below it.
            # If min_deg >= min_deg_required, we might still want to add more edges 
            # to make it more robust, or just stop if we are sure it's C4-free (it is by construction).
            # Actually, if min_deg >= 8, we have a valid graph. We can return immediately?
            # Yes, the condition is "min degree AT LEAST n-39".
            # So if we reach min_deg >= 8, we are done!
            
            if min_deg >= min_deg_required:
                # Verify C4-free (it should be, but just in case)
                is_c4_free = True
                for a in range(n):
                    for c in range(a+1, n):
                        if c in adj[a]:
                            continue
                        # Check common neighbors
                        # Optimized check
                        common_count = 0
                        # Iterate over smaller set
                        if len(adj[a]) < len(adj[c]):
                            for x in adj[a]:
                                if x in adj[c]:
                                    common_count += 1
                                    if common_count >= 2:
                                        is_c4_free = False
                                        break
                        else:
                            for x in adj[c]:
                                if x in adj[a]:
                                    common_count += 1
                                    if common_count >= 2:
                                        is_c4_free = False
                                        break
                        if not is_c4_free:
                            break
                    if not is_c4_free:
                        break
                
                if is_c4_free:
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append([u, v])
                    return {'n': n, 'edges': edges}
            
            # If min_deg < required, we must add an edge incident to a min-degree vertex.
            # Find all vertices with min degree
            min_vertices = [i for i in range(n) if len(adj[i]) == min_deg]
            
            # Try to add an edge to one of these vertices
            progress_made = False
            # Shuffle to avoid bias
            random.shuffle(min_vertices)
            
            for u in min_vertices:
                # Find candidates v for u
                # v must be non-adjacent to u
                # Adding (u,v) must not create a C4.
                # A C4 is created if there is a vertex x such that 
                # x is adjacent to u and x is adjacent to v.
                # So N(u) and N(v) must be disjoint.
                
                candidates = []
                for v in range(n):
                    if v == u:
                        continue
                    if v in adj[u]:
                        continue
                    
                    # Check if N(u) and N(v) are disjoint
                    # If they are not disjoint, adding (u,v) creates a C4.
                    # Specifically, if x in N(u) \cap N(v), then u-x-v-u is a triangle? No.
                    # u-x, x-v, v-u (new). This is a triangle.
                    # C4 is u-x-w-v-u.
                    # If x in N(u) and x in N(v), then u-x-v is a path of length 2.
                    # Adding (u,v) creates triangle u-x-v.
                    # Does it create C4?
                    # A C4 involves 4 distinct vertices.
                    # If N(u) \cap N(v) is non-empty, say x, then u-x-v-u is a 3-cycle.
                    # This does NOT create a C4 directly.
                    # Wait.
                    # If we add (u,v), a C4 is formed if there exists a path of length 3 between u and v in the existing graph.
                    # Path: u - a - b - v.
                    # This requires a in N(u) and b in N(v) and a-b is an edge.
                    # So we need to check if there is an edge (a,b) such that a in N(u) and b in N(v).
                    
                    # Efficient check:
                    # For each a in N(u), check if any neighbor of a is in N(v).
                    # Or: For each a in N(u), for each b in N(a), if b in N(v), then C4.
                    
                    conflict = False
                    for a in adj[u]:
                        for b in adj[a]:
                            if b in adj[v]:
                                conflict = True
                                break
                        if conflict:
                            break
                    
                    if not conflict:
                        candidates.append(v)
                
                if candidates:
                    # Pick a candidate randomly
                    v = random.choice(candidates)
                    adj[u].add(v)
                    adj[v].add(u)
                    progress_made = True
                    break
            
            if not progress_made:
                # No min-degree vertex can accept any new edge without creating a C4.
                # We are stuck.
                # Try next trial.
                break

    # Fallback to n=46 if 47 fails
    n = 46
    min_deg_required = n - 39  # 7
    
    for trial in range(500):
        random.seed(trial + 1000)
        adj = [set() for _ in range(n)]
        
        max_steps = 5000
        steps = 0
        while steps < max_steps:
            steps += 1
            min_deg = min(len(adj[i]) for i in range(n))
            
            if min_deg >= min_deg_required:
                # Verify
                is_c4_free = True
                for a in range(n):
                    for c in range(a+1, n):
                        if c in adj[a]:
                            continue
                        common_count = 0
                        if len(adj[a]) < len(adj[c]):
                            for x in adj[a]:
                                if x in adj[c]:
                                    common_count += 1
                                    if common_count >= 2:
                                        is_c4_free = False
                                        break
                        else:
                            for x in adj[c]:
                                if x in adj[a]:
                                    common_count += 1
                                    if common_count >= 2:
                                        is_c4_free = False
                                        break
                        if not is_c4_free:
                            break
                    if not is_c4_free:
                        break
                
                if is_c4_free:
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append([u, v])
                    return {'n': n, 'edges': edges}
            
            min_vertices = [i for i in range(n) if len(adj[i]) == min_deg]
            random.shuffle(min_vertices)
            
            progress_made = False
            for u in min_vertices:
                candidates = []
                for v in range(n):
                    if v == u:
                        continue
                    if v in adj[u]:
                        continue
                    
                    conflict = False
                    for a in adj[u]:
                        for b in adj[a]:
                            if b in adj[v]:
                                conflict = True
                                break
                        if conflict:
                            break
                    
                    if not conflict:
                        candidates.append(v)
                
                if candidates:
                    v = random.choice(candidates)
                    adj[u].add(v)
                    adj[v].add(u)
                    progress_made = True
                    break
            
            if not progress_made:
                break

    return {'n': 39, 'edges': []}
