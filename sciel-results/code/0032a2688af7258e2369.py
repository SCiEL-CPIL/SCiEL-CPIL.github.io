<<<<<<< SEARCH
    def has_q3_with_edge(u, v):
        """Check if adding edge (u,v) creates a Q3 subgraph."""
        u_neighbors = list(adj[u] - {v})
        v_neighbors = list(adj[v] - {u})
        
        # Iterate over pairs of neighbors of u
        for i in range(len(u_neighbors)):
            a = u_neighbors[i]
            for j in range(i+1, len(u_neighbors)):
                b = u_neighbors[j]
                # In Q3, neighbors of a vertex are not adjacent to each other
                if b in adj[a]:
                    continue
                
                # Iterate over pairs of neighbors of v
                for k in range(len(v_neighbors)):
                    c = v_neighbors[k]
                    for l in range(k+1, len(v_neighbors)):
                        d = v_neighbors[l]
                        # In Q3, neighbors of a vertex are not adjacent to each other
                        if d in adj[c]:
                            continue
                        
                        # Ensure all 6 vertices are distinct
                        if len({a, b, c, d}) < 4:
                            continue
                        if len({a, b, c, d} & {u, v}) > 0:
                            continue

                        a_c = c in adj[a]
                        a_d = d in adj[a]
                        b_c = c in adj[b]
                        b_d = d in adj[b]
                        
                        valid_pattern = False
                        # Pattern 1: a-c and b-d exist
                        if a_c and b_d and not a_d and not b_c:
                            valid_pattern = True
                        # Pattern 2: a-d and b-c exist
                        elif a_d and b_c and not a_c and not b_d:
                            valid_pattern = True

                        if not valid_pattern:
                            continue

                        # Find e and f
                        used = {u, v, a, b, c, d}
                        candidates_e = []
                        for x in range(n):
                            if x not in used and a in adj[x] and b in adj[x]:
                                candidates_e.append(x)
                        
                        for e in candidates_e:
                            candidates_f = []
                            for x in range(n):
                                if x not in used and x != e and c in adj[x] and d in adj[x]:
                                    candidates_f.append(x)
                            for f in candidates_f:
                                if e in adj[f]:
                                    return True
        return False
=======
    def has_q3_with_edge(u, v):
        """Check if adding edge (u,v) creates a Q3 subgraph."""
        # Get neighbors as lists of indices
        u_neighbors = [x for x in range(n) if (adj[u] >> x) & 1 and x != v]
        v_neighbors = [x for x in range(n) if (adj[v] >> x) & 1 and x != u]
        
        for i in range(len(u_neighbors)):
            a = u_neighbors[i]
            for j in range(i+1, len(u_neighbors)):
                b = u_neighbors[j]
                # Check a-b not edge
                if (adj[a] >> b) & 1:
                    continue
                
                for k in range(len(v_neighbors)):
                    c = v_neighbors[k]
                    for l in range(k+1, len(v_neighbors)):
                        d = v_neighbors[l]
                        # Check c-d not edge
                        if (adj[c] >> d) & 1:
                            continue
                        
                        # Check distinctness
                        if a == c or a == d or b == c or b == d:
                            continue
                        if a == u or a == v or b == u or b == v or c == u or c == v or d == u or d == v:
                            continue

                        a_c = (adj[a] >> c) & 1
                        a_d = (adj[a] >> d) & 1
                        b_c = (adj[b] >> c) & 1
                        b_d = (adj[b] >> d) & 1
                        
                        valid_pattern = False
                        if a_c and b_d and not a_d and not b_c:
                            valid_pattern = True
                        elif a_d and b_c and not a_c and not b_d:
                            valid_pattern = True

                        if not valid_pattern:
                            continue

                        used_mask = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d)
                        
                        candidates_e = []
                        for x in range(n):
                            if not (used_mask >> x) & 1:
                                if (adj[x] >> a) & 1 and (adj[x] >> b) & 1:
                                    candidates_e.append(x)
                        
                        for e in candidates_e:
                            candidates_f = []
                            for x in range(n):
                                if not (used_mask >> x) & 1 and x != e:
                                    if (adj[x] >> c) & 1 and (adj[x] >> d) & 1:
                                        candidates_f.append(x)
                            for f in candidates_f:
                                if (adj[e] >> f) & 1:
                                    return True
        return False
>>>>>>> REPLACE

And update the main loop to use bitmasks:

