def construct():
    import time
    start_time = time.time()
    DEADLINE = 115.0
    n = 16
    
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                adj[i] |= (1 << j)
                
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            edges.add((i, j))

    def find_q3_bitset(u, v):
        """
        Finds a Q3 subgraph containing edge (u, v).
        Returns a set of 8 vertices if found, else None.
        """
        # Neighbors of u excluding v
        nbrs_u_mask = adj[u] & ~(1 << v)
        # Neighbors of v excluding u
        nbrs_v_mask = adj[v] & ~(1 << u)
        
        # Get list of neighbors
        nbrs_u = [i for i in range(n) if (nbrs_u_mask >> i) & 1]
        nbrs_v = [i for i in range(n) if (nbrs_v_mask >> i) & 1]
        
        # Q3 structure relative to u=0, v=1
        # u (0) connects to two (2), four (4)
        # v (1) connects to three (3), five (5)
        # two (2) connects to three (3), six (6)
        # four (4) connects to five (5), six (6)
        # three (3) connects to seven (7)
        # five (5) connects to seven (7)
        # six (6) connects to seven (7)
        
        for two in nbrs_u:
            for four in nbrs_u:
                if four == two:
                    continue
                # Check if (two, four) is NOT an edge? 
                # In Q3, 2 and 4 are not connected.
                # If they are connected in G, it doesn't prevent them from being part of a Q3, 
                # but the subgraph must be isomorphic to Q3. 
                # The definition of "contains a subgraph isomorphic to Q3" means there exists 
                # a mapping of Q3 vertices to G vertices such that edges of Q3 map to edges of G. 
                # Non-edges in Q3 do NOT need to map to non-edges in G. 
                # So (two, four) can be an edge in G.
                
                for three in nbrs_v:
                    if three in (u, v, two, four):
                        continue
                    if ((adj[two] >> three) & 1) == 0:
                        continue
                    for five in nbrs_v:
                        if five in (u, v, two, four, three):
                            continue
                        if ((adj[four] >> five) & 1) == 0:
                            continue
                        
                        # Now we have u, v, two, four, three, five
                        # Need six: neighbor of two and four, not in used
                        # Need seven: neighbor of three, five, six, not in used
                        
                        used = {u, v, two, four, three, five}
                        
                        # Find six
                        six_mask = adj[two] & adj[four]
                        # Remove used vertices from six_mask
                        for x in used:
                            six_mask &= ~(1 << x)
                        
                        # Iterate six
                        six_list = [i for i in range(n) if (six_mask >> i) & 1]
                        for six in six_list:
                            # Find seven
                            seven_mask = adj[three] & adj[five] & adj[six]
                            used_with_six = used | {six}
                            for x in used_with_six:
                                seven_mask &= ~(1 << x)
                                
                            # Check if any seven exists
                            if seven_mask != 0:
                                # Find the actual vertex
                                # We just need to return the set
                                # Get one bit
                                seven = (seven_mask & -seven_mask).bit_length() - 1
                                return {u, v, two, four, three, five, six, seven}
        return None

    while time.time() - start_time < DEADLINE:
        found_q3 = False
        # Iterate over a copy of edges to avoid modification issues during iteration
        # But we break after first removal, so it's okay
        for u, v in list(edges):
            if time.time() - start_time > DEADLINE:
                break
            q3_verts = find_q3_bitset(u, v)
            if q3_verts:
                # Remove edge (u, v)
                # Or remove an edge from q3_verts that is in edges
                # Let's just remove (u, v) for simplicity
                edges.discard((u, v))
                edges.discard((v, u)) # edges are stored as (min, max) or just set of tuples? 
                # My edges set contains (i, j) with i < j? 
                # Let's ensure edges are stored consistently.
                # I added (i, j) for i < j.
                # So discard (min(u,v), max(u,v))
                a, b = (u, v) if u < v else (v, u)
                edges.discard((a, b))
                
                # Update adj
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
                found_q3 = True
                break # Restart the loop to check for other Q3s
                
        if not found_q3:
            break

    # Convert edges set to list of lists
    return [[u, v] for u, v in edges]
