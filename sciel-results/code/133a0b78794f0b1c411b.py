    import time
    import random

    def construct():
        start = time.time()
        deadline = start + 110 # Safety margin
        n = 16
        
        # Q3 edges in canonical form (8 vertices, 12 edges)
        q3e = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
        q3_adj = {v: [] for v in range(8)}
        for a, b in q3e:
            q3_adj[a].append(b)
            q3_adj[b].append(a)
            
        # Pre-compute all permutations of 8 vertices? No.
        
        def find_q3(edges):
            """Find a Q3 subgraph in the graph defined by 'edges'. Returns set of edges or None."""
            # Randomize vertex order to help search
            verts = list(range(n))
            random.shuffle(verts)
            
            # Try to map Q3 vertices to graph vertices
            # We pick 8 distinct vertices from 'verts'
            # This is a subset selection + mapping problem.
            
            # Heuristic: Pick a random 8-subset, then try to map Q3 to it.
            # If it fails, try another subset.
            
            for _ in range(500): # Try 500 random subsets
                if time.time() > deadline - 10:
                    return None
                subset = random.sample(verts, 8)
                # Try to find an isomorphism from Q3 to this subset
                # Using backtracking on the mapping
                mapping = {}
                used = set()
                remaining = [i for i in range(8)]
                
                def bt(idx):
                    if time.time() > deadline - 10:
                        return None
                    if idx == len(remaining):
                        # Check if all Q3 edges are in 'edges'
                        for a, b in q3e:
                            u, v = mapping[a], mapping[b]
                            if (min(u,v), max(u,v)) not in edges:
                                return None
                        return { (min(mapping[a], mapping[b]), max(mapping[a], mapping[b])) for a, b in q3e }
                    
                    qv = remaining[idx]
                    # Candidates are vertices in subset not yet used
                    cands = [v for v in subset if v not in used]
                    random.shuffle(cands)
                    
                    for cand in cands:
                        ok = True
                        for nb in q3_adj[qv]:
                            if nb in mapping:
                                x = mapping[nb]
                                if (min(cand, x), max(cand, x)) not in edges:
                                    ok = False
                                    break
                        if ok:
                            mapping[qv] = cand
                            used.add(cand)
                            res = bt(idx + 1)
                            if res:
                                return res
                            del mapping[qv]
                            used.remove(cand)
                    return None
                
                res = bt(0)
                if res:
                    return res
            return None

        # Start with a dense random graph
        random.seed(42)
        edges = set()
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.7:
                    edges.add((u, v))
        
        # Phase 1: Remove edges to destroy Q3s
        while time.time() < deadline - 30:
            q3_found = find_q3(edges)
            if not q3_found:
                break
            # Remove a random edge from the found Q3
            edge_to_remove = random.choice(list(q3_found))
            edges.remove(edge_to_remove)
            
        # Phase 2: Greedy addition of missing edges
        all_edges = set((i, j) for i in range(n) for j in range(i+1, n))
        missing = list(all_edges - edges)
        random.shuffle(missing)
        
        def creates_q3(edge_set, new_edge):
            # Reuse logic from current but optimized?
            # For now, use find_q3 on edge_set U {new_edge}
            # This is slow if called many times.
            # Better: use the specific check from before but with bitmasks?
            # Let's stick to the previous creates_q3 logic but make it faster.
            
            u, v = new_edge
            # Try to find Q3 containing (u,v)
            # ... similar to before
            
            # Let's just use find_q3 for simplicity in this edit, but it might be slow.
            # If it's too slow, we revert to previous creates_q3.
            
            # Actually, let's use the previous creates_q3 logic but ensure it's efficient.
            # The previous one was okay.
            
            # Let's implement a faster creates_q3 using the fact that (u,v) is in the Q3.
            # There are 24 embeddings of (u,v) in Q3?
            # Edges in Q3: 12.
            # For each edge (a,b) in Q3, we can map a->u, b->v or a->v, b->u.
            # 12 * 2 = 24 cases.
            
            for a, b in q3e:
                for perm in [(a, b), (b, a)]:
                    pa, pb = perm
                    assign = {pa: u, pb: v}
                    used = {u, v}
                    remaining = [i for i in range(8) if i not in (pa, pb)]
                    
                    def bt(idx):
                        if time.time() > deadline:
                            return False
                        if idx == len(remaining):
                            for ea, eb in q3e:
                                x, y = assign[ea], assign[eb]
                                if (x, y) == new_edge or (y, x) == new_edge:
                                    continue
                                if (min(x,y), max(x,y)) not in edge_set:
                                    return False
                            return True
                        qv = remaining[idx]
                        # Order candidates by degree?
                        for cand in range(n):
                            if cand in used:
                                continue
                            ok = True
                            for nb in q3_adj[qv]:
                                if nb in assign:
                                    x = assign[nb]
                                    if (cand, x) == new_edge or (x, cand) == new_edge:
                                        continue
                                    if (min(cand, x), max(cand, x)) not in edge_set:
                                        ok = False
                                        break
                            if ok:
                                assign[qv] = cand
                                used.add(cand)
                                res = bt(idx + 1)
                                if res:
                                    return True
                                del assign[qv]
                                used.remove(cand)
                        return False
                    
                    if bt(0):
                        return True
            return False

        for e in missing:
            if time.time() > deadline - 5:
                break
            if not creates_q3(edges, e):
                edges.add(e)
                
        return [[u, v] for u, v in edges]
    