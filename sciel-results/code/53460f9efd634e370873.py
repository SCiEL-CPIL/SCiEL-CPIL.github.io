def construct():
    from random import Random
    from itertools import combinations

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def clique_partition(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set(edges)
        
        # Build neighbor sets for I vertices
        neighbor_sets = {}
        for v in range(k, n):
            neighbors = [u for u in range(k) if (u, v) in edge_set or (v, u) in edge_set]
            neighbor_sets[v] = set(neighbors)
        
        # Initial state: each v has one clique N(v)
        # cliques[v] is a list of sets, initially [N(v)]
        cliques = {v: [set(neighbor_sets[v])] for v in range(k, n)}
        
        # Count how many times each pair in C is covered
        # pair_count[(u, w)] = count
        pair_count = {}
        for v in range(k, n):
            S = neighbor_sets[v]
            for u, w in combinations(S, 2):
                key = (min(u, w), max(u, w))
                pair_count[key] = pair_count.get(key, 0) + 1
        
        # Conflicts are pairs with count > 1
        # We want to resolve conflicts by splitting cliques
        # Greedy loop
        while True:
            conflicts = [key for key, cnt in pair_count.items() if cnt > 1]
            if not conflicts:
                break
            
            # Find the best split
            # A split of a clique S (from some v) into A and B (disjoint, union S)
            # Resolves conflicts for pairs (u,w) where u in A, w in B
            # We need to evaluate potential splits.
            # To keep it efficient, we only consider splits that separate a conflicting pair.
            # For each conflict (u, w), find v's that cover it.
            # For each such v, try splitting N(v) such that u and w are separated.
            
            best_split = None
            best_gain = 0
            best_v = None
            best_A = None
            best_B = None
            
            # Iterate through conflicts
            for u, w in conflicts:
                # Find v's covering this pair
                covering_vs = [v for v in range(k, n) if u in neighbor_sets[v] and w in neighbor_sets[v]]
                for v in covering_vs:
                    S = neighbor_sets[v]
                    # We want to split S into A, B such that u in A, w in B (or vice versa)
                    # To maximize gain, we should separate as many other conflicting pairs as possible?
                    # Or just separate u and w?
                    # A simple heuristic: Put u in A, w in B. 
                    # What about other elements?
                    # To minimize future conflicts, maybe put elements that are often paired with u in A, and with w in B?
                    # For now, let's try a random or simple split: 
                    # A = {u} U {x in S | x != u, w}, B = {w}
                    # This separates u and w.
                    # Let's evaluate the gain: number of conflicts resolved.
                    # A conflict (x,y) is resolved if x in A, y in B or x in B, y in A.
                    
                    # Let's try two specific splits:
                    # 1. A={u}, B=S\{u} (w is in B)
                    # 2. A=S\{w}, B={w} (u is in A)
                    # 3. A={u, ...}, B={w, ...}
                    
                    # Let's just try putting u in A and w in B, and distributing others arbitrarily?
                    # Actually, the gain depends on which pairs are separated.
                    # Pairs (u, z) where z in B are separated.
                    # Pairs (w, z) where z in A are separated.
                    
                    # Let's try a simple split: A = {u}, B = S - {u}
                    A1 = {u}
                    B1 = S - {u}
                    gain1 = 0
                    for x, y in conflicts:
                        if (x in A1 and y in B1) or (x in B1 and y in A1):
                            # Check if this pair is currently covered by v
                            if x in S and y in S:
                                gain1 += 1
                            
                    # Try A = S - {w}, B = {w}
                    A2 = S - {w}
                    B2 = {w}
                    gain2 = 0
                    for x, y in conflicts:
                        if (x in A2 and y in B2) or (x in B2 and y in A2):
                            if x in S and y in S:
                                gain2 += 1

                    # Try a balanced split?
                    # Let's stick to the best of these two for now.
                    if gain1 > best_gain:
                        best_gain = gain1
                        best_split = (A1, B1)
                        best_v = v
                    if gain2 > best_gain:
                        best_gain = gain2
                        best_split = (A2, B2)
                        best_v = v
            
            if best_gain == 0:
                # No split can resolve any conflict?
                # This might happen if conflicts are "intrinsic" or we ran out of options?
                # Or if my gain calculation is wrong.
                # If we can't resolve by splitting, we must leave them and cover with C-cliques.
                # But wait, if count > 1, it's invalid. We MUST resolve.
                # So there must be a split.
                # Maybe my gain calc missed something?
                # If u, w are in S, splitting S must separate them if we put them in different sets.
                # So gain should be at least 1 for the pair (u,w) itself.
                # Why would gain be 0?
                # Ah, the loop `for x, y in conflicts` checks if x,y in S.
                # If (u,w) is in conflicts, and u,w in S, then in split A1={u}, B1=S-{u},
                # u in A1, w in B1. So (u,w) is separated. gain should be >= 1.
                # Unless `u` or `w` is not in `conflicts` list? No, we iterated over conflicts.
                # So best_gain should be > 0.
                
                # If it is 0, break to avoid infinite loop, but result will be invalid.
                break

            # Apply the best split
            v = best_v
            A, B = best_split
            S = neighbor_sets[v]
            
            # Update cliques[v]: replace S with A and B
            # Find index of S in cliques[v]
            # Since we use sets, we can just replace the list
            cliques[v] = [A, B] # Note: this replaces ALL previous splits for v?
            # No, we need to replace ONLY the set S in the list cliques[v].
            # But cliques[v] might have multiple sets if v was split before.
            # My structure `cliques[v]` is a list of sets.
            # I need to find the set that is equal to S.
            
            # Since S is the original neighbor set, and we only split original sets or their children?
            # If I split S into A, B, and later split A into A1, A2, 
            # then cliques[v] contains A1, A2, B.
            # The `neighbor_sets[v]` is always the original full set.
            # So I cannot use `neighbor_sets[v]` to identify the current set to split.
            
            # I need to track the current sets.
            # Let's restructure:
            # active_cliques = list of (v, S) pairs.
            
            # Let's restart the logic with a list of active cliques.
            pass
        
        # The above logic is getting complex to patch. 
        # Let's implement a cleaner version from scratch in the replacement.
        
        # REIMPLEMENTATION
        active_cliques = []
        for v in range(k, n):
            if neighbor_sets[v]:
                active_cliques.append((v, set(neighbor_sets[v])))
        
        # Recalculate pair counts based on active_cliques
        def calc_pair_counts(cliques_list):
            counts = {}
            for v, S in cliques_list:
                for u, w in combinations(S, 2):
                    key = (min(u, w), max(u, w))
                    counts[key] = counts.get(key, 0) + 1
            return counts
        
        pair_counts = calc_pair_counts(active_cliques)
        
        while True:
            conflicts = [key for key, cnt in pair_counts.items() if cnt > 1]
            if not conflicts:
                break
            
            best_gain = 0
            best_idx = -1
            best_A = None
            best_B = None
            
            for i, (v, S) in enumerate(active_cliques):
                if len(S) < 2:
                    continue
                # Try to find a split of S that resolves the most conflicts
                # Heuristic: Pick a conflict (u,w) that is in S.
                # Split S into {u} and S\{u} or similar.
                # To be more efficient, we can try a few random splits or specific ones.
                # Given small k, we can try to separate a specific conflict.
                
                # Find a conflict involving elements of S
                target_conflict = None
                for u, w in conflicts:
                    if u in S and w in S:
                        target_conflict = (u, w)
                        break
                
                if not target_conflict:
                    continue
                
                u, w = target_conflict
                # Try split: A={u}, B=S\{u}
                A1 = {u}
                B1 = S - {u}
                # Calculate gain: number of conflicts resolved
                # A conflict (x,y) is resolved if it was in S and is now separated.
                gain1 = 0
                for x, y in conflicts:
                    if x in S and y in S:
                        if (x in A1 and y in B1) or (x in B1 and y in A1):
                            gain1 += 1
                            
                # Try split: A=S\{w}, B={w}
                A2 = S - {w}
                B2 = {w}
                gain2 = 0
                for x, y in conflicts:
                    if x in S and y in S:
                        if (x in A2 and y in B2) or (x in B2 and y in A2):
                            gain2 += 1

                if gain1 > best_gain:
                    best_gain = gain1
                    best_idx = i
                    best_A = A1
                    best_B = B1
                if gain2 > best_gain:
                    best_gain = gain2
                    best_idx = i
                    best_A = A2
                    best_B = B2
            
            if best_gain == 0:
                # Cannot resolve any conflict by splitting?
                # This implies we should stop and cover remaining with C-cliques?
                # But conflicts mean double coverage, which is invalid.
                # So we MUST resolve.
                # If gain is 0, it means no split separates any conflicting pair within S?
                # That's impossible if target_conflict is in S.
                # Unless my gain calc is flawed.
                # If u, w in S, and we split {u} | S-{u}, then u in A, w in B.
                # So (u,w) is separated. Gain should be >= 1.
                # So best_gain should be > 0.
                break

            # Apply split
            v, S = active_cliques[best_idx]
            # Remove old clique
            active_cliques.pop(best_idx)
            # Add new cliques
            if best_A:
                active_cliques.append((v, best_A))
            if best_B:
                active_cliques.append((v, best_B))
                
            # Update pair counts
            pair_counts = calc_pair_counts(active_cliques)

        # Now, all pairs are covered at most once by I-cliques.
        # Some pairs might be covered 0 times.
        # These must be covered by C-cliques.
        
        covered_edges = set()
        for v, S in active_cliques:
            for u, w in combinations(S, 2):
                covered_edges.add((min(u, w), max(u, w)))
        
        # Find uncovered internal edges
        all_internal_edges = set()
        for u in range(k):
            for w in range(u+1, k):
                all_internal_edges.add((u, w))
                
        uncovered_internal = all_internal_edges - covered_edges
        
        # Partition uncovered_internal into cliques
        # Greedy: pick largest clique in remaining graph
        remaining_edges = set(uncovered_internal)
        c_cliques = []
        
        while remaining_edges:
            # Find largest clique in graph defined by remaining_edges
            # Vertices involved
            vertices = set()
            for u, v in remaining_edges:
                vertices.add(u)
                vertices.add(v)
            
            # Try to find a large clique
            # Start with a random edge or vertex
            # Since k is small, we can try to build a clique greedily
            best_clique = None
            best_size = 0
            
            # Try starting from each vertex
            for start in vertices:
                # Greedily add vertices that are connected to all in current clique
                current = {start}
                candidates = [v for v in vertices if v != start]
                
                # Sort candidates by degree in remaining graph?
                # Or just iterate
                # We want to maximize size.
                # Simple heuristic: add vertex if it connects to all in current
                # To get a good clique, we might need to be smarter.
                # Given small size, let's try a simple greedy:
                # Sort candidates by number of connections to current clique (desc)
                
                while candidates:
                    # Score candidates
                    scored = []
                    for v in candidates:
                        # Check if v connects to all in current
                        ok = True
                        for u in current:
                            e = (min(u, v), max(u, v))
                            if e not in remaining_edges:
                                ok = False
                                break
                        if ok:
                            # Count edges to current
                            cnt = sum(1 for u in current if (min(u, v), max(u, v)) in remaining_edges)
                            scored.append((cnt, v))
                    
                    if not scored:
                        break
                    
                    # Pick best
                    scored.sort(reverse=True)
                    _, best_v = scored[0]
                    current.add(best_v)
                    candidates.remove(best_v)
                
                if len(current) > best_size:
                    best_size = len(current)
                    best_clique = current
                    
            if best_clique is None or best_size < 2:
                # Should not happen if remaining_edges is not empty
                # If no edge, loop ends.
                break
            
            # Add clique
            c_cliques.append(sorted(list(best_clique)))
            
            # Remove edges covered by this clique
            for u, w in combinations(best_clique, 2):
                remaining_edges.discard((min(u, w), max(u, w)))
        
        # Combine results
        result = []
        for v, S in active_cliques:
            if len(S) >= 2:
                result.append(sorted(list(S) + [v]))
        
        for c in c_cliques:
            result.append(c)
            
        return result

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    return [clique_partition(*spec) for spec in specs]
