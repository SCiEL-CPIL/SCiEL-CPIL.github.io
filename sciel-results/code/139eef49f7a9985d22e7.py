    def calc_exact_in(selected_indices):
        local_map = {idx: i for i, idx in enumerate(selected_indices)}
        triples_local = set()
        for line in lines:
            local_line = [local_map[idx] for idx in line if idx in local_map]
            if len(local_line) >= 3:
                for combo in combinations(local_line, 3):
                    triples_local.add(combo)
        
        n = 20
        best = 0
        
        # Precompute for each node, the list of pairs (u, v) such that (node, u, v) is a triple
        # Actually, just store the triples.
        
        def bt(available, selected_count, selected_set):
            nonlocal best
            if selected_count + len(available) <= best:
                return
            if not available:
                if selected_count > best:
                    best = selected_count
                return
                
            # Pick a node from available
            # Heuristic: pick the one with most constraints?
            # For speed, just pick the first one.
            node = available[0]
            rest = available[1:]
            
            # Option 1: Skip node
            bt(rest, selected_count, selected_set)
            
            # Option 2: Select node
            # Check if node conflicts with selected_set
            # Conflict if there exist u, v in selected_set such that (node, u, v) in triples
            conflict = False
            for t in triples_local:
                if node in t:
                    other_two = [x for x in t if x != node]
                    if other_two[0] in selected_set and other_two[1] in selected_set:
                        conflict = True
                        break
            
            if not conflict:
                new_selected = selected_set | {node}
                # New available: remove node.
                # Also, we don't need to remove others yet, the conflict check handles it.
                bt(rest, selected_count + 1, new_selected)
                
        bt(list(range(20)), 0, set())
        return best
