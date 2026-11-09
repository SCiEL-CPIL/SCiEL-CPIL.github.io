def pebble(dag, fast_memory_budget):
    import time
    import heapq
    from collections import deque, defaultdict
    
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    # Fallback to baseline
    baseline_ops = None
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        baseline_ops = pebble_baseline(dag, fast_memory_budget)
    except:
        baseline_ops = None
    
    try:
        # Extract DAG info
        if isinstance(dag, dict):
            parents_dict = dict(dag.get('parents', {}))
            inputs = list(dag.get('inputs', []))
            outputs = set(dag.get('outputs', []))
        else:
            parents_dict = dict(getattr(dag, 'parents', {}))
            inputs = list(getattr(dag, 'inputs', []))
            outputs = set(getattr(dag, 'outputs', []))
        
        # Compute topological order
        all_vertices = set(inputs)
        for v, ps in parents_dict.items():
            all_vertices.add(v)
            all_vertices.update(ps)
        
        in_degree = {v: len(parents_dict.get(v, [])) for v in all_vertices}
        children = {v: [] for v in all_vertices}
        for v, ps in parents_dict.items():
            for p in ps:
                children[p].append(v)
        
        queue = deque([v for v in all_vertices if in_degree[v] == 0])
        topo_order = []
        while queue:
            v = queue.popleft()
            topo_order.append(v)
            for c in children[v]:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
        
        if len(topo_order) != len(all_vertices):
            if baseline_ops is not None:
                return baseline_ops
            return []
        
        # Precompute topo index
        topo_idx = {v: i for i, v in enumerate(topo_order)}
        
        # Compute use positions for each vertex (which topo positions need it)
        use_positions = {v: [] for v in all_vertices}
        for i, v in enumerate(topo_order):
            ps = parents_dict.get(v, [])
            for p in ps:
                use_positions[p].append(i)
        
        # Next use and last use
        next_use = {}
        last_use = {}
        for v in all_vertices:
            if use_positions[v]:
                next_use[v] = use_positions[v][0]
                last_use[v] = use_positions[v][-1]
            else:
                next_use[v] = float('inf')
                last_use[v] = -1
        
        # Future use count (total number of times vertex is needed)
        future_uses = {v: len(use_positions[v]) for v in all_vertices}
        
        # Track remaining uses as we progress
        remaining_uses = dict(future_uses)
        
        # Greedy pebbling with lookahead-aware eviction
        in_fast_memory = set()
        operations = []
        
        # Lookahead window size for anticipating needs
        LOOKAHEAD = 50
        
        for idx, v in enumerate(topo_order):
            if time.time() > deadline:
                break
            
            # Determine which vertices will be needed in the lookahead window
            # This helps prioritize keeping frequently-needed vertices in memory
            lookahead_set = set()
            for j in range(idx, min(idx + LOOKAHEAD, len(topo_order))):
                future_v = topo_order[j]
                ps = parents_dict.get(future_v, [])
                for p in ps:
                    lookahead_set.add(p)
            
            if v in inputs:
                if v not in in_fast_memory:
                    if len(in_fast_memory) >= fast_memory_budget:
                        # Evict: find best candidate to remove
                        # Score: lower is better to evict
                        # Priority 1: No future uses (score 0)
                        # Priority 2: Farthest next use (larger distance = better to evict)
                        # Priority 3: Not in lookahead window
                        best_evict = None
                        best_score = float('inf')
                        
                        for u in list(in_fast_memory):
                            if u in outputs:
                                continue  # Don't evict outputs before storing
                            
                            ru = remaining_uses.get(u, 0)
                            if ru == 0:
                                score = 0  # Best to evict
                            else:
                                # Distance to next use
                                dist = next_use.get(u, float('inf')) - idx
                                # In lookahead?
                                in_lookahead = 1 if u in lookahead_set else 0
                                # Combine: prefer evicting those far away and not in lookahead
                                score = dist * 10 + in_lookahead * 1000
                            
                            if score < best_score:
                                best_score = score
                                best_evict = u
                        
                        if best_evict is not None:
                            in_fast_memory.discard(best_evict)
                            operations.append(['delete', best_evict])
                    
                    in_fast_memory.add(v)
                    operations.append(['load', v])
            else:
                ps = parents_dict.get(v, [])
                for p in ps:
                    if p not in in_fast_memory:
                        if len(in_fast_memory) >= fast_memory_budget:
                            # Evict: find best candidate to remove
                            best_evict = None
                            best_score = float('inf')
                            
                            for u in list(in_fast_memory):
                                if u in outputs:
                                    continue
                                if u == p:
                                    continue  # Don't evict the vertex we're trying to load
                                
                                ru = remaining_uses.get(u, 0)
                                if ru == 0:
                                    score = 0
                                else:
                                    dist = next_use.get(u, float('inf')) - idx
                                    in_lookahead = 1 if u in lookahead_set else 0
                                    score = dist * 10 + in_lookahead * 1000
                                
                                if score < best_score:
                                    best_score = score
                                    best_evict = u
                            
                            if best_evict is not None:
                                in_fast_memory.discard(best_evict)
                                operations.append(['delete', best_evict])
                        
                        in_fast_memory.add(p)
                        operations.append(['load', p])
                
                in_fast_memory.add(v)
                operations.append(['compute', v])
            
            # Update remaining uses: decrement for each child that uses v
            for c in children.get(v, []):
                if remaining_uses.get(v, 0) > 0:
                    remaining_uses[v] -= 1
            
            # Delete if no longer needed and not an output
            if remaining_uses.get(v, 0) == 0 and v not in outputs:
                if v in in_fast_memory:
                    in_fast_memory.discard(v)
                    operations.append(['delete', v])
            elif v in outputs:
                if v in in_fast_memory:
                    in_fast_memory.discard(v)
                    operations.append(['store', v])
        
        # Ensure all outputs are stored
        for v in outputs:
            if v in in_fast_memory:
                in_fast_memory.discard(v)
                operations.append(['store', v])
        
        # Count transfers
        my_transfers = sum(1 for op in operations if op[0] in ['load', 'store'])
        
        # Compare with baseline
        if baseline_ops is not None:
            baseline_transfers = sum(1 for op in baseline_ops if op[0] in ['load', 'store'])
            if my_transfers < baseline_transfers:
                return operations
            else:
                return baseline_ops
        else:
            return operations
    
    except:
        if baseline_ops is not None:
            return baseline_ops
        return []
