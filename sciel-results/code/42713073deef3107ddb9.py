def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    deadline = start_time + 85
    
    # Fallback to baseline
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        baseline_ops = pebble_baseline(dag, fast_memory_budget)
    except:
        baseline_ops = None
    
    # Try to improve
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
        
        from collections import deque
        queue = deque([v for v in all_vertices if in_degree[v] == 0])
        topo_order = []
        while queue:
            v = queue.popleft()
            topo_order.append(v)
            for c in children[v]:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
        
        # Position of each vertex in topological order
        pos_in_topo = {v: i for i, v in enumerate(topo_order)}
        
        # Compute next use position for each vertex (when it's next needed as a parent)
        # For each vertex, find the earliest position where it's used as a parent
        next_use_pos = {v: len(topo_order) for v in all_vertices}  # default: never needed again
        for i, v in enumerate(topo_order):
            ps = parents_dict.get(v, [])
            for p in ps:
                # p is needed at position i
                if i < next_use_pos.get(p, len(topo_order)):
                    next_use_pos[p] = i
        
        # Also track total future uses for tie-breaking
        future_uses = {v: 0 for v in all_vertices}
        for v in topo_order:
            for c in children[v]:
                future_uses[v] += 1
        
        # Greedy pebbling with improved memory management
        in_fast_memory = set()
        operations = []
        
        def should_evict(u, current_pos):
            """Score for eviction: lower score = better to evict.
            Use next_use_pos as primary, future_uses as secondary."""
            if u in outputs:
                return float('inf')  # Never evict outputs
            # Primary: how far away is next use (larger = better to evict)
            # Secondary: fewer future uses = better to evict
            next_use = next_use_pos.get(u, len(topo_order))
            distance_to_next_use = max(0, next_use - current_pos)
            # Combine: prefer evicting vertices with far next use and few total uses
            # Use a weighted combination
            score = -distance_to_next_use * 10 - future_uses.get(u, 0)
            return score
        
        for idx, v in enumerate(topo_order):
            if time.time() > deadline - 5:
                break
            
            if v in inputs:
                if v not in in_fast_memory:
                    # Need to load input
                    if len(in_fast_memory) >= fast_memory_budget:
                        # Find best vertex to evict
                        best_evict = None
                        best_score = float('-inf')
                        for u in list(in_fast_memory):
                            score = should_evict(u, idx)
                            if score > best_score:
                                best_score = score
                                best_evict = u
                        if best_evict is not None:
                            in_fast_memory.discard(best_evict)
                            operations.append(['delete', best_evict])
                    in_fast_memory.add(v)
                    operations.append(['load', v])
            else:
                ps = parents_dict.get(v, [])
                # Ensure parents are in fast memory
                for p in ps:
                    if p not in in_fast_memory:
                        # Need to free space if at budget
                        if len(in_fast_memory) >= fast_memory_budget:
                            # Find best vertex to evict
                            best_evict = None
                            best_score = float('-inf')
                            for u in list(in_fast_memory):
                                if u != p:
                                    score = should_evict(u, idx)
                                    if score > best_score:
                                        best_score = score
                                        best_evict = u
                            if best_evict is not None:
                                in_fast_memory.discard(best_evict)
                                operations.append(['delete', best_evict])
                        
                        in_fast_memory.add(p)
                        operations.append(['load', p])
                
                in_fast_memory.add(v)
                operations.append(['compute', v])
            
            # Delete if no longer needed and not an output
            if future_uses.get(v, 0) == 0 and v not in outputs:
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
