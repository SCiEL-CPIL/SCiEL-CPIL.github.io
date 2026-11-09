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
        
        # Compute next_use: for each vertex, the index in topo_order where it is next needed as a parent
        # Also compute last_use: the last index where it's needed
        next_use = {}
        last_use = {}
        # For each vertex, find the positions in topo_order where it appears as a parent
        # We'll compute for each vertex the list of topo indices where it's used as a parent
        vertex_use_positions = {v: [] for v in all_vertices}
        for i, v in enumerate(topo_order):
            ps = parents_dict.get(v, [])
            for p in ps:
                vertex_use_positions[p].append(i)
        
        for v in all_vertices:
            if vertex_use_positions[v]:
                next_use[v] = vertex_use_positions[v][0]
                last_use[v] = vertex_use_positions[v][-1]
            else:
                next_use[v] = float('inf')
                last_use[v] = -1
        
        # Greedy pebbling with memory management
        in_fast_memory = set()
        operations = []
        
        for idx, v in enumerate(topo_order):
            if v in inputs:
                if v not in in_fast_memory:
                    # Need to load input
                    if len(in_fast_memory) >= fast_memory_budget:
                        # Find best vertex to evict
                        best_evict = None
                        best_score = float('inf')
                        for u in list(in_fast_memory):
                            if u not in outputs:
                                # Score based on when it's next needed (later is better to evict)
                                # Use next_use as proxy: later next_use = better to evict
                                score = next_use.get(u, float('inf'))
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
                # Ensure parents are in fast memory
                for p in ps:
                    if p not in in_fast_memory:
                        # Need to free space if at budget
                        if len(in_fast_memory) >= fast_memory_budget:
                            # Find best vertex to evict - prefer those with later next_use or no future uses
                            best_evict = None
                            best_score = float('inf')
                            for u in list(in_fast_memory):
                                if u not in outputs and u != p:
                                    # Score: later next_use is better to evict
                                    score = next_use.get(u, float('inf'))
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
            
            # Delete if no longer needed and not an output
            if last_use.get(v, -1) < idx and v not in outputs:
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
