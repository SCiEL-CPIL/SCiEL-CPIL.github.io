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
        
        # Build position map
        topo_pos = {v: i for i, v in enumerate(topo_order)}
        
        # Count future uses
        future_uses = {v: 0 for v in all_vertices}
        for v in topo_order:
            for c in children[v]:
                future_uses[v] += 1
        
        # For each vertex, compute when it's next needed
        next_needed = {}
        for v in all_vertices:
            if children.get(v):
                next_needed[v] = min(topo_pos[c] for c in children[v])
            else:
                next_needed[v] = float('inf')
        
        # Greedy pebbling with improved eviction
        in_fast_memory = set()
        operations = []
        
        input_set = set(inputs)
        
        def make_space_if_needed():
            """Make space in fast memory if at budget. Returns True if space was made."""
            if len(in_fast_memory) < fast_memory_budget:
                return True
            # Find best vertex to evict
            candidates = [u for u in in_fast_memory if u not in outputs]
            if not candidates:
                return False
            # Evict the vertex that is least urgently needed
            # Priority: evict non-outputs with largest next_needed (least urgent) and fewest future uses
            candidates.sort(key=lambda u: (next_needed.get(u, float('inf')), -future_uses.get(u, 0), str(u)))
            u_to_delete = candidates[0]
            in_fast_memory.discard(u_to_delete)
            operations.append(['delete', u_to_delete])
            return True
        
        for v in topo_order:
            if v in input_set:
                if v not in in_fast_memory:
                    if make_space_if_needed():
                        in_fast_memory.add(v)
                        operations.append(['load', v])
            else:
                ps = parents_dict.get(v, [])
                # Ensure parents are in fast memory
                for p in ps:
                    if p not in in_fast_memory:
                        if make_space_if_needed():
                            in_fast_memory.add(p)
                            operations.append(['load', p])
                
                in_fast_memory.add(v)
                operations.append(['compute', v])
            
            # Clean up: delete vertices that are no longer needed
            if v not in outputs:
                if future_uses.get(v, 0) == 0 and v in in_fast_memory:
                    in_fast_memory.discard(v)
                    operations.append(['delete', v])
            else:
                # Store outputs
                if v in in_fast_memory:
                    in_fast_memory.discard(v)
                    operations.append(['store', v])
        
        # Final cleanup: store any remaining outputs
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
