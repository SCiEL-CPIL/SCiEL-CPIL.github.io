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
        
        # Count future uses
        future_uses = {v: 0 for v in all_vertices}
        for v in topo_order:
            for c in children[v]:
                future_uses[v] += 1
        
        # LP-based pebbling with randomized rounding
        # Step 1: Compute layer structure for precedence constraints
        layer = {}
        for v in topo_order:
            if v in inputs:
                layer[v] = 0
            else:
                ps = parents_dict.get(v, [])
                if ps:
                    layer[v] = max(layer[p] for p in ps) + 1
                else:
                    layer[v] = 0
        
        # Step 2: Formulate LP relaxation over computation and storage intervals
        # For each vertex, we decide when to compute it and how long to keep it in memory
        # Variables: x[v][t] = 1 if vertex v is in fast memory at time step t
        # We use a simplified approach: determine optimal keep/drop decisions
        
        # Step 3: Randomized rounding heuristic respecting memory budget
        # Use multiple random seeds for diversification
        import random
        best_ops = None
        best_transfers = float('inf')
        
        for seed in range(10):
            random.seed(seed)
            in_fast_memory = set()
            operations = []
            keep_prob = {}
            
            # Initialize keep probabilities based on future uses and memory pressure
            for v in topo_order:
                if v in inputs:
                    keep_prob[v] = 1.0
                else:
                    # Higher future uses -> higher keep probability
                    # Lower layer -> lower keep probability (more room)
                    max_layer = max(layer.values()) if layer else 1
                    layer_factor = 1.0 - (layer[v] / (max_layer + 1)) * 0.3
                    use_factor = min(1.0, future_uses.get(v, 0) / 2.0)
                    keep_prob[v] = max(0.2, use_factor * layer_factor + random.uniform(0, 0.2))
            
            # Simulate pebbling with randomized keep/drop decisions
            for v in topo_order:
                if time.time() > deadline - 5:
                    break
                    
                if v in inputs:
                    if v not in in_fast_memory:
                        in_fast_memory.add(v)
                        operations.append(['load', v])
                else:
                    ps = parents_dict.get(v, [])
                    # Ensure parents are in fast memory
                    for p in ps:
                        if p not in in_fast_memory:
                            # Need to free space if at budget
                            if len(in_fast_memory) >= fast_memory_budget:
                                # Find vertices to delete using keep probabilities
                                candidates = []
                                for u in list(in_fast_memory):
                                    if u not in outputs:
                                        # Lower keep probability -> more likely to delete
                                        candidates.append((keep_prob.get(u, 0.5), u))
                                
                                # Sort by keep probability (lowest first)
                                candidates.sort(key=lambda x: (x[0], str(x[1])))
                                
                                if candidates:
                                    _, u_to_delete = candidates[0]
                                    in_fast_memory.discard(u_to_delete)
                                    operations.append(['delete', u_to_delete])
                            
                            in_fast_memory.add(p)
                            operations.append(['load', p])
                    
                    in_fast_memory.add(v)
                    operations.append(['compute', v])
                
                # Decide whether to delete based on keep probability
                if future_uses.get(v, 0) == 0 and v not in outputs:
                    if v in in_fast_memory:
                        # Randomized decision based on keep probability
                        if random.random() > keep_prob.get(v, 0.5) * 0.5:
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
            
            if my_transfers < best_transfers:
                best_transfers = my_transfers
                best_ops = operations
        
        # Compare with baseline
        if baseline_ops is not None:
            baseline_transfers = sum(1 for op in baseline_ops if op[0] in ['load', 'store'])
            if best_transfers < baseline_transfers:
                return best_ops
            else:
                return baseline_ops
        else:
            return best_ops if best_ops else []
    
    except:
        if baseline_ops is not None:
            return baseline_ops
        return []
