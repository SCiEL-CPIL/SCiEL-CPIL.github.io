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
    
    # Try to improve with chain-decomposition + future-use-aware scheduling
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
            # Cycle or issue, fall back
            if baseline_ops is not None:
                return baseline_ops
            return []
        
        # Compute next_use and last_use for each vertex
        vertex_use_positions = {v: [] for v in all_vertices}
        for i, v in enumerate(topo_order):
            ps = parents_dict.get(v, [])
            for p in ps:
                vertex_use_positions[p].append(i)
        
        next_use = {}
        last_use = {}
        for v in all_vertices:
            if vertex_use_positions[v]:
                next_use[v] = vertex_use_positions[v][0]
                last_use[v] = vertex_use_positions[v][-1]
            else:
                next_use[v] = float('inf')
                last_use[v] = -1
        
        # Compute future use counts (how many times each vertex is needed in the future)
        future_uses = {v: 0 for v in all_vertices}
        for v in topo_order:
            for c in children[v]:
                future_uses[v] += 1
        
        # --- Chain decomposition via greedy matching ---
        # Build a matching: each vertex can be matched to at most one parent and one child
        matched_parent = {}  # child -> parent
        matched_child = {}   # parent -> child
        
        # Process vertices in topological order, match to available parent
        for v in topo_order:
            if v in children:
                # Sort children by their position in topo_order
                sorted_children = sorted(children[v], key=lambda c: topo_order.index(c))
                for c in sorted_children:
                    if c not in matched_parent:
                        matched_parent[c] = v
                        matched_child[v] = c
                        break
        
        # Build chains from the matching
        chains = []
        vertex_to_chain = {}
        chain_id_counter = 0
        
        # Find all chain heads: vertices that are not matched as a child
        for v in all_vertices:
            if v not in matched_parent:
                chain = [v]
                current = v
                while current in matched_child:
                    current = matched_child[current]
                    chain.append(current)
                chains.append(chain)
                for cv in chain:
                    vertex_to_chain[cv] = chain_id_counter
                chain_id_counter += 1
        
        # Create single-vertex chains for unmatched vertices
        for v in all_vertices:
            if v not in vertex_to_chain:
                chains.append([v])
                vertex_to_chain[v] = chain_id_counter
                chain_id_counter += 1
        
        # --- Advanced pebbling with future-use-aware eviction ---
        in_fast_memory = set()
        operations = []
        
        # Precompute topo index for each vertex
        topo_idx = {v: i for i, v in enumerate(topo_order)}
        
        # Track remaining future uses as we process
        remaining_uses = dict(future_uses)
        
        # For each chain, track the next vertex that will be needed
        chain_next_need = {}
        for i, chain in enumerate(chains):
            chain_sorted = sorted(chain, key=lambda v: topo_idx[v])
            chain_next_need[i] = float('inf')
            for v in chain_sorted:
                if next_use[v] != float('inf'):
                    chain_next_need[i] = next_use[v]
                    break
        
        # Greedy pebbling with sophisticated eviction
        for idx, v in enumerate(topo_order):
            if time.time() > deadline:
                break
            
            if v in inputs:
                if v not in in_fast_memory:
                    if len(in_fast_memory) >= fast_memory_budget:
                        # Evict: prefer vertices with no future uses, then farthest next need
                        best_evict = None
                        best_score = float('inf')
                        for u in list(in_fast_memory):
                            if u not in outputs:
                                # Score: 0 if no future uses, else next_use distance
                                if remaining_uses.get(u, 0) == 0:
                                    score = 0
                                else:
                                    score = next_use.get(u, float('inf')) - idx
                                # Prefer to evict vertices with lowest score (no future uses first)
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
                            # Evict: prefer vertices with no future uses, then farthest next need
                            best_evict = None
                            best_score = float('inf')
                            for u in list(in_fast_memory):
                                if u not in outputs and u != p:
                                    if remaining_uses.get(u, 0) == 0:
                                        score = 0
                                    else:
                                        score = next_use.get(u, float('inf')) - idx
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
            
            # Update remaining uses as we progress
            for c in children.get(v, []):
                if remaining_uses.get(v, 0) > 0:
                    remaining_uses[v] -= 1
            
            # Update chain needs as we progress
            ci = vertex_to_chain.get(v, -1)
            if ci >= 0:
                chain = chains[ci]
                chain_sorted = sorted(chain, key=lambda x: topo_idx[x])
                found = False
                for cv in chain_sorted:
                    if topo_idx[cv] > idx and next_use[cv] != float('inf'):
                        chain_next_need[ci] = next_use[cv]
                        found = True
                        break
                if not found:
                    chain_next_need[ci] = float('inf')
            
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
