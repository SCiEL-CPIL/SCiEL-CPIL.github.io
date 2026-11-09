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
        
        if len(topo_order) < len(all_vertices):
            if baseline_ops is not None:
                return baseline_ops
            return []
        
        pos_in_topo = {v: i for i, v in enumerate(topo_order)}
        n = len(topo_order)
        
        # Precompute for each vertex: last use position (global)
        last_use = {v: pos_in_topo[v] for v in topo_order}
        for v in topo_order:
            for c in children.get(v, []):
                last_use[v] = max(last_use[v], pos_in_topo[c])
        
        # For the entire DAG, outputs must be kept until the end
        for v in outputs:
            last_use[v] = n - 1
        
        # Memoization for solve_range
        memo = {}
        
        # Precompute child positions for all vertices
        child_positions = {v: [pos_in_topo[c] for c in children.get(v, [])] for v in topo_order}
        
        # Precompute for each vertex: the range of child positions
        child_min_pos = {v: min(child_positions[v]) if child_positions[v] else n for v in topo_order}
        child_max_pos = {v: max(child_positions[v]) if child_positions[v] else -1 for v in topo_order}
        
        def solve_range(start_idx, end_idx, pre_loaded_set):
            """
            Pebble the sub-DAG defined by topo_order[start_idx:end_idx].
            pre_loaded_set: Set of vertices already in fast memory at start.
            Returns: (ops, final_mem_set, transfer_count)
            """
            if start_idx >= end_idx:
                return [], pre_loaded_set, 0
            
            # Create a memo key based on the range and a hash of pre_loaded_set
            # For efficiency, only memoize if pre_loaded_set is small enough
            if len(pre_loaded_set) <= 20:
                memo_key = (start_idx, end_idx, frozenset(pre_loaded_set))
                if memo_key in memo:
                    return memo[memo_key]
            
            sub_topo = topo_order[start_idx:end_idx]
            sub_set = set(sub_topo)
            
            # Determine external inputs and outputs for this sub-range
            ext_inputs = set()
            ext_outputs = set()
            
            for v in sub_topo:
                for p in parents_dict.get(v, []):
                    if p not in sub_set:
                        ext_inputs.add(p)
                for c in children.get(v, []):
                    if c not in sub_set:
                        ext_outputs.add(v)
            
            # Try linear pass first with smart deletion
            in_mem = set(pre_loaded_set)
            simulated_ops = []
            fits = True
            transfer_count = 0
            
            # Precompute last use within sub-range for each vertex
            last_use_in_sub = {}
            for i, v in enumerate(sub_topo):
                last_use_in_sub[v] = i
            for i, v in enumerate(sub_topo):
                for c in children.get(v, []):
                    if c in sub_set:
                        last_use_in_sub[v] = max(last_use_in_sub[v], pos_in_topo[c] - start_idx)
            for v in ext_outputs:
                last_use_in_sub[v] = len(sub_topo) - 1
            
            for i, v in enumerate(sub_topo):
                # Load inputs not in memory
                for p in parents_dict.get(v, []):
                    if p not in in_mem:
                        if len(in_mem) >= fast_memory_budget:
                            fits = False
                            break
                        in_mem.add(p)
                        simulated_ops.append(['load', p])
                        transfer_count += 1
                    if not fits:
                        break
                
                if not fits:
                    break
                
                # Compute v
                in_mem.add(v)
                simulated_ops.append(['compute', v])
                
                # Delete if not needed later in this sub-range and not an ext_output
                if last_use_in_sub.get(v, i) < i + 1:
                    in_mem.discard(v)
                    simulated_ops.append(['delete', v])
                
                if len(in_mem) > fast_memory_budget:
                    fits = False
                    break
            
            if fits:
                result = (simulated_ops, in_mem, transfer_count)
                if len(pre_loaded_set) <= 20:
                    memo[memo_key] = result
                return result
            
            # Linear doesn't fit, need to split
            # Find optimal split point using efficient cut size computation
            
            range_size = end_idx - start_idx
            best_k = -1
            best_cost = float('inf')
            
            # For efficient cut size computation, we can use a sweep approach
            # cut_size(k) = |{v in [start_idx, k) : exists c in children[v] with c in [k, end_idx)}|
            
            # Build candidates
            if range_size <= 200:
                candidates = list(range(start_idx + 1, end_idx))
            elif range_size <= 1000:
                candidates = list(range(start_idx + 1, end_idx, max(1, range_size // 50)))
                if candidates[-1] != end_idx - 1:
                    candidates.append(end_idx - 1)
            else:
                candidates = list(range(start_idx + 1, end_idx, max(1, range_size // 100)))
                if candidates[-1] != end_idx - 1:
                    candidates.append(end_idx - 1)
            
            # For each candidate k, compute cut size efficiently
            # A vertex v in [start_idx, k) contributes to cut_size(k) if it has a child in [k, end_idx)
            # This is equivalent to: child_max_pos[v] >= k and child_min_pos[v] >= start_idx
            
            # Precompute which vertices have children in the right part
            # For a given k: cut_size = count of v in [start_idx, k) where child_max_pos[v] >= k
            
            # We can compute this by iterating through vertices in [start_idx, k)
            # For efficiency, let's just compute directly for each candidate
            
            for k in candidates:
                cut_size = 0
                for v in topo_order[start_idx:k]:
                    if child_max_pos[v] >= k and child_max_pos[v] < end_idx:
                        cut_size += 1
                
                if cut_size > fast_memory_budget:
                    continue
                
                # Better cost estimate: consider memory pressure on both sides
                # Left side: needs to keep cut_size vertices at the end
                # Right side: starts with cut_size vertices pre-loaded
                # Estimate transfers based on segment sizes and cut size
                left_size = k - start_idx
                right_size = end_idx - k
                
                # Heuristic: cost is proportional to cut_size * (1/left_size + 1/right_size)
                # Plus a balance penalty
                balance_penalty = abs(left_size - right_size) / max(1, range_size)
                cut_penalty = cut_size * (1.0 / max(1, left_size) + 1.0 / max(1, right_size))
                cost = cut_penalty * 100 + balance_penalty * 10
                
                if cost < best_cost:
                    best_cost = cost
                    best_k = k
            
            if best_k == -1:
                # No valid split found. Try to find any k with cut_size <= budget
                for k in range(start_idx + 1, end_idx):
                    cut_size = 0
                    for v in topo_order[start_idx:k]:
                        if child_max_pos[v] >= k and child_max_pos[v] < end_idx:
                            cut_size += 1
                    if cut_size <= fast_memory_budget:
                        best_k = k
                        break
            
            if best_k == -1:
                # No valid split at all. Return linear ops (may be invalid) or baseline
                if baseline_ops is not None:
                    return baseline_ops, pre_loaded_set, 0
                return simulated_ops, in_mem, transfer_count
            
            # Recurse
            ops1, mem1, tc1 = solve_range(start_idx, best_k, pre_loaded_set)
            ops2, mem2, tc2 = solve_range(best_k, end_idx, mem1)
            
            result = (ops1 + ops2, mem2, tc1 + tc2)
            
            # Memoize if pre_loaded_set is small
            if len(pre_loaded_set) <= 20:
                memo[memo_key] = result
            
            return result
        
        # Initial call
        ops, final_mem, total_transfers = solve_range(0, n, set())
        
        # Ensure outputs are stored
        for v in outputs:
            if v in final_mem:
                ops.append(['store', v])
                final_mem.discard(v)
                total_transfers += 1
        
        # Check operation limit
        if len(ops) > 20000:
            if baseline_ops is not None:
                return baseline_ops
            return []
        
        # Check time
        if time.time() > deadline:
            if baseline_ops is not None:
                return baseline_ops
            return []
        
        return ops
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        if baseline_ops is not None:
            return baseline_ops
        return []
