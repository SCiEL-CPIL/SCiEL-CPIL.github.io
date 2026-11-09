def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
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
        
        # Precompute child positions
        child_positions = {v: [pos_in_topo[c] for c in children.get(v, [])] for v in topo_order}
        
        # Precompute for each vertex: max child position (for cut size computation)
        child_max_pos = {}
        for v in topo_order:
            if child_positions[v]:
                child_max_pos[v] = max(child_positions[v])
            else:
                child_max_pos[v] = -1
        
        # For efficient cut size computation:
        # cut_size(k) = |{v in [0, k) : child_max_pos[v] >= k}|
        # We can precompute an array where for each position p, we count how many vertices
        # with child_max_pos[v] >= p are in [0, k) for various k.
        
        # Alternative: For a given range [start, end), cut_size(k) for k in (start, end)
        # = |{v in [start, k) : child_max_pos[v] >= k and child_max_pos[v] < end}|
        
        # Let's precompute for each vertex v, the range of k values where v contributes to cut_size
        # v contributes to cut_size(k) if start <= v < k <= child_max_pos[v] < end
        
        # For simplicity and speed, we'll compute cut sizes on the fly but with early termination
        
        memo = {}
        
        def get_cut_size(start_idx, end_idx, k):
            """Compute cut size at position k for range [start_idx, end_idx)"""
            count = 0
            for v in topo_order[start_idx:k]:
                cmax = child_max_pos[v]
                if cmax >= k and cmax < end_idx:
                    count += 1
            return count
        
        def solve_range(start_idx, end_idx, pre_loaded_set):
            """
            Pebble the sub-DAG defined by topo_order[start_idx:end_idx].
            Returns: (ops, final_mem_set, transfer_count)
            """
            if time.time() > deadline:
                return [], pre_loaded_set, 0
            
            if start_idx >= end_idx:
                return [], pre_loaded_set, 0
            
            # Memo key - use a simpler hash for speed
            if len(pre_loaded_set) <= 15:
                memo_key = (start_idx, end_idx, tuple(sorted(pre_loaded_set)))
                if memo_key in memo:
                    return memo[memo_key]
            else:
                memo_key = None
            
            sub_topo = topo_order[start_idx:end_idx]
            sub_set = set(sub_topo)
            range_size = end_idx - start_idx
            
            # Try linear pass first with smart deletion
            in_mem = set(pre_loaded_set)
            simulated_ops = []
            fits = True
            transfer_count = 0
            
            # Precompute last use within sub-range
            last_use_in_sub = {}
            for i, v in enumerate(sub_topo):
                last_use_in_sub[v] = i
            for i, v in enumerate(sub_topo):
                for c in children.get(v, []):
                    if c in sub_set:
                        last_use_in_sub[v] = max(last_use_in_sub[v], pos_in_topo[c] - start_idx)
            
            # Outputs must be kept until end
            for v in outputs:
                if v in sub_set:
                    last_use_in_sub[v] = len(sub_topo) - 1
            
            for i, v in enumerate(sub_topo):
                # Load parents not in memory
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
                
                # Delete if not needed later
                if last_use_in_sub.get(v, i) < i + 1 and v not in outputs:
                    in_mem.discard(v)
                    simulated_ops.append(['delete', v])
                
                if len(in_mem) > fast_memory_budget:
                    fits = False
                    break
            
            if fits:
                result = (simulated_ops, in_mem, transfer_count)
                if memo_key is not None:
                    memo[memo_key] = result
                return result
            
            # Linear doesn't fit, need to split
            # Find optimal split point
            best_k = -1
            best_cost = float('inf')
            
            # Generate candidate split points
            if range_size <= 100:
                candidates = list(range(start_idx + 1, end_idx))
            elif range_size <= 500:
                step = max(1, range_size // 30)
                candidates = list(range(start_idx + 1, end_idx, step))
                if candidates[-1] != end_idx - 1:
                    candidates.append(end_idx - 1)
            else:
                step = max(1, range_size // 50)
                candidates = list(range(start_idx + 1, end_idx, step))
                if candidates[-1] != end_idx - 1:
                    candidates.append(end_idx - 1)
            
            # For each candidate, compute cut size and estimate cost
            for k in candidates:
                cut_size = get_cut_size(start_idx, end_idx, k)
                
                if cut_size > fast_memory_budget:
                    continue
                
                left_size = k - start_idx
                right_size = end_idx - k
                
                if left_size == 0 or right_size == 0:
                    continue
                
                # Better cost estimate: 
                # Each vertex in the cut will be loaded/stored at least once per side
                # Estimated transfers ~ cut_size * (1 + log(left_size) + log(right_size))
                # Plus overhead for the sub-problems
                import math
                cut_penalty = cut_size * (math.log(max(2, left_size)) + math.log(max(2, right_size)))
                balance_penalty = abs(left_size - right_size) / max(1, range_size)
                cost = cut_penalty + balance_penalty * 5
                
                if cost < best_cost:
                    best_cost = cost
                    best_k = k
            
            # If no good split found with sampled candidates, try finer search around best
            if best_k == -1:
                # Try all positions if range is small enough
                if range_size <= 200:
                    for k in range(start_idx + 1, end_idx):
                        cut_size = get_cut_size(start_idx, end_idx, k)
                        if cut_size <= fast_memory_budget:
                            left_size = k - start_idx
                            right_size = end_idx - k
                            import math
                            cut_penalty = cut_size * (math.log(max(2, left_size)) + math.log(max(2, right_size)))
                            balance_penalty = abs(left_size - right_size) / max(1, range_size)
                            cost = cut_penalty + balance_penalty * 5
                            if cost < best_cost:
                                best_cost = cost
                                best_k = k
                else:
                    # Try a few more candidates near the middle
                    mid = (start_idx + end_idx) // 2
                    for k in range(max(start_idx + 1, mid - 10), min(end_idx, mid + 11)):
                        cut_size = get_cut_size(start_idx, end_idx, k)
                        if cut_size <= fast_memory_budget:
                            left_size = k - start_idx
                            right_size = end_idx - k
                            import math
                            cut_penalty = cut_size * (math.log(max(2, left_size)) + math.log(max(2, right_size)))
                            balance_penalty = abs(left_size - right_size) / max(1, range_size)
                            cost = cut_penalty + balance_penalty * 5
                            if cost < best_cost:
                                best_cost = cost
                                best_k = k
            
            if best_k == -1:
                # No valid split found. Return linear ops (may be invalid) or baseline
                if baseline_ops is not None:
                    return baseline_ops, pre_loaded_set, 0
                return simulated_ops, in_mem, transfer_count
            
            # Recurse
            ops1, mem1, tc1 = solve_range(start_idx, best_k, pre_loaded_set)
            ops2, mem2, tc2 = solve_range(best_k, end_idx, mem1)
            
            result = (ops1 + ops2, mem2, tc1 + tc2)
            
            if memo_key is not None:
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
        
        # Polish pass: try to reduce transfers by merging consecutive loads/stores
        # and removing redundant operations
        polished_ops = []
        for op in ops:
            # Skip redundant delete/load pairs
            if polished_ops and polished_ops[-1] == ['delete', op[1]] and op[0] == 'load':
                continue
            if polished_ops and polished_ops[-1] == ['load', op[1]] and op[0] == 'delete':
                polished_ops.pop()
                continue
            polished_ops.append(op)
        
        if len(polished_ops) <= 20000:
            ops = polished_ops
        
        return ops
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        if baseline_ops is not None:
            return baseline_ops
        return []
