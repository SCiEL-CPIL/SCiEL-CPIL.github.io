def pebble(dag, fast_memory_budget):
    import time
    import math
    from collections import deque
    
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
        
        # Precompute child positions and max child position
        child_positions = {v: [pos_in_topo[c] for c in children.get(v, [])] for v in topo_order}
        
        child_max_pos = {}
        for v in topo_order:
            if child_positions[v]:
                child_max_pos[v] = max(child_positions[v])
            else:
                child_max_pos[v] = -1
        
        # Precompute future uses count for each vertex
        future_uses = {v: 0 for v in all_vertices}
        for v in topo_order:
            for c in children.get(v, []):
                future_uses[v] += 1
        
        memo = {}
        
        def get_cut_size(start_idx, end_idx, k):
            """Compute cut size at position k for range [start_idx, end_idx)"""
            count = 0
            for v in topo_order[start_idx:k]:
                cmax = child_max_pos[v]
                if cmax >= k and cmax < end_idx:
                    count += 1
            return count
        
        def simulate_linear(start_idx, end_idx, pre_loaded_set):
            """
            Try to pebble the sub-DAG linearly with smart deletion.
            Returns (ops, final_mem, transfer_count, success)
            """
            sub_topo = topo_order[start_idx:end_idx]
            sub_set = set(sub_topo)
            
            in_mem = set(pre_loaded_set)
            simulated_ops = []
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
                            return simulated_ops, in_mem, transfer_count, False
                        in_mem.add(p)
                        simulated_ops.append(['load', p])
                        transfer_count += 1
                
                # Compute v
                in_mem.add(v)
                simulated_ops.append(['compute', v])
                
                # Delete if not needed later
                if last_use_in_sub.get(v, i) < i + 1 and v not in outputs:
                    in_mem.discard(v)
                    simulated_ops.append(['delete', v])
                
                if len(in_mem) > fast_memory_budget:
                    return simulated_ops, in_mem, transfer_count, False
            
            return simulated_ops, in_mem, transfer_count, True
        
        def solve_range(start_idx, end_idx, pre_loaded_set):
            """
            Pebble the sub-DAG defined by topo_order[start_idx:end_idx].
            Returns: (ops, final_mem_set, transfer_count)
            """
            if time.time() > deadline:
                return [], pre_loaded_set, 0
            
            if start_idx >= end_idx:
                return [], pre_loaded_set, 0
            
            # Memo key
            if len(pre_loaded_set) <= 15:
                memo_key = (start_idx, end_idx, tuple(sorted(pre_loaded_set)))
                if memo_key in memo:
                    return memo[memo_key]
            else:
                memo_key = None
            
            range_size = end_idx - start_idx
            
            # Try linear pass first
            ops_lin, mem_lin, tc_lin, fits = simulate_linear(start_idx, end_idx, pre_loaded_set)
            
            if fits:
                result = (ops_lin, mem_lin, tc_lin)
                if memo_key is not None:
                    memo[memo_key] = result
                return result
            
            # Linear doesn't fit, need to split
            best_k = -1
            best_cost = float('inf')
            
            # Generate candidate split points
            if range_size <= 150:
                candidates = list(range(start_idx + 1, end_idx))
            elif range_size <= 600:
                step = max(1, range_size // 40)
                candidates = list(range(start_idx + 1, end_idx, step))
                if candidates[-1] != end_idx - 1:
                    candidates.append(end_idx - 1)
            else:
                step = max(1, range_size // 60)
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
                
                # Weighted interval scheduling-inspired cost model:
                # The cost of a split is dominated by the cut vertices that must be
                # stored/reloaded. We weight by the "interval length" each cut vertex
                # spans, which correlates with recompute savings.
                # Use a smoothed objective: minimize cut_size * (log-left + log-right)
                # with a balance penalty to encourage good recursion depth.
                cut_penalty = cut_size * (math.log(max(2, left_size)) + math.log(max(2, right_size)))
                balance_penalty = abs(left_size - right_size) / max(1, range_size)
                cost = cut_penalty + balance_penalty * 3.0
                
                if cost < best_cost:
                    best_cost = cost
                    best_k = k
            
            # If no good split found with sampled candidates, try finer search around middle
            if best_k == -1:
                mid = (start_idx + end_idx) // 2
                search_lo = max(start_idx + 1, mid - 20)
                search_hi = min(end_idx, mid + 21)
                for k in range(search_lo, search_hi):
                    cut_size = get_cut_size(start_idx, end_idx, k)
                    if cut_size <= fast_memory_budget:
                        left_size = k - start_idx
                        right_size = end_idx - k
                        cut_penalty = cut_size * (math.log(max(2, left_size)) + math.log(max(2, right_size)))
                        balance_penalty = abs(left_size - right_size) / max(1, range_size)
                        cost = cut_penalty + balance_penalty * 3.0
                        if cost < best_cost:
                            best_cost = cost
                            best_k = k
            
            if best_k == -1:
                # No valid split found. Return linear ops (may be invalid) or baseline
                if baseline_ops is not None:
                    return baseline_ops, pre_loaded_set, 0
                return ops_lin, mem_lin, tc_lin
            
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
        
        # Polish pass: remove redundant operations
        polished_ops = []
        for op in ops:
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
