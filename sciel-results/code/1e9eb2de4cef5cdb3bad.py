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
        
        def solve_range(start_idx, end_idx, pre_loaded_set):
            """
            Pebble the sub-DAG defined by topo_order[start_idx:end_idx].
            pre_loaded_set: Set of vertices already in fast memory at start.
            Returns: (ops, final_mem_set, transfer_count)
            """
            if start_idx >= end_idx:
                return [], pre_loaded_set, 0
            
            # Check memo - but pre_loaded_set matters, so we can't simply memoize
            # For now, no memoization due to varying pre_loaded_set
            
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
            
            # Try linear pass first
            in_mem = set(pre_loaded_set)
            simulated_ops = []
            fits = True
            transfer_count = 0
            
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
                needed_later = False
                for c in children.get(v, []):
                    if c in sub_set and pos_in_topo[c] > i:
                        needed_later = True
                        break
                if v in ext_outputs:
                    needed_later = True
                
                if not needed_later:
                    in_mem.discard(v)
                    simulated_ops.append(['delete', v])
                
                if len(in_mem) > fast_memory_budget:
                    fits = False
                    break
            
            if fits:
                return simulated_ops, in_mem, transfer_count
            
            # Linear doesn't fit, need to split
            # Find optimal split point
            # Cut at k: vertices in [start_idx, k) that have children in [k, end_idx)
            
            # Precompute for efficiency: for each k, the cut set
            # We want to minimize total cost: cost_left + cost_right + cut_transfer_cost
            
            # For a given split k:
            # - Left: [start_idx, k), pre_loaded = pre_loaded_set
            # - Right: [k, end_idx), pre_loaded = cut_vertices (those kept from left)
            # - Transfer cost: number of cut vertices that need to be loaded for right side
            #   (they're already in memory from left, so no load needed if we keep them)
            #   Actually, the cut vertices are kept in memory, so no extra transfer.
            #   The transfer cost is just the sum of transfers in left and right.
            
            # But wait: the cut vertices must be kept in memory at the end of left.
            # So the left problem has ext_outputs = cut_vertices.
            # The right problem has pre_loaded = cut_vertices.
            
            # To minimize total transfers, we want to find k that minimizes:
            #   transfers_left(k) + transfers_right(k)
            
            # This is complex. Let's use a heuristic: find k that minimizes cut size,
            # but also consider the balance.
            
            # For layered DAGs, the cut size varies. Let's find all valid k (cut_size <= budget)
            # and pick the one that minimizes a cost estimate.
            
            # Cost estimate: cut_size * 2 (rough estimate of transfers per cut vertex)
            # Plus imbalance penalty.
            
            best_k = -1
            best_cost = float('inf')
            
            # To speed up, sample k values or use a smarter search
            # For now, try all k if range is small, else sample
            
            range_size = end_idx - start_idx
            if range_size <= 500:
                candidates = list(range(start_idx + 1, end_idx))
            else:
                # Sample candidates
                candidates = list(range(start_idx + 1, end_idx, max(1, range_size // 100)))
                if candidates[-1] != end_idx - 1:
                    candidates.append(end_idx - 1)
            
            # Precompute: for each vertex v, which children are in which range
            # This is expensive, so let's compute cut_size incrementally
            
            # Alternative: just compute cut_size for each candidate k
            # cut_size(k) = |{v in [start_idx, k) : exists c in children[v] with c in [k, end_idx)}|
            
            # For efficiency, precompute for each v the set of children positions
            child_positions = {v: [pos_in_topo[c] for c in children.get(v, [])] for v in topo_order}
            
            for k in candidates:
                # Compute cut size
                cut_size = 0
                for v in topo_order[start_idx:k]:
                    for cp in child_positions[v]:
                        if cp >= k and cp < end_idx:
                            cut_size += 1
                            break
                
                if cut_size > fast_memory_budget:
                    continue
                
                # Cost estimate: cut_size is a proxy for memory pressure
                # Smaller cut = less memory needed, potentially fewer transfers
                # Also prefer balanced splits
                balance_penalty = abs(k - start_idx - (end_idx - k))
                cost = cut_size * 10 + balance_penalty
                
                if cost < best_cost:
                    best_cost = cost
                    best_k = k
            
            if best_k == -1:
                # No valid split found. Try to find any k with cut_size <= budget
                for k in range(start_idx + 1, end_idx):
                    cut_size = 0
                    for v in topo_order[start_idx:k]:
                        for cp in child_positions[v]:
                            if cp >= k and cp < end_idx:
                                cut_size += 1
                                break
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
            
            return ops1 + ops2, mem2, tc1 + tc2
        
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
