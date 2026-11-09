def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit

    # Fallback to baseline
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        baseline_ops = pebble_baseline(dag, fast_memory_budget)
    except Exception:
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
        
        # Precompute children positions for fast cut size calculation
        child_positions = {}
        for v in topo_order:
            child_positions[v] = [pos_in_topo[c] for c in children.get(v, [])]

        # Precompute last use for each vertex in the global topo order
        # This helps in the linear pass to know when to delete
        last_use = {v: pos_in_topo[v] for v in topo_order}
        for v in topo_order:
            for c in children.get(v, []):
                last_use[v] = max(last_use[v], pos_in_topo[c])
        for v in outputs:
            last_use[v] = n - 1

        # Memoization for solve_range
        # Key: (start_idx, end_idx, frozenset(pre_loaded_set))
        # Note: pre_loaded_set can be large, so we use a hash of the sorted tuple
        memo = {}

        def solve_range(start_idx, end_idx, pre_loaded_set):
            """
            Pebble the sub-DAG defined by topo_order[start_idx:end_idx].
            pre_loaded_set: Set of vertices already in fast memory at start.
            Returns: (ops, final_mem_set, transfer_count)
            """
            if time.time() > deadline:
                return [], pre_loaded_set, 0 # Timeout guard inside recursion

            if start_idx >= end_idx:
                return [], pre_loaded_set, 0

            key = (start_idx, end_idx, tuple(sorted(pre_loaded_set)))
            if key in memo:
                return memo[key]

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
                result = (simulated_ops, in_mem, transfer_count)
                memo[key] = result
                return result

            # Linear doesn't fit, need to split
            # Find optimal split point k
            # We want to minimize total cost: cost_left + cost_right
            # Heuristic: Minimize cut_size, then balance sizes
            
            best_k = -1
            best_cost = float('inf')
            
            range_size = end_idx - start_idx
            
            # To ensure we find a good split, we scan candidates.
            # If range is small, scan all. If large, scan more densely.
            step = 1 if range_size <= 100 else max(1, range_size // 200)
            
            candidates = list(range(start_idx + 1, end_idx, step))
            if candidates[-1] != end_idx - 1:
                candidates.append(end_idx - 1)

            for k in candidates:
                # Compute cut size: vertices in [start, k) that have children in [k, end)
                cut_size = 0
                for v in topo_order[start_idx:k]:
                    for cp in child_positions[v]:
                        if cp >= k and cp < end_idx:
                            cut_size += 1
                            break
                
                if cut_size > fast_memory_budget:
                    continue
                
                # Cost heuristic:
                # 1. Cut size is the primary driver of memory pressure and potential re-computation if not managed well.
                # 2. Balance is secondary.
                left_size = k - start_idx
                right_size = end_idx - k
                balance_penalty = abs(left_size - right_size)
                
                # We weight cut size heavily because a larger cut means more vertices are "active" 
                # at the boundary, which might force more loads/stores if the subproblems 
                # are not perfectly aligned.
                cost = cut_size * 100 + balance_penalty
                
                if cost < best_cost:
                    best_cost = cost
                    best_k = k

            if best_k == -1:
                # No valid split found with the sampled candidates.
                # Try a full scan if range is not too huge
                if range_size <= 2000:
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
                    # Fallback: return linear ops (which might be invalid if it didn't fit, 
                    # but we have no choice) or baseline
                    if baseline_ops is not None:
                        return baseline_ops, pre_loaded_set, 0
                    return simulated_ops, in_mem, transfer_count

            # Recurse
            ops1, mem1, tc1 = solve_range(start_idx, best_k, pre_loaded_set)
            ops2, mem2, tc2 = solve_range(best_k, end_idx, mem1)
            
            result = (ops1 + ops2, mem2, tc1 + tc2)
            memo[key] = result
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
