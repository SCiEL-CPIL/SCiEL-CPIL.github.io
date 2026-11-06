# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque, defaultdict

_DEADLINE_SECONDS = 85.0


def _toposort_vertices(dag):
    if 'vertices' in dag:
        vertices = list(dag['vertices'])
    elif 'nodes' in dag:
        vertices = list(dag['nodes'])
    else:
        all_v = set()
        parents = dag.get('parents', {})
        for v, ps in parents.items():
            all_v.add(v)
            for p in ps:
                all_v.add(p)
        vertices = sorted(all_v)

    if 'parents' in dag:
        parents = {k: list(v) for k, v in dag['parents'].items()}
    elif 'edges' in dag:
        parents = {}
        for edge in dag['edges']:
            if isinstance(edge, (list, tuple)) and len(edge) == 2:
                p, c = edge
                parents.setdefault(c, []).append(p)
    else:
        parents = {}

    if 'outputs' in dag:
        outputs = set(dag['outputs'])
    elif 'output' in dag:
        outputs = set(dag['output']) if isinstance(dag['output'], (list, set, tuple)) else {dag['output']}
    else:
        children = {}
        for v, ps in parents.items():
            for p in ps:
                children.setdefault(p, []).append(v)
        outputs = {v for v in vertices if v not in children}

    if 'inputs' in dag:
        inputs = set(dag['inputs'])
    elif 'blue' in dag:
        inputs = set(dag['blue'])
    elif 'blue_vertices' in dag:
        inputs = set(dag['blue_vertices'])
    else:
        inputs = {v for v in vertices if not parents.get(v)}

    return vertices, parents, outputs, inputs


def _validate_ops(ops, dag, fast_memory_budget):
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    blue = set(inputs)
    fast_mem = set()
    available = set(inputs)

    for i, (op, v) in enumerate(ops):
        if op == 'load':
            if v not in inputs:
                return False
            if v in fast_mem:
                return False
            if len(fast_mem) >= fast_memory_budget:
                return False
            fast_mem.add(v)
            available.add(v)
        elif op == 'store':
            if v not in fast_mem and v not in blue:
                return False
            if v in fast_mem:
                fast_mem.remove(v)
            blue.add(v)
        elif op == 'compute':
            ps = parents.get(v, [])
            for p in ps:
                if p not in available:
                    return False
            if len(fast_mem) >= fast_memory_budget:
                return False
            fast_mem.add(v)
            available.add(v)
        elif op == 'delete':
            if v not in fast_mem:
                return False
            fast_mem.remove(v)
            if v not in blue:
                available.discard(v)
        else:
            return False

    if not outputs.issubset(blue):
        return False

    return True


def _count_transfers(ops):
    """Count load+store operations (the metric that matters)."""
    return sum(1 for op, v in ops if op in ('load', 'store'))


def _dijkstra_pebble_bitmask(dag, fast_memory_budget, time_limit=30.0):
    """A* over bitmask states for small/medium instances with heuristic pruning."""
    start_time = time.time()
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    n = len(vertices)
    
    if n > 22:
        return None

    v_to_idx = {v: i for i, v in enumerate(vertices)}
    idx_to_v = {i: v for i, v in enumerate(vertices)}

    inputs_mask = 0
    for v in inputs:
        inputs_mask |= (1 << v_to_idx[v])
    
    outputs_mask = 0
    for v in outputs:
        outputs_mask |= (1 << v_to_idx[v])
    
    computed_mask = (~inputs_mask) & ((1 << n) - 1)

    parent_masks = []
    for v in vertices:
        ps = parents.get(v, [])
        mask = 0
        for p in ps:
            mask |= (1 << v_to_idx[p])
        parent_masks.append(mask)

    # Precompute for each vertex: how many children it has
    children_count = [0] * n
    for v_idx, v in enumerate(vertices):
        for c in vertices:
            if v in parents.get(c, []):
                children_count[v_idx] += 1

    init_blue = inputs_mask
    init_fast = 0
    init_state = (init_blue, init_fast)

    def is_goal(blue, fast):
        return (blue & outputs_mask) == outputs_mask

    if is_goal(init_blue, init_fast):
        return []

    # Heuristic: estimate remaining transfers needed
    def heuristic(blue, fast):
        # Count outputs not yet in blue
        missing_outputs = outputs_mask & ~blue
        # Each missing output needs at least one transfer (store)
        h = missing_outputs.bit_count()
        # Also account for inputs that need to be loaded if not in fast and not in blue
        # But this is complex; keep it simple
        return h

    best_g = {}
    best_g[init_state] = 0
    parent_map = {}
    h0 = heuristic(init_blue, init_fast)
    pq = [(h0, 0, 0, init_state)]  # (f, g, counter, state)

    max_states = 200000
    states_explored = 0
    best_found = None
    best_found_cost = float('inf')
    counter = 0

    while pq:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        if states_explored >= max_states:
            break

        f, g, _, state = heapq.heappop(pq)
        if g > best_g.get(state, float('inf')):
            continue

        states_explored += 1
        blue, fast = state

        if is_goal(blue, fast):
            if g < best_found_cost:
                best_found_cost = g
                best_found = state
                break

        available = blue | fast
        h = heuristic(blue, fast)

        # 1. load(v)
        load_candidates = inputs_mask & ~fast
        while load_candidates:
            bit = load_candidates & (-load_candidates)
            v_idx = bit.bit_length() - 1
            load_candidates ^= bit
            new_fast = fast | bit
            if new_fast.bit_count() <= fast_memory_budget:
                new_state = (blue, new_fast)
                new_g = g + 1
                if new_g < best_g.get(new_state, float('inf')):
                    best_g[new_state] = new_g
                    parent_map[new_state] = (state, 'load', idx_to_v[v_idx])
                    counter += 1
                    heapq.heappush(pq, (new_g + heuristic(blue, new_fast), new_g, counter, new_state))

        # 2. store(v) - only if it helps (reduces missing outputs or frees space)
        store_candidates = fast & outputs_mask  # Only store outputs
        while store_candidates:
            bit = store_candidates & (-store_candidates)
            v_idx = bit.bit_length() - 1
            store_candidates ^= bit
            new_fast = fast & ~bit
            new_blue = blue | bit
            new_state = (new_blue, new_fast)
            new_g = g + 1
            if new_g < best_g.get(new_state, float('inf')):
                best_g[new_state] = new_g
                parent_map[new_state] = (state, 'store', idx_to_v[v_idx])
                counter += 1
                heapq.heappush(pq, (new_g + heuristic(new_blue, new_fast), new_g, counter, new_state))

        # 3. compute(v)
        if fast.bit_count() < fast_memory_budget:
            compute_candidates = computed_mask & ~fast
            while compute_candidates:
                bit = compute_candidates & (-compute_candidates)
                v_idx = bit.bit_length() - 1
                compute_candidates ^= bit
                pm = parent_masks[v_idx]
                if (pm & ~available) == 0:
                    new_fast = fast | bit
                    new_state = (blue, new_fast)
                    new_g = g
                    if new_g < best_g.get(new_state, float('inf')):
                        best_g[new_state] = new_g
                        parent_map[new_state] = (state, 'compute', idx_to_v[v_idx])
                        counter += 1
                        heapq.heappush(pq, (new_g + heuristic(blue, new_fast), new_g, counter, new_state))

        # 4. delete(v) - only if it's not an output that's already stored
        delete_candidates = fast & ~(outputs_mask & blue)  # Don't delete outputs already in blue
        while delete_candidates:
            bit = delete_candidates & (-delete_candidates)
            v_idx = bit.bit_length() - 1
            delete_candidates ^= bit
            new_fast = fast & ~bit
            new_state = (blue, new_fast)
            new_g = g
            if new_g < best_g.get(new_state, float('inf')):
                best_g[new_state] = new_g
                parent_map[new_state] = (state, 'delete', idx_to_v[v_idx])
                counter += 1
                heapq.heappush(pq, (new_g + heuristic(blue, new_fast), new_g, counter, new_state))

    if best_found is not None:
        path = []
        cur = best_found
        while cur != init_state:
            prev, op, v = parent_map[cur]
            path.append((op, v))
            cur = prev
        path.reverse()
        return path

    return None


def _local_search_polish(ops, dag, fast_memory_budget, time_limit=5.0):
    """Reduce transfers by removing redundant operations and reordering."""
    start_time = time.time()
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    
    if not ops:
        return ops
    
    improved = True
    max_iterations = 10
    iteration = 0
    while improved and (time.time() - start_time) < time_limit and iteration < max_iterations:
        improved = False
        iteration += 1
        new_ops = list(ops)
        
        # Pattern 1: delete(v) immediately followed by load(v) where v is an input
        i = 0
        while i < len(new_ops) - 1:
            op1, v1 = new_ops[i]
            op2, v2 = new_ops[i+1]
            if op1 == 'delete' and op2 == 'load' and v1 == v2 and v1 in inputs:
                del new_ops[i]
                del new_ops[i]
                improved = True
                continue
            i += 1
        
        # Pattern 2: load(v) immediately followed by delete(v) where v is an input
        # and v is not needed later in the remaining ops
        i = 0
        while i < len(new_ops) - 1:
            op1, v1 = new_ops[i]
            op2, v2 = new_ops[i+1]
            if op1 == 'load' and op2 == 'delete' and v1 == v2 and v1 in inputs:
                # Check if v is used later (as a parent for any compute)
                remaining_ops = new_ops[i+2:]
                v_used_later = False
                for op, vtx in remaining_ops:
                    if op == 'compute':
                        if v1 in parents.get(vtx, []):
                            v_used_later = True
                            break
                    elif op == 'load' and vtx == v1:
                        v_used_later = True
                        break
                
                if not v_used_later:
                    del new_ops[i]
                    del new_ops[i]
                    improved = True
                    continue
            i += 1
        
        # Pattern 3: Remove trailing deletes that aren't needed
        last_store_idx = -1
        for i, (op, v) in enumerate(new_ops):
            if op == 'store':
                last_store_idx = i
        
        if last_store_idx >= 0 and last_store_idx < len(new_ops) - 1:
            # Remove all deletes after the last store
            new_ops = new_ops[:last_store_idx+1]
            improved = True
        
        # Pattern 4: Merge consecutive loads of the same input
        i = 0
        while i < len(new_ops) - 1:
            op1, v1 = new_ops[i]
            op2, v2 = new_ops[i+1]
            if op1 == 'load' and op2 == 'load' and v1 == v2:
                del new_ops[i]
                improved = True
                continue
            i += 1
        
        # Pattern 5: Remove redundant stores (store same vertex twice)
        seen_stores = set()
        new_ops2 = []
        for op, v in new_ops:
            if op == 'store':
                if v in seen_stores:
                    improved = True
                    continue
                seen_stores.add(v)
            new_ops2.append((op, v))
        new_ops = new_ops2
        
        ops = new_ops
    
    # Final validation and cleanup
    if not _validate_ops(ops, dag, fast_memory_budget):
        # Return original if polished version is invalid
        return ops
    
    return ops


def _greedy_pebble_improved(dag, fast_memory_budget):
    """Improved greedy with lookahead-based eviction policy."""
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    
    children = {}
    for v in vertices:
        for p in parents.get(v, []):
            children.setdefault(p, []).append(v)
    
    v_to_pos = {v: i for i, v in enumerate(vertices)}
    
    # Precompute: for each vertex, how many times it will be needed in the future
    # This is the number of children that haven't been processed yet
    remaining_children = {v: 0 for v in vertices}
    for v in vertices:
        for c in children.get(v, []):
            if v_to_pos.get(c, 0) > v_to_pos[v]:
                remaining_children[v] += 1

    ops = []
    fast_mem = set()
    blue = set(inputs)
    available = set(inputs)
    
    # Track which vertices are "active" (needed soon)
    # A vertex is active if it has remaining children
    def get_eviction_priority(v):
        """Lower priority = more likely to be evicted.
        
        Priority factors:
        1. remaining_uses: fewer uses = better to evict
        2. is_input: inputs can be reloaded (cheaper to lose), computed values must be recomputed
        3. is_output: outputs should be kept if possible
        4. distance_to_next_use: closer next use = keep longer
        """
        rc = remaining_children.get(v, 0)
        is_input = 1 if v in inputs else 0
        is_output = 1 if v in outputs else 0
        
        # Find distance to next use (smallest position among remaining children)
        next_use_dist = float('inf')
        for c in children.get(v, []):
            if v_to_pos.get(c, 0) > v_to_pos.get(v, 0):
                dist = v_to_pos[c] - v_to_pos[v]
                next_use_dist = min(next_use_dist, dist)
        
        # Priority: we want to evict vertices with:
        # - Low remaining uses
        # - Are inputs (reloadable)
        # - Are not outputs
        # - Have far next use
        # So lower tuple = better to evict
        return (rc, is_input, -is_output, next_use_dist)
    
    for v in vertices:
        if v in inputs:
            if v not in available:
                if len(fast_mem) >= fast_memory_budget:
                    best_evict = None
                    best_priority = None
                    for fm in fast_mem:
                        pr = get_eviction_priority(fm)
                        if best_priority is None or pr < best_priority:
                            best_priority = pr
                            best_evict = fm
                    
                    if best_evict is not None:
                        ops.append(('delete', best_evict))
                        fast_mem.discard(best_evict)
                        if best_evict not in blue:
                            available.discard(best_evict)
                
                if len(fast_mem) < fast_memory_budget:
                    ops.append(('load', v))
                    fast_mem.add(v)
                    available.add(v)
        else:
            ps = parents.get(v, [])
            
            # Check if all parents are available
            parents_available = all(p in available for p in ps)
            
            if not parents_available:
                # Need to load missing parents
                for p in ps:
                    if p not in available:
                        if p in inputs:
                            if len(fast_mem) >= fast_memory_budget:
                                best_evict = None
                                best_priority = None
                                for fm in fast_mem:
                                    pr = get_eviction_priority(fm)
                                    if best_priority is None or pr < best_priority:
                                        best_priority = pr
                                        best_evict = fm
                                
                                if best_evict is not None:
                                    ops.append(('delete', best_evict))
                                    fast_mem.discard(best_evict)
                                    if best_evict not in blue:
                                        available.discard(best_evict)
                            
                            if len(fast_mem) < fast_memory_budget:
                                ops.append(('load', p))
                                fast_mem.add(p)
                                available.add(p)
            
            if len(fast_mem) >= fast_memory_budget:
                best_evict = None
                best_priority = None
                for fm in fast_mem:
                    pr = get_eviction_priority(fm)
                    if best_priority is None or pr < best_priority:
                        best_priority = pr
                        best_evict = fm
                
                if best_evict is not None:
                    ops.append(('delete', best_evict))
                    fast_mem.discard(best_evict)
                    if best_evict not in blue:
                        available.discard(best_evict)
            
            if len(fast_mem) < fast_memory_budget:
                ops.append(('compute', v))
                fast_mem.add(v)
                available.add(v)
        
        for p in parents.get(v, []):
            remaining_children[p] = remaining_children.get(p, 0) - 1
        
        if v in outputs and v in fast_mem:
            ops.append(('store', v))
            fast_mem.discard(v)
            blue.add(v)
    
    for v in list(fast_mem):
        ops.append(('delete', v))
        fast_mem.discard(v)
    
    return ops


def _layered_greedy_pebble(dag, fast_memory_budget):
    """Layer-based greedy as fallback."""
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    
    depth = {}
    for v in vertices:
        ps = parents.get(v, [])
        if not ps:
            depth[v] = 0
        else:
            depth[v] = max(depth.get(p, 0) for p in ps) + 1
    
    max_depth = max(depth.values()) if depth else 0
    layers = {}
    for v in vertices:
        layers.setdefault(depth[v], []).append(v)
    
    ops = []
    fast_mem = set()
    blue = set(inputs)
    available = set(inputs)
    remaining_outputs = set(outputs)
    
    for d in range(max_depth + 1):
        layer_vertices = layers.get(d, [])
        for v in layer_vertices:
            if v in inputs:
                if v not in fast_mem:
                    if len(fast_mem) >= fast_memory_budget:
                        to_delete = None
                        for fm in fast_mem:
                            if fm not in outputs and fm not in remaining_outputs:
                                to_delete = fm
                                break
                        if to_delete is None and fast_mem:
                            to_delete = next(iter(fast_mem))
                        if to_delete:
                            ops.append(('delete', to_delete))
                            fast_mem.discard(to_delete)
                            if to_delete not in blue:
                                available.discard(to_delete)
                    if len(fast_mem) < fast_memory_budget:
                        ops.append(('load', v))
                        fast_mem.add(v)
                        available.add(v)
            else:
                ps = parents.get(v, [])
                if len(fast_mem) >= fast_memory_budget:
                    to_delete = None
                    for fm in fast_mem:
                        if fm not in outputs and fm not in remaining_outputs:
                            to_delete = fm
                            break
                    if to_delete is None and fast_mem:
                        to_delete = next(iter(fast_mem))
                    if to_delete:
                        ops.append(('delete', to_delete))
                        fast_mem.discard(to_delete)
                        if to_delete not in blue:
                            available.discard(to_delete)
                if len(fast_mem) < fast_memory_budget:
                    ops.append(('compute', v))
                    fast_mem.add(v)
                    available.add(v)
        
        for v in layer_vertices:
            if v in outputs and v in fast_mem:
                ops.append(('store', v))
                fast_mem.discard(v)
                blue.add(v)
                remaining_outputs.discard(v)
    
    for v in remaining_outputs:
        if v in fast_mem:
            ops.append(('store', v))
            fast_mem.discard(v)
            blue.add(v)
    
    for v in list(fast_mem):
        ops.append(('delete', v))
    
    return ops


def pebble(dag, fast_memory_budget):
    start_time = time.time()
    
    try:
        vertices, parents, outputs, inputs = _toposort_vertices(dag)
        n_vertices = len(vertices)
        
        # For small instances, try exact A* with bitmask
        if n_vertices <= 20:
            time_budget = min(35.0, 1.5 * n_vertices)
            remaining_time = _DEADLINE_SECONDS - (time.time() - start_time)
            time_budget = min(time_budget, remaining_time - 10.0)
            
            if time_budget > 2.0:
                result = _dijkstra_pebble_bitmask(dag, fast_memory_budget, time_limit=time_budget)
                if result is not None and len(result) <= 20000:
                    if _validate_ops(result, dag, fast_memory_budget):
                        # Polish the result
                        polished = _local_search_polish(result, dag, fast_memory_budget, time_limit=3.0)
                        if _validate_ops(polished, dag, fast_memory_budget) and len(polished) <= 20000:
                            return polished
                        return result
        
        # For larger instances or if A* failed, use improved greedy
        remaining_time = _DEADLINE_SECONDS - (time.time() - start_time)
        if remaining_time < 8.0:
            # Emergency fallback
            try:
                from autoresearch_suite.science_tcs_expansion import pebble_baseline
                return pebble_baseline(dag, fast_memory_budget)
            except Exception:
                ops = []
                for v in vertices:
                    if v in inputs:
                        ops.append(('load', v))
                    else:
                        ops.append(('compute', v))
                for v in outputs:
                    ops.append(('store', v))
                return ops
        
        # Try multiple greedy strategies and pick the best
        candidates = []
        
        # Strategy 1: Improved greedy
        result1 = _greedy_pebble_improved(dag, fast_memory_budget)
        if result1 and _validate_ops(result1, dag, fast_memory_budget) and len(result1) <= 20000:
            candidates.append(result1)
        
        # Strategy 2: Layered greedy
        result2 = _layered_greedy_pebble(dag, fast_memory_budget)
        if result2 and _validate_ops(result2, dag, fast_memory_budget) and len(result2) <= 20000:
            candidates.append(result2)
        
        # Polish each candidate
        best_ops = None
        best_transfers = float('inf')
        
        for cand in candidates:
            polished = _local_search_polish(cand, dag, fast_memory_budget, time_limit=2.0)
            if _validate_ops(polished, dag, fast_memory_budget) and len(polished) <= 20000:
                transfers = _count_transfers(polished)
                if transfers < best_transfers:
                    best_transfers = transfers
                    best_ops = polished
        
        if best_ops is not None:
            return best_ops
        
        # Fallback
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            ops = []
            for v in vertices:
                if v in inputs:
                    ops.append(('load', v))
                else:
                    ops.append(('compute', v))
            for v in outputs:
                ops.append(('store', v))
            return ops
    
    except Exception:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            try:
                vertices, parents, outputs, inputs = _toposort_vertices(dag)
                ops = []
                for v in vertices:
                    if v in inputs:
                        ops.append(('load', v))
                    else:
                        ops.append(('compute', v))
                for v in outputs:
                    ops.append(('store', v))
                return ops
            except Exception:
                return []

# EVOLVE-BLOCK-END
