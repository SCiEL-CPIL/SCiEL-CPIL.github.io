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


def _dijkstra_pebble_bitmask(dag, fast_memory_budget, time_limit=30.0):
    """Dijkstra over bitmask states for small/medium instances."""
    start_time = time.time()
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    n = len(vertices)
    
    if n > 24:
        return None

    # Map vertices to indices
    v_to_idx = {v: i for i, v in enumerate(vertices)}
    idx_to_v = {i: v for i, v in enumerate(vertices)}

    # Bitmasks
    inputs_mask = 0
    for v in inputs:
        inputs_mask |= (1 << v_to_idx[v])
    
    outputs_mask = 0
    for v in outputs:
        outputs_mask |= (1 << v_to_idx[v])
    
    computed_mask = (~inputs_mask) & ((1 << n) - 1)

    # Parent masks for each vertex
    parent_masks = []
    for v in vertices:
        ps = parents.get(v, [])
        mask = 0
        for p in ps:
            mask |= (1 << v_to_idx[p])
        parent_masks.append(mask)

    # State: (blue_mask, fast_mask)
    # blue: values in slow memory (inputs + stored)
    # fast: values in fast memory
    # available = blue | fast
    # Goal: outputs ⊆ blue

    init_blue = inputs_mask
    init_fast = 0
    init_state = (init_blue, init_fast)

    def is_goal(blue, fast):
        return (blue & outputs_mask) == outputs_mask

    if is_goal(init_blue, init_fast):
        return []

    best_cost = {}
    best_cost[init_state] = 0
    parent_map = {}
    pq = [(0, 0, init_state)]

    max_states = 200000
    states_explored = 0
    best_found = None
    best_found_cost = float('inf')

    while pq:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        if states_explored >= max_states:
            break

        cost, ops_count, state = heapq.heappop(pq)
        if cost > best_cost.get(state, float('inf')):
            continue

        states_explored += 1
        blue, fast = state

        if is_goal(blue, fast):
            if cost < best_found_cost:
                best_found_cost = cost
                best_found = state
                # Can stop since Dijkstra guarantees optimal
                break

        available = blue | fast

        # 1. load(v): v in inputs, v not in fast, len(fast) < budget
        load_candidates = inputs_mask & ~fast
        while load_candidates:
            bit = load_candidates & (-load_candidates)
            v_idx = bit.bit_length() - 1
            load_candidates ^= bit
            new_fast = fast | bit
            if new_fast.bit_count() <= fast_memory_budget:
                new_state = (blue, new_fast)
                new_cost = cost + 1
                if new_cost < best_cost.get(new_state, float('inf')):
                    best_cost[new_state] = new_cost
                    parent_map[new_state] = (state, 'load', idx_to_v[v_idx])
                    heapq.heappush(pq, (new_cost, ops_count + 1, new_state))

        # 2. store(v): v in fast
        store_candidates = fast
        while store_candidates:
            bit = store_candidates & (-store_candidates)
            v_idx = bit.bit_length() - 1
            store_candidates ^= bit
            new_fast = fast & ~bit
            new_blue = blue | bit
            new_state = (new_blue, new_fast)
            new_cost = cost + 1
            if new_cost < best_cost.get(new_state, float('inf')):
                best_cost[new_state] = new_cost
                parent_map[new_state] = (state, 'store', idx_to_v[v_idx])
                heapq.heappush(pq, (new_cost, ops_count + 1, new_state))

        # 3. compute(v): v not in inputs, v not in fast, all parents available, len(fast) < budget
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
                    new_cost = cost
                    if new_cost < best_cost.get(new_state, float('inf')):
                        best_cost[new_state] = new_cost
                        parent_map[new_state] = (state, 'compute', idx_to_v[v_idx])
                        heapq.heappush(pq, (new_cost, ops_count + 1, new_state))

        # 4. delete(v): v in fast (free operation)
        delete_candidates = fast
        while delete_candidates:
            bit = delete_candidates & (-delete_candidates)
            v_idx = bit.bit_length() - 1
            delete_candidates ^= bit
            new_fast = fast & ~bit
            new_state = (blue, new_fast)
            new_cost = cost
            if new_cost < best_cost.get(new_state, float('inf')):
                best_cost[new_state] = new_cost
                parent_map[new_state] = (state, 'delete', idx_to_v[v_idx])
                heapq.heappush(pq, (new_cost, ops_count + 1, new_state))

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
    """Try to reduce operations by removing redundant delete/load pairs and reordering."""
    start_time = time.time()
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    
    if not ops:
        return ops
    
    # Try to remove redundant operations
    # A delete followed by a load of the same vertex can be removed if the vertex is still in blue
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        new_ops = list(ops)
        
        # Look for patterns: delete(v), ..., load(v) where v is in inputs and still available
        # This is complex; simpler approach: try removing individual deletes that aren't needed
        
        # Simpler: try to merge consecutive delete+load of same vertex
        i = 0
        while i < len(new_ops) - 1:
            op1, v1 = new_ops[i]
            op2, v2 = new_ops[i+1]
            
            # If we delete v and immediately load v, and v is an input, we can remove both
            if op1 == 'delete' and op2 == 'load' and v1 == v2 and v1 in inputs:
                del new_ops[i]
                del new_ops[i]
                improved = True
                continue
            i += 1
        
        ops = new_ops
    
    # Try to remove trailing deletes that aren't necessary
    # Only keep deletes that are needed to make room for future loads/computes
    # This is hard to determine without lookahead, so skip for now
    
    return ops


def _greedy_pebble_improved(dag, fast_memory_budget):
    """Improved greedy: topological order with smart eviction using priority queue."""
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    
    # Build children map and compute future-need counts
    children = {}
    for v in vertices:
        for p in parents.get(v, []):
            children.setdefault(p, []).append(v)
    
    # For each vertex, compute how many times it will be needed as a parent
    # in remaining vertices (in topological order)
    v_to_pos = {v: i for i, v in enumerate(vertices)}
    
    # remaining_children[v] = number of children of v that appear after v in topo order
    remaining_children = {}
    for v in vertices:
        count = 0
        for c in children.get(v, []):
            if v_to_pos.get(c, 0) > v_to_pos[v]:
                count += 1
        remaining_children[v] = count

    ops = []
    fast_mem = set()
    blue = set(inputs)
    available = set(inputs)
    
    # Priority queue for eviction: (remaining_uses, is_output, vertex)
    # Lower remaining_uses = better candidate for eviction
    # is_output: 1 if output (should not be evicted if possible)
    
    def get_eviction_priority(v):
        """Get eviction priority: lower is more likely to be evicted."""
        rc = remaining_children.get(v, 0)
        is_out = 1 if v in outputs else 0
        return (rc, is_out)
    
    for v in vertices:
        if v in inputs:
            if v not in available:
                # Need to load v
                if len(fast_mem) >= fast_memory_budget:
                    # Evict: find the best candidate using priority
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
            # Compute v
            ps = parents.get(v, [])
            
            if len(fast_mem) >= fast_memory_budget:
                # Evict: find the best candidate using priority
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
        
        # Update remaining_children for parents of v
        for p in parents.get(v, []):
            remaining_children[p] = remaining_children.get(p, 0) - 1
        
        # Store outputs immediately
        if v in outputs and v in fast_mem:
            ops.append(('store', v))
            fast_mem.discard(v)
            blue.add(v)
    
    # Cleanup
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
                for p in ps:
                    if p not in available:
                        pass
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
        
        # For small instances, try exact Dijkstra with bitmask
        if n_vertices <= 20:
            time_budget = min(40.0, 2.0 * n_vertices)
            remaining_time = _DEADLINE_SECONDS - (time.time() - start_time)
            time_budget = min(time_budget, remaining_time - 5.0)
            
            if time_budget > 2.0:
                result = _dijkstra_pebble_bitmask(dag, fast_memory_budget, time_limit=time_budget)
                if result is not None and len(result) <= 20000:
                    if _validate_ops(result, dag, fast_memory_budget):
                        return result
        
        # For larger instances or if Dijkstra failed, use improved greedy
        remaining_time = _DEADLINE_SECONDS - (time.time() - start_time)
        if remaining_time < 5.0:
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
        
        result = _greedy_pebble_improved(dag, fast_memory_budget)
        if result and _validate_ops(result, dag, fast_memory_budget) and len(result) <= 20000:
            return result
        
        # Fallback to layered greedy
        result = _layered_greedy_pebble(dag, fast_memory_budget)
        if result and _validate_ops(result, dag, fast_memory_budget) and len(result) <= 20000:
            return result
        
        # Last resort
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
