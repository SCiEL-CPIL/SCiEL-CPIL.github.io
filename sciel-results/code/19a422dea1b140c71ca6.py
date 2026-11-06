# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque

_DEADLINE_SECONDS = 70.0


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


def _dijkstra_pebble_min_transfers(dag, fast_memory_budget, time_limit=30.0):
    """Dijkstra over bitmask states minimizing transfer cost (load/store)."""
    start_time = time.time()
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    n = len(vertices)
    
    if n > 22: # Slightly larger limit as cost is sparse
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

    init_blue = inputs_mask
    init_fast = 0
    init_state = (init_blue, init_fast)

    def is_goal(blue, fast):
        return (blue & outputs_mask) == outputs_mask

    if is_goal(init_blue, init_fast):
        return []

    # Cost: 1 for load/store, 0 for compute/delete
    best_cost = {}
    best_cost[init_state] = 0
    parent_map = {}
    # Heap: (cost, ops_count, state)
    pq = [(0, 0, init_state)]

    max_states = 300000
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
                break

        available = blue | fast

        # 1. load(v): v in inputs, v not in fast, len(fast) < budget. Cost 1
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

        # 2. store(v): v in fast. Cost 1
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

        # 3. compute(v): v not in inputs, v not in fast, all parents available, len(fast) < budget. Cost 0
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

        # 4. delete(v): v in fast. Cost 0
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


def _greedy_pebble_min_transfers(dag, fast_memory_budget):
    """Greedy minimizing transfers by prioritizing reuse."""
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    
    children = {}
    for v in vertices:
        for p in parents.get(v, []):
            children.setdefault(p, []).append(v)
    
    v_to_pos = {v: i for i, v in enumerate(vertices)}
    
    # Future need: number of children not yet processed
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
    
    # We want to minimize loads/stores.
    # Strategy: Keep vertices in fast_mem as long as they are needed.
    # Evict the one with the least future need.
    
    def evict_one():
        if not fast_mem:
            return
        # Prioritize evicting:
        # 1. Not an output
        # 2. 0 remaining children
        # 3. Lowest remaining children
        
        best_evict = None
        best_score = float('inf')
        
        for fm in fast_mem:
            is_output = 1 if fm in outputs else 0
            rem = remaining_children.get(fm, 0)
            # Score: lower is better to evict
            # We want to keep outputs (high score), keep high reuse (high score)
            # So evict low score.
            # Score = (is_output * 100) + rem
            score = (is_output * 100) + rem
            if score < best_score:
                best_score = score
                best_evict = fm
        
        if best_evict is not None:
            ops.append(('delete', best_evict))
            fast_mem.discard(best_evict)
            if best_evict not in blue:
                available.discard(best_evict)

    for v in vertices:
        if v in inputs:
            if v not in available:
                if len(fast_mem) >= fast_memory_budget:
                    evict_one()
                if len(fast_mem) < fast_memory_budget:
                    ops.append(('load', v))
                    fast_mem.add(v)
                    available.add(v)
        else:
            ps = parents.get(v, [])
            # Ensure parents are available
            for p in ps:
                if p not in available:
                    # This implies we need to reload p, which is bad.
                    # In a valid topo order with this greedy, this shouldn't happen 
                    # if we don't evict needed parents. 
                    # But if we did, we need to reload.
                    if p in inputs:
                        if len(fast_mem) >= fast_memory_budget:
                            evict_one()
                        if len(fast_mem) < fast_memory_budget:
                            ops.append(('load', p))
                            fast_mem.add(p)
                            available.add(p)
                    else:
                        # Recompute p? Complex. 
                        # For now, assume this doesn't happen or handle by recomputing if possible.
                        pass
            
            if len(fast_mem) >= fast_memory_budget:
                evict_one()
            
            if len(fast_mem) < fast_memory_budget:
                ops.append(('compute', v))
                fast_mem.add(v)
                available.add(v)
        
        # Update remaining children
        for p in parents.get(v, []):
            remaining_children[p] = remaining_children.get(p, 0) - 1
        
        # Store outputs immediately to satisfy goal, but this costs a transfer.
        # We must store outputs.
        if v in outputs and v in fast_mem:
            ops.append(('store', v))
            fast_mem.discard(v)
            blue.add(v)
    
    # Cleanup
    for v in list(fast_mem):
        ops.append(('delete', v))
        fast_mem.discard(v)
    
    return ops


def pebble(dag, fast_memory_budget):
    start_time = time.time()
    
    try:
        vertices, parents, outputs, inputs = _toposort_vertices(dag)
        n_vertices = len(vertices)
        
        # Try exact min-transfer Dijkstra for small instances
        if n_vertices <= 22:
            remaining_time = _DEADLINE_SECONDS - (time.time() - start_time)
            time_budget = min(40.0, remaining_time - 5.0)
            
            if time_budget > 2.0:
                result = _dijkstra_pebble_min_transfers(dag, fast_memory_budget, time_limit=time_budget)
                if result is not None and len(result) <= 20000:
                    if _validate_ops(result, dag, fast_memory_budget):
                        return result
        
        # Fallback to greedy
        remaining_time = _DEADLINE_SECONDS - (time.time() - start_time)
        if remaining_time < 5.0:
            # Emergency fallback
            ops = []
            for v in vertices:
                if v in inputs:
                    ops.append(('load', v))
                else:
                    ops.append(('compute', v))
            for v in outputs:
                ops.append(('store', v))
            return ops
        
        result = _greedy_pebble_min_transfers(dag, fast_memory_budget)
        if result and _validate_ops(result, dag, fast_memory_budget) and len(result) <= 20000:
            return result
        
        # Last resort
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
