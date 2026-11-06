# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque

_DEADLINE_SECONDS = 80.0


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
            if v not in fast_mem:
                return False
            fast_mem.remove(v)
            blue.add(v)
        elif op == 'compute':
            ps = parents.get(v, [])
            for p in ps:
                if p not in available:
                    return False
            if v in fast_mem:
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
    """Dijkstra over bitmask states for small/medium instances with dominance pruning."""
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

    init_blue = inputs_mask
    init_fast = 0
    init_state = (init_blue, init_fast)

    def is_goal(blue, fast):
        return (blue & outputs_mask) == outputs_mask

    if is_goal(init_blue, init_fast):
        return []

    # State: (blue_mask, fast_mask)
    # Dominance: state A dominates state B if:
    #   A.blue ⊇ B.blue AND A.fast ⊇ B.fast (A has more info in both memories)
    # We use a simpler approach: track best cost per state, and prune if we've seen
    # a state with same fast but more blue (or same blue but more fast) at lower cost.
    
    best_cost = {}
    best_cost[init_state] = 0
    parent_map = {}
    pq = [(0, init_state)]

    max_states = 500000
    states_explored = 0
    best_found = None
    best_found_cost = float('inf')

    while pq:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        if states_explored >= max_states:
            break

        cost, state = heapq.heappop(pq)
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

        # Generate neighbors
        neighbors = []

        # 1. load(v): v in inputs, v not in fast, len(fast) < budget
        load_candidates = inputs_mask & ~fast
        if fast.bit_count() < fast_memory_budget:
            while load_candidates:
                bit = load_candidates & (-load_candidates)
                v_idx = bit.bit_length() - 1
                load_candidates ^= bit
                new_fast = fast | bit
                neighbors.append((blue, new_fast, 'load', v_idx, 1))

        # 2. store(v): v in fast, v not in blue (storing already-blue is redundant)
        store_candidates = fast & ~blue
        while store_candidates:
            bit = store_candidates & (-store_candidates)
            v_idx = bit.bit_length() - 1
            store_candidates ^= bit
            new_fast = fast & ~bit
            new_blue = blue | bit
            neighbors.append((new_blue, new_fast, 'store', v_idx, 1))

        # 3. compute(v): v not in inputs, v not in fast, all parents available, len(fast) < budget
        if fast.bit_count() < fast_memory_budget:
            compute_candidates = computed_mask & ~fast & ~blue
            while compute_candidates:
                bit = compute_candidates & (-compute_candidates)
                v_idx = bit.bit_length() - 1
                compute_candidates ^= bit
                pm = parent_masks[v_idx]
                if (pm & ~available) == 0:
                    new_fast = fast | bit
                    neighbors.append((blue, new_fast, 'compute', v_idx, 0))

        # 4. delete(v): v in fast (free operation)
        delete_candidates = fast
        while delete_candidates:
            bit = delete_candidates & (-delete_candidates)
            v_idx = bit.bit_length() - 1
            delete_candidates ^= bit
            new_fast = fast & ~bit
            neighbors.append((blue, new_fast, 'delete', v_idx, 0))

        for new_blue, new_fast, op, v_idx, transfer_cost in neighbors:
            new_state = (new_blue, new_fast)
            new_cost = cost + transfer_cost
            
            if new_cost >= best_found_cost:
                continue
                
            if new_cost < best_cost.get(new_state, float('inf')):
                best_cost[new_state] = new_cost
                parent_map[new_state] = (state, op, idx_to_v[v_idx])
                heapq.heappush(pq, (new_cost, new_state))

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


def _layered_dag_pebble(dag, fast_memory_budget):
    """Specialized pebbling for layered DAGs: process layer by layer, 
    keeping current and next layer in fast memory."""
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    
    if not vertices:
        return []

    # Compute depth/layer for each vertex
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
    
    # Track which vertices have been stored
    stored = set(inputs)
    
    for d in range(max_depth + 1):
        layer_vertices = layers.get(d, [])
        if not layer_vertices:
            continue
        
        # Determine which vertices in this layer need to be computed/loaded
        for v in layer_vertices:
            if v in inputs:
                if v not in available:
                    # Need to load v
                    if len(fast_mem) >= fast_memory_budget:
                        # Evict vertices that are no longer needed
                        evicted = False
                        for fm in list(fast_mem):
                            # Check if fm is still needed by any remaining vertex
                            still_needed = False
                            for rd in range(d, max_depth + 1):
                                for rv in layers.get(rd, []):
                                    if fm in parents.get(rv, []):
                                        still_needed = True
                                        break
                                if still_needed:
                                    break
                            if not still_needed and fm not in outputs:
                                ops.append(('delete', fm))
                                fast_mem.discard(fm)
                                if fm not in blue:
                                    available.discard(fm)
                                evicted = True
                                break
                        if not evicted and fast_mem:
                            fm = next(iter(fast_mem))
                            ops.append(('delete', fm))
                            fast_mem.discard(fm)
                            if fm not in blue:
                                available.discard(fm)
                    
                    if len(fast_mem) < fast_memory_budget:
                        ops.append(('load', v))
                        fast_mem.add(v)
                        available.add(v)
            else:
                # Compute v
                ps = parents.get(v, [])
                # Ensure all parents are available
                for p in ps:
                    if p not in available:
                        # Try to load p if it's an input
                        if p in inputs:
                            if len(fast_mem) >= fast_memory_budget:
                                evicted = False
                                for fm in list(fast_mem):
                                    still_needed = False
                                    for rd in range(d, max_depth + 1):
                                        for rv in layers.get(rd, []):
                                            if fm in parents.get(rv, []):
                                                still_needed = True
                                                break
                                        if still_needed:
                                            break
                                    if not still_needed and fm not in outputs:
                                        ops.append(('delete', fm))
                                        fast_mem.discard(fm)
                                        if fm not in blue:
                                            available.discard(fm)
                                        evicted = True
                                        break
                                if not evicted and fast_mem:
                                    fm = next(iter(fast_mem))
                                    ops.append(('delete', fm))
                                    fast_mem.discard(fm)
                                    if fm not in blue:
                                        available.discard(fm)
                            if len(fast_mem) < fast_memory_budget:
                                ops.append(('load', p))
                                fast_mem.add(p)
                                available.add(p)
                
                if len(fast_mem) >= fast_memory_budget:
                    evicted = False
                    for fm in list(fast_mem):
                        still_needed = False
                        for rd in range(d, max_depth + 1):
                            for rv in layers.get(rd, []):
                                if fm in parents.get(rv, []):
                                    still_needed = True
                                    break
                            if still_needed:
                                break
                        if not still_needed and fm not in outputs:
                            ops.append(('delete', fm))
                            fast_mem.discard(fm)
                            if fm not in blue:
                                available.discard(fm)
                            evicted = True
                            break
                    if not evicted and fast_mem:
                        fm = next(iter(fast_mem))
                        ops.append(('delete', fm))
                        fast_mem.discard(fm)
                        if fm not in blue:
                            available.discard(fm)
                
                if len(fast_mem) < fast_memory_budget:
                    ops.append(('compute', v))
                    fast_mem.add(v)
                    available.add(v)
        
        # Store outputs from this layer
        for v in layer_vertices:
            if v in outputs and v in fast_mem:
                ops.append(('store', v))
                fast_mem.discard(v)
                blue.add(v)
                stored.add(v)
    
    # Final cleanup
    for v in list(fast_mem):
        ops.append(('delete', v))
        fast_mem.discard(v)
    
    return ops


def _greedy_pebble_improved(dag, fast_memory_budget):
    """Improved greedy with better eviction based on remaining uses."""
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    
    children = {}
    for v in vertices:
        for p in parents.get(v, []):
            children.setdefault(p, []).append(v)
    
    v_to_pos = {v: i for i, v in enumerate(vertices)}
    
    # Compute remaining uses: for each vertex, how many times it will be needed
    remaining_uses = {}
    for v in vertices:
        count = 0
        for c in children.get(v, []):
            if v_to_pos.get(c, 0) > v_to_pos[v]:
                count += 1
        if v in outputs:
            count = max(count, 1)  # Outputs must be stored
        remaining_uses[v] = count

    ops = []
    fast_mem = set()
    blue = set(inputs)
    available = set(inputs)
    
    for v in vertices:
        if v in inputs:
            if v not in available:
                if len(fast_mem) >= fast_memory_budget:
                    evicted = False
                    # Evict vertex with fewest remaining uses and not an output
                    best_evict = None
                    best_uses = float('inf')
                    for fm in list(fast_mem):
                        if fm not in outputs:
                            uses = remaining_uses.get(fm, 0)
                            if uses < best_uses:
                                best_uses = uses
                                best_evict = fm
                    if best_evict is not None:
                        ops.append(('delete', best_evict))
                        fast_mem.discard(best_evict)
                        if best_evict not in blue:
                            available.discard(best_evict)
                        evicted = True
                    if not evicted and fast_mem:
                        fm = next(iter(fast_mem))
                        ops.append(('delete', fm))
                        fast_mem.discard(fm)
                        if fm not in blue:
                            available.discard(fm)
                
                if len(fast_mem) < fast_memory_budget:
                    ops.append(('load', v))
                    fast_mem.add(v)
                    available.add(v)
        else:
            ps = parents.get(v, [])
            for p in ps:
                if p not in available:
                    if p in inputs:
                        if len(fast_mem) >= fast_memory_budget:
                            best_evict = None
                            best_uses = float('inf')
                            for fm in list(fast_mem):
                                if fm not in outputs:
                                    uses = remaining_uses.get(fm, 0)
                                    if uses < best_uses:
                                        best_uses = uses
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
                best_uses = float('inf')
                for fm in list(fast_mem):
                    if fm not in outputs:
                        uses = remaining_uses.get(fm, 0)
                        if uses < best_uses:
                            best_uses = uses
                            best_evict = fm
                if best_evict is not None:
                    ops.append(('delete', best_evict))
                    fast_mem.discard(best_evict)
                    if best_evict not in blue:
                        available.discard(best_evict)
                elif fast_mem:
                    fm = next(iter(fast_mem))
                    ops.append(('delete', fm))
                    fast_mem.discard(fm)
                    if fm not in blue:
                        available.discard(fm)
            
            if len(fast_mem) < fast_memory_budget:
                ops.append(('compute', v))
                fast_mem.add(v)
                available.add(v)
        
        # Update remaining uses for parents of v
        for p in parents.get(v, []):
            remaining_uses[p] = remaining_uses.get(p, 0) - 1
        
        if v in outputs and v in fast_mem:
            ops.append(('store', v))
            fast_mem.discard(v)
            blue.add(v)
    
    for v in list(fast_mem):
        ops.append(('delete', v))
        fast_mem.discard(v)
    
    return ops


def _count_transfers(ops):
    return sum(1 for op, v in ops if op in ('load', 'store'))


def _naive_schedule_ops(dag, fast_memory_budget):
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    ops = []
    for v in vertices:
        if v in inputs:
            ops.append(('load', v))
        else:
            ops.append(('compute', v))
    for v in outputs:
        ops.append(('store', v))
    for v in reversed(vertices):
        if v not in inputs and v not in outputs:
            ops.append(('delete', v))
    transfers = sum(1 for op, v in ops if op in ('load', 'store'))
    return ops, transfers


def pebble(dag, fast_memory_budget):
    start_time = time.time()
    
    try:
        vertices, parents, outputs, inputs = _toposort_vertices(dag)
        n_vertices = len(vertices)
        
        best_ops = None
        best_transfers = float('inf')
        
        # Strategy 1: For small instances, try exact Dijkstra
        if n_vertices <= 20:
            remaining_time = _DEADLINE_SECONDS - (time.time() - start_time)
            time_budget = min(40.0, 2.0 * n_vertices, remaining_time - 5.0)
            
            if time_budget > 2.0:
                result = _dijkstra_pebble_bitmask(dag, fast_memory_budget, time_limit=time_budget)
                if result is not None and len(result) <= 20000:
                    if _validate_ops(result, dag, fast_memory_budget):
                        best_ops = result
                        best_transfers = _count_transfers(result)
        
        # Strategy 2: Layered DAG pebbling (good for layered structures)
        remaining_time = _DEADLINE_SECONDS - (time.time() - start_time)
        if remaining_time > 3.0:
            result = _layered_dag_pebble(dag, fast_memory_budget)
            if result and _validate_ops(result, dag, fast_memory_budget) and len(result) <= 20000:
                transfers = _count_transfers(result)
                if transfers < best_transfers:
                    best_ops = result
                    best_transfers = transfers
        
        # Strategy 3: Improved greedy
        remaining_time = _DEADLINE_SECONDS - (time.time() - start_time)
        if remaining_time > 2.0:
            result = _greedy_pebble_improved(dag, fast_memory_budget)
            if result and _validate_ops(result, dag, fast_memory_budget) and len(result) <= 20000:
                transfers = _count_transfers(result)
                if transfers < best_transfers:
                    best_ops = result
                    best_transfers = transfers
        
        if best_ops is not None:
            return best_ops
        
        # Fallback: simple valid schedule
        ops = []
        for v in vertices:
            if v in inputs:
                ops.append(('load', v))
            else:
                ops.append(('compute', v))
        for v in outputs:
            ops.append(('store', v))
        for v in reversed(vertices):
            if v not in inputs and v not in outputs:
                ops.append(('delete', v))
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
