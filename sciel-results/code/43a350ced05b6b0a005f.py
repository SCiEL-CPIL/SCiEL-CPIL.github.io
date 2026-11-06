# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque, defaultdict
import sys

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
    return sum(1 for op, v in ops if op in ('load', 'store'))

def _dijkstra_pebble_bitmask(dag, fast_memory_budget, time_limit=30.0):
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

    best_cost = {}
    best_cost[init_state] = 0
    parent_map = {}
    pq = [(0, 0, init_state)]

    max_states = 150000
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

        # 1. load(v)
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

        # 2. store(v)
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
                    new_cost = cost
                    if new_cost < best_cost.get(new_state, float('inf')):
                        best_cost[new_state] = new_cost
                        parent_map[new_state] = (state, 'compute', idx_to_v[v_idx])
                        heapq.heappush(pq, (new_cost, ops_count + 1, new_state))

        # 4. delete(v)
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
    start_time = time.time()
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    
    if not ops:
        return ops
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
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
        
        # Pattern 2: load(v) ... delete(v) ... load(v) where v is an input and no use in between
        # Simplified: consecutive load+delete of same input
        i = 0
        while i < len(new_ops) - 1:
            op1, v1 = new_ops[i]
            op2, v2 = new_ops[i+1]
            if op1 == 'load' and op2 == 'delete' and v1 == v2 and v1 in inputs:
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
            new_ops = new_ops[:last_store_idx+1]
            improved = True
        
        ops = new_ops
    
    return ops

def _greedy_pebble_advanced(dag, fast_memory_budget):
    """
    Advanced greedy using priority queue to minimize evictions.
    State: 
    - available: set of vertices in fast_mem or blue
    - fast_mem: set of vertices in fast memory
    - blue: set of vertices in slow memory (inputs + stored)
    - remaining_uses: dict v -> count of children not yet processed
    """
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
            
    # Pre-compute remaining uses (number of children)
    remaining_uses = {v: len(children[v]) for v in vertices}
    
    # Priority Queue for ready vertices
    # Key: (estimated_cost, vertex)
    # estimated_cost: 0 if input (load cost 1 later), 1 if computed (compute cost 0, but eviction cost?)
    # Actually, we want to process vertices that keep memory pressure low.
    # Heuristic: Process vertices with high remaining_uses first? Or low?
    # If we process a vertex with high remaining_uses, we keep its value in memory longer.
    # If we process a vertex with low remaining_uses, we can evict it sooner.
    # Standard strategy: Process vertices in topological order, but when choosing which to compute/load,
    # prioritize those that are "critical" or have many dependents to avoid reloading.
    
    # Let's stick to a simpler, robust greedy:
    # 1. Maintain a set of "ready" vertices (all parents available).
    # 2. If memory is full, evict the vertex with the lowest "value".
    #    Value = remaining_uses. If tie, prefer evicting inputs (cheaper to reload).
    
    ops = []
    fast_mem = set()
    blue = set(inputs)
    available = set(inputs)
    
    # Ready vertices: inputs are initially ready (parents empty)
    # But we don't load them immediately. We load them when needed or when memory allows?
    # Better: Load inputs as they become ready?
    # Let's use a queue of ready vertices.
    
    ready = deque()
    in_ready = set()
    
    for v in vertices:
        if not parents.get(v):
            ready.append(v)
            in_ready.add(v)
            
    processed = 0
    total = len(vertices)
    
    # We need to ensure all parents are available before computing a child.
    # This greedy approach processes vertices as they become ready.
    
    while processed < total:
        if not ready:
            # Deadlock? Should not happen in DAG
            break
            
        # Choose next vertex from ready
        # Heuristic: Pick vertex with highest remaining_uses? 
        # Or pick vertex that is an input?
        # Let's pick the vertex that minimizes the number of new loads required.
        # Actually, just pick the first one? No, let's pick one with high degree to keep it in memory.
        
        # Find best candidate
        best_v = None
        best_score = -1
        for v in list(ready):
            # Score: higher remaining_uses is better to keep in memory
            # But if we have to evict, we evict the one with lowest remaining_uses.
            # So processing order doesn't directly dictate eviction, but it dictates when we need space.
            # Let's just process in topological order but try to optimize eviction.
            score = remaining_uses.get(v, 0)
            if score > best_score:
                best_score = score
                best_v = v
                
        if best_v is None:
            best_v = ready[0]
            
        ready.remove(best_v)
        in_ready.discard(best_v)
        v = best_v
        
        # Process v
        if v in inputs:
            # Need to load v
            if v not in fast_mem:
                if len(fast_mem) >= fast_memory_budget:
                    # Evict
                    # Find eviction candidate in fast_mem
                    # Prefer evicting:
                    # 1. Inputs (can reload)
                    # 2. Vertices with low remaining_uses
                    evict_v = None
                    min_score = float('inf')
                    for fm in fast_mem:
                        is_input = 1 if fm in inputs else 0
                        # We want to minimize the "cost" of eviction.
                        # Evicting an input is cheaper (load cost 1) than evicting a computed value (recompute cost > 1).
                        # So we prefer to evict inputs first?
                        # Wait, if we evict an input, we have to load it again later.
                        # If we evict a computed value, we have to recompute it.
                        # Recompute is generally more expensive than load.
                        # So we should KEEP computed values and EVICT inputs?
                        # No, inputs are small. Computed values might be needed for many children.
                        # Let's use remaining_uses as primary metric.
                        # Lower remaining_uses = less useful = evict first.
                        # Tie-breaker: Evict inputs (easier to restore) or computed?
                        # Actually, if we evict an input, we just load it. If we evict a computed, we recompute.
                        # Load is cheaper. So we prefer to evict inputs?
                        # BUT, if the input has high remaining_uses, we need it a lot.
                        # So: Score = (remaining_uses, is_input). 
                        # We want to evict the one with LOWEST remaining_uses.
                        # If tie, evict the one that is an INPUT? (Because load is cheap)
                        # Or evict the one that is COMPUTED? (Because recompute is expensive, so we should NOT evict it)
                        # So: Prefer to evict Inputs.
                        # Score for eviction: (remaining_uses, 0 if input else 1). Minimize this.
                        s = (remaining_uses.get(fm, 0), 0 if fm in inputs else 1)
                        if s < (min_score, 0) if isinstance(min_score, tuple) else False:
                             # Simplify comparison
                             pass
                        # Let's just use a tuple
                        if evict_v is None or s < (remaining_uses.get(evict_v, 0), 0 if evict_v in inputs else 1):
                            evict_v = fm
                            
                    if evict_v:
                        ops.append(('delete', evict_v))
                        fast_mem.discard(evict_v)
                        if evict_v not in blue:
                            available.discard(evict_v)
                
                if len(fast_mem) < fast_memory_budget:
                    ops.append(('load', v))
                    fast_mem.add(v)
                    available.add(v)
        else:
            # Compute v
            # Ensure parents are available
            # They should be, since v is ready
            if len(fast_mem) >= fast_memory_budget:
                evict_v = None
                for fm in fast_mem:
                    if evict_v is None or (remaining_uses.get(fm, 0), 0 if fm in inputs else 1) < (remaining_uses.get(evict_v, 0), 0 if evict_v in inputs else 1):
                        evict_v = fm
                if evict_v:
                    ops.append(('delete', evict_v))
                    fast_mem.discard(evict_v)
                    if evict_v not in blue:
                        available.discard(evict_v)
                        
            if len(fast_mem) < fast_memory_budget:
                ops.append(('compute', v))
                fast_mem.add(v)
                available.add(v)
                
        processed += 1
        
        # Update children
        for c in children.get(v, []):
            # Check if all parents of c are available
            # If so, add c to ready
            all_parents_available = True
            for p in parents.get(c, []):
                if p not in available:
                    all_parents_available = False
                    break
            if all_parents_available and c not in in_ready and c not in [r for r in ready]:
                ready.append(c)
                in_ready.add(c)
                
        # If v is an output, store it?
        # We should store outputs to satisfy the goal.
        # But storing frees up fast memory.
        # Let's store immediately if it's an output and in fast_mem.
        if v in outputs and v in fast_mem:
            ops.append(('store', v))
            fast_mem.discard(v)
            blue.add(v)
            
    # Final cleanup: Store any remaining outputs in fast_mem
    for v in fast_mem:
        if v in outputs:
            ops.append(('store', v))
            fast_mem.discard(v)
            blue.add(v)
            
    # Delete remaining fast_mem items? Not strictly necessary for validity if we don't need to end with empty fast_mem,
    # but good practice. The validator only checks if outputs are blue.
    # However, if we don't delete, they stay in fast_mem. That's fine.
    # But wait, the validator checks `if not outputs.issubset(blue)`.
    # It does NOT require fast_mem to be empty.
    # So we can skip final deletes to save ops? 
    # "at most 20000 operations".
    # Deletes don't count as transfers. So they are "free" in the score, but count towards op limit.
    # We should probably delete them to be clean, but it's not required for score.
    # Let's delete them to avoid any potential issue with "state" if the evaluator is strict.
    for v in list(fast_mem):
        ops.append(('delete', v))
        fast_mem.discard(v)
        
    return ops

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    
    try:
        vertices, parents, outputs, inputs = _toposort_vertices(dag)
        n_vertices = len(vertices)
        
        # For small instances, try exact Dijkstra with bitmask
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
        
        # For larger instances or if Dijkstra failed, use improved greedy
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
        
        # Strategy 1: Advanced greedy
        result1 = _greedy_pebble_advanced(dag, fast_memory_budget)
        if result1 and _validate_ops(result1, dag, fast_memory_budget) and len(result1) <= 20000:
            candidates.append(result1)
        
        # Strategy 2: Simple topological greedy (fallback)
        result2 = []
        fast_mem = set()
        blue = set(inputs)
        available = set(inputs)
        for v in vertices:
            if v in inputs:
                if v not in fast_mem:
                    if len(fast_mem) >= fast_memory_budget:
                        # Evict first in fast_mem (FIFO-ish)
                        to_del = next(iter(fast_mem))
                        result2.append(('delete', to_del))
                        fast_mem.discard(to_del)
                        if to_del not in blue:
                            available.discard(to_del)
                    if len(fast_mem) < fast_memory_budget:
                        result2.append(('load', v))
                        fast_mem.add(v)
                        available.add(v)
            else:
                if len(fast_mem) >= fast_memory_budget:
                    to_del = next(iter(fast_mem))
                    result2.append(('delete', to_del))
                    fast_mem.discard(to_del)
                    if to_del not in blue:
                        available.discard(to_del)
                if len(fast_mem) < fast_memory_budget:
                    result2.append(('compute', v))
                    fast_mem.add(v)
                    available.add(v)
            if v in outputs and v in fast_mem:
                result2.append(('store', v))
                fast_mem.discard(v)
                blue.add(v)
        for v in fast_mem:
            result2.append(('delete', v))
            
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
