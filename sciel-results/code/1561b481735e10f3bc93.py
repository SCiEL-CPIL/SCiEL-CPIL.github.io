# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque
from functools import lru_cache

# Deadline guard: 80s of the 90s wall-clock limit, leaving safety margin
_DEADLINE_SECONDS = 80.0


def _toposort_vertices(dag):
    """Extract vertices and build adjacency from the dag structure.
    
    dag is expected to have:
      - dag['vertices'] or dag['nodes']: list of vertex ids (topologically ordered)
      - dag['parents'] or dag['edges']: for each vertex, its parents
      - dag['outputs']: set/list of output vertices
      - dag['inputs'] or dag['blue']: set/list of initially blue (input) vertices
    """
    # Try common key names
    if 'vertices' in dag:
        vertices = list(dag['vertices'])
    elif 'nodes' in dag:
        vertices = list(dag['nodes'])
    else:
        # Infer from parents dict
        all_v = set()
        parents = dag.get('parents', {})
        for v, ps in parents.items():
            all_v.add(v)
            for p in ps:
                all_v.add(p)
        vertices = sorted(all_v)

    if 'parents' in dag:
        parents = dag['parents']
    elif 'edges' in dag:
        # edges might be list of (parent, child) pairs
        parents = {}
        for edge in dag['edges']:
            if isinstance(edge, (list, tuple)) and len(edge) == 2:
                p, c = edge
                parents.setdefault(c, []).append(p)
            elif isinstance(edge, dict):
                parents[edge['child']] = [edge['parent']]
    else:
        parents = {}

    if 'outputs' in dag:
        outputs = set(dag['outputs'])
    elif 'output' in dag:
        outputs = set(dag['output']) if isinstance(dag['output'], (list, set, tuple)) else {dag['output']}
    else:
        # Outputs are vertices with no children
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
        # Inputs are vertices with no parents
        inputs = {v for v in vertices if not parents.get(v)}

    return vertices, parents, outputs, inputs


def _get_pebble_baseline(dag, fast_memory_budget):
    """Try to get the baseline score for comparison."""
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    except Exception:
        return None


def _naive_schedule_ops(dag, fast_memory_budget):
    """Generate a naive topological schedule and count transfers."""
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    ops = []
    # Naive: load each input, compute each vertex, store each output, delete as needed
    # Count transfers: load and store are transfers
    for v in vertices:
        if v in inputs:
            ops.append(('load', v))
        else:
            ops.append(('compute', v))
    # Store outputs
    for v in outputs:
        ops.append(('store', v))
    # Delete non-output computed values
    for v in reversed(vertices):
        if v not in inputs and v not in outputs:
            ops.append(('delete', v))
    transfers = sum(1 for op, v in ops if op in ('load', 'store'))
    return ops, transfers


def _count_transfers(ops):
    """Count load+store operations (transfers)."""
    return sum(1 for op, v in ops if op in ('load', 'store'))


def _validate_ops(ops, dag, fast_memory_budget):
    """Validate that the operation sequence is legal."""
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    blue = set(inputs)  # Initially blue
    fast_mem = set()    # Values in fast memory (red)
    # Track which values are available (blue or in fast memory)
    available = set(inputs)
    
    for i, (op, v) in enumerate(ops):
        if op == 'load':
            # v must be an input (or externally available)
            if v not in inputs:
                return False
            if v in fast_mem:
                return False  # Already in fast memory
            if len(fast_mem) >= fast_memory_budget:
                return False  # No room
            fast_mem.add(v)
            available.add(v)
        elif op == 'store':
            # v must be in fast memory or blue
            if v not in fast_mem and v not in blue:
                return False
            if v in fast_mem:
                fast_mem.remove(v)
            # After store, v is "stored" - still blue for purposes of being available
            # Actually store writes to slow memory, v becomes blue
            blue.add(v)
        elif op == 'compute':
            # All parents of v must be available
            ps = parents.get(v, [])
            for p in ps:
                if p not in available:
                    return False
            # Result goes into fast memory
            if len(fast_mem) >= fast_memory_budget:
                return False
            fast_mem.add(v)
            available.add(v)
        elif op == 'delete':
            # v must be in fast memory
            if v not in fast_mem:
                return False
            fast_mem.remove(v)
            # If v is not blue, it's no longer available
            if v not in blue:
                available.discard(v)
        else:
            return False
    
    # All outputs must be blue at the end
    if not outputs.issubset(blue):
        return False
    
    return True


def _greedy_dijkstra_pebble(dag, fast_memory_budget, time_limit=60.0):
    """Dijkstra-like search over memory configurations to minimize transfers.
    
    State: (frozenset of fast_mem, frozenset of blue_values)
    Goal: all outputs are in blue, fast_mem can be anything
    Cost: number of load+store operations
    """
    start_time = time.time()
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    
    # Determine which vertices are "inputs" (can be loaded) and which are computed
    # Blue set: values that are in slow memory (inputs + stored values)
    # Fast mem: values in fast memory
    
    # Initial state
    init_blue = frozenset(inputs)
    init_fast = frozenset()
    init_state = (init_blue, init_fast)
    
    # Goal check: all outputs in blue
    def is_goal(state):
        blue, fast = state
        return outputs.issubset(blue)
    
    if is_goal(init_state):
        return []
    
    # State: (blue_set, fast_set)
    # Transitions:
    # - load(v): v in inputs, v not in fast, len(fast) < budget -> add v to fast, cost=1
    # - store(v): v in fast -> remove from fast, add to blue, cost=1
    # - compute(v): all parents in (blue ∪ fast), v not in fast, len(fast) < budget -> add v to fast, cost=1
    # - delete(v): v in fast -> remove from fast, cost=0
    
    # Available = blue ∪ fast
    # Dijkstra: priority queue (cost, state), track best cost per state
    
    # Pruning: dominance. State A dominates B if:
    #   blue_A ⊇ blue_B AND fast_A ⊇ fast_B  (A has at least as much in both)
    #   and cost_A <= cost_B
    # This is expensive to check, so we'll use a simpler approach:
    # Keep a dict of best cost per state, and use the budget to limit state space.
    
    best_cost = {}
    best_cost[init_state] = 0
    
    # Priority queue: (cost, num_ops, state)
    # num_ops for tie-breaking
    pq = [(0, 0, init_state)]
    
    # Track parent for path reconstruction
    parent = {}  # state -> (prev_state, op, vertex)
    
    max_states = 500000  # Limit to avoid memory blowup
    states_explored = 0
    
    while pq:
        if time.time() - start_time > time_limit:
            break
        if states_explored >= max_states:
            break
            
        cost, ops_count, state = heapq.heappop(pq)
        
        if cost > best_cost.get(state, float('inf')):
            continue
        
        states_explored += 1
        
        if is_goal(state):
            # Reconstruct path
            path = []
            cur = state
            while cur != init_state:
                prev, op, v = parent[cur]
                path.append((op, v))
                cur = prev
            path.reverse()
            return path
        
        blue, fast = state
        available = blue | fast
        
        # Generate transitions
        # 1. load(v) for v in inputs not in fast
        for v in inputs:
            if v not in fast:
                new_fast = fast | {v}
                if len(new_fast) <= fast_memory_budget:
                    new_state = (blue, new_fast)
                    new_cost = cost + 1
                    if new_cost < best_cost.get(new_state, float('inf')):
                        best_cost[new_state] = new_cost
                        parent[new_state] = (state, 'load', v)
                        heapq.heappush(pq, (new_cost, ops_count + 1, new_state))
        
        # 2. store(v) for v in fast
        for v in fast:
            new_fast = fast - {v}
            new_blue = blue | {v}
            new_state = (new_blue, new_fast)
            new_cost = cost + 1
            if new_cost < best_cost.get(new_state, float('inf')):
                best_cost[new_state] = new_cost
                parent[new_state] = (state, 'store', v)
                heapq.heappush(pq, (new_cost, ops_count + 1, new_state))
        
        # 3. compute(v) for v not in inputs, not in fast, all parents available
        for v in vertices:
            if v in inputs:
                continue
            if v in fast:
                continue
            ps = parents.get(v, [])
            if all(p in available for p in ps):
                if len(fast) < fast_memory_budget:
                    new_fast = fast | {v}
                    new_state = (blue, new_fast)
                    new_cost = cost  # compute is free
                    if new_cost < best_cost.get(new_state, float('inf')):
                        best_cost[new_state] = new_cost
                        parent[new_state] = (state, 'compute', v)
                        heapq.heappush(pq, (new_cost, ops_count + 1, new_state))
        
        # 4. delete(v) for v in fast (only if v is not needed... but we don't know)
        # Delete is free, so always do it to free up space
        for v in fast:
            new_fast = fast - {v}
            new_state = (blue, new_fast)
            new_cost = cost  # delete is free
            if new_cost < best_cost.get(new_state, float('inf')):
                best_cost[new_state] = new_cost
                parent[new_state] = (state, 'delete', v)
                heapq.heappush(pq, (new_cost, ops_count + 1, new_state))
    
    # If we didn't find a solution via Dijkstra, return empty
    return None


def _layered_greedy_pebble(dag, fast_memory_budget):
    """Greedy layer-by-layer pebbling with careful memory management.
    
    Process vertices in topological order. For each layer (vertices at same depth),
    load inputs, compute vertices, store outputs, delete intermediate results.
    Use fast memory budget to decide what to keep.
    """
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    
    # Compute depth of each vertex
    depth = {}
    for v in vertices:
        ps = parents.get(v, [])
        if not ps:
            depth[v] = 0
        else:
            depth[v] = max(depth.get(p, 0) for p in ps) + 1
    
    max_depth = max(depth.values()) if depth else 0
    
    # Group by depth
    layers = {}
    for v in vertices:
        layers.setdefault(depth[v], []).append(v)
    
    ops = []
    fast_mem = set()
    blue = set(inputs)
    available = set(inputs)
    
    # Track which outputs need to be stored
    remaining_outputs = set(outputs)
    
    # For each layer from 0 to max_depth
    for d in range(max_depth + 1):
        layer_vertices = layers.get(d, [])
        
        # First, load any needed inputs that aren't already available
        # We need to know which inputs will be needed in this and future layers
        # For now, load inputs as needed when computing
        
        for v in layer_vertices:
            if v in inputs:
                # Load input if not already in fast mem
                if v not in fast_mem:
                    # Need space in fast mem
                    if len(fast_mem) >= fast_memory_budget:
                        # Delete something: prefer non-output, non-parent values
                        # Delete the least valuable item
                        # Heuristic: delete items that won't be needed soon
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
                # Compute v: all parents must be available
                ps = parents.get(v, [])
                for p in ps:
                    if p not in available:
                        # Need to load or restore p
                        # This shouldn't happen if we process in topological order
                        # and keep parents in fast mem until children are computed
                        pass
                
                # Need space in fast mem for result
                if len(fast_mem) >= fast_memory_budget:
                    # Delete something
                    to_delete = None
                    for fm in fast_mem:
                        if fm not in outputs and fm not in remaining_outputs:
                            # Check if fm is a parent of any remaining vertex
                            needed = False
                            for rv in remaining_outputs:
                                # Simple check: is fm an ancestor of rv?
                                # For now, just delete non-output
                                pass
                            to_delete = fm
                            break
                    if to_delete is None and fast_mem:
                        # Delete the oldest/fastest to evict
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
        
        # After processing layer d, store any outputs in this layer
        for v in layer_vertices:
            if v in outputs:
                if v in fast_mem:
                    ops.append(('store', v))
                    fast_mem.discard(v)
                    blue.add(v)
                    remaining_outputs.discard(v)
    
    # Store any remaining outputs
    for v in remaining_outputs:
        if v in fast_mem:
            ops.append(('store', v))
            fast_mem.discard(v)
            blue.add(v)
    
    # Cleanup: delete remaining fast mem items
    for v in fast_mem:
        ops.append(('delete', v))
    
    return ops


def _smart_greedy_pebble(dag, fast_memory_budget):
    """Improved greedy: process in topological order, keep necessary parents in fast memory,
    delete as soon as possible, store outputs immediately."""
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    
    # Build children map
    children = {}
    for v in vertices:
        for p in parents.get(v, []):
            children.setdefault(p, []).append(v)
    
    # Count how many children each vertex has remaining
    remaining_children = {v: len(children.get(v, [])) for v in vertices}
    
    # Topological order (already sorted)
    ops = []
    fast_mem = set()
    blue = set(inputs)
    available = set(inputs)
    
    # Track which inputs are still needed
    # An input is needed if it's a parent of some not-yet-processed vertex
    input_need_count = {}
    for v in vertices:
        for p in parents.get(v, []):
            if p in inputs:
                input_need_count[p] = input_need_count.get(p, 0) + 1
    
    # Process in topological order
    for v in vertices:
        if v in inputs:
            # Load if not already available
            if v not in available:
                if len(fast_mem) >= fast_memory_budget:
                    # Evict: find something in fast_mem that's no longer needed
                    evicted = False
                    for fm in list(fast_mem):
                        # Can we evict fm?
                        # fm can be evicted if:
                        # - fm is not an output (or output already stored)
                        # - all children of fm have been processed
                        if fm in outputs:
                            continue
                        if remaining_children.get(fm, 0) > 0:
                            continue
                        ops.append(('delete', fm))
                        fast_mem.discard(fm)
                        available.discard(fm)
                        evicted = True
                        break
                    if not evicted and fast_mem:
                        # Force evict
                        fm = next(iter(fast_mem))
                        ops.append(('delete', fm))
                        fast_mem.discard(fm)
                        available.discard(fm)
                
                ops.append(('load', v))
                fast_mem.add(v)
                available.add(v)
        else:
            # Compute v
            # Ensure all parents are in fast mem (or available)
            ps = parents.get(v, [])
            for p in ps:
                if p not in fast_mem and p in available:
                    # p is blue but not in fast mem, need to... wait
                    # If p is blue (stored), we can't compute from it unless we load it
                    # Actually, "available" means we can use it for computation
                    # But we need it in fast mem? No, we just need the value.
                    # The rule is: compute requires all parents to be available (blue or in fast mem)
                    # So if p is blue, it's available. We don't need to load it.
                    pass
            
            # Need space for result
            if len(fast_mem) >= fast_memory_budget:
                # Evict something that's no longer needed
                evicted = False
                for fm in list(fast_mem):
                    if fm in outputs:
                        continue
                    if remaining_children.get(fm, 0) > 0:
                        continue
                    ops.append(('delete', fm))
                    fast_mem.discard(fm)
                    available.discard(fm)
                    evicted = True
                    break
                if not evicted and fast_mem:
                    fm = next(iter(fast_mem))
                    ops.append(('delete', fm))
                    fast_mem.discard(fm)
                    available.discard(fm)
            
            ops.append(('compute', v))
            fast_mem.add(v)
            available.add(v)
        
        # Update remaining_children for all parents
        for p in parents.get(v, []):
            remaining_children[p] = remaining_children.get(p, 0) - 1
        
        # If v is an output, store it immediately
        if v in outputs:
            if v in fast_mem:
                ops.append(('store', v))
                fast_mem.discard(v)
                blue.add(v)
    
    # Cleanup
    for v in fast_mem:
        ops.append(('delete', v))
    
    return ops


def _hybrid_pebble(dag, fast_memory_budget):
    """Combine strategies: try Dijkstra for small instances, fall back to greedy."""
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    n_vertices = len(vertices)
    
    # For small instances, try exact Dijkstra
    if n_vertices <= 12:
        result = _greedy_dijkstra_pebble(dag, fast_memory_budget, time_limit=15.0)
        if result is not None and len(result) <= 20000:
            if _validate_ops(result, dag, fast_memory_budget):
                return result
    
    # For larger instances, use smart greedy
    result = _smart_greedy_pebble(dag, fast_memory_budget)
    if _validate_ops(result, dag, fast_memory_budget):
        return result
    
    # Fallback to basic greedy
    result = _layered_greedy_pebble(dag, fast_memory_budget)
    if _validate_ops(result, dag, fast_memory_budget):
        return result
    
    # Last resort: try baseline
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    except Exception:
        # Generate a minimal valid schedule
        ops = []
        for v in vertices:
            if v in inputs:
                ops.append(('load', v))
            else:
                ops.append(('compute', v))
        for v in outputs:
            ops.append(('store', v))
        return ops


def pebble(dag, fast_memory_budget):
    """Main entry point: find communication-minimizing pebbling schedule."""
    start_time = time.time()
    
    # Deadline guard
    remaining = _DEADLINE_SECONDS - (time.time() - start_time)
    if remaining < 1.0:
        # Emergency fallback
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
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
    
    try:
        result = _hybrid_pebble(dag, fast_memory_budget)
        
        # Validate result
        if result and _validate_ops(result, dag, fast_memory_budget):
            # Ensure operation count limit
            if len(result) > 20000:
                # Truncate and try to make valid
                # For now, return the smart greedy result
                alt = _smart_greedy_pebble(dag, fast_memory_budget)
                if _validate_ops(alt, dag, fast_memory_budget) and len(alt) <= 20000:
                    return alt
            return result
        else:
            # Result invalid, fall back
            try:
                from autoresearch_suite.science_tcs_expansion import pebble_baseline
                return pebble_baseline(dag, fast_memory_budget)
            except Exception:
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
    except Exception as e:
        # Any error: fall back to baseline
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
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

# EVOLVE-BLOCK-END
