# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque, defaultdict
from functools import lru_cache
import math

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


def _dijkstra_pebble_bitmask(dag, fast_memory_budget, time_limit=20.0):
    """Dijkstra search using bitmask state encoding for small/medium instances.
    
    Encodes (blue_mask, fast_mask) as a single integer for fast hashing.
    Only viable when n_vertices <= 20 and budget is small.
    """
    start_time = time.time()
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    n = len(vertices)
    
    if n > 20:
        return None
    
    # Map vertices to indices
    v_idx = {v: i for i, v in enumerate(vertices)}
    
    # Bitmasks
    init_blue_mask = 0
    for v in inputs:
        init_blue_mask |= (1 << v_idx[v])
    
    outputs_mask = 0
    for v in outputs:
        outputs_mask |= (1 << v_idx[v])
    
    inputs_mask = 0
    for v in inputs:
        inputs_mask |= (1 << v_idx[v])
    
    # Precompute parent masks
    parent_mask = {}
    for v in vertices:
        pm = 0
        for p in parents.get(v, []):
            pm |= (1 << v_idx[p])
        parent_mask[v] = pm
    
    budget = fast_memory_budget
    if budget > n:
        budget = n
    
    # State: (blue_mask, fast_mask) encoded as blue_mask * (1 << n) + fast_mask
    init_state = (init_blue_mask, 0)
    
    def is_goal(state):
        blue_m, fast_m = state
        return (blue_m & outputs_mask) == outputs_mask
    
    if is_goal(init_state):
        return []
    
    best_cost = {}
    best_cost[init_state] = 0
    
    pq = [(0, 0, init_state)]  # (cost, ops_count, state)
    parent = {}
    
    max_states = 200000
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
            path = []
            cur = state
            while cur != init_state:
                prev, op, v = parent[cur]
                path.append((op, v))
                cur = prev
            path.reverse()
            return path
        
        blue_m, fast_m = state
        avail_m = blue_m | fast_m
        
        # 1. load(v) for input v not in fast
        for v in inputs:
            bit = 1 << v_idx[v]
            if not (fast_m & bit):
                new_fast_m = fast_m | bit
                if bin(new_fast_m).count('1') <= budget:
                    new_state = (blue_m, new_fast_m)
                    new_cost = cost + 1
                    if new_cost < best_cost.get(new_state, float('inf')):
                        best_cost[new_state] = new_cost
                        parent[new_state] = (state, 'load', v)
                        heapq.heappush(pq, (new_cost, ops_count + 1, new_state))
        
        # 2. store(v) for v in fast
        fm = fast_m
        while fm:
            lb = fm & (-fm)
            v_idx_bit = lb.bit_length() - 1
            v = vertices[v_idx_bit]
            new_fast_m = fast_m & ~lb
            new_blue_m = blue_m | lb
            new_state = (new_blue_m, new_fast_m)
            new_cost = cost + 1
            if new_cost < best_cost.get(new_state, float('inf')):
                best_cost[new_state] = new_cost
                parent[new_state] = (state, 'store', v)
                heapq.heappush(pq, (new_cost, ops_count + 1, new_state))
            fm &= fm - 1
        
        # 3. compute(v) for non-input v, all parents available, v not in fast
        for v in vertices:
            if v in inputs:
                continue
            bit = 1 << v_idx[v]
            if fast_m & bit:
                continue
            if (parent_mask[v] & ~avail_m) != 0:
                continue
            if bin(fast_m).count('1') < budget:
                new_fast_m = fast_m | bit
                new_state = (blue_m, new_fast_m)
                new_cost = cost
                if new_cost < best_cost.get(new_state, float('inf')):
                    best_cost[new_state] = new_cost
                    parent[new_state] = (state, 'compute', v)
                    heapq.heappush(pq, (new_cost, ops_count + 1, new_state))
        
        # 4. delete(v) for v in fast
        fm = fast_m
        while fm:
            lb = fm & (-fm)
            v_idx_bit = lb.bit_length() - 1
            v = vertices[v_idx_bit]
            new_fast_m = fast_m & ~lb
            new_state = (blue_m, new_fast_m)
            new_cost = cost
            if new_cost < best_cost.get(new_state, float('inf')):
                best_cost[new_state] = new_cost
                parent[new_state] = (state, 'delete', v)
                heapq.heappush(pq, (new_cost, ops_count + 1, new_state))
            fm &= fm - 1
    
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
    """Improved greedy: process in topological order with optimal eviction strategy.
    
    Key improvements:
    - Track exact number of remaining children for each vertex
    - Evict vertices whose last child has been processed (no longer needed)
    - Prioritize evicting vertices with no remaining children
    - Store outputs immediately to free fast memory
    """
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    
    # Build children map
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Count how many children each vertex has remaining
    remaining_children = {v: len(children.get(v, [])) for v in vertices}
    
    # Process in topological order
    ops = []
    fast_mem = set()
    blue = set(inputs)
    available = set(inputs)
    
    # Track which vertices are still needed (have unprocessed children)
    # A vertex is "needed" if it has remaining_children > 0 or is an output not yet stored
    
    def can_evict(fm):
        """Check if fm can be safely evicted from fast memory."""
        # If fm is an output and not yet stored, we can't evict
        if fm in outputs and fm not in blue:
            return False
        # If fm still has children to process, we can't evict
        if remaining_children.get(fm, 0) > 0:
            return False
        return True
    
    def find_evictable():
        """Find a vertex in fast_mem that can be safely evicted."""
        for fm in fast_mem:
            if can_evict(fm):
                return fm
        return None
    
    for v in vertices:
        if v in inputs:
            # Load if not already available
            if v not in available:
                if len(fast_mem) >= fast_memory_budget:
                    # Need to evict
                    evictable = find_evictable()
                    if evictable is None:
                        # Force evict: pick the one with fewest remaining children
                        best = None
                        best_count = float('inf')
                        for fm in fast_mem:
                            rc = remaining_children.get(fm, 0)
                            if fm in outputs and fm not in blue:
                                continue
                            if rc < best_count:
                                best_count = rc
                                best = fm
                        if best is not None:
                            evictable = best
                        else:
                            evictable = next(iter(fast_mem))
                    
                    ops.append(('delete', evictable))
                    fast_mem.discard(evictable)
                    if evictable not in blue:
                        available.discard(evictable)
                
                ops.append(('load', v))
                fast_mem.add(v)
                available.add(v)
        else:
            # Compute v
            # Need space for result
            if len(fast_mem) >= fast_memory_budget:
                # Need to evict
                evictable = find_evictable()
                if evictable is None:
                    # Force evict: pick the one with fewest remaining children
                    best = None
                    best_count = float('inf')
                    for fm in fast_mem:
                        if fm in outputs and fm not in blue:
                            continue
                        rc = remaining_children.get(fm, 0)
                        if rc < best_count:
                            best_count = rc
                            best = fm
                    if best is not None:
                        evictable = best
                    else:
                        evictable = next(iter(fast_mem))
                
                ops.append(('delete', evictable))
                fast_mem.discard(evictable)
                if evictable not in blue:
                    available.discard(evictable)
            
            ops.append(('compute', v))
            fast_mem.add(v)
            available.add(v)
        
        # Update remaining_children for all parents of v
        for p in parents.get(v, []):
            remaining_children[p] = remaining_children.get(p, 0) - 1
        
        # If v is an output, store it immediately
        if v in outputs:
            if v in fast_mem:
                ops.append(('store', v))
                fast_mem.discard(v)
                blue.add(v)
    
    # Cleanup: delete remaining fast mem items
    for v in list(fast_mem):
        ops.append(('delete', v))
        fast_mem.discard(v)
    
    return ops


def _local_search_polish(ops, dag, fast_memory_budget, max_iterations=50):
    """Try to reduce transfers by local search: swap adjacent operations or remove redundant ones.
    
    Returns improved ops if found, else original.
    """
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    
    current_ops = list(ops)
    current_transfers = _count_transfers(current_ops)
    
    for _ in range(max_iterations):
        improved = False
        
        # Try removing redundant delete+load pairs
        new_ops = []
        i = 0
        while i < len(current_ops):
            if i + 1 < len(current_ops):
                op1, v1 = current_ops[i]
                op2, v2 = current_ops[i + 1]
                # If we delete v then immediately load v, skip both
                if op1 == 'delete' and op2 == 'load' and v1 == v2 and v2 in inputs:
                    i += 2
                    continue
                # If we load v then immediately delete v, skip both
                if op1 == 'load' and op2 == 'delete' and v1 == v2 and v1 in inputs:
                    i += 2
                    continue
            new_ops.append(current_ops[i])
            i += 1
        
        if len(new_ops) < len(current_ops):
            if _validate_ops(new_ops, dag, fast_memory_budget):
                current_ops = new_ops
                improved = True
        
        # Try swapping adjacent non-transfer operations
        for i in range(len(current_ops) - 1):
            op1, v1 = current_ops[i]
            op2, v2 = current_ops[i + 1]
            # Only swap if neither is a transfer
            if op1 in ('compute', 'delete') and op2 in ('compute', 'delete'):
                swapped = list(current_ops)
                swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
                if _validate_ops(swapped, dag, fast_memory_budget):
                    # Check if this reduces transfers (it won't directly, but may enable future reductions)
                    current_ops = swapped
                    improved = True
                    break
        
        if not improved:
            break
    
    return current_ops


def _hybrid_pebble(dag, fast_memory_budget):
    """Combine strategies: try bitmask Dijkstra for small instances, then greedy with polish."""
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    n_vertices = len(vertices)
    
    # For small instances, try exact Dijkstra with bitmask encoding
    if n_vertices <= 20 and fast_memory_budget <= 8:
        result = _dijkstra_pebble_bitmask(dag, fast_memory_budget, time_limit=15.0)
        if result is not None and len(result) <= 20000:
            if _validate_ops(result, dag, fast_memory_budget):
                # Apply local search polish
                polished = _local_search_polish(result, dag, fast_memory_budget)
                if _validate_ops(polished, dag, fast_memory_budget) and len(polished) <= 20000:
                    return polished
                return result
    
    # For medium instances, try set-based Dijkstra with tighter limit
    if n_vertices <= 15:
        result = _dijkstra_pebble_bitmask(dag, fast_memory_budget, time_limit=10.0)
        if result is not None and len(result) <= 20000:
            if _validate_ops(result, dag, fast_memory_budget):
                polished = _local_search_polish(result, dag, fast_memory_budget)
                if _validate_ops(polished, dag, fast_memory_budget) and len(polished) <= 20000:
                    return polished
                return result
    
    # For larger instances, use smart greedy with polish
    result = _smart_greedy_pebble(dag, fast_memory_budget)
    if _validate_ops(result, dag, fast_memory_budget) and len(result) <= 20000:
        # Apply local search polish
        polished = _local_search_polish(result, dag, fast_memory_budget)
        if _validate_ops(polished, dag, fast_memory_budget) and len(polished) <= 20000:
            return polished
        return result
    
    # Fallback to basic greedy
    result = _layered_greedy_pebble(dag, fast_memory_budget)
    if _validate_ops(result, dag, fast_memory_budget) and len(result) <= 20000:
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
    if time.time() - start_time > _DEADLINE_SECONDS - 1.0:
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
    except Exception:
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
