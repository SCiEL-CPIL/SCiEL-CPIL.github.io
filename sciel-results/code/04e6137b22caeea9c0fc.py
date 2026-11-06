import time
from collections import defaultdict, deque
import heapq
import sys

def pebble(dag, fast_memory_budget):
    """
    Minimize communication (load/store) operations for DAG pebbling.
    Returns: list of [operation, vertex] where operation in {load, store, compute, delete}
    """
    deadline = time.time() + 50  # Safety margin before 60s limit
    
    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('nodes', dag.get('V', None)))
        parents = dag.get('parents', dag.get('edges', dag.get('pred', None)))
        inputs = dag.get('inputs', dag.get('sources', dag.get('input_nodes', None)))
        outputs = dag.get('outputs', dag.get('sinks', dag.get('output_nodes', None)))
    else:
        vertices = None
        parents = None
        inputs = None
        outputs = None

    if vertices is None or parents is None:
        return []

    # Normalize to sets/lists
    if isinstance(vertices, (list, tuple, range)):
        vertex_list = list(vertices)
    elif isinstance(vertices, int):
        vertex_list = list(range(vertices))
    else:
        vertex_list = list(vertices)

    num_vertices = len(vertex_list)

    # Normalize parents to dict
    if isinstance(parents, dict):
        parent_dict = dict(parents)
    elif isinstance(parents, (list, tuple)):
        parent_dict = {i: list(p) for i, p in enumerate(parents)}
    else:
        parent_dict = {}

    for v in vertex_list:
        if v not in parent_dict:
            parent_dict[v] = []

    # Normalize inputs and outputs
    if inputs is None:
        inputs = set(v for v in vertex_list if not parent_dict.get(v, []))
    elif isinstance(inputs, (set, list, tuple)):
        inputs = set(inputs)
    else:
        inputs = set(inputs)

    if outputs is None:
        children = {v: [] for v in vertex_list}
        for v in vertex_list:
            for p in parent_dict.get(v, []):
                children[p].append(v)
        outputs = set(v for v in vertex_list if not children[v])
    elif isinstance(outputs, (set, list, tuple)):
        outputs = set(outputs)
    else:
        outputs = set(outputs)

    # Build children map
    children = {v: [] for v in vertex_list}
    for v in vertex_list:
        for p in parent_dict.get(v, []):
            children[p].append(v)

    # Determine needed vertices (ancestors of outputs)
    needed = set(outputs)
    for v in reversed(vertex_list):
        if v in needed:
            for p in parent_dict.get(v, []):
                needed.add(p)

    needed_vertices = [v for v in vertex_list if v in needed]
    needed_set = set(needed_vertices)
    
    n_needed = len(needed_vertices)
    if n_needed == 0:
        return []

    v_to_local = {v: i for i, v in enumerate(needed_vertices)}
    
    # Local parents/children
    local_parents = {i: [v_to_local[p] for p in parent_dict.get(needed_vertices[i], []) if p in needed_set] for i in range(n_needed)}
    local_children = {i: [v_to_local[c] for c in children.get(needed_vertices[i], []) if c in needed_set] for i in range(n_needed)}
    
    local_inputs = set(v_to_local[v] for v in inputs if v in needed_set)
    local_outputs = set(v_to_local[v] for v in outputs if v in needed_set)

    # Compute last use index for each local vertex
    last_child_idx = {}
    for i in range(n_needed):
        last_child_idx[i] = -1
        for c in local_children[i]:
            last_child_idx[i] = max(last_child_idx[i], c)

    # Detect layered structure
    layers = _detect_layers(local_parents, local_children, n_needed)
    
    # Strategy Selection
    # 1. A* for small problems (optimal or near-optimal)
    if fast_memory_budget <= 10 and n_needed <= 50:
        result = _astar_pebble(needed_vertices, local_parents, local_children, local_inputs, local_outputs,
                               fast_memory_budget, deadline)
        if result is not None:
            return result

    # 2. Layered divide-and-conquer for layered DAGs
    if layers is not None and len(layers) >= 2:
        result = _layered_pebble(needed_vertices, local_parents, local_children, local_inputs, local_outputs,
                                 fast_memory_budget, deadline, layers)
        if result is not None:
            return result

    # 3. Greedy for large
    return _greedy_pebble(needed_vertices, local_parents, local_children, local_inputs, local_outputs,
                          fast_memory_budget, deadline)

def _detect_layers(local_parents, local_children, n):
    """Detect if the DAG has a clear layered structure. Returns list of layers (each a set of vertex indices) or None."""
    if n == 0:
        return None
    
    # Compute "depth" (longest path from any input)
    depth = [0] * n
    for i in range(n):
        for p in local_parents.get(i, []):
            depth[i] = max(depth[i], depth[p] + 1)
    
    # Group by depth
    from collections import defaultdict
    depth_groups = defaultdict(set)
    for i in range(n):
        depth_groups[depth[i]].add(i)
    
    if len(depth_groups) < 2:
        return None
    
    # Check if it's a proper layered structure: each vertex's parents are all in one lower layer
    # and children are all in one higher layer
    max_depth = max(depth_groups.keys())
    layers = []
    for d in range(max_depth + 1):
        if d in depth_groups:
            layer = depth_groups[d]
            # Verify: all parents of vertices in this layer are in layer d-1 or lower
            # All children are in layer d+1 or higher
            valid = True
            for v in layer:
                for p in local_parents.get(v, []):
                    if depth[p] >= depth[v]:
                        valid = False
                        break
                for c in local_children.get(v, []):
                    if depth[c] <= depth[v]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                return None
            layers.append(layer)
    
    if len(layers) < 2:
        return None
    
    return layers

def _astar_pebble(needed_vertices, local_parents, local_children, local_inputs, local_outputs,
                  fast_memory_budget, deadline):
    """
    A* search with admissible heuristic for small problems.
    Uses bitsets for efficiency when n <= 64.
    """
    n = len(needed_vertices)
    if n == 0:
        return []
    
    # Use bitset representation for n <= 64
    if n <= 64:
        return _astar_bitset(needed_vertices, local_parents, local_children, local_inputs, local_outputs,
                             fast_memory_budget, deadline)
    else:
        return None

def _astar_bitset(needed_vertices, local_parents, local_children, local_inputs, local_outputs,
                  fast_memory_budget, deadline):
    """A* with bitset state representation for n <= 64."""
    n = len(needed_vertices)
    if n == 0:
        return []
    
    init_fast = 0
    init_computed = 0
    for v in local_inputs:
        init_computed |= (1 << v)
    
    init_state = (init_fast, init_computed)
    
    dist = {init_state: 0}
    parent = {}
    # (f_score, g_score, fast_mem, computed)
    pq = [(0, 0, init_fast, init_computed)]
    
    best_final_g = float('inf')
    best_final_state = None
    
    steps = 0
    max_steps = 100000
    
    # Precompute parent bitmasks
    parent_mask = [0] * n
    for i in range(n):
        mask = 0
        for p in local_parents.get(i, []):
            mask |= (1 << p)
        parent_mask[i] = mask
    
    # Precompute children lists
    children_list = [local_children.get(i, []) for i in range(n)]
    
    input_mask = 0
    for v in local_inputs:
        input_mask |= (1 << v)
    
    output_mask = 0
    for v in local_outputs:
        output_mask |= (1 << v)
    
    while pq:
        if time.time() > deadline:
            break
        if steps >= max_steps:
            break
        steps += 1
        
        f, cost, fast_mem, computed = heapq.heappop(pq)
        state = (fast_mem, computed)
        
        if dist.get(state, float('inf')) < cost:
            continue
        
        # Check if done
        if (fast_mem & output_mask) == output_mask:
            if cost < best_final_g:
                best_final_g = cost
                best_final_state = state
            continue
        
        # Find computable vertices
        computable = []
        for v in range(n):
            if computed & (1 << v):
                continue
            if (fast_mem & parent_mask[v]) == parent_mask[v]:
                computable.append(v)
        
        # Action 1: Compute
        for v in computable:
            new_computed = computed | (1 << v)
            new_fast = fast_mem | (1 << v)
            if bin(new_fast).count('1') <= fast_memory_budget:
                new_state = (new_fast, new_computed)
                new_cost = cost
                if dist.get(new_state, float('inf')) > new_cost:
                    dist[new_state] = new_cost
                    parent[new_state] = (state, 'compute', v)
                    h = _compute_heuristic_bitset(new_fast, new_computed, parent_mask, children_list, 
                                                   input_mask, output_mask, n, fast_memory_budget)
                    heapq.heappush(pq, (new_cost + h, new_cost, new_fast, new_computed))
        
        # Action 2: Load
        loadable = []
        for u in range(n):
            if fast_mem & (1 << u):
                continue
            if (input_mask & (1 << u)) or (computed & (1 << u)):
                loadable.append(u)
        
        for u in loadable:
            if bin(fast_mem).count('1') < fast_memory_budget:
                new_fast = fast_mem | (1 << u)
                new_state = (new_fast, computed)
                new_cost = cost + 1
                if dist.get(new_state, float('inf')) > new_cost:
                    dist[new_state] = new_cost
                    parent[new_state] = (state, 'load', u)
                    h = _compute_heuristic_bitset(new_fast, computed, parent_mask, children_list,
                                                   input_mask, output_mask, n, fast_memory_budget)
                    heapq.heappush(pq, (new_cost + h, new_cost, new_fast, computed))
            else:
                for u_evict in range(n):
                    if not (fast_mem & (1 << u_evict)):
                        continue
                    if input_mask & (1 << u_evict):
                        evict_cost = 0
                        evict_op = 'delete'
                    else:
                        evict_cost = 1
                        evict_op = 'store'
                    
                    new_fast = (fast_mem & ~(1 << u_evict)) | (1 << u)
                    new_state = (new_fast, computed)
                    new_cost = cost + 1 + evict_cost
                    if dist.get(new_state, float('inf')) > new_cost:
                        dist[new_state] = new_cost
                        parent[new_state] = (state, f'load_evict_{u_evict}_{evict_op}', u)
                        h = _compute_heuristic_bitset(new_fast, computed, parent_mask, children_list,
                                                       input_mask, output_mask, n, fast_memory_budget)
                        heapq.heappush(pq, (new_cost + h, new_cost, new_fast, computed))
        
        # Action 3: Store
        for u in range(n):
            if not (fast_mem & (1 << u)):
                continue
            if (computed & (1 << u)) and not (input_mask & (1 << u)):
                new_fast = fast_mem & ~(1 << u)
                new_state = (new_fast, computed)
                new_cost = cost + 1
                if dist.get(new_state, float('inf')) > new_cost:
                    dist[new_state] = new_cost
                    parent[new_state] = (state, 'store', u)
                    h = _compute_heuristic_bitset(new_fast, computed, parent_mask, children_list,
                                                   input_mask, output_mask, n, fast_memory_budget)
                    heapq.heappush(pq, (new_cost + h, new_cost, new_fast, computed))
        
        # Action 4: Delete
        for u in range(n):
            if not (fast_mem & (1 << u)):
                continue
            if input_mask & (1 << u):
                new_fast = fast_mem & ~(1 << u)
                new_state = (new_fast, computed)
                new_cost = cost
                if dist.get(new_state, float('inf')) > new_cost:
                    dist[new_state] = new_cost
                    parent[new_state] = (state, 'delete', u)
                    h = _compute_heuristic_bitset(new_fast, computed, parent_mask, children_list,
                                                   input_mask, output_mask, n, fast_memory_budget)
                    heapq.heappush(pq, (new_cost + h, new_cost, new_fast, computed))
            elif computed & (1 << u):
                has_future = False
                for c in children_list[u]:
                    if not (computed & (1 << c)):
                        has_future = True
                        break
                if not has_future:
                    new_fast = fast_mem & ~(1 << u)
                    new_state = (new_fast, computed)
                    new_cost = cost
                    if dist.get(new_state, float('inf')) > new_cost:
                        dist[new_state] = new_cost
                        parent[new_state] = (state, 'delete', u)
                        h = _compute_heuristic_bitset(new_fast, computed, parent_mask, children_list,
                                                       input_mask, output_mask, n, fast_memory_budget)
                        heapq.heappush(pq, (new_cost + h, new_cost, new_fast, computed))

    if best_final_state is None:
        return None
    
    ops = []
    state = best_final_state
    while state in parent:
        prev_state, action, vertex_local = parent[state]
        original_vertex = needed_vertices[vertex_local]
        
        if action == 'compute':
            ops.append(['compute', original_vertex])
        elif action == 'load':
            ops.append(['load', original_vertex])
        elif action.startswith('load_evict_'):
            parts = action.split('_')
            evict_local = int(parts[2])
            evict_op = parts[3]
            evict_vertex = needed_vertices[evict_local]
            ops.append([evict_op, evict_vertex])
            ops.append(['load', original_vertex])
        elif action == 'store':
            ops.append(['store', original_vertex])
        elif action == 'delete':
            ops.append(['delete', original_vertex])
        
        state = prev_state
    
    ops.reverse()
    
    if _verify_solution(ops, needed_vertices, local_parents, local_children, local_inputs, local_outputs, 
                        fast_memory_budget, set(needed_vertices)):
        return ops
    else:
        return None

def _compute_heuristic_bitset(fast_mem, computed, parent_mask, children_list, input_mask, output_mask, n, budget):
    """Fast heuristic for bitset states."""
    h = 0
    # Missing outputs
    missing_outputs = output_mask & ~fast_mem
    h += bin(missing_outputs).count('1')
    
    # Missing parents for uncomputed vertices
    missing_parents = 0
    for v in range(n):
        if computed & (1 << v):
            continue
        pm = parent_mask[v]
        missing = pm & ~fast_mem
        if missing:
            missing_parents |= missing
    
    h += bin(missing_parents).count('1')
    
    return h

def _layered_pebble(needed_vertices, local_parents, local_children, local_inputs, local_outputs,
                    fast_memory_budget, deadline, layers):
    """
    Exploit layered structure: process layer by layer, keeping boundary vertices in fast memory.
    """
    n = len(needed_vertices)
    in_fast = set()
    in_slow = set(local_inputs)
    computed = set(local_inputs)
    operations = []
    
    def load(v):
        operations.append(['load', needed_vertices[v]])
        in_fast.add(v)
        
    def store(v):
        operations.append(['store', needed_vertices[v]])
        in_fast.discard(v)
        in_slow.add(v)
        
    def compute(v):
        operations.append(['compute', needed_vertices[v]])
        computed.add(v)
        in_fast.add(v)
        
    def delete(v):
        operations.append(['delete', needed_vertices[v]])
        in_fast.discard(v)
        
    def evict_one(protected=None):
        if protected is None:
            protected = set()
        
        best_evict = None
        best_score = -float('inf')
        
        for u in list(in_fast):
            if u in protected:
                continue
            
            score = 0.0
            future_uses = 0
            for c in local_children.get(u, []):
                if c not in computed:
                    future_uses += 1
            
            if future_uses == 0:
                score += 1000
            else:
                score -= future_uses * 50
            
            if u in in_slow:
                score += 100
            else:
                score -= 100
            
            if u in local_inputs:
                score += 50
            
            if score > best_score:
                best_score = score
                best_evict = u

        if best_evict is not None:
            if best_evict in in_slow:
                delete(best_evict)
            else:
                store(best_evict)
            return best_evict
        return None
    
    # Process each layer
    for layer_idx, layer in enumerate(layers):
        if time.time() > deadline:
            break
        
        # Determine which vertices in this layer need to be computed
        layer_vertices = sorted(layer)
        
        # Identify boundary: vertices in this layer that are parents of next layer
        if layer_idx + 1 < len(layers):
            next_layer = layers[layer_idx + 1]
            boundary = set()
            for v in layer_vertices:
                for c in local_children.get(v, []):
                    if c in next_layer:
                        boundary.add(v)
                        break
        else:
            boundary = set(local_outputs) # Keep outputs
        
        # Compute vertices in this layer
        for v in layer_vertices:
            if v in computed:
                continue
            
            parents = local_parents.get(v, [])
            
            # Ensure parents are in fast memory
            for p in parents:
                if p in in_fast:
                    continue
                
                if len(in_fast) >= fast_memory_budget:
                    protected = set(parents) | boundary
                    evict_one(protected)
                
                load(p)
            
            # Compute v
            if len(in_fast) >= fast_memory_budget:
                protected = boundary
                evict_one(protected)
            
            compute(v)
        
        # After processing layer, evict vertices no longer needed (except boundary)
        if layer_idx + 1 < len(layers):
            next_layer_set = layers[layer_idx + 1]
            for u in list(in_fast):
                if u in boundary:
                    continue
                # Check if u is needed by any future vertex
                needed_future = False
                for c in local_children.get(u, []):
                    if c not in computed:
                        needed_future = True
                        break
                if not needed_future:
                    if u in in_slow:
                        delete(u)
                    else:
                        store(u)
    
    # Ensure outputs in fast
    for v in local_outputs:
        if v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                protected = set(local_outputs)
                evict_one(protected)
            load(v)
    
    if len(operations) <= 20000:
        if _verify_solution(operations, needed_vertices, local_parents, local_children, local_inputs, local_outputs,
                           fast_memory_budget, set(needed_vertices)):
            return operations
    
    return None

def _verify_solution(ops, needed_vertices, local_parents, local_children, local_inputs, local_outputs,
                     fast_memory_budget, needed_set):
    """Verify that a solution is valid."""
    if len(ops) > 20000:
        return False
    
    in_fast = set()
    in_slow = set(local_inputs)
    computed = set(local_inputs)
    
    v_to_local = {v: i for i, v in enumerate(needed_vertices)}
    
    for op, v in ops:
        if v not in v_to_local:
            return False
        v_local = v_to_local[v]
        
        if op == 'load':
            if v_local not in in_slow:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v_local)
        elif op == 'store':
            if v_local not in in_fast:
                return False
            in_fast.discard(v_local)
            in_slow.add(v_local)
        elif op == 'compute':
            parents_local = local_parents.get(v_local, [])
            if not all(p in in_fast for p in parents_local):
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v_local)
            computed.add(v_local)
        elif op == 'delete':
            if v_local not in in_fast:
                return False
            in_fast.discard(v_local)
    
    for o in local_outputs:
        if o not in in_fast:
            return False
    
    return True

def _greedy_pebble(needed_vertices, local_parents, local_children, local_inputs, local_outputs,
                   fast_memory_budget, deadline):
    """Improved greedy pebbling with better eviction heuristic."""
    n = len(needed_vertices)
    in_fast = set()
    in_slow = set(local_inputs)
    computed = set(local_inputs)
    operations = []

    def load(v):
        operations.append(['load', needed_vertices[v]])
        in_fast.add(v)
        
    def store(v):
        operations.append(['store', needed_vertices[v]])
        in_fast.discard(v)
        in_slow.add(v)
        
    def compute(v):
        operations.append(['compute', needed_vertices[v]])
        computed.add(v)
        in_fast.add(v)
        
    def delete(v):
        operations.append(['delete', needed_vertices[v]])
        in_fast.discard(v)

    def evict_one(protected=None):
        if protected is None:
            protected = set()
        
        best_evict = None
        best_score = -float('inf')
        
        for u in list(in_fast):
            if u in protected:
                continue
            
            score = 0.0
            future_uses = 0
            for c in local_children.get(u, []):
                if c not in computed:
                    future_uses += 1
            
            if future_uses == 0:
                score += 1000
            else:
                score -= future_uses * 50
            
            if u in in_slow:
                score += 100
            else:
                score -= 100
            
            if u in local_inputs:
                score += 50
            
            if score > best_score:
                best_score = score
                best_evict = u

        if best_evict is not None:
            if best_evict in in_slow:
                delete(best_evict)
            else:
                store(best_evict)
            return best_evict
        return None

    for pos in range(n):
        if time.time() > deadline:
            break
            
        v = pos
        parents = local_parents.get(v, [])

        for p in parents:
            if p in in_fast:
                continue
            
            if len(in_fast) >= fast_memory_budget:
                protected = set(parents)
                evict_one(protected)

            load(p)

        if v not in in_fast:
            if v in local_inputs:
                if len(in_fast) >= fast_memory_budget:
                    protected = set()
                    evict_one(protected)
                load(v)
            else:
                compute(v)

        for u in list(in_fast):
            if u in local_outputs:
                continue
            future_uses = 0
            for c in local_children.get(u, []):
                if c not in computed:
                    future_uses += 1
            if future_uses == 0:
                if u in in_slow:
                    delete(u)
                else:
                    store(u)

    for v in local_outputs:
        if v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                protected = set(local_outputs)
                evict_one(protected)
            load(v)

    return operations
