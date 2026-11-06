"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import heapq

def pebble(dag, fast_memory_budget):
    """
    Minimize communication (load/store) operations for DAG pebbling.
    
    Uses Dijkstra's algorithm over memory configurations for small DAGs,
    falling back to an improved greedy strategy for larger ones.
    
    dag structure (assumed):
      - dag['vertices'] or dag['nodes']: list of vertex ids or range
      - dag['parents']: dict or list mapping vertex -> list of parent vertices
      - dag['inputs']: set/list of input vertices (start blue)
      - dag['outputs']: set/list of output vertices (must finish blue)
      - Or similar structure
      
    Returns: list of [operation, vertex] where operation in {load, store, compute, delete}
    """
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit
    
    # Parse DAG structure - try common formats
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
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
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
        children_map = {v: [] for v in vertex_list}
        for v in vertex_list:
            for p in parent_dict.get(v, []):
                children_map[p].append(v)
        outputs = set(v for v in vertex_list if not children_map[v])
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
    
    vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}
    
    # Determine which vertices are inputs (start blue)
    input_set = inputs & needed_set
    
    # For Dijkstra: determine if a vertex is an input
    is_input = {}
    for v in needed_set:
        is_input[v] = v in input_set
    
    # Try Dijkstra for small problems
    if num_vertices <= 15 and fast_memory_budget >= 1:
        result = _dijkstra_pebble(needed_vertices, parent_dict, needed_set, 
                                  input_set, outputs, fast_memory_budget, deadline)
        if result is not None:
            return result
    
    # Fall back to improved greedy
    return _greedy_pebble(needed_vertices, parent_dict, needed_set, input_set,
                          outputs, fast_memory_budget, children, vertex_to_idx, deadline)


def _dijkstra_pebble(needed_vertices, parent_dict, needed_set, input_set, 
                     outputs, budget, deadline):
    """
    Dijkstra over memory configurations.
    State: (frozenset of vertices in fast memory, frozenset of vertices computed)
    Actions: load(v), store(v), compute(v), delete(v)
    Cost: 1 for load/store, 0 for compute/delete
    Goal: all outputs in fast memory
    """
    # Map vertices to indices for compact representation
    v_to_i = {v: i for i, v in enumerate(needed_vertices)}
    n = len(needed_vertices)
    
    # Precompute parent indices
    parent_indices = {}
    for v in needed_vertices:
        parent_indices[v] = tuple(v_to_i[p] for p in parent_dict.get(v, []) if p in needed_set)
    
    # State: (fast_mask, computed_mask) where masks are integers
    # fast_mask: bits set for vertices in fast memory
    # computed_mask: bits set for vertices that have been computed (value exists)
    
    start_fast = 0
    start_computed = 0
    for v in needed_vertices:
        if v in input_set:
            # Inputs start blue - they're already computed and can be loaded
            start_computed |= (1 << v_to_i[v])
    
    # Dijkstra
    # dist[state] = (cost, fast_mask, computed_mask)
    # Use heap: (cost, fast_mask, computed_mask)
    
    start_state = (start_fast, start_computed)
    
    # Check if start is already goal
    goal_check = True
    for v in outputs:
        if v in needed_set:
            if not (start_fast & (1 << v_to_i[v])):
                goal_check = False
                break
    if goal_check:
        return []
    
    dist = {start_state: 0}
    parent_state = {}  # state -> (prev_state, op, vertex)
    
    heap = [(0, start_fast, start_computed)]
    
    max_states = 500000  # limit to prevent memory explosion
    states_explored = 0
    
    while heap:
        if time.time() > deadline:
            return None
        if len(dist) > max_states:
            return None
        
        cost, fast_mask, comp_mask = heapq.heappop(heap)
        state = (fast_mask, comp_mask)
        
        if dist.get(state, float('inf')) < cost:
            continue
        
        states_explored += 1
        
        # Check goal
        is_goal = True
        for v in outputs:
            if v in needed_set:
                if not (fast_mask & (1 << v_to_i[v])):
                    is_goal = False
                    break
        
        if is_goal:
            # Reconstruct path
            ops = []
            cur = state
            while cur in parent_state:
                prev, op, vertex = parent_state[cur]
                ops.append([op, vertex])
                cur = prev
            ops.reverse()
            return ops
        
        # Generate successor states
        # 1. Compute a vertex: all parents in fast memory, vertex not yet computed
        for i, v in enumerate(needed_vertices):
            if comp_mask & (1 << i):
                continue  # already computed
            # Check if all parents are in fast memory
            can_compute = True
            for p_idx in parent_indices[v]:
                if not (fast_mask & (1 << p_idx)):
                    can_compute = False
                    break
            if not can_compute:
                continue
            
            new_comp = comp_mask | (1 << i)
            new_fast = fast_mask | (1 << i)  # result goes to fast memory
            
            # Check budget
            if bin(new_fast).count('1') > budget:
                continue
            
            new_state = (new_fast, new_comp)
            new_cost = cost  # compute is free
            if new_cost < dist.get(new_state, float('inf')):
                dist[new_state] = new_cost
                parent_state[new_state] = (state, 'compute', v)
                heapq.heappush(heap, (new_cost, new_fast, new_comp))
        
        # 2. Load a vertex: computed but not in fast memory
        for i, v in enumerate(needed_vertices):
            if comp_mask & (1 << i):
                if not (fast_mask & (1 << i)):
                    new_fast = fast_mask | (1 << i)
                    if bin(new_fast).count('1') <= budget:
                        new_state = (new_fast, comp_mask)
                        new_cost = cost + 1
                        if new_cost < dist.get(new_state, float('inf')):
                            dist[new_state] = new_cost
                            parent_state[new_state] = (state, 'load', v)
                            heapq.heappush(heap, (new_cost, new_fast, comp_mask))
        
        # 3. Store a vertex: in fast memory, not an output (or even output if we'll reload)
        #    Actually, we can store any vertex in fast memory
        for i, v in enumerate(needed_vertices):
            if fast_mask & (1 << i):
                new_fast = fast_mask & ~(1 << i)
                new_state = (new_fast, comp_mask)
                new_cost = cost + 1
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    parent_state[new_state] = (state, 'store', v)
                    heapq.heappush(heap, (new_cost, new_fast, comp_mask))
        
        # 4. Delete a vertex: in fast memory, remove from computed too
        #    This is only useful if we want to recompute later
        for i, v in enumerate(needed_vertices):
            if fast_mask & (1 << i):
                new_fast = fast_mask & ~(1 << i)
                new_comp = comp_mask & ~(1 << i)
                new_state = (new_fast, new_comp)
                new_cost = cost  # delete is free
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    parent_state[new_state] = (state, 'delete', v)
                    heapq.heappush(heap, (new_cost, new_fast, new_comp))
    
    return None


def _greedy_pebble(needed_vertices, parent_dict, needed_set, input_set,
                   outputs, budget, children, vertex_to_idx, deadline):
    """
    Improved greedy pebbling with better eviction strategy.
    """
    in_fast = set()
    computed = set()
    operations = []
    
    # Track last use for each vertex
    last_child_idx = {}
    for v in needed_vertices:
        last_child_idx[v] = -1
        for child in children.get(v, []):
            if child in needed_set:
                last_child_idx[v] = max(last_child_idx[v], vertex_to_idx.get(child, -1))
    
    # Also consider: if v is an output, it must stay until the end
    for v in outputs:
        if v in needed_set:
            last_child_idx[v] = max(last_child_idx[v], len(vertex_list) if 'vertex_list' in dir() else 10**9)
    
    # Actually, let's use a large sentinel for outputs
    max_idx = max(vertex_to_idx.values()) if vertex_to_idx else 0
    for v in outputs:
        if v in needed_set:
            last_child_idx[v] = max(last_child_idx[v], max_idx + 1)
    
    v_idx = {v: i for i, v in enumerate(needed_vertices)}
    
    for v in needed_vertices:
        if time.time() > deadline:
            break
        
        v_pos = v_idx[v]
        parents = [p for p in parent_dict.get(v, []) if p in needed_set]
        
        # Ensure parents are in fast memory
        for p in parents:
            if p in in_fast:
                continue
            
            # Need to load p
            if len(in_fast) >= budget:
                # Evict: prefer vertices whose last use is before current position
                # among non-essential vertices
                candidates = []
                for u in in_fast:
                    if u == v:
                        continue
                    if u in parents:
                        continue
                    u_last = last_child_idx.get(u, -1)
                    candidates.append((u_last, u))
                
                # Sort by last use (earliest first)
                candidates.sort()
                
                # Try to evict one that's no longer needed
                evicted = False
                for u_last, u in candidates:
                    if u_last < v_pos:
                        operations.append(['store', u])
                        in_fast.discard(u)
                        evicted = True
                        break
                
                if not evicted and candidates:
                    # Evict the one with earliest last use
                    u_last, u = candidates[0]
                    operations.append(['store', u])
                    in_fast.discard(u)
            
            operations.append(['load', p])
            in_fast.add(p)
        
        # Compute v
        if v in input_set:
            # Input: just make sure it's in fast memory (already handled above if it's a parent)
            # If v itself is an input and not loaded yet, load it
            if v not in in_fast:
                if len(in_fast) >= budget:
                    # Evict
                    candidates = []
                    for u in in_fast:
                        if u in parents:
                            continue
                        u_last = last_child_idx.get(u, -1)
                        candidates.append((u_last, u))
                    candidates.sort()
                    if candidates:
                        u_last, u = candidates[0]
                        operations.append(['store', u])
                        in_fast.discard(u)
                operations.append(['load', v])
                in_fast.add(v)
            computed.add(v)
        else:
            operations.append(['compute', v])
            computed.add(v)
            in_fast.add(v)
        
        # After computing, if v is not needed by future vertices and not an output,
        # we could store it to free space
        # But let's be conservative and only store if we're at capacity and v is truly done
        v_last = last_child_idx.get(v, -1)
        if v_last < v_pos and v not in outputs:
            # v is no longer needed
            # Store it to free up space
            operations.append(['store', v])
            in_fast.discard(v)
    
    # Ensure all outputs are in fast memory
    for v in outputs:
        if v in needed_set and v not in in_fast:
            if v in computed:
                # Need to load
                if len(in_fast) >= budget:
                    # Evict non-output
                    for u in list(in_fast):
                        if u not in outputs:
                            operations.append(['store', u])
                            in_fast.discard(u)
                            break
                operations.append(['load', v])
                in_fast.add(v)
    
    return operations

# EVOLVE-BLOCK-END
