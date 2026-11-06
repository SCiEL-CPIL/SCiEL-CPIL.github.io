"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import heapq
from collections import defaultdict

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
    
    # Try Dijkstra for small problems with dominance pruning
    if num_vertices <= 20 and fast_memory_budget >= 1:
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
    Dijkstra over memory configurations with dominance pruning.
    State: (fast_mask, computed_mask) where masks are integers
    Actions: load(v), store(v), compute(v), delete(v)
    Cost: 1 for load/store, 0 for compute/delete
    Goal: all outputs in fast memory
    
    Dominance: State A dominates B if A.fast ⊇ B.fast and A.comp ⊇ B.comp and A.cost <= B.cost
    We prune B if any explored state A dominates it.
    """
    v_to_i = {v: i for i, v in enumerate(needed_vertices)}
    n = len(needed_vertices)
    
    # Precompute parent indices
    parent_indices = {}
    for v in needed_vertices:
        parent_indices[v] = tuple(v_to_i[p] for p in parent_dict.get(v, []) if p in needed_set)
    
    # Precompute output mask
    output_mask = 0
    for v in outputs:
        if v in needed_set:
            output_mask |= (1 << v_to_i[v])
    
    # Start state
    start_fast = 0
    start_computed = 0
    for v in needed_vertices:
        if v in input_set:
            start_computed |= (1 << v_to_i[v])
    
    start_state = (start_fast, start_computed)
    
    # Check if start is already goal
    if (start_fast & output_mask) == output_mask:
        return []
    
    # dist[state] = cost
    # For dominance: keep list of (cost, fast, comp) for explored states
    dist = {start_state: 0}
    parent_state = {}  # state -> (prev_state, op, vertex)
    
    # For dominance checking: store all visited states with their costs
    # visited_states: list of (cost, fast_mask, comp_mask)
    visited_states = [(0, start_fast, start_computed)]
    
    heap = [(0, start_fast, start_computed)]
    
    max_states = 2000000  # higher limit with dominance pruning
    states_explored = 0
    max_heap_size = 5000000
    
    while heap:
        if time.time() > deadline:
            return None
        if len(dist) > max_states or len(heap) > max_heap_size:
            return None
        
        cost, fast_mask, comp_mask = heapq.heappop(heap)
        state = (fast_mask, comp_mask)
        
        if dist.get(state, float('inf')) < cost:
            continue
        
        states_explored += 1
        
        # Check goal
        if (fast_mask & output_mask) == output_mask:
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
        successors = []
        
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
                successors.append((new_cost, new_fast, new_comp))
        
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
                            successors.append((new_cost, new_fast, comp_mask))
        
        # 3. Store a vertex: in fast memory
        for i, v in enumerate(needed_vertices):
            if fast_mask & (1 << i):
                new_fast = fast_mask & ~(1 << i)
                new_state = (new_fast, comp_mask)
                new_cost = cost + 1
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    parent_state[new_state] = (state, 'store', v)
                    successors.append((new_cost, new_fast, comp_mask))
        
        # 4. Delete a vertex: in fast memory, remove from computed too
        for i, v in enumerate(needed_vertices):
            if fast_mask & (1 << i):
                new_fast = fast_mask & ~(1 << i)
                new_comp = comp_mask & ~(1 << i)
                new_state = (new_fast, new_comp)
                new_cost = cost  # delete is free
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    parent_state[new_state] = (state, 'delete', v)
                    successors.append((new_cost, new_fast, new_comp))
        
        # Push successors to heap, applying dominance pruning
        for scost, sfast, scomp in successors:
            # Check if this state is dominated by any previously explored state
            dominated = False
            for ecost, efast, ecomp in visited_states:
                if ecost <= scost and (efast | sfast) == efast and (ecomp | scomp) == ecomp:
                    # efast ⊇ sfast and ecomp ⊇ scomp and ecost <= scost
                    dominated = True
                    break
            if not dominated:
                heapq.heappush(heap, (scost, sfast, scomp))
                visited_states.append((scost, sfast, scomp))
    
    return None


def _greedy_pebble(needed_vertices, parent_dict, needed_set, input_set,
                   outputs, budget, children, vertex_to_idx, deadline):
    """
    Improved greedy pebbling with better eviction strategy.
    Uses topological order and evicts based on last-use analysis.
    """
    in_fast = set()
    computed = set()
    operations = []
    
    v_idx = {v: i for i, v in enumerate(needed_vertices)}
    n = len(needed_vertices)
    
    # Track last use for each vertex (position in needed_vertices order)
    last_use = {}
    for v in needed_vertices:
        last_use[v] = -1
        for child in children.get(v, []):
            if child in needed_set:
                last_use[v] = max(last_use[v], v_idx.get(child, -1))
    
    # Outputs must stay until the end
    for v in outputs:
        if v in needed_set:
            last_use[v] = max(last_use[v], n)
    
    # Process vertices in topological order
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
                candidates = []
                for u in in_fast:
                    if u in parents:
                        continue
                    u_last = last_use.get(u, -1)
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
            # Input: just make sure it's in fast memory
            if v not in in_fast:
                if len(in_fast) >= budget:
                    # Evict
                    candidates = []
                    for u in in_fast:
                        if u in parents:
                            continue
                        u_last = last_use.get(u, -1)
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
        # store it to free space
        v_last = last_use.get(v, -1)
        if v_last < v_pos and v not in outputs:
            operations.append(['store', v])
            in_fast.discard(v)
    
    # Ensure all outputs are in fast memory
    for v in outputs:
        if v in needed_set and v not in in_fast:
            if v in computed:
                if len(in_fast) >= budget:
                    for u in list(in_fast):
                        if u not in outputs:
                            operations.append(['store', u])
                            in_fast.discard(u)
                            break
                operations.append(['load', v])
                in_fast.add(v)
    
    return operations

# EVOLVE-BLOCK-END
