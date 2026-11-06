"""Rewritten program for Communication-minimizing DAG pebbling."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict
import heapq

def pebble(dag, fast_memory_budget):
    """
    Minimize communication (load/store) operations for DAG pebbling.
    Returns: list of [operation, vertex] where operation in {load, store, compute, delete}
    """
    deadline = time.time() + 55  # Safety margin before 60s limit

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
    vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}

    # Try optimal Dijkstra-based search for small problems
    if len(needed_vertices) <= 12 and fast_memory_budget <= 4:
        result = _optimal_pebble(needed_vertices, parent_dict, needed_set,
                                 inputs, outputs, fast_memory_budget, deadline)
        if result is not None:
            return result

    # Fallback to greedy FNU approach
    return _greedy_pebble(needed_vertices, parent_dict, needed_set,
                          inputs, outputs, fast_memory_budget, deadline,
                          vertex_to_idx)


def _optimal_pebble(needed_vertices, parent_dict, needed_set,
                    inputs, outputs, fast_memory_budget, deadline):
    """
    Dijkstra-based search over memory states to find minimum-transfer pebbling.
    State: (frozenset of vertices in fast memory, frozenset of computed vertices)
    Cost: number of transfers (load/store operations)
    """
    # Initial state: no fast memory, no computed vertices
    init_fast = frozenset()
    init_computed = frozenset()
    init_state = (init_fast, init_computed)

    # Dijkstra's algorithm
    # dist[state] = minimum transfers to reach this state
    dist = {init_state: 0}
    # parent tracking: state -> (prev_state, operation, vertex)
    parent = {}
    
    # Priority queue: (cost, fast_frozen, computed_frozen)
    pq = [(0, init_fast, init_computed)]
    
    # Precompute which vertices can be computed given a set of computed vertices
    # A vertex v can be computed if all its needed parents are in computed set
    # and v is not already computed
    
    # Track number of states explored for timeout
    states_explored = 0
    max_states = 50000  # Limit to avoid timeout
    
    while pq:
        if time.time() > deadline - 2:
            return None
        if states_explored > max_states:
            return None
            
        cost, fast, computed = heapq.heappop(pq)
        
        if cost > dist.get((fast, computed), float('inf')):
            continue
            
        states_explored += 1
        
        # Check if we've reached goal: all outputs are in fast memory
        if outputs.issubset(fast):
            # Reconstruct path
            ops = []
            state = (fast, computed)
            while state in parent:
                prev_state, op, vertex = parent[state]
                ops.append([op, vertex])
                state = prev_state
            ops.reverse()
            return ops
        
        # Generate successor states
        # Actions: load, store, compute, delete
        
        # 1. LOAD: bring a vertex from slow memory to fast
        #    Can load: inputs not in fast, or computed vertices not in fast
        #    Must have space in fast memory
        if len(fast) < fast_memory_budget:
            # Vertices that can be loaded: inputs not in fast, or computed not in fast
            loadable = (inputs | computed) - fast
            for v in loadable:
                if v not in needed_set:
                    continue
                new_fast = fast | {v}
                new_state = (new_fast, computed)
                new_cost = cost + 1
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    parent[new_state] = ((fast, computed), 'load', v)
                    heapq.heappush(pq, (new_cost, new_fast, computed))
        
        # 2. STORE: move a vertex from fast to slow memory
        #    Can store: any vertex in fast that is either an input or computed
        #    (not a freshly computed result that hasn't been stored yet? 
        #     Actually, any vertex in fast can be stored if it's been computed)
        #    After storing, the vertex is no longer in fast
        for v in fast:
            # Can only store if v is an input or has been computed
            # (a vertex in fast that hasn't been computed yet doesn't make sense
            #  unless it was loaded, but loaded vertices are either inputs or computed)
            if v in inputs or v in computed:
                new_fast = fast - {v}
                new_state = (new_fast, computed)
                new_cost = cost + 1
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    parent[new_state] = ((fast, computed), 'store', v)
                    heapq.heappush(pq, (new_cost, new_fast, computed))
        
        # 3. COMPUTE: compute a vertex whose parents are all in fast memory
        #    Result goes into fast memory (uses one slot)
        #    Must have space, or we need to evict first (but that's a separate action)
        #    Actually, compute requires parents in fast, and result takes a slot
        #    So we need len(fast) < budget OR we can evict as part of this step?
        #    Let's keep it simple: compute only if there's room
        for v in needed_set:
            if v in computed:
                continue
            # Check if all needed parents are in fast
            v_parents = [p for p in parent_dict.get(v, []) if p in needed_set]
            if all(p in fast for p in v_parents):
                if len(fast) < fast_memory_budget:
                    new_fast = fast | {v}
                    new_computed = computed | {v}
                    new_state = (new_fast, new_computed)
                    new_cost = cost  # compute doesn't cost a transfer
                    if new_cost < dist.get(new_state, float('inf')):
                        dist[new_state] = new_cost
                        parent[new_state] = ((fast, computed), 'compute', v)
                        heapq.heappush(pq, (new_cost, new_fast, new_computed))
        
        # 4. DELETE: remove a vertex from fast memory without storing
        #    Only valid for vertices that are NOT inputs and NOT needed later
        #    (i.e., computed vertices that won't be used again)
        #    For simplicity, allow deleting any computed vertex that's not an input
        #    This frees up space
        for v in fast:
            if v in computed and v not in inputs:
                new_fast = fast - {v}
                new_state = (new_fast, computed)
                new_cost = cost  # delete doesn't cost a transfer
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    parent[new_state] = ((fast, computed), 'delete', v)
                    heapq.heappush(pq, (new_cost, new_fast, computed))
    
    return None  # No solution found within limits


def _greedy_pebble(needed_vertices, parent_dict, needed_set,
                   inputs, outputs, fast_memory_budget, deadline, vertex_to_idx):
    """Greedy FNU-based pebbling as fallback."""
    # Precompute next-use positions for each vertex
    next_use = defaultdict(list)
    for i, v in enumerate(needed_vertices):
        for p in parent_dict.get(v, []):
            if p in needed_set:
                next_use[p].append(i)

    # State
    in_fast = set()
    in_slow = set(inputs)  # Inputs start in slow memory
    computed = set()
    operations = []
    
    # Track current position for FNU
    current_pos = 0

    def get_next_use(u, pos):
        """Get the next use position of u after position pos, or infinity if never used again."""
        uses = next_use.get(u, [])
        # Binary search for first use > pos
        lo, hi = 0, len(uses)
        while lo < hi:
            mid = (lo + hi) // 2
            if uses[mid] <= pos:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(uses):
            return uses[lo]
        return float('inf')

    def load(v):
        operations.append(['load', v])
        in_fast.add(v)
        
    def store(v):
        operations.append(['store', v])
        in_fast.discard(v)
        in_slow.add(v)
        
    def compute(v):
        operations.append(['compute', v])
        computed.add(v)
        in_fast.add(v)
        
    def delete(v):
        operations.append(['delete', v])
        in_fast.discard(v)

    def evict_one(protect_set):
        """Evict one vertex from fast memory using FNU strategy.
        protect_set: set of vertices that should not be evicted."""
        if not in_fast or len(in_fast) <= 0:
            return None
        
        best_evict = None
        best_next_use = -1  # We want to maximize next_use (furthest in future)
        
        for u in list(in_fast):
            if u in protect_set:
                continue
            nu = get_next_use(u, current_pos)
            # FNU: evict the one with the furthest next use
            if nu > best_next_use:
                best_next_use = nu
                best_evict = u
        
        if best_evict is None:
            # All protected, pick any
            for u in in_fast:
                best_evict = u
                break
        
        if best_evict is not None:
            if best_evict in in_slow:
                delete(best_evict)
            else:
                store(best_evict)
        
        return best_evict

    # Process vertices in topological order
    for v in needed_vertices:
        if time.time() > deadline:
            break
        
        current_pos = vertex_to_idx.get(v, current_pos)
        
        parents = [p for p in parent_dict.get(v, []) if p in needed_set]
        v_idx = vertex_to_idx.get(v, -1)

        # Ensure parents are in fast memory
        for p in parents:
            if p in in_fast:
                continue
            
            # Need to load parent p
            if len(in_fast) >= fast_memory_budget:
                # Evict: Find best candidate to remove using FNU
                # Protect current parents that we still need to load
                remaining_parents = set(parents) - {p}
                evict_one(remaining_parents | {v})
            
            load(p)

        # Compute v
        if v not in in_fast:
            compute(v)
        else:
            # v was already in fast? Maybe from a previous computation?
            if v in inputs:
                pass # It's an input, already in fast (loaded above)
            else:
                # This case shouldn't happen in a standard topo pass
                pass

    # Ensure all outputs are in fast memory at the end
    # First, evict non-outputs to make room if needed
    for v in outputs:
        if v in needed_set and v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                # Evict something that's not an output
                evict_one(outputs)
            
            load(v)

    # Final check: ensure all outputs are in fast
    for v in outputs:
        if v in needed_set and v not in in_fast:
            if len(in_fast) < fast_memory_budget:
                load(v)
            else:
                # Emergency: evict anything
                if in_fast:
                    u = next(iter(in_fast))
                    if u in in_slow:
                        delete(u)
                    else:
                        store(u)
                    load(v)

    return operations
# EVOLVE-BLOCK-END
