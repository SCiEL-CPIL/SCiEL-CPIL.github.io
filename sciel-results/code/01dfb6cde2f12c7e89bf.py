"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import heapq
import itertools
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Solve the DAG pebbling problem using Dijkstra's algorithm over the state space.
    
    State: (frozen_set, red_set) where:
      - frozen_set: vertices that are "blue" (committed to fast memory, cannot be deleted)
      - red_set: vertices currently in fast memory as red (computed but not yet frozen)
    
    Initially: all inputs are blue (frozen), no reds.
    Target: all outputs are blue (frozen), no reds.
    
    Operations:
      - load(v): v must be an input vertex, not in frozen_set, not in red_set. 
                 Moves v from "available input" to red_set. Cost: 1 transfer.
      - compute(v): v's parents must all be in red_set or frozen_set. v not in red_set, not in frozen_set.
                    Adds v to red_set. Cost: 0 transfers (but consumes memory).
      - store(v): v must be in red_set. Moves v from red_set to frozen_set. Cost: 1 transfer.
      - delete(v): v must be in red_set. Removes v from red_set. Cost: 1 transfer.
    
    Constraint: len(red_set) + len(frozen_set) <= fast_memory_budget at all times.
    """
    # Parse the DAG structure
    # dag is expected to have:
    #   - 'vertices' or similar list of vertex IDs
    #   - 'parents' mapping: vertex -> list of parent vertices
    #   - 'inputs' list of input vertices
    #   - 'outputs' list of output vertices
    #   - Possibly 'edges' or other structure
    
    # Try to extract structure from dag
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        parents = dag.get('parents', {})
        inputs = dag.get('inputs', [])
        outputs = dag.get('outputs', [])
    else:
        # Try other formats
        raise ValueError("Unknown DAG format")
    
    # Convert to sets for faster lookup
    input_set = set(inputs)
    output_set = set(outputs)
    all_vertices = set(vertices) if vertices else set(parents.keys())
    
    # Build children mapping
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
    
    # Initial state: all inputs are frozen (blue), no reds
    initial_frozen = frozenset(inputs)
    initial_red = frozenset()
    initial_state = (initial_frozen, initial_red)
    
    # Target state: all outputs are frozen, no reds
    # Note: other vertices may or may not be frozen, but outputs must be
    target_frozen_required = frozenset(outputs)
    
    def is_target(frozen, red):
        return target_frozen_required.issubset(frozen) and len(red) == 0
    
    # Dijkstra's algorithm
    # Priority queue: (cost, frozen, red, path)
    # To limit memory, we don't store full paths in the heap; we reconstruct later
    # But for simplicity with 20000 op limit, let's store parent pointers
    
    # State space can be large, but for small DAGs it's manageable
    # We'll use a dict to track best cost for each state
    best_cost = {}
    parent_map = {}  # state -> (prev_state, operation, vertex)
    
    start_cost = 0
    best_cost[initial_state] = start_cost
    parent_map[initial_state] = None
    
    heap = []
    heapq.heappush(heap, (start_cost, 0, initial_frozen, initial_red))
    counter = 1  # Tie-breaker for heap
    
    max_ops = 20000
    found_target = None
    
    while heap:
        cost, _, frozen, red = heapq.heappop(heap)
        state = (frozen, red)
        
        if best_cost.get(state, float('inf')) < cost:
            continue
        
        if is_target(frozen, red):
            found_target = state
            break
        
        # Generate neighbors
        # 1. load(v): v in inputs, v not in frozen, v not in red
        for v in inputs:
            if v not in frozen and v not in red:
                new_red = frozenset(red | {v})
                # Check memory: frozen + new_red <= budget
                if len(frozen) + len(new_red) <= fast_memory_budget:
                    new_state = (frozen, new_red)
                    new_cost = cost + 1
                    if new_cost < best_cost.get(new_state, float('inf')):
                        best_cost[new_state] = new_cost
                        parent_map[new_state] = (state, 'load', v)
                        heapq.heappush(heap, (new_cost, counter, frozen, new_red))
                        counter += 1
        
        # 2. compute(v): all parents in (red | frozen), v not in red, v not in frozen
        for v in all_vertices:
            if v in red or v in frozen:
                continue
            ps = parents.get(v, [])
            all_parents_satisfied = all(p in red or p in frozen for p in ps)
            if all_parents_satisfied:
                new_red = frozenset(red | {v})
                if len(frozen) + len(new_red) <= fast_memory_budget:
                    new_state = (frozen, new_red)
                    new_cost = cost + 0  # compute is free
                    if new_cost < best_cost.get(new_state, float('inf')):
                        best_cost[new_state] = new_cost
                        parent_map[new_state] = (state, 'compute', v)
                        heapq.heappush(heap, (new_cost, counter, frozen, new_red))
                        counter += 1
        
        # 3. store(v): v in red, move to frozen
        for v in red:
            new_red = frozenset(red - {v})
            new_frozen = frozenset(frozen | {v})
            if len(new_frozen) + len(new_red) <= fast_memory_budget:
                new_state = (new_frozen, new_red)
                new_cost = cost + 1
                if new_cost < best_cost.get(new_state, float('inf')):
                    best_cost[new_state] = new_cost
                    parent_map[new_state] = (state, 'store', v)
                    heapq.heappush(heap, (new_cost, counter, new_frozen, new_red))
                    counter += 1
        
        # 4. delete(v): v in red, remove from red
        for v in red:
            new_red = frozenset(red - {v})
            if len(frozen) + len(new_red) <= fast_memory_budget:
                new_state = (frozen, new_red)
                new_cost = cost + 1
                if new_cost < best_cost.get(new_state, float('inf')):
                    best_cost[new_state] = new_cost
                    parent_map[new_state] = (state, 'delete', v)
                    heapq.heappush(heap, (new_cost, counter, frozen, new_red))
                    counter += 1
    
    if found_target is None:
        # Fallback: return empty or raise
        raise ValueError("No solution found")
    
    # Reconstruct path
    operations = []
    state = found_target
    while parent_map[state] is not None:
        prev_state, op, v = parent_map[state]
        operations.append([op, v])
        state = prev_state
    
    operations.reverse()
    
    # Check if operations exceed limit
    if len(operations) > max_ops:
        # Truncate? No, this shouldn't happen if we found the optimal
        # But just in case, return as is
        pass
    
    return operations

# EVOLVE-BLOCK-END
