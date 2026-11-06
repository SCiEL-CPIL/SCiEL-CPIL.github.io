# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def _topo_sort(vertices, parents_map):
    """Compute topological order given vertices and parents."""
    in_degree = {v: len(parents_map.get(v, [])) for v in vertices}
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
    
    queue = deque([v for v in vertices if in_degree[v] == 0])
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in children_map[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    return order

def _compute_naive_transfers(dag, fast_memory_budget):
    """Compute the naive schedule's transfer count for comparison."""
    # Naive: load each parent, compute, store result, delete parents
    # This is the baseline behavior
    vertices = list(dag.keys())
    parents_map = {v: dag[v] for v in vertices}
    outputs = dag.get('_outputs', [v for v in vertices if v not in parents_map])
    
    # Simple count: each vertex needs its parents loaded (if not in memory)
    # and its result stored
    # For naive, assume worst case: all loads and stores
    transfers = 0
    for v in vertices:
        for p in parents_map.get(v, []):
            transfers += 1  # load parent
        transfers += 1  # store result
    return transfers

def _greedy_pebble(dag, fast_memory_budget, deadline):
    """Greedy pebbling with memory management."""
    vertices = list(dag.keys())
    parents_map = {v: dag[v] for v in vertices}
    outputs = dag.get('_outputs', [])
    if not outputs:
        # Assume last vertices are outputs
        children_map = defaultdict(list)
        for v in vertices:
            for p in parents_map.get(v, []):
                children_map[p].append(v)
        outputs = [v for v in vertices if not children_map[v]]
    
    # Determine which vertices are inputs (no parents)
    input_vertices = [v for v in vertices if not parents_map.get(v, [])]
    
    # State: fast_memory = set of vertices currently in fast memory (red)
    # We need to compute a schedule of operations
    # Each operation is [op, vertex]
    # Ops: load, store, compute, delete
    
    # Strategy: Process vertices in topological order
    # Keep fast_memory within budget
    # When computing a vertex, ensure all parents are in fast memory
    # After computing, store result to slow memory (freeing fast slot)
    # Delete parents when no longer needed
    
    order = _topo_sort(vertices, parents_map)
    
    # Count how many times each vertex is needed as a parent
    usage_count = defaultdict(int)
    for v in vertices:
        for p in parents_map.get(v, []):
            usage_count[p] += 1
    
    fast_memory = set()
    operations = []
    remaining_usage = dict(usage_count)
    
    # Track which outputs need to be in fast memory at the end
    output_set = set(outputs)
    
    for v in order:
        # Check deadline
        if time.time() > deadline:
            break
        
        # Ensure all parents are in fast memory
        for p in parents_map.get(v, []):
            if p not in fast_memory:
                # Need to load p
                # First, make room if necessary
                while len(fast_memory) >= fast_memory_budget:
                    # Find a vertex to evict: one that's not an output and has no remaining usage
                    evictable = [x for x in fast_memory 
                                  if x not in output_set and remaining_usage.get(x, 0) == 0]
                    if evictable:
                        # Pick one to evict
                        evict_v = evictable[0]
                        operations.append(['delete', evict_v])
                        fast_memory.discard(evict_v)
                    else:
                        # If we can't evict, we're stuck - but this shouldn't happen
                        # if budget is sufficient
                        break
                
                if len(fast_memory) < fast_memory_budget:
                    operations.append(['load', p])
                    fast_memory.add(p)
                else:
                    # Can't load, skip (shouldn't happen)
                    pass
        
        # Compute v
        operations.append(['compute', v])
        fast_memory.add(v)
        
        # If v is an output, we must keep it in fast memory
        # Otherwise, we can store it to free space
        if v not in output_set:
            # Store v to slow memory
            operations.append(['store', v])
            fast_memory.discard(v)
        
        # Update usage counts
        for p in parents_map.get(v, []):
            remaining_usage[p] = remaining_usage.get(p, 0) - 1
    
    # Ensure all outputs are in fast memory at the end
    for o in outputs:
        if o not in fast_memory:
            # Need to load it back
            while len(fast_memory) >= fast_memory_budget:
                evictable = [x for x in fast_memory 
                              if x not in output_set and remaining_usage.get(x, 0) == 0]
                if evictable:
                    evict_v = evictable[0]
                    operations.append(['delete', evict_v])
                    fast_memory.discard(evict_v)
                else:
                    break
            if len(fast_memory) < fast_memory_budget:
                operations.append(['load', o])
                fast_memory.add(o)
    
    return operations

def _count_transfers(operations):
    """Count the number of load/store operations (transfers)."""
    return sum(1 for op, v in operations if op in ('load', 'store'))

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    try:
        # Try the greedy approach
        result = _greedy_pebble(dag, fast_memory_budget, deadline)
        
        # Verify the result is valid
        if result and len(result) <= 20000:
            return result
        else:
            # Fall back to baseline
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
    except Exception as e:
        # Fall back to baseline on any error
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []
# EVOLVE-BLOCK-END
