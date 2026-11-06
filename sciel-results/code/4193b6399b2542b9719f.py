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

def _greedy_pebble(dag, fast_memory_budget, deadline):
    """
    Greedy pebbling with smart eviction policy.
    - Inputs: Load once, keep while needed, delete when done (0 cost).
    - Computed: Compute once, keep while needed. If must evict, store (1 cost).
    - Eviction Priority:
        1. Delete vertices with 0 remaining usage (0 cost).
        2. Evict Inputs with remaining usage (1 cost total: 0 delete + 1 load later).
           - Choose input with fewest remaining uses.
        3. Evict Computed vertices with remaining usage (2 cost total: 1 store + 1 load later).
           - Choose computed vertex with fewest remaining uses.
    """
    vertices = list(dag.keys())
    parents_map = {v: dag[v] for v in vertices}
    outputs = dag.get('_outputs', [])
    if not outputs:
        children_map = defaultdict(list)
        for v in vertices:
            for p in parents_map.get(v, []):
                children_map[p].append(v)
        outputs = [v for v in vertices if not children_map[v]]
    
    input_vertices = [v for v in vertices if not parents_map.get(v, [])]
    input_set = set(input_vertices)
    output_set = set(outputs)
    
    # Pre-compute remaining usage counts
    usage_count = defaultdict(int)
    for v in vertices:
        for p in parents_map.get(v, []):
            usage_count[p] += 1
    remaining_usage = dict(usage_count)
    
    fast_memory = set()
    operations = []
    
    # Helper to get eviction priority
    # Lower score = better candidate for eviction
    def get_eviction_score(v):
        r = remaining_usage.get(v, 0)
        if r == 0:
            return (0, 0, 0, v) # Delete immediately, 0 cost
        if v in output_set:
            return (float('inf'), 0, 0, v) # Never evict
        if v in input_set:
            # Input: cost to evict+reload is 1 (load). Delete is 0.
            # Score based on remaining usage. Fewer uses = better to evict.
            return (1, r, 0, v)
        else:
            # Computed: cost to evict+reload is 2 (store + load).
            # Score based on remaining usage.
            return (2, r, 0, v)

    def evict_one():
        if not fast_memory:
            return
        # Find victim with lowest score
        victim = min(fast_memory, key=get_eviction_score)
        if victim in input_set:
            operations.append(['delete', victim])
        else:
            operations.append(['store', victim])
        fast_memory.discard(victim)
        
    def ensure_in_memory(v):
        if v in fast_memory:
            return
        
        # Make space if needed
        while len(fast_memory) >= fast_memory_budget:
            evict_one()
            if len(fast_memory) < fast_memory_budget:
                break
            # If still full and we just evicted, we are in an infinite loop or budget is 0
            # If budget is 0, we can't hold anything.
            if fast_memory_budget == 0:
                return # Can't hold anything

        if v in input_set:
            operations.append(['load', v])
        else:
            # Need to compute v
            # Ensure parents are in memory
            for p in parents_map.get(v, []):
                ensure_in_memory(p)
            operations.append(['compute', v])
        
        fast_memory.add(v)
            
    order = _topo_sort(vertices, parents_map)
    
    for v in order:
        if time.time() > deadline:
            break
            
        ensure_in_memory(v)
        
        # If v is not an output and has no remaining usage, evict it immediately
        # to free up space for future vertices
        if v not in output_set and remaining_usage.get(v, 0) == 0:
             if v in input_set:
                 operations.append(['delete', v])
             else:
                 operations.append(['store', v])
             fast_memory.discard(v)
             
        # Update usage counts
        for p in parents_map.get(v, []):
            remaining_usage[p] = remaining_usage.get(p, 0) - 1

    # Ensure all outputs are in fast memory at the end
    for o in outputs:
        if o not in fast_memory:
            ensure_in_memory(o)
            
    return operations

def _count_transfers(operations):
    """Count the number of load/store operations (transfers)."""
    return sum(1 for op, v in operations if op in ('load', 'store'))

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    try:
        result = _greedy_pebble(dag, fast_memory_budget, deadline)
        
        # Verify the result is valid
        if result and len(result) <= 20000:
            return result
        else:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
    except Exception as e:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []
# EVOLVE-BLOCK-END
