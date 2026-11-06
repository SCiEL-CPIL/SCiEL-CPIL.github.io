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
    vertices = list(dag.keys())
    parents_map = {v: dag[v] for v in vertices}
    outputs = dag.get('_outputs', [v for v in vertices if v not in parents_map])
    
    transfers = 0
    for v in vertices:
        for p in parents_map.get(v, []):
            transfers += 1  # load parent
        transfers += 1  # store result
    return transfers

def _greedy_pebble(dag, fast_memory_budget, deadline):
    """Greedy pebbling with memory management and recomputation."""
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
    
    # Pre-compute recompute cost proxy: number of parents
    # Inputs have cost 0 (just load)
    # Computed nodes have cost > 0
    parent_count = {v: len(parents_map.get(v, [])) for v in vertices}
    
    # Usage counts
    usage_count = defaultdict(int)
    for v in vertices:
        for p in parents_map.get(v, []):
            usage_count[p] += 1
    remaining_usage = dict(usage_count)
    
    fast_memory = set()
    operations = []
    
    # Min-heap for eviction candidates: (remaining_uses, is_computed, parent_count, vertex)
    # We want to evict the "least valuable" to keep.
    # Low remaining uses is good for eviction.
    # Inputs (is_computed=0) are cheaper to reload, so they are "less valuable" to keep in fast memory than computed nodes?
    # Actually, if we evict an input, we pay 1 load later.
    # If we evict a computed node, we pay Store now (1) + Load later (1) = 2.
    # So Inputs are cheaper to evict (Delete, 0 now, 1 later = 1 total).
    # So we prefer to evict Inputs first.
    # Then we prefer to evict nodes with fewer remaining uses.
    # Then nodes with fewer parents (cheaper to recompute if we chose delete, but we store computed).
    
    # To make this dynamic, we don't use a heap that updates easily.
    # Instead, when we need to evict, we scan fast_memory.
    # Since fast_memory_budget is likely small, scanning is O(B).
    
    def get_eviction_score(v):
        # Lower score = better candidate for eviction
        # 1. Remaining usage: Lower is better to evict
        # 2. Type: Input (0) is better to evict than Computed (1) because Delete is cheaper than Store+Load
        # 3. Parent count: Lower is better (cheaper recompute if we ever delete)
        r = remaining_usage.get(v, 0)
        is_comp = 1 if v not in input_set else 0
        pc = parent_count[v]
        return (r, is_comp, pc, v)

    def evict_one():
        if not fast_memory:
            return
        # Find victim
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
        if len(fast_memory) >= fast_memory_budget:
            evict_one()
            # If still full (budget 0?), we are stuck, but assume budget >= 1
            if len(fast_memory) >= fast_memory_budget:
                 # Try to evict again? Or fail?
                 # If budget is 0, we can't hold anything.
                 pass

        if v in input_set:
            operations.append(['load', v])
            fast_memory.add(v)
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
             # Evict v
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
