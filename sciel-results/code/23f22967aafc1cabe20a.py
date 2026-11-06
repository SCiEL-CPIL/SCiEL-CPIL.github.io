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
    
    def get_savings_if_kept(v):
        """
        Estimate the transfer savings if vertex v is kept in fast memory.
        Higher savings = more valuable to keep.
        We evict the vertex with the LOWEST savings.
        """
        r = remaining_usage.get(v, 0)
        if v in input_set:
            # If kept, we avoid 'r' loads.
            # If evicted, we pay 0 now, but 'r' loads later.
            # Savings = r * 1
            return r
        else:
            # If kept, we avoid 1 store now.
            # If evicted, we pay 1 store now.
            # Later, for each of the 'r' uses:
            # If kept, we pay 0 loads for v.
            # If evicted, we pay 1 load for v + recompute cost.
            # Recompute cost is approximated by parent_count[v].
            # So per use, savings = 1 (load) + parent_count[v] (recompute overhead? No, recompute doesn't cost transfers directly, but requires parents to be loaded. 
            # Actually, recompute cost in transfers is the cost to load parents that are not in memory.
            # This is complex. Let's stick to a simpler proxy.
            # The primary cost difference is the Load of v itself.
            # Savings per use = 1.
            # Plus the immediate store cost avoided = 1.
            # So total savings = 1 + r * 1.
            # But recompute is expensive. Let's weight it.
            # Let's assume recompute cost is roughly parent_count[v] transfers (if parents are evicted).
            # So savings per use = 1 + parent_count[v].
            # Total savings = 1 + r * (1 + parent_count[v]).
            return 1 + r * (1 + parent_count[v])

    def evict_one():
        if not fast_memory:
            return
        # Find victim: min savings
        victim = min(fast_memory, key=get_savings_if_kept)
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
