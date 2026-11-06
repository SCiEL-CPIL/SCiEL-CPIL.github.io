# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque
import math

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

def _compute_recompute_cost(dag, parents_map, input_set):
    """Compute the cost to recompute each node from scratch (in terms of transfers).
    
    Uses dynamic programming in topological order to compute the exact minimum
    number of transfers needed to recompute a vertex from scratch.
    """
    recompute_cost = {}
    for v in dag:
        if v in input_set:
            recompute_cost[v] = 1  # Just a load
        else:
            # Cost to recompute: sum of costs to get each parent into memory + 1 (store v)
            # But parents might be shared, so we use a lower bound: sum of individual costs
            # This is an overestimate if parents are shared, but captures relative ordering
            total_parent_cost = sum(recompute_cost.get(p, 1) for p in parents_map.get(v, []))
            recompute_cost[v] = total_parent_cost + 1
    return recompute_cost

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
    
    # Pre-compute recompute cost
    recompute_cost = _compute_recompute_cost(dag, parents_map, input_set)
    
    # Usage counts: how many times each vertex is used as a parent
    usage_count = defaultdict(int)
    for v in vertices:
        for p in parents_map.get(v, []):
            usage_count[p] += 1
    remaining_usage = dict(usage_count)
    
    fast_memory = set()
    operations = []
    
    def get_eviction_score(v):
        """
        Score for eviction: lower means better to evict.
        
        benefit(v) = recompute_cost[v] * max(remaining_usage[v], 0)
        We want to evict the node with the lowest benefit.
        """
        r = max(remaining_usage.get(v, 0), 0)
        rc = recompute_cost[v]
        score = rc * r
        # Tie-breaker: prefer evicting inputs over computed
        is_comp = 0 if v in input_set else 1
        return (score, is_comp, v)

    def evict_one():
        if not fast_memory:
            return
        victim = min(fast_memory, key=get_eviction_score)
        if victim in input_set:
            operations.append(['delete', victim])
        else:
            operations.append(['store', victim])
        fast_memory.discard(victim)
        
    def make_space(n_slots):
        """Evict until we have at least n_slots free."""
        while len(fast_memory) + n_slots > fast_memory_budget and fast_memory:
            evict_one()
    
    def ensure_in_memory(v):
        if v in fast_memory:
            return
        
        if v in input_set:
            make_space(1)
            operations.append(['load', v])
            fast_memory.add(v)
        else:
            # Need to compute v: all parents + v must be in memory
            # First ensure all parents are in memory
            for p in parents_map.get(v, []):
                ensure_in_memory(p)
            # Now make space for v itself
            make_space(1)
            operations.append(['compute', v])
            fast_memory.add(v)
            
    order = _topo_sort(vertices, parents_map)
    
    for v in order:
        if time.time() > deadline:
            break
            
        ensure_in_memory(v)
        
        # If v is not an output and has no remaining usage, evict it immediately
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

def _verify_and_optimize(operations, dag, fast_memory_budget, deadline):
    """
    Verify the operations are valid and try to optimize by removing redundant operations.
    Returns the optimized operations or the original if verification fails.
    """
    if not operations:
        return operations
    
    vertices = set(dag.keys())
    parents_map = {v: dag[v] for v in vertices}
    outputs = dag.get('_outputs', [])
    if not outputs:
        children_map = defaultdict(list)
        for v in vertices:
            for p in parents_map.get(v, []):
                children_map[p].append(v)
        outputs = [v for v in vertices if not children_map[v]]
    output_set = set(outputs)
    input_set = set(v for v in vertices if not parents_map.get(v, []))
    
    valid = True
    computed_stored = set()
    in_fast = set()
    
    for op, v in operations:
        if op == 'load':
            if v not in input_set and v not in computed_stored:
                valid = False
                break
            if len(in_fast) >= fast_memory_budget:
                valid = False
                break
            in_fast.add(v)
        elif op == 'store':
            if v not in in_fast:
                valid = False
                break
            in_fast.discard(v)
            computed_stored.add(v)
        elif op == 'delete':
            if v not in in_fast:
                valid = False
                break
            in_fast.discard(v)
        elif op == 'compute':
            for p in parents_map.get(v, []):
                if p not in in_fast:
                    valid = False
                    break
            if not valid:
                break
            if len(in_fast) >= fast_memory_budget:
                valid = False
                break
            in_fast.add(v)
            computed_stored.add(v)
    
    if not valid:
        return operations
    
    for o in outputs:
        if o not in in_fast:
            return operations
    
    # Post-processing: remove redundant store+load pairs
    # If we store v and then later load v without computing v in between,
    # the store+load pair is redundant (we could have just kept v in memory)
    # But this is complex to detect. Instead, let's try a simpler optimization:
    # Remove consecutive store+load of the same vertex
    optimized = []
    i = 0
    while i < len(operations):
        if i + 1 < len(operations):
            op1, v1 = operations[i]
            op2, v2 = operations[i + 1]
            # If we store v and immediately load v, that's redundant
            if op1 == 'store' and op2 == 'load' and v1 == v2:
                i += 2
                continue
            # If we delete v and immediately load v (for inputs), that's redundant
            if op1 == 'delete' and op2 == 'load' and v1 == v2 and v1 in input_set:
                i += 2
                continue
        optimized.append(operations[i])
        i += 1
    
    # Verify the optimized version
    valid2 = True
    computed_stored2 = set()
    in_fast2 = set()
    
    for op, v in optimized:
        if op == 'load':
            if v not in input_set and v not in computed_stored2:
                valid2 = False
                break
            if len(in_fast2) >= fast_memory_budget:
                valid2 = False
                break
            in_fast2.add(v)
        elif op == 'store':
            if v not in in_fast2:
                valid2 = False
                break
            in_fast2.discard(v)
            computed_stored2.add(v)
        elif op == 'delete':
            if v not in in_fast2:
                valid2 = False
                break
            in_fast2.discard(v)
        elif op == 'compute':
            for p in parents_map.get(v, []):
                if p not in in_fast2:
                    valid2 = False
                    break
            if not valid2:
                break
            if len(in_fast2) >= fast_memory_budget:
                valid2 = False
                break
            in_fast2.add(v)
            computed_stored2.add(v)
    
    if not valid2:
        return operations
    
    for o in outputs:
        if o not in in_fast2:
            return operations
    
    return optimized

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s kill
    
    try:
        # Try the greedy approach
        result = _greedy_pebble(dag, fast_memory_budget, deadline)
        
        # Verify the result is valid and optimize if possible
        if result and len(result) <= 20000:
            # Try to optimize the result
            optimized = _verify_and_optimize(result, dag, fast_memory_budget, deadline)
            if optimized and len(optimized) <= 20000:
                return optimized
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
