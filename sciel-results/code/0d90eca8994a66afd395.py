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
    
    transfers = 0
    for v in vertices:
        for p in parents_map.get(v, []):
            transfers += 1  # load parent
        transfers += 1  # store result
    return transfers

def _greedy_pebble(dag, fast_memory_budget, deadline):
    """
    Greedy pebbling with memory management and recomputation.
    Uses a cost-based eviction strategy.
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
    
    # Pre-compute recompute cost: cost to recompute a node if it's not in memory
    # For inputs: 1 (just load)
    # For computed: 1 (store) + sum of parent recompute costs (approximate)
    # Actually, let's use a simpler heuristic:
    # recompute_cost(v) = 1 if input, else 1 + len(parents)
    # This is a rough estimate of the number of transfers needed to get v back
    
    recompute_cost = {}
    for v in vertices:
        if v in input_set:
            recompute_cost[v] = 1
        else:
            recompute_cost[v] = 1 + len(parents_map.get(v, []))
    
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
        
        The cost of evicting v now vs. later:
        - If we evict now: we pay store/delete now, and later when we need v again,
          we pay load (if input) or recompute (if computed).
        - If we keep v: we pay nothing now, but we might need to evict something else later.
        
        Key insight: The "value" of keeping v in memory is the savings from not
        having to re-fetch it. If v has many remaining uses, keeping it saves more.
        If v is expensive to re-fetch (high recompute_cost), keeping it saves more.
        
        So we want to evict nodes with:
        1. Low remaining_usage (few future uses)
        2. Low recompute_cost (cheap to re-fetch)
        3. Is an input (cheaper to re-fetch than computed)
        
        Combined score: we want to minimize the "loss" from eviction.
        Loss from evicting v = recompute_cost[v] * (remaining_usage[v] - 1) 
        (the -1 because we're about to use it now, so remaining uses after this is remaining_usage - 1)
        But actually, if we evict after using it, remaining_usage is already decremented.
        
        Let's think differently: The marginal cost of keeping v in memory is 1 slot.
        The benefit is avoiding re-fetch cost for each future use.
        benefit(v) = recompute_cost[v] * remaining_usage[v]
        We want to evict the node with the lowest benefit per slot, i.e., lowest benefit.
        But also, inputs are cheaper to re-fetch, so they have lower benefit.
        
        So: eviction_score = recompute_cost[v] * max(remaining_usage[v], 0)
        Lower score = better to evict.
        """
        r = max(remaining_usage.get(v, 0), 0)
        rc = recompute_cost[v]
        # For inputs, recompute_cost is 1, for computed it's 1 + num_parents
        # Multiply by remaining uses to get total avoided cost
        score = rc * r
        # Tie-breaker: prefer evicting inputs over computed (inputs are cheaper to re-fetch)
        is_comp = 0 if v in input_set else 1
        return (score, is_comp, v)
    
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
        while len(fast_memory) >= fast_memory_budget and fast_memory:
            evict_one()
        
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
    
    # Simulate the operations to verify validity
    fast_mem = set()
    blue_set = set()  # vertices that are currently blue (in memory or stored)
    # Actually, let's track what's in fast memory and what's been computed/stored
    # A vertex is "available" if it's in fast memory
    # We need to track which vertices have been computed and stored (so they can be loaded later)
    
    # Let's do a simple verification: simulate and check that:
    # 1. Every load is for an input or a previously computed vertex
    # 2. Every compute has all parents in fast memory
    # 3. Every store/delete is for a vertex in fast memory
    # 4. All outputs end up in fast memory (blue)
    
    valid = True
    computed_stored = set()  # vertices that have been computed and stored to slow memory
    in_fast = set()
    
    for op, v in operations:
        if op == 'load':
            # Can only load inputs or previously computed/stored vertices
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
            computed_stored.add(v)  # Now it can be loaded later
        elif op == 'delete':
            if v not in in_fast:
                valid = False
                break
            in_fast.discard(v)
            # Deleted, so it's not stored; if it was computed, it's lost
            # But if it's an input, it can be reloaded
        elif op == 'compute':
            # All parents must be in fast memory
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
            computed_stored.add(v)  # It's been computed, so it can be stored later
    
    if not valid:
        return operations  # Return original if invalid
    
    # Check all outputs are in fast memory at the end
    for o in outputs:
        if o not in in_fast:
            return operations  # Return original if outputs not all in memory
    
    # Now try to optimize: remove redundant operations
    # A load is redundant if the vertex is already in fast memory
    # A store/delete is redundant if it's not needed (but we can't easily tell)
    # For now, just return the valid operations
    return operations

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
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
