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

def _greedy_pebble_heap(dag, fast_memory_budget, deadline):
    """
    Heap-based greedy pebbling.
    Prioritizes processing vertices that are ready (all parents resolved) 
    and have high 'value' (high remaining usage, expensive to recompute).
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
    
    # Build children map
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
            
    # Pre-compute metrics
    # remaining_usage: how many children still need this vertex
    remaining_usage = {v: len(children_map[v]) for v in vertices}
    
    # in_degree: number of unprocessed parents
    in_degree = {v: len(parents_map.get(v, [])) for v in vertices}
    
    order = _topo_sort(vertices, parents_map)
    order_index = {v: i for i, v in enumerate(order)}
    
    # Fast memory state
    fast_mem = set()
    # State: 'in_slow', 'in_fast', 'gone', 'stored'
    # Inputs start in 'in_slow' (available to load)
    # Computed start in 'gone' (must compute)
    state = {v: 'in_slow' if v in input_set else 'gone' for v in vertices}
    
    ops = []
    
    def get_eviction_priority(v):
        """
        Returns a tuple for eviction. Lower is evicted first.
        """
        r = remaining_usage.get(v, 0)
        if r == 0:
            return (0, 0, 0, 0, v)
        
        # Distance to next use (approximate by min index of children not yet processed)
        # But we process in topo order, so "next use" is just the next child in order.
        # Actually, remaining_usage counts *total* children.
        # A better heuristic: prefer evicting things with LOW remaining usage.
        # If remaining usage is 0, evict immediately.
        
        # Type: Input (0) vs Computed (1). 
        # Evict Inputs first (cheaper to reload).
        is_comp = 1 if v not in input_set else 0
        
        # Parent count (proxy for recompute cost)
        pc = len(parents_map.get(v, []))
        
        # Remaining usage
        return (r, is_comp, pc, v)

    def evict_one():
        if not fast_mem:
            return False
        
        # Find victim
        victim = min(fast_mem, key=lambda v: get_eviction_priority(v))
        
        if remaining_usage.get(victim, 0) == 0:
            # No more uses, just delete
            ops.append(['delete', victim])
        else:
            # Has more uses
            if victim in input_set:
                ops.append(['delete', victim])
            else:
                ops.append(['store', victim])
        
        fast_mem.discard(victim)
        state[victim] = 'in_slow' if victim in input_set else 'stored'
        return True

    def ensure_in_memory(v):
        if v in fast_mem:
            return
        
        # Make space
        while len(fast_mem) >= fast_memory_budget:
            if not evict_one():
                break
            if time.time() > deadline:
                return False
        
        if len(fast_mem) >= fast_memory_budget:
            return False
            
        if v in input_set:
            ops.append(['load', v])
            fast_mem.add(v)
            state[v] = 'in_fast'
        else:
            # Check if it was stored
            if state.get(v) == 'stored':
                ops.append(['load', v])
                fast_mem.add(v)
                state[v] = 'in_fast'
            else:
                # Must compute
                # Ensure parents are in memory
                for p in parents_map.get(v, []):
                    if not ensure_in_memory(p):
                        return False
                    if time.time() > deadline:
                        return False
                ops.append(['compute', v])
                fast_mem.add(v)
                state[v] = 'in_fast'
        return True

    # Process in Topological Order
    for v in order:
        if time.time() > deadline:
            break
            
        if not ensure_in_memory(v):
            break
        
        # After computing/loading v, it is in fast memory.
        # Update remaining_usage of parents?
        # No, remaining_usage is "how many children still need this vertex".
        # When we process v, we satisfy one need for its parents.
        for p in parents_map.get(v, []):
            remaining_usage[p] -= 1
            
    # Ensure outputs are in memory
    for o in outputs:
        if o not in fast_mem:
            if not ensure_in_memory(o):
                break
            if time.time() > deadline:
                break
            
    return ops

def _count_transfers(operations):
    return sum(1 for op, v in operations if op in ('load', 'store'))

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80
    
    try:
        result = _greedy_pebble_heap(dag, fast_memory_budget, deadline)
        
        if result and len(result) <= 20000:
            # Verify validity loosely
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
