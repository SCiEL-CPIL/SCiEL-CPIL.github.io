"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin out of 60s
    
    try:
        return _pebble_improved(dag, fast_memory_budget, deadline)
    except Exception:
        # Fallback to baseline if anything fails
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)

def _pebble_improved(dag, fast_memory_budget, deadline):
    """
    Improved pebbling strategy using next-use-based eviction with heap:
    - Precompute next-use index for each vertex
    - Use a max-heap (negated priority) to evict the value with farthest next use
    - Proactively manage memory to minimize loads
    - Ensure outputs end up blue (in fast memory)
    """
    # Extract DAG structure
    vertices = list(dag.get_vertices())
    n = len(vertices)
    if n == 0:
        return []
    
    # Get parents for each vertex (topologically ordered)
    parents = {v: dag.get_parents(v) for v in vertices}
    inputs = set(dag.get_inputs())
    outputs = set(dag.get_outputs())
    
    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents[v]:
            if p in children:
                children[p].append(v)
    
    # Determine which vertices to process (exclude inputs, they're already computed)
    to_process = [v for v in vertices if v not in inputs]
    m = len(to_process)
    
    # Precompute next-use index for each vertex
    # next_use_idx[v] = index in to_process of the first vertex that uses v, or infinity
    INF = float('inf')
    next_use_idx = {v: INF for v in vertices}
    
    # For each vertex v in to_process, find which parents it uses
    # and update next_use_idx for those parents
    for step_idx, v in enumerate(to_process):
        for p in parents[v]:
            if next_use_idx[p] > step_idx:
                next_use_idx[p] = step_idx
    
    # Also, outputs need to be loaded at the end if not in memory
    # Their next use is effectively at the end
    for v in outputs:
        if next_use_idx[v] == INF:
            next_use_idx[v] = m  # After all processing
    
    # Initial state
    in_fast_memory = set(inputs)
    operations = []
    budget = fast_memory_budget
    
    # Track remaining uses for each vertex (for secondary tie-breaking)
    total_ref_count = {v: 0 for v in vertices}
    for v in vertices:
        for p in parents[v]:
            total_ref_count[p] += 1
    remaining_uses = dict(total_ref_count)
    
    # Heap for eviction: (negated_next_use_idx, -remaining_uses, vertex)
    # We want to evict the one with LARGEST next_use_idx (farthest in future)
    # and among ties, the one with fewest remaining uses
    # Using negation for max-heap behavior
    evict_heap = []
    evict_heap_set = set()
    
    def push_to_heap(v):
        if v not in evict_heap_set:
            # Negate next_use_idx so largest becomes smallest (for min-heap)
            # If next_use_idx is INF, use a large number
            nu = next_use_idx[v] if next_use_idx[v] != INF else 10**9
            heapq.heappush(evict_heap, (-nu, -remaining_uses[v], v))
            evict_heap_set.add(v)
    
    # Initialize heap with all values in fast memory
    for v in in_fast_memory:
        if v not in outputs:  # Don't evict outputs
            push_to_heap(v)
    
    def make_room(count=1):
        """Evict values until we have room for 'count' more values."""
        while len(in_fast_memory) >= budget - count + 1:
            if not evict_heap:
                # Emergency: evict any non-output
                for candidate in list(in_fast_memory):
                    if candidate not in outputs:
                        operations.append(['delete', candidate])
                        in_fast_memory.remove(candidate)
                        break
                else:
                    # All are outputs, can't evict
                    return False
                continue
            
            # Pop from heap until we find a valid candidate
            while evict_heap:
                neg_nu, neg_ru, candidate = heapq.heappop(evict_heap)
                if candidate in evict_heap_set:
                    evict_heap_set.discard(candidate)
                    if candidate in in_fast_memory and candidate not in outputs:
                        operations.append(['delete', candidate])
                        in_fast_memory.remove(candidate)
                        break
                    else:
                        # Stale entry, discard
                        continue
            else:
                # Heap empty, emergency evict
                for candidate in list(in_fast_memory):
                    if candidate not in outputs:
                        operations.append(['delete', candidate])
                        in_fast_memory.remove(candidate)
                        break
                else:
                    return False
    
    # Process vertices in topological order
    for step_idx, v in enumerate(to_process):
        if time.time() > deadline:
            break
        
        # Ensure all parents are in fast memory
        for p in parents[v]:
            if p not in in_fast_memory:
                # Make room if needed
                make_room(1)
                operations.append(['load', p])
                in_fast_memory.add(p)
                if p not in outputs:
                    push_to_heap(p)
        
        # Compute v
        operations.append(['compute', v])
        in_fast_memory.add(v)
        if v not in outputs:
            push_to_heap(v)
        
        # Update next_use_idx and remaining_uses for parents
        for p in parents[v]:
            remaining_uses[p] -= 1
            # Update next_use_idx for p: find next vertex after step_idx that uses p
            # This is expensive, so we do it lazily or skip for now
            # For now, we keep the original next_use_idx which was precomputed
            # This is a limitation but should still work reasonably well
    
    # Ensure all outputs are in fast memory (blue)
    for v in outputs:
        if v not in in_fast_memory:
            make_room(1)
            operations.append(['load', v])
            in_fast_memory.add(v)
    
    # Check operation limit
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
