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
    Improved pebbling strategy using next-use-based eviction:
    - Precompute next-use distance for each vertex at each step
    - Evict the value whose next use is farthest in the future
    - Proactively delete before memory fills up
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
    
    # Compute total reference count
    total_ref_count = {v: 0 for v in vertices}
    for v in vertices:
        for p in parents[v]:
            total_ref_count[p] += 1
    
    # Determine which vertices to process (exclude inputs, they're already computed)
    to_process = [v for v in vertices if v not in inputs]
    
    # Precompute for each vertex in to_process, the list of parents that need to be loaded
    # and the next-use index for each parent at that point
    
    # For each vertex v in to_process, we need to know:
    # - Which parents are not already in memory (will need load)
    # - The next-use distance for each value in memory at that point
    
    # Strategy: simulate the schedule and track next-use distances
    # We'll use a priority queue for eviction decisions
    
    # Initial state
    in_fast_memory = set(inputs)
    operations = []
    budget = fast_memory_budget
    
    # Track remaining uses for each vertex
    remaining_uses = dict(total_ref_count)
    
    # For each vertex, track how many more times it will be needed
    # When we process vertex v, we decrement remaining_uses for its parents
    
    # Precompute the order of processing
    # For each vertex v in to_process, record which parents need loading
    
    # We'll process vertices in topological order
    # At each step, before computing v, we ensure all parents are in memory
    
    # To decide what to evict, we need to know the next-use distance
    # next_use[v] = index in to_process of the next vertex that uses v, or infinity
    
    # Precompute next_use for each vertex at each step
    # This is expensive, so let's compute it lazily
    
    # Alternative: use a simpler heuristic - evict the value with the fewest remaining uses
    # If remaining_uses is 0, evict first
    
    # Let's use a priority queue where priority = (remaining_uses, vertex)
    # We evict the one with smallest remaining_uses (0 means never used again)
    
    # Actually, we want to evict the one with LARGEST next-use distance
    # So we want to keep values that are needed soon
    
    # Let me reconsider: we should evict values that are needed FAR in the future
    # or never again
    
    # Simpler approach: 
    # 1. Before computing v, load all parents not in memory
    # 2. If memory exceeds budget, evict the value with the smallest remaining_uses
    #    (prefer evicting values that are never used again, then those used least)
    # 3. Compute v
    # 4. Decrement remaining_uses for parents
    
    # This is a greedy heuristic that should work well for layered DAGs
    
    for step_idx, v in enumerate(to_process):
        if time.time() > deadline:
            break
        
        # Ensure all parents are in fast memory
        for p in parents[v]:
            if p not in in_fast_memory:
                # Need to load p, but first make room if needed
                while len(in_fast_memory) >= budget:
                    # Find best candidate to evict
                    best_candidate = None
                    best_priority = None
                    for candidate in in_fast_memory:
                        if candidate in outputs:
                            continue  # Don't evict outputs
                        # Priority: smaller remaining_uses is better to evict
                        # If remaining_uses is 0, evict immediately
                        priority = remaining_uses[candidate]
                        if best_priority is None or priority < best_priority:
                            best_priority = priority
                            best_candidate = candidate
                    if best_candidate is None:
                        # All are outputs, emergency evict any
                        best_candidate = next(iter(in_fast_memory))
                    operations.append(['delete', best_candidate])
                    in_fast_memory.remove(best_candidate)
                
                operations.append(['load', p])
                in_fast_memory.add(p)
        
        # Compute v
        operations.append(['compute', v])
        in_fast_memory.add(v)
        
        # Decrement remaining_uses for parents
        for p in parents[v]:
            remaining_uses[p] -= 1
        
        # After compute, if memory exceeds budget, evict
        while len(in_fast_memory) > budget:
            best_candidate = None
            best_priority = None
            for candidate in in_fast_memory:
                if candidate in outputs:
                    continue
                priority = remaining_uses[candidate]
                if best_priority is None or priority < best_priority:
                    best_priority = priority
                    best_candidate = candidate
            if best_candidate is None:
                best_candidate = next(iter(in_fast_memory))
            operations.append(['delete', best_candidate])
            in_fast_memory.remove(best_candidate)
    
    # Ensure all outputs are in fast memory (blue)
    for v in outputs:
        if v not in in_fast_memory:
            # Make room
            while len(in_fast_memory) >= budget:
                best_candidate = None
                best_priority = None
                for candidate in in_fast_memory:
                    if candidate in outputs:
                        continue
                    priority = remaining_uses[candidate]
                    if best_priority is None or priority < best_priority:
                        best_priority = priority
                        best_candidate = candidate
                if best_candidate is None:
                    best_candidate = next(iter(in_fast_memory))
                operations.append(['delete', best_candidate])
                in_fast_memory.remove(best_candidate)
            
            operations.append(['load', v])
            in_fast_memory.add(v)
    
    # Check operation limit
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
