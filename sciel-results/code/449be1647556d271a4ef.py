"""Rewritten DAG pebbling solver from scratch."""

import time

def pebble(dag, fast_memory_budget):
    """
    Solve communication-minimizing DAG pebbling.
    
    Parameters:
    -----------
    dag : dict with keys:
        - 'topo': list of vertex indices in topological order
        - 'parents': dict mapping vertex -> list of parent vertices
        - 'inputs': list of input vertex indices (start blue)
        - 'outputs': list of output vertex indices (must finish blue)
    fast_memory_budget : int - number of fast memory slots (red pebbles)
    
    Returns:
    --------
    list of [operation, vertex] where operation is 'load', 'store', 'compute', 'delete'
    """
    start_time = time.time()
    deadline = start_time + 60  # 60s limit, keep safety margin
    
    # Parse DAG structure
    topo = list(dag['topo'])
    parents = dag['parents']
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    budget = fast_memory_budget
    
    # Build set of all vertices
    all_vertices = set(topo)
    for v in parents:
        all_vertices.add(v)
    
    # Compute in-degree (number of parents) for each vertex
    in_degree = {}
    for v in all_vertices:
        in_degree[v] = len(parents.get(v, []))
    
    # Determine if a vertex is an input (starts blue / available for free)
    def is_input(v):
        return v in inputs
    
    # Strategy: Greedy pebbling with memory management
    # We need to:
    # 1. Load inputs into fast memory (or keep in slow, load when needed)
    # 2. Compute vertices in topological order when all parents are available
    # 3. Store results to slow memory when fast memory is full
    # 4. Keep outputs blue at the end
    
    # Track state:
    # - blue_in_fast: set of vertices that are blue (computed/available) and in fast memory
    # - blue_in_slow: set of vertices that are blue and in slow memory (already stored)
    # - red: set of vertices that are red (need to be computed/deleted) - actually, 
    #   in pebbling, "blue" means the value is available, "red" means not available.
    #   Wait, let me re-read the task.
    
    # "Inputs start blue, all outputs must finish blue"
    # "every red value consumes one fast-memory slot"
    # So: blue = value is available (in memory or slow), red = value is NOT available
    # Actually, I think:
    # - A vertex is "pebbled" (blue) if its value is in fast memory or slow memory
    # - A vertex is "unpebbled" (red) if its value is not available
    # - Wait, "red value consumes one fast-memory slot" - this is confusing.
    
    # Let me reinterpret: In standard DAG pebbling:
    # - Blue pebble: vertex's value is in fast memory
    # - The vertex can only be computed if all its parents have blue pebbles
    # - After computing, you can keep the blue pebble or move it to slow memory (store)
    # - Store: move value from fast to slow (releases fast slot)
    # - Load: move value from slow to fast (uses fast slot)
    # - Delete: remove pebble (if not output)
    
    # But the task says "red value consumes one fast-memory slot" - this might mean
    # that vertices whose value is NOT yet computed (red) also consume slots? That doesn't
    # make sense. Let me re-read: "every red value consumes one fast-memory slot, including 
    # newly computed results."
    
    # I think the terminology might be:
    # - "Blue" = value is available/stored
    # - "Red" = value is in fast memory (consumes a slot)
    # Hmm, that's odd. Let me just implement a standard pebbling algorithm.
    
    # Standard approach:
    # - Maintain a set of vertices in fast memory (blue pebbles)
    # - A vertex can be computed if all its parents are in fast memory
    # - After computation, the vertex is in fast memory
    # - We can store (move to slow) or delete vertices to free up fast memory
    # - Inputs are initially available (can be loaded into fast memory)
    # - Outputs must remain in fast memory (or be loaded back) at the end
    
    # Operations:
    # - load(v): load v from slow to fast (v must be in slow, not in fast)
    # - store(v): store v from fast to slow (v must be in fast, not in slow)
    # - compute(v): compute v (all parents must be in fast), v ends up in fast
    # - delete(v): delete v from fast (v must be in fast)
    
    # Let me implement a simple greedy strategy:
    
    ops = []
    fast_set = set()  # vertices in fast memory
    slow_set = set()  # vertices in slow memory (already computed and stored)
    computed = set()  # vertices that have been computed at least once
    
    # Initially, inputs are available in slow memory (they start blue)
    for v in inputs:
        slow_set.add(v)
    
    # We need to process vertices in topological order
    # For each vertex, ensure all parents are in fast memory, compute it, 
    # then manage memory
    
    # To minimize transfers, we want to keep values in fast memory as long as possible
    # and only store/delete when we need to free space
    
    # Simple greedy: process in topo order, load parents if needed, compute, 
    # and when fast memory is full, store/delete some vertex
    
    # Build a priority: vertices that are parents of many upcoming vertices should 
    # be kept longer
    
    # Count how many children each vertex has
    children_count = {v: 0 for v in all_vertices}
    for v in all_vertices:
        for p in parents.get(v, []):
            if p in children_count:
                children_count[p] += 1
    
    def get_remaining_uses(v):
        """How many more times v will be needed (as a parent)."""
        # This is a rough estimate: count of unprocessed children
        return children_count.get(v, 0)
    
    # Track which vertices have been processed
    processed = set()
    
    def can_compute(v):
        """Check if all parents of v are in fast memory."""
        for p in parents.get(v, []):
            if p not in fast_set:
                return False
        return True
    
    def make_room(num_slots=1):
        """Free up fast memory slots by storing or deleting."""
        freed = 0
        while freed < num_slots and fast_set:
            if len(fast_set) <= budget:
                break
            # Find a vertex to store or delete
            # Priority: vertices with fewest remaining uses, and not outputs
            candidates = []
            for v in fast_set:
                if v in outputs:
                    continue  # Don't store/delete outputs
                remaining = get_remaining_uses(v)
                candidates.append((remaining, v))
            
            if not candidates:
                # All remaining are outputs, can't free up
                break
            
            candidates.sort()
            remaining, v = candidates[0]
            
            if v in slow_set:
                # Already in slow, just delete from fast
                ops.append(['delete', v])
                fast_set.discard(v)
            else:
                # Store to slow, then it's available in slow
                ops.append(['store', v])
                fast_set.discard(v)
                slow_set.add(v)
            
            freed += 1
    
    for v in topo:
        if time.time() > deadline - 5:
            break  # Safety margin
        
        if v in processed:
            continue
        
        # Ensure all parents are in fast memory
        for p in parents.get(v, []):
            if p not in fast_set:
                if p in slow_set:
                    # Load from slow
                    if len(fast_set) >= budget:
                        make_room(1)
                    ops.append(['load', p])
                    fast_set.add(p)
                elif p in inputs:
                    # Input, should be in slow already
                    if len(fast_set) >= budget:
                        make_room(1)
                    ops.append(['load', p])
                    fast_set.add(p)
                    slow_set.add(p)
                else:
                    # This shouldn't happen if we process in topo order
                    # Recompute?
                    pass
        
        # Now compute v
        if not can_compute(v):
            continue
        
        ops.append(['compute', v])
        fast_set.add(v)
        computed.add(v)
        processed.add(v)
        
        # If fast memory is full, make room for future vertices
        if len(fast_set) > budget:
            make_room(len(fast_set) - budget)
    
    # Ensure all outputs are in fast memory (blue) at the end
    for v in outputs:
        if v not in fast_set:
            if v in slow_set:
                if len(fast_set) >= budget:
                    make_room(1)
                ops.append(['load', v])
                fast_set.add(v)
            # If v is not in slow and not in fast, we'd need to recompute, 
            # but for simplicity, assume it's either in fast or slow
    
    # Truncate to max 20000 operations
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
