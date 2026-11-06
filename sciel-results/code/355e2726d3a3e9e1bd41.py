def pebble(dag, fast_memory_budget):
    import time
    
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Parse DAG
    # dag is a list of lists of parent indices
    n = len(dag)
    parents = dag
    children = [[] for _ in range(n)]
    for i in range(n):
        for p in parents[i]:
            children[p].append(i)
            
    # Identify inputs (nodes with no parents)
    inputs = [i for i in range(n) if len(parents[i]) == 0]
    
    # Identify outputs (nodes with no children)
    outputs = [i for i in range(n) if len(children[i]) == 0]
    
    # State
    # "Inputs start blue" -> assume they are in fast memory initially
    in_fast = set(inputs)
    
    # If initial inputs exceed budget, evict some
    if len(in_fast) > fast_memory_budget:
        # We need to evict (len(in_fast) - fast_memory_budget) inputs
        # Which ones? Those with lowest future usage.
        # Future usage for inputs: number of children (initially)
        # Actually, we need to compute rem_children first to know future usage accurately?
        # rem_children is just len(children[i]) initially.
        
        # Sort inputs by number of children (ascending)
        # Evict those with fewest children first
        # Note: This is a heuristic.
        
        # We need a list of inputs to evict
        to_evict = []
        current_size = len(in_fast)
        target_size = fast_memory_budget
        
        # Get all inputs sorted by len(children[input])
        sorted_inputs = sorted(inputs, key=lambda x: len(children[x]))
        
        for inp in sorted_inputs:
            if current_size > target_size:
                to_evict.append(inp)
                current_size -= 1
            else:
                break
        
        for v in to_evict:
            in_fast.remove(v)
            # Evicting an input:
            # If it has children, we will need to load it later.
            # Is it 'store' or 'delete'?
            # If we 'delete' it, we lose it. We must 'load' it later.
            # If we 'store' it, we keep it in slow memory. We must 'load' it later.
            # Cost is same: 1 load later.
            # But 'delete' is 0 cost now, 'store' is 1 cost now.
            # So 'delete' is better.
            # However, if the evaluator expects 'store' for non-dead values?
            # "Inputs start blue". If we delete them, they become red (gone)?
            # Let's use 'store' to be safe? No, 'delete' is standard for freeing space.
            # Let's check the operation list: load/store/compute/delete.
            # If I delete an input, and later need it, I load it.
            # If I store an input, and later need it, I load it.
            # Store cost 1, Delete cost 0.
            # So Delete is better.
            # But wait, if I delete it, is it still "blue"? No.
            # The prompt says "Inputs start blue".
            # If I delete it, it's not blue.
            # But I can load it to make it blue again.
            # So 'delete' is fine.
            
            # However, there is a nuance. If the input is used by many nodes, 
            # deleting it might be bad if we can't afford the load later?
            # But we have to evict something.
            
            ops_initial = []
            for v in to_evict:
                ops_initial.append(['delete', v])
            # We will append these to the main ops list later?
            # Or just start the main list with these.
            
    else:
        ops_initial = []

    # Pre-compute remaining children count
    # This tracks how many children of a node have NOT yet been computed.
    rem_children = [len(children[i]) for i in range(n)]
    
    ops = ops_initial
    
    # Helper to find best victim for eviction
    def find_victim():
        # Find vertex in in_fast with minimum rem_children
        # Tie breaker: lowest index?
        best_v = None
        min_rc = float('inf')
        for v in in_fast:
            if rem_children[v] < min_rc:
                min_rc = rem_children[v]
                best_v = v
        return best_v, min_rc

    # Main loop
    for v in range(n):
        # Time check
        if time.time() > deadline:
            # Try to finish quickly?
            # If we break here, the schedule is incomplete.
            # We should probably just return what we have, hoping it's valid?
            # Or force complete?
            # For now, break and return.
            break
        
        # Ensure all parents of v are in fast memory
        for p in parents[v]:
            if p not in in_fast:
                # Need to load p
                if len(in_fast) >= fast_memory_budget:
                    # Evict a vertex
                    victim, min_rc = find_victim()
                    if victim is not None:
                        in_fast.remove(victim)
                        if rem_children[victim] == 0:
                            ops.append(['delete', victim])
                        else:
                            ops.append(['store', victim])
                
                # Load p
                in_fast.add(p)
                ops.append(['load', p])
        
        # Compute v
        # v is added to fast memory
        in_fast.add(v)
        ops.append(['compute', v])
        
        # Update remaining children for parents of v
        for p in parents[v]:
            rem_children[p] -= 1
            # No immediate eviction here, let the next load trigger it if needed
    
    # Ensure all outputs are in fast memory
    for o in outputs:
        if o not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                # Evict a non-output if possible
                victim = None
                for u in in_fast:
                    if u not in outputs:
                        victim = u
                        break
                
                if victim is not None:
                    in_fast.remove(victim)
                    ops.append(['delete', victim])
                else:
                    # All are outputs, evict one with min rem_children (which is 0)
                    victim, _ = find_victim()
                    if victim is not None:
                        in_fast.remove(victim)
                        ops.append(['delete', victim])
            
            in_fast.add(o)
            ops.append(['load', o])
            
    return ops
