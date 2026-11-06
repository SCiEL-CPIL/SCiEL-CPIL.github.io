import time
import heapq
import bisect

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50.0
    
    try:
        return _pebble_optimized(dag, fast_memory_budget, deadline)
    except Exception:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)

def _pebble_optimized(dag, fast_memory_budget, deadline):
    vertices = list(dag.get_vertices())
    n = len(vertices)
    if n == 0:
        return []
    
    parents = {v: dag.get_parents(v) for v in vertices}
    inputs = set(dag.get_inputs())
    outputs = set(dag.get_outputs())
    
    to_process = [v for v in vertices if v not in inputs]
    
    # Precompute use indices
    use_indices = {v: [] for v in vertices}
    for step_idx, v in enumerate(to_process):
        for p in parents[v]:
            use_indices[p].append(step_idx)
    
    in_fast_memory = set(inputs)
    operations = []
    budget = fast_memory_budget
    
    # Heap for eviction: (next_use_step, vertex)
    # We need to handle the case where next_use_step is inf
    INF = float('inf')
    
    # Initialize heap with inputs
    # For inputs, next use is the first step they are used
    heap = []
    for v in inputs:
        next_use = INF
        if use_indices[v]:
            next_use = use_indices[v][0]
        heapq.heappush(heap, (next_use, v))
        
    # Track current next use index pointer for each vertex to avoid bisecting
    # next_use_ptr[v] = index in use_indices[v] of the next use
    next_use_ptr = {v: 0 for v in vertices}
    
    # If budget is smaller than inputs, we have a problem.
    # Assume budget >= max(1, len(inputs)) or handle by evicting inputs immediately?
    # The problem says "Inputs start blue". If budget < len(inputs), it's invalid?
    # We'll assume valid inputs.
    
    if len(in_fast_memory) > budget:
        # Evict until within budget
        while len(in_fast_memory) > budget:
            next_use, v = heapq.heappop(heap)
            if v not in in_fast_memory:
                continue
            # Check if next_use is still valid
            # Since we haven't processed any steps, next_use_ptr is 0.
            # The value in heap should match use_indices[v][0]
            operations.append(['delete', v])
            in_fast_memory.remove(v)
            # Push back if it has more uses? No, it's deleted.
            # If it's an input, it can be loaded later.
            # If it's not an input, it's gone.
            # But inputs are always available to load.
            # Non-inputs must be computed.
            # So we just remove it.
            
    for step_idx, v in enumerate(to_process):
        if time.time() > deadline:
            break
            
        # Ensure parents are in memory
        for p in parents[v]:
            if p not in in_fast_memory:
                # Need to load p
                # First, make room
                while len(in_fast_memory) >= budget:
                    # Evict best candidate
                    evicted = False
                    while heap:
                        next_use, candidate = heapq.heappop(heap)
                        if candidate not in in_fast_memory:
                            continue
                        # Verify next_use is still correct
                        # next_use_ptr[candidate] points to the next use
                        # The next use should be use_indices[candidate][next_use_ptr[candidate]]
                        # If the heap value matches, it's valid
                        if use_indices[candidate] and next_use_ptr[candidate] < len(use_indices[candidate]):
                            current_next_use = use_indices[candidate][next_use_ptr[candidate]]
                            if current_next_use != next_use:
                                # Stale entry, push back with correct value
                                heapq.heappush(heap, (current_next_use, candidate))
                                continue
                        else:
                            # No more uses, next_use should be INF
                            if next_use != INF:
                                heapq.heappush(heap, (INF, candidate))
                                continue
                        
                        # Valid candidate
                        if candidate in outputs:
                            # Don't evict outputs if possible, but if we must...
                            # The problem says outputs must finish blue.
                            # If we evict an output, we must load it later.
                            # It's better to evict non-outputs.
                            # If all are outputs, we have to evict one.
                            pass 
                        
                        # Evict this candidate
                        operations.append(['delete', candidate])
                        in_fast_memory.remove(candidate)
                        evicted = True
                        break
                    
                    if not evicted:
                        # Heap empty but memory full? Should not happen.
                        # Force evict any
                        if in_fast_memory:
                            candidate = next(iter(in_fast_memory))
                            operations.append(['delete', candidate])
                            in_fast_memory.remove(candidate)
                
                operations.append(['load', p])
                in_fast_memory.add(p)
                
                # Add p to heap with its next use
                # The next use for p is the current step_idx (since it's a parent of v)
                # But wait, use_indices[p] contains step indices.
                # The current step_idx is the one where p is used.
                # So the *next* use after this one is at next_use_ptr[p] + 1?
                # No, we are loading p *for* this step.
                # After this step, the next use is the one after the current one.
                
                # Update next_use_ptr for p?
                # The heap entry for p should reflect its *next* future use.
                # Since we are using it now, the next use is the one after this.
                
                # Let's simplify:
                # When we load p, we push it into the heap with its *next* use step.
                # The current step is step_idx.
                # The next use is use_indices[p][next_use_ptr[p] + 1] if it exists, else INF.
                # But we must increment next_use_ptr[p] because we are consuming the current use?
                # No, the use is at step_idx.
                # We should have advanced next_use_ptr[p] when we *processed* the step where p was used.
                
                # Let's refactor the logic:
                # 1. Before processing step_idx, ensure parents are in memory.
                # 2. When we load a parent p, we are satisfying the use at step_idx.
                # 3. The *next* use of p is at use_indices[p][next_use_ptr[p] + 1].
                # 4. So we push (next_use, p) into heap.
                # 5. We increment next_use_ptr[p] by 1? No, we increment it when we *pass* the use.
                #    Since we are *at* the use, we should consider it consumed for the purpose of *future* eviction decisions.
                #    So, next use is index next_use_ptr[p] + 1.
                
                # Actually, let's just use bisect in the heap push to be safe.
                # It's O(log N) and N is small.
                
                next_use_val = INF
                if use_indices[p]:
                    # Find first use >= step_idx + 1
                    idx = bisect.bisect_right(use_indices[p], step_idx)
                    if idx < len(use_indices[p]):
                        next_use_val = use_indices[p][idx]
                
                heapq.heappush(heap, (next_use_val, p))
                
        # Compute v
        operations.append(['compute', v])
        in_fast_memory.add(v)
        
        # Add v to heap with its next use
        next_use_val = INF
        if use_indices[v]:
            idx = bisect.bisect_right(use_indices[v], step_idx)
            if idx < len(use_indices[v]):
                next_use_val = use_indices[v][idx]
        
        heapq.heappush(heap, (next_use_val, v))
        
        # Evict if over budget
        while len(in_fast_memory) > budget:
            evicted = False
            while heap:
                next_use, candidate = heapq.heappop(heap)
                if candidate not in in_fast_memory:
                    continue
                
                # Verify next_use
                # The next use should be the first use > step_idx
                # We can re-verify with bisect
                actual_next_use = INF
                if use_indices[candidate]:
                    idx = bisect.bisect_right(use_indices[candidate], step_idx)
                    if idx < len(use_indices[candidate]):
                        actual_next_use = use_indices[candidate][idx]
                
                if actual_next_use != next_use:
                    heapq.heappush(heap, (actual_next_use, candidate))
                    continue
                
                # Evict
                operations.append(['delete', candidate])
                in_fast_memory.remove(candidate)
                evicted = True
                break
            
            if not evicted:
                if in_fast_memory:
                    candidate = next(iter(in_fast_memory))
                    operations.append(['delete', candidate])
                    in_fast_memory.remove(candidate)
    
    # Ensure outputs are in memory
    for v in outputs:
        if v not in in_fast_memory:
            while len(in_fast_memory) >= budget:
                evicted = False
                while heap:
                    next_use, candidate = heapq.heappop(heap)
                    if candidate not in in_fast_memory:
                        continue
                    # Evict non-output if possible
                    if candidate in outputs:
                        continue
                    # Verify next use (should be INF or future)
                    operations.append(['delete', candidate])
                    in_fast_memory.remove(candidate)
                    evicted = True
                    break
                if not evicted:
                    # Evict any output if necessary
                    if in_fast_memory:
                        candidate = next(iter(in_fast_memory))
                        operations.append(['delete', candidate])
                        in_fast_memory.remove(candidate)
            
            operations.append(['load', v])
            in_fast_memory.add(v)
            
    if len(operations) > 20000:
        operations = operations[:20000]
        
    return operations
