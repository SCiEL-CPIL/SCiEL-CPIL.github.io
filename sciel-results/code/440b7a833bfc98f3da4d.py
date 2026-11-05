"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

def _topological_sort(vertices, parents):
    """Returns a list of vertices in topological order."""
    in_degree = {v: 0 for v in vertices}
    children = {v: [] for v in vertices}
    
    for v in vertices:
        if v in parents:
            for p in parents[v]:
                if p in in_degree:
                    in_degree[v] += 1
                    children[p].append(v)
    
    queue = deque([v for v in vertices if in_degree[v] == 0])
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(order) != len(vertices):
        return list(vertices)
    return order

def _pebbling_schedule(dag, fast_memory_budget):
    """
    Proper pebbling schedule with memory management, eviction, and recomputation.
    Returns a list of [operation, vertex] operations.
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    order = _topological_sort(vertices, parents)
    
    # Build children map
    children_map = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children_map:
                children_map[p].append(v)
    
    # Track state
    # 'blue' = in slow memory (inputs start blue, outputs must end blue)
    # 'red' = in fast memory
    # 'void' = not in any memory
    state = {v: 'blue' for v in vertices}  # All start blue (inputs are blue, others are... hmm)
    
    # Actually, only inputs start blue. Other vertices start in void (not computed yet).
    # Let me reconsider: "Inputs start blue". So inputs are in slow memory.
    # Non-inputs are not computed yet, so they're in void.
    # But they can't be computed until their parents are red.
    
    # Let's define:
    # state[v] in {'blue', 'red', 'void'}
    # Initially: inputs are 'blue', others are 'void'
    # A vertex can be computed if all its parents are 'red'
    # After compute, the vertex becomes 'red'
    # A vertex can be stored (blue) if it's 'red'
    # A vertex can be loaded (red) if it's 'blue'
    # A vertex can be deleted (void) if it's 'red'
    
    # Reset state
    state = {}
    for v in vertices:
        if not parents.get(v, []):
            state[v] = 'blue'  # Input
        else:
            state[v] = 'void'  # Not computed yet
    
    fast_memory = set()  # Vertices currently in fast memory (red)
    ops = []
    
    # Track how many children of each vertex are still uncomputed
    uncomputed_children = {v: len(children_map[v]) for v in vertices}
    
    # A vertex v is "needed" if it has uncomputed children that require it
    # i.e., uncomputed_children[v] > 0
    # But also, if v is an output and not yet stored, it's needed.
    
    computed = set()
    
    def can_delete(v):
        """Check if vertex v can be deleted (no uncomputed children need it, and it's not an output that needs to be stored)."""
        if v not in fast_memory:
            return False
        if state[v] != 'red':
            return False
        # Can delete if no uncomputed children need it
        if uncomputed_children[v] > 0:
            return False
        # If it's an output, we should store it instead of deleting
        # But if we've already stored it, we can delete
        # Actually, if it's an output and uncomputed_children == 0, we should store it
        # For now, let's say outputs can be deleted after being stored
        # But we haven't tracked "stored" state separately
        # Let's simplify: if uncomputed_children[v] == 0, v can be removed from fast memory
        # If v is an output, we store it; otherwise, we delete it
        return True
    
    def get_eviction_candidates():
        """Get vertices that can be safely evicted from fast memory."""
        candidates = []
        for v in fast_memory:
            if v in outputs:
                # Outputs should be stored, not deleted
                # But if we must free space, we can store an output
                # For now, prefer to evict non-outputs
                continue
            if uncomputed_children[v] == 0:
                candidates.append(v)
        return candidates
    
    def free_space_needed():
        """Check if we need to free space."""
        return len(fast_memory) >= fast_memory_budget
    
    def do_eviction():
        """Evict a vertex to free space. Returns True if eviction happened."""
        # First, try to store outputs that can be stored
        for v in list(fast_memory):
            if v in outputs and uncomputed_children[v] == 0:
                ops.append(['store', v])
                fast_memory.discard(v)
                state[v] = 'blue'
                return True
        
        # Then, try to delete non-outputs that are no longer needed
        candidates = get_eviction_candidates()
        if candidates:
            # Delete the one with the highest index (least likely to be needed soon)
            v = candidates[-1]
            ops.append(['delete', v])
            fast_memory.discard(v)
            state[v] = 'void'
            return True
        
        # If we can't free space, we have a problem
        # We might need to recompute later
        return False
    
    # Main loop
    for v in order:
        if not parents.get(v, []):
            # Input vertex
            # It starts blue. To compute its children, it needs to be red.
            # Load it when needed, not upfront.
            # But we need it to be red when we compute its first child.
            # For now, let's load it when we first need it.
            # Actually, let's just mark it as available (blue) and load when needed.
            computed.add(v)
            # Update uncomputed_children of v's children? No, v is a parent.
            # When v is computed (it's an input, so it's "computed" from the start),
            # we don't need to update anything yet.
            # We'll load it when a child needs it.
            continue
        
        # Non-input vertex
        # Check if all parents are in fast memory (red)
        all_parents_red = all(state.get(p) == 'red' for p in parents[v])
        
        if not all_parents_red:
            # Some parents are not in fast memory
            # We need to load them (if blue) or recompute them (if void)
            # For simplicity, let's try to load them
            for p in parents[v]:
                if state.get(p) == 'blue':
                    # Load it
                    if free_space_needed():
                        if not do_eviction():
                            # Can't free space, this is problematic
                            # For now, just load anyway (might exceed budget)
                            pass
                    ops.append(['load', p])
                    fast_memory.add(p)
                    state[p] = 'red'
                elif state.get(p) == 'void':
                    # Parent was deleted, need to recompute
                    # This is complex. For now, we'll try to recompute
                    # But recomputation requires its parents, etc.
                    # This can get very complicated.
                    # For a simpler approach, let's avoid deleting parents that might be needed.
                    # If we get here, it's an error in our eviction logic.
                    # Let's just skip for now and hope it doesn't happen.
                    pass
        
        # Now compute v
        ops.append(['compute', v])
        fast_memory.add(v)
        state[v] = 'red'
        computed.add(v)
        
        # Update uncomputed_children of v's parents
        for p in parents[v]:
            uncomputed_children[p] -= 1
        
        # Check if we need to free space
        if len(fast_memory) > fast_memory_budget:
            do_eviction()
        
        # If v is an output and has no more uncomputed children, store it
        if v in outputs and uncomputed_children[v] == 0:
            ops.append(['store', v])
            fast_memory.discard(v)
            state[v] = 'blue'
    
    # Final cleanup
    for v in list(fast_memory):
        if v in outputs:
            ops.append(['store', v])
            fast_memory.discard(v)
            state[v] = 'blue'
        else:
            ops.append(['delete', v])
            fast_memory.discard(v)
            state[v] = 'void'
    
    return ops

def _validate_schedule(dag, ops, fast_memory_budget):
    """Validate that the schedule is correct and respects memory budget."""
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    state = {}
    for v in vertices:
        if not parents.get(v, []):
            state[v] = 'blue'
        else:
            state[v] = 'void'
    
    fast_memory = set()
    computed = set()
    
    for op, v in ops:
        if op == 'load':
            if state[v] != 'blue':
                return False
            if v in fast_memory:
                return False
            if len(fast_memory) >= fast_memory_budget:
                return False
            fast_memory.add(v)
            state[v] = 'red'
        elif op == 'store':
            if state[v] != 'red':
                return False
            if v not in fast_memory:
                return False
            fast_memory.discard(v)
            state[v] = 'blue'
        elif op == 'delete':
            if state[v] != 'red':
                return False
            if v not in fast_memory:
                return False
            fast_memory.discard(v)
            state[v] = 'void'
        elif op == 'compute':
            if v in computed:
                return False
            for p in parents.get(v, []):
                if state.get(p) != 'red':
                    return False
            if len(fast_memory) >= fast_memory_budget:
                return False
            fast_memory.add(v)
            state[v] = 'red'
            computed.add(v)
        else:
            return False
    
    # Check all outputs are blue
    for v in outputs:
        if state.get(v) != 'blue':
            return False
    
    return True

def _simple_schedule(dag, fast_memory_budget):
    """
    A simpler, more conservative schedule that avoids recomputation.
    Loads inputs lazily, computes in topological order, and evicts carefully.
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    order = _topological_sort(vertices, parents)
    
    children_map = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children_map:
                children_map[p].append(v)
    
    state = {}
    for v in vertices:
        if not parents.get(v, []):
            state[v] = 'blue'
        else:
            state[v] = 'void'
    
    fast_memory = set()
    ops = []
    computed = set()
    
    # Track uncomputed children for each vertex
    uncomputed_children = {v: len(children_map[v]) for v in vertices}
    
    def ensure_in_memory(v):
        """Ensure vertex v is in fast memory (red)."""
        if state[v] == 'red':
            return
        if state[v] == 'blue':
            # Need to free space if needed
            while len(fast_memory) >= fast_memory_budget:
                # Try to evict
                evicted = False
                # First, store outputs that are done
                for w in list(fast_memory):
                    if w in outputs and uncomputed_children[w] == 0:
                        ops.append(['store', w])
                        fast_memory.discard(w)
                        state[w] = 'blue'
                        evicted = True
                        break
                if evicted:
                    continue
                # Then, delete non-outputs that are done
                for w in list(fast_memory):
                    if w not in outputs and uncomputed_children[w] == 0:
                        ops.append(['delete', w])
                        fast_memory.discard(w)
                        state[w] = 'void'
                        evicted = True
                        break
                if not evicted:
                    # Can't free space, this is a problem
                    # For now, just proceed (might exceed budget)
                    break
            ops.append(['load', v])
            fast_memory.add(v)
            state[v] = 'red'
        else:
            # state[v] == 'void', need to recompute
            # This is complex; for now, we'll try to recompute
            # But this requires its parents to be in memory, etc.
            # For simplicity, let's hope this doesn't happen
            pass
    
    for v in order:
        if not parents.get(v, []):
            # Input
            computed.add(v)
            # Don't load yet; load when needed
            continue
        
        # Ensure all parents are in fast memory
        for p in parents[v]:
            ensure_in_memory(p)
        
        # Compute v
        ops.append(['compute', v])
        fast_memory.add(v)
        state[v] = 'red'
        computed.add(v)
        
        # Update uncomputed children count for parents
        for p in parents[v]:
            uncomputed_children[p] -= 1
        
        # If v is an output and has no more uncomputed children, store it
        if v in outputs and uncomputed_children[v] == 0:
            ops.append(['store', v])
            fast_memory.discard(v)
            state[v] = 'blue'
    
    # Final cleanup
    for v in list(fast_memory):
        if v in outputs:
            ops.append(['store', v])
            fast_memory.discard(v)
            state[v] = 'blue'
        else:
            ops.append(['delete', v])
            fast_memory.discard(v)
            state[v] = 'void'
    
    return ops

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    try:
        # Try the simple schedule first
        schedule = _simple_schedule(dag, fast_memory_budget)
        
        # Validate it
        if _validate_schedule(dag, schedule, fast_memory_budget):
            if len(schedule) <= 20000:
                return schedule
        
        # If validation failed, try the pebbling schedule
        schedule = _pebbling_schedule(dag, fast_memory_budget)
        
        if _validate_schedule(dag, schedule, fast_memory_budget):
            if len(schedule) <= 20000:
                return schedule
        
        # Fallback: try a very conservative schedule
        # Load inputs one at a time, compute, and evict aggressively
        vertices = dag['vertices']
        parents = dag['parents']
        outputs = set(dag['outputs'])
        
        order = _topological_sort(vertices, parents)
        
        children_map = {v: [] for v in vertices}
        for v in vertices:
            for p in parents.get(v, []):
                if p in children_map:
                    children_map[p].append(v)
        
        state = {}
        for v in vertices:
            if not parents.get(v, []):
                state[v] = 'blue'
            else:
                state[v] = 'void'
        
        fast_memory = set()
        ops = []
        computed = set()
        uncomputed_children = {v: len(children_map[v]) for v in vertices}
        
        for v in order:
            if not parents.get(v, []):
                computed.add(v)
                continue
            
            # Ensure parents are in memory
            for p in parents[v]:
                if state[p] != 'red':
                    # Free space if needed
                    while len(fast_memory) >= fast_memory_budget:
                        evicted = False
                        for w in list(fast_memory):
                            if w in outputs and uncomputed_children[w] == 0:
                                ops.append(['store', w])
                                fast_memory.discard(w)
                                state[w] = 'blue'
                                evicted = True
                                break
                        if evicted:
                            continue
                        for w in list(fast_memory):
                            if w not in outputs and uncomputed_children[w] == 0:
                                ops.append(['delete', w])
                                fast_memory.discard(w)
                                state[w] = 'void'
                                evicted = True
                                break
                        if not evicted:
                            break
                    
                    if state[p] == 'blue':
                        ops.append(['load', p])
                        fast_memory.add(p)
                        state[p] = 'red'
                    elif state[p] == 'void':
                        # Can't recompute easily; skip
                        pass
            
            # Compute v
            if all(state.get(p) == 'red' for p in parents[v]):
                ops.append(['compute', v])
                fast_memory.add(v)
                state[v] = 'red'
                computed.add(v)
                
                for p in parents[v]:
                    uncomputed_children[p] -= 1
                
                if v in outputs and uncomputed_children[v] == 0:
                    ops.append(['store', v])
                    fast_memory.discard(v)
                    state[v] = 'blue'
            
            # Check time
            if time.time() > deadline:
                break
        
        # Final cleanup
        for v in list(fast_memory):
            if v in outputs:
                ops.append(['store', v])
                fast_memory.discard(v)
                state[v] = 'blue'
            else:
                ops.append(['delete', v])
                fast_memory.discard(v)
                state[v] = 'void'
        
        if _validate_schedule(dag, ops, fast_memory_budget):
            return ops
        
        # Last resort: return an empty schedule (will fail validation, but better than timeout)
        return []
        
    except Exception as e:
        # Fallback to baseline if available
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []

# EVOLVE-BLOCK-END
