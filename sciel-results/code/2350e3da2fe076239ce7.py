"""Optimized DAG pebbling with communication-minimizing strategy."""
# EVOLVE-BLOCK-START
import time
import sys

def pebble(dag, fast_memory_budget):
    """
    Generate a pebbling schedule for a DAG that minimizes communication.
    
    Args:
        dag: A dict with 'parents' (list of lists, topologically ordered) and 
             'outputs' (list of vertex indices that must finish blue).
             Vertices are indexed 0..n-1. Vertex i's parents are dag['parents'][i].
             Inputs are vertices with no parents (or explicitly given).
        fast_memory_budget: Maximum number of red values that can be held in fast memory.
    
    Returns:
        List of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'.
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    # Parse the DAG
    parents = dag['parents']
    n = len(parents)
    outputs = dag.get('outputs', None)
    
    if outputs is None:
        # Outputs are vertices with no children
        children = [[] for _ in range(n)]
        for i in range(n):
            for p in parents[i]:
                children[p].append(i)
        outputs = [i for i in range(n) if not children[i]]
    
    # Identify inputs (vertices with no parents)
    inputs = [i for i in range(n) if not parents[i]]
    
    # Build children mapping
    children = [[] for _ in range(n)]
    for i in range(n):
        for p in parents[i]:
            children[p].append(i)
    
    # Count how many times each vertex is needed as a parent
    # (i.e., how many children it has that need its value)
    # For the schedule, we need to know when a value is last needed
    
    # Topological order is already given (vertices 0..n-1 are in topo order)
    # parents[i] are all < i
    
    # Strategy: We'll use a dynamic programming / greedy approach.
    # The key insight for communication minimization:
    # - Each input must be loaded at least once (unless recomputed, but inputs can't be recomputed)
    # - Each computed vertex's result must be stored if it will be needed again after being evicted
    # - We want to keep frequently-needed values in fast memory
    
    # For a layered DAG, the optimal strategy often involves:
    # 1. Loading all inputs needed for the current layer
    # 2. Computing the layer
    # 3. Storing results that will be needed in future layers
    # 4. Deleting values that are no longer needed
    
    # Let's compute for each vertex the "last use" - the last child that needs it
    # and the "first use" - the first child that needs it
    
    # For each vertex, find the range of vertices that depend on it
    last_needed = [-1] * n  # last vertex index that needs this vertex as a parent
    for i in range(n):
        for p in parents[i]:
            if i > last_needed[p]:
                last_needed[p] = i
    
    # First needed (for inputs, this is the first child)
    first_needed = [n] * n
    for i in range(n):
        for p in parents[i]:
            if i < first_needed[p]:
                first_needed[p] = i
    
    # Generate schedule using a greedy approach with memory management
    # State: which vertices are in fast memory (red)
    # We process vertices in topological order
    
    # For each vertex, we need its parents to be available when we compute it.
    # After computing, the result goes into fast memory (if budget allows).
    # We can delete a vertex's value after all its children have been computed.
    
    # Let's think about this more carefully.
    # The naive schedule would be:
    # - Load all inputs
    # - Compute each vertex in order, loading parents if not in memory
    # - Store outputs
    # - Delete everything else
    
    # A better schedule minimizes loads/stores by keeping values in memory as long as useful.
    
    # Let's implement a proper pebbling algorithm.
    # We'll use a recursive strategy for the DAG.
    
    # For simplicity and correctness, let's use a straightforward approach:
    # 1. Process vertices in topological order
    # 2. Maintain a set of vertices in fast memory
    # 3. Before computing vertex i, ensure all parents are in memory (load if needed)
    # 4. Compute vertex i (adds it to fast memory)
    # 5. After all children of a vertex are computed, we can delete it
    # 6. At the end, ensure outputs are stored
    
    # But we need to be smart about when to delete to avoid re-loading.
    
    # Let's use a priority-based eviction: keep values that are needed soonest.
    
    schedule = []
    in_memory = set()  # vertices currently in fast memory
    computed = set()   # vertices that have been computed (value exists somewhere)
    stored = set()     # vertices whose values have been stored to slow memory
    
    # We need to track for each vertex when it can be deleted from fast memory
    # A vertex v can be deleted after all its children that need it have been computed
    
    # pending_children[v] = number of children of v not yet computed
    pending_children = [len(children[v]) for v in range(n)]
    
    def can_delete(v):
        """Check if vertex v can be deleted from fast memory."""
        return pending_children[v] == 0
    
    def needs_load(v):
        """Check if vertex v needs to be loaded (not in memory and not an input not yet loaded)."""
        if v in in_memory:
            return False
        if v in inputs:
            # Inputs can be loaded from slow memory
            return True
        # For computed vertices, they should be in memory or need reload
        return v not in in_memory
    
    # Process in topological order
    # We need to be careful: when we compute vertex i, we need all parents[i] in memory.
    # If a parent is not in memory, we load it.
    # After computing i, i enters memory.
    # We can delete vertices that are no longer needed.
    
    # To minimize communication, we want to:
    # - Load a value only when needed
    # - Keep it in memory as long as it's useful (has pending children)
    # - Delete it as soon as it's no longer needed (to free memory)
    # - If memory is full, evict the value that is least useful (last needed soonest? or never needed again?)
    
    # Let's implement with eviction based on last_needed time.
    
    for i in range(n):
        if time.time() - start_time > DEADLINE:
            # Emergency: just store all outputs and return
            for o in outputs:
                if o not in stored:
                    schedule.append(['store', o])
                    stored.add(o)
            return schedule
        
        # Ensure all parents of i are in memory
        for p in parents[i]:
            if p not in in_memory:
                # Need to load p
                # First, check if we need to evict something
                if len(in_memory) >= fast_memory_budget:
                    # Evict a vertex that is no longer needed or least useful
                    # Prefer to evict vertices with no pending children
                    evictable = [v for v in in_memory if can_delete(v)]
                    if evictable:
                        # Evict the one that was needed least recently (or just any)
                        # Choose the one with smallest last_needed (least future use)
                        evict_v = min(evictable, key=lambda v: last_needed[v])
                    else:
                        # All have pending children, evict the one with latest last_needed
                        evict_v = max(in_memory, key=lambda v: last_needed[v])
                    schedule.append(['delete', evict_v])
                    in_memory.discard(evict_v)
                
                schedule.append(['load', p])
                in_memory.add(p)
        
        # Compute vertex i
        # If i is an input, it's already loaded
        if i not in in_memory and i in inputs:
            # Shouldn't happen if inputs are loaded when needed
            if len(in_memory) >= fast_memory_budget:
                evictable = [v for v in in_memory if can_delete(v)]
                if evictable:
                    evict_v = min(evictable, key=lambda v: last_needed[v])
                else:
                    evict_v = max(in_memory, key=lambda v: last_needed[v])
                schedule.append(['delete', evict_v])
                in_memory.discard(evict_v)
            schedule.append(['load', i])
            in_memory.add(i)
        elif i not in in_memory:
            # i is not an input, so it must be computed
            if len(in_memory) >= fast_memory_budget:
                evictable = [v for v in in_memory if can_delete(v)]
                if evictable:
                    evict_v = min(evictable, key=lambda v: last_needed[v])
                else:
                    evict_v = max(in_memory, key=lambda v: last_needed[v])
                schedule.append(['delete', evict_v])
                in_memory.discard(evict_v)
            schedule.append(['compute', i])
            in_memory.add(i)
            computed.add(i)
        
        # Update pending children for parents of i
        for p in parents[i]:
            pending_children[p] -= 1
        
        # Try to delete vertices that are no longer needed (to free memory)
        # Only delete if memory is tight
        if len(in_memory) >= fast_memory_budget:
            for v in list(in_memory):
                if can_delete(v) and v != i:
                    schedule.append(['delete', v])
                    in_memory.discard(v)
                if len(in_memory) < fast_memory_budget:
                    break
    
    # Ensure all outputs are stored
    for o in outputs:
        if o not in stored:
            if o not in in_memory:
                # Load it first
                if len(in_memory) >= fast_memory_budget:
                    evictable = [v for v in in_memory if can_delete(v)]
                    if evictable:
                        evict_v = min(evictable, key=lambda v: last_needed[v])
                    else:
                        evict_v = max(in_memory, key=lambda v: last_needed[v])
                    schedule.append(['delete', evict_v])
                    in_memory.discard(evict_v)
                schedule.append(['load', o])
                in_memory.add(o)
            schedule.append(['store', o])
            stored.add(o)
    
    return schedule
# EVOLVE-BLOCK-END
