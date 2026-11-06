"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin out of 60s
    
    try:
        return _pebble_optimized(dag, fast_memory_budget, deadline)
    except Exception:
        # Fallback to baseline if anything fails
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)

def _pebble_optimized(dag, fast_memory_budget, deadline):
    """
    Optimized pebbling strategy:
    - Uses topological order.
    - Inputs start in fast memory.
    - Computes vertices, loads parents as needed.
    - Evicts based on 'next use' distance (LRU-like).
    - Deletes nodes that are no longer needed (no future dependents).
    - Ensures outputs are in fast memory at the end.
    """
    vertices = list(dag.get_vertices())
    n = len(vertices)
    if n == 0:
        return []
    
    parents = {v: dag.get_parents(v) for v in vertices}
    inputs = set(dag.get_inputs())
    outputs = set(dag.get_outputs())
    
    # Map vertex to index for quick lookup
    v_to_idx = {v: i for i, v in enumerate(vertices)}
    
    # Build dependents list: for each vertex p, list of indices of vertices that depend on p
    dependents = {v: [] for v in vertices}
    for v in vertices:
        for p in parents[v]:
            dependents[p].append(v_to_idx[v])
            
    # Sort dependents lists (they should be in topological order already if we iterate vertices in order)
    for p in vertices:
        dependents[p].sort()
        
    # Pointers to next dependent
    ptr = {v: 0 for v in vertices}
    
    # Initial state
    in_fast = set(inputs)
    in_slow = set() # Values stored to slow memory
    operations = []
    
    budget = fast_memory_budget
    
    # Helper to get next use index for a vertex
    def get_next_use(v):
        if ptr[v] < len(dependents[v]):
            return dependents[v][ptr[v]]
        return None

    # Process vertices in topological order
    for i, v in enumerate(vertices):
        if time.time() > deadline:
            break
            
        if v in inputs:
            continue
        
        # Load parents if not in fast memory
        for p in parents[v]:
            if p not in in_fast:
                if p in in_slow:
                    operations.append(['load', p])
                    in_slow.remove(p)
                    in_fast.add(p)
                else:
                    # This should not happen if we correctly store/delete
                    pass
        
        # Compute v
        operations.append(['compute', v])
        in_fast.add(v)
        
        # Update pointers for parents of v
        # v is a dependent of its parents
        for p in parents[v]:
            # Advance ptr[p] past the current index i
            while ptr[p] < len(dependents[p]) and dependents[p][ptr[p]] <= i:
                ptr[p] += 1
        
        # Proactive deletion of nodes that are no longer needed
        # Check parents of v and potentially others?
        # Just checking parents of v is a good heuristic to keep the set clean.
        # Actually, let's delete any node in in_fast that has no more dependents.
        # To be efficient, we can check the set of nodes we touched recently or just iterate through a small subset.
        # But iterating through all in_fast is O(Budget).
        # Let's try to delete nodes that have no future dependents.
        
        to_delete = []
        for c in in_fast:
            if get_next_use(c) is None and c not in outputs:
                to_delete.append(c)
        
        for c in to_delete:
            operations.append(['delete', c])
            in_fast.remove(c)
            if c in in_slow:
                in_slow.remove(c)

        # Eviction phase
        while len(in_fast) > budget:
            # Find best candidate to evict
            best_candidate = None
            best_priority = -1 # Higher is better to evict
            best_action = None
            
            for c in in_fast:
                next_use = get_next_use(c)
                
                if next_use is None:
                    # Not needed again
                    if c in outputs:
                        # Must keep in fast until end?
                        # If we evict, we must store it.
                        # Then load it at the end.
                        # Prefer to keep outputs in fast if budget allows?
                        # But if budget is tight, we might have to store.
                        # Priority: Lower than non-outputs?
                        # Let's say non-outputs with next_use None are priority 100.
                        # Outputs with next_use None are priority 50 (store them).
                        priority = 50
                        action = 'store'
                    else:
                        priority = 100
                        action = 'delete'
                else:
                    # Needed again at index next_use
                    # Distance = next_use - i
                    # Larger distance = better candidate to evict (store)
                    distance = next_use - i
                    # Priority: Larger distance is better
                    # But 'delete' (priority 100) is always better than 'store' (priority < 100)
                    priority = distance
                    action = 'store'
                    
                if priority > best_priority:
                    best_priority = priority
                    best_candidate = c
                    best_action = action
            
            if best_candidate is None:
                break
                
            # Execute eviction
            if best_action == 'delete':
                operations.append(['delete', best_candidate])
                in_fast.remove(best_candidate)
                if best_candidate in in_slow:
                    in_slow.remove(best_candidate)
            else: # store
                operations.append(['store', best_candidate])
                in_fast.remove(best_candidate)
                in_slow.add(best_candidate)
                
    # Ensure all outputs are in fast memory (blue)
    for v in outputs:
        if v not in in_fast:
            if v in in_slow:
                operations.append(['load', v])
                in_slow.remove(v)
                in_fast.add(v)
            else:
                # Should not happen
                pass
            # Make room if needed
            while len(in_fast) > budget:
                deleted = False
                for candidate in list(in_fast):
                    if candidate not in outputs:
                        operations.append(['delete', candidate])
                        in_fast.remove(candidate)
                        if candidate in in_slow:
                            in_slow.remove(candidate)
                        deleted = True
                        break
                if not deleted:
                    break
    
    if len(operations) > 20000:
        operations = operations[:20000]
        
    return operations
# EVOLVE-BLOCK-END
