"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict
import bisect

def pebble(dag, fast_memory_budget):
    """
    Minimize communication in DAG pebbling using Belady's optimal paging algorithm.
    """
    start_time = time.time()
    
    # Parse DAG structure
    if hasattr(dag, 'num_vertices') and hasattr(dag, 'parents'):
        n = dag.num_vertices
        parents = [list(p) for p in dag.parents]
        outputs = list(dag.outputs) if hasattr(dag, 'outputs') else []
        inputs = list(dag.inputs) if hasattr(dag, 'inputs') else []
    elif isinstance(dag, dict):
        n = dag.get('num_vertices', len(dag.get('parents', [])))
        parents = [list(p) for p in dag.get('parents', [])]
        outputs = list(dag.get('outputs', []))
        inputs = list(dag.get('inputs', []))
    elif isinstance(dag, list):
        parents = [list(p) for p in dag]
        n = len(dag)
        outputs = []
        inputs = []
    else:
        try:
            parents = [list(p) for p in dag]
            n = len(dag)
            outputs = []
            inputs = []
        except:
            return []
    
    if n == 0:
        return []
    
    if not inputs:
        inputs = [v for v in range(n) if not parents[v]]
    if not outputs:
        children = defaultdict(list)
        for v in range(n):
            for p in parents[v]:
                children[p].append(v)
        outputs = [v for v in range(n) if not children[v]]
    
    children = defaultdict(list)
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    is_input = [False] * n
    for i in inputs:
        if i < n:
            is_input[i] = True
    is_output = [False] * n
    for o in outputs:
        if o < n:
            is_output[o] = True
    
    # Build access sequence and track which step each access belongs to
    # Step v: need parents[v] in fast memory, then compute v, then v is in fast memory
    # The "request" at step v is for all parents[v]
    
    # For Belady's, we treat each parent access as a separate request
    # But we can batch: at step v, we request all parents[v] at once
    
    # Let's build a list of (step, vertex) for all accesses
    # Access order: for v=0 to n-1, for each p in parents[v]: access p
    # Then after computing v, v is added to fast memory
    
    # Precompute next_use for each vertex at each step
    # next_use[v][step] = the next step >= step where v is needed as a parent
    # This is expensive. Instead, let's compute on the fly.
    
    # For each vertex u, precompute the list of steps where u is needed as a parent
    # u is needed at step v if u is in parents[v]
    needed_at = defaultdict(list)  # u -> list of steps v where u is a parent of v
    for v in range(n):
        if is_input[v]:
            continue
        for p in parents[v]:
            needed_at[p].append(v)
    
    # Now simulate
    fast_mem = set()
    operations = []
    
    # For each step, we need to ensure all parents are in fast memory
    # Use Belady's for eviction
    
    # Precompute: for each vertex u, the sorted list of steps where u is needed
    # We already have needed_at[u] sorted (since v increases)
    
    # For Belady's: at step v, for a cached vertex u, the next use is the first
    # element in needed_at[u] that is >= v
    # We can use a pointer for each u to track the current position in needed_at[u]
    
    # But we also need to consider that u might be needed as an output (stored at the end)
    # Actually, outputs are stored immediately after computation, so they don't need
    # to be in fast memory after that step.
    
    # Let's also consider: after computing v, v is in fast memory. v is needed at
    # steps in needed_at[v]. If v is an output, v is stored and removed from fast memory.
    
    # Simulate step by step
    for v in range(n):
        if time.time() - start_time > 85:
            break
        
        if is_input[v]:
            continue
        
        # Load parents not in fast memory
        for p in parents[v]:
            if p not in fast_mem:
                # Need to load p
                # If fast memory is full, evict using Belady's
                if len(fast_mem) >= fast_memory_budget:
                    # Find the vertex in fast_mem with farthest next use
                    best_to_evict = None
                    farthest = -1
                    for u in fast_mem:
                        # Find next use of u at or after step v
                        uses = needed_at.get(u, [])
                        # Binary search for first use >= v
                        idx = bisect.bisect_left(uses, v)
                        if idx < len(uses):
                            next_use = uses[idx]
                        else:
                            next_use = float('inf')
                        if next_use > farthest:
                            farthest = next_use
                            best_to_evict = u
                    if best_to_evict is not None:
                        operations.append(['delete', best_to_evict])
                        fast_mem.discard(best_to_evict)
                operations.append(['load', p])
                fast_mem.add(p)
        
        # Compute v
        operations.append(['compute', v])
        fast_mem.add(v)
        
        # Handle overflow after adding v
        while len(fast_mem) > fast_memory_budget:
            best_to_evict = None
            farthest = -1
            for u in fast_mem:
                uses = needed_at.get(u, [])
                idx = bisect.bisect_left(uses, v)
                if idx < len(uses):
                    next_use = uses[idx]
                else:
                    next_use = float('inf')
                if next_use > farthest:
                    farthest = next_use
                    best_to_evict = u
            if best_to_evict is None:
                break
            operations.append(['delete', best_to_evict])
            fast_mem.discard(best_to_evict)
        
        # Store v
        operations.append(['store', v])
        
        # If v is an output, remove from fast memory
        if is_output[v]:
            fast_mem.discard(v)
    
    # Ensure all outputs are stored (they should be from the loop)
    # Safety: if we broke early, some outputs might not be stored
    stored_outputs = set(op[1] for op in operations if op[0] == 'store' and is_output.get(op[1], False))
    for o in outputs:
        if o not in stored_outputs:
            # Need to store o
            # o should be in slow memory from a previous store, or we need to handle it
            # For safety, just add a store (it's already in slow memory)
            operations.append(['store', o])
    
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations

# EVOLVE-BLOCK-END
