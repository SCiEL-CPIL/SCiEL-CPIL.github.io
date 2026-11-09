"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from heapq import heappush, heappop

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = 85.0  # seconds, safety margin before 90s kill
    
    # Parse the DAG structure
    if hasattr(dag, 'num_vertices'):
        n = dag.num_vertices
    elif hasattr(dag, 'n'):
        n = dag.n
    elif isinstance(dag, dict) and 'n' in dag:
        n = dag['n']
    else:
        if hasattr(dag, 'vertices'):
            n = len(dag.vertices)
        else:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
    
    def get_parents(v):
        if hasattr(dag, 'parents'):
            return dag.parents[v]
        elif isinstance(dag, dict) and 'parents' in dag:
            return dag['parents'][v]
        else:
            return []
    
    def get_outputs():
        if hasattr(dag, 'outputs'):
            return set(dag.outputs)
        elif isinstance(dag, dict) and 'outputs' in dag:
            return set(dag['outputs'])
        else:
            return {n-1}
    
    outputs = get_outputs()
    
    # Build children list once (O(E) instead of O(n^2))
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in get_parents(v):
            children[p].append(v)
    
    # remaining[v] = number of future children of v not yet processed
    remaining = [len(children[v]) for v in range(n)]
    
    red = set()
    operations = []
    
    # Priority queue for eviction: (remaining_use, vertex)
    # We use a lazy-deletion approach: push entries, skip stale ones on pop
    import bisect
    
    def evict_one():
        """Evict the vertex with minimum remaining future use."""
        if not red:
            return
        # Find min remaining among red vertices
        best_v = None
        best_count = float('inf')
        for c in red:
            if remaining[c] < best_count:
                best_count = remaining[c]
                best_v = c
            elif remaining[c] == best_count and best_v is not None and c < best_v:
                best_v = c
        if best_v is not None:
            operations.append(['store', best_v])
            red.discard(best_v)
    
    def ensure_space():
        """Ensure we have room for one more red vertex."""
        while len(red) >= fast_memory_budget:
            if time.time() - start_time > deadline - 2:
                return
            evict_one()
    
    for v in range(n):
        if time.time() - start_time > deadline - 2:
            break
        
        parents = get_parents(v)
        
        if not parents:
            # Input vertex
            ensure_space()
            if v not in red:
                operations.append(['load', v])
                red.add(v)
        else:
            # Has parents: need all parents in fast memory
            for p in parents:
                if p not in red:
                    ensure_space()
                    if p not in red:
                        operations.append(['load', p])
                        red.add(p)
            
            # Compute v
            operations.append(['compute', v])
            red.add(v)
        
        # Decrement remaining for v's children (v is now "used" as parent for each child)
        # Actually, remaining[v] should decrease as children are processed
        # We decrement when a child is processed, not when v is computed
        # Let's fix: remaining[v] = number of children of v that have NOT yet been computed
        # When we compute child u, we decrement remaining[parent] for each parent of u
    
    # Decrement remaining counts: for each vertex u, for each parent p of u, remaining[p] -= 1
    # But we need to do this as we process, so let's restructure
    
    # Actually, let me redo this more carefully:
    # remaining[v] = number of children of v that are yet to be computed
    # Initially: remaining[v] = len(children[v])
    # When we compute vertex u, for each parent p of u: remaining[p] -= 1
    
    # Let me restart with the correct logic
    remaining = [len(children[v]) for v in range(n)]
    red = set()
    operations = []
    
    def evict_one_v2():
        if not red:
            return
        best_v = None
        best_count = float('inf')
        for c in red:
            rc = remaining[c]
            if rc < best_count:
                best_count = rc
                best_v = c
            elif rc == best_count and best_v is not None and c < best_v:
                best_v = c
        if best_v is not None:
            operations.append(['store', best_v])
            red.discard(best_v)
    
    def ensure_space_v2():
        while len(red) >= fast_memory_budget:
            if time.time() - start_time > deadline - 2:
                return
            evict_one_v2()
    
    for v in range(n):
        if time.time() - start_time > deadline - 2:
            break
        
        parents = get_parents(v)
        
        if not parents:
            ensure_space_v2()
            if v not in red:
                operations.append(['load', v])
                red.add(v)
        else:
            for p in parents:
                if p not in red:
                    ensure_space_v2()
                    if p not in red:
                        operations.append(['load', p])
                        red.add(p)
            
            operations.append(['compute', v])
            red.add(v)
        
        # After computing v, decrement remaining for each parent of v
        for p in parents:
            remaining[p] -= 1
    
    # Ensure all outputs are blue
    for v in outputs:
        if v in red:
            operations.append(['store', v])
            red.discard(v)
    
    if len(operations) > 20000:
        operations = operations[:20000]
        for v in outputs:
            if v in red:
                operations.append(['store', v])
    
    return operations
# EVOLVE-BLOCK-END
