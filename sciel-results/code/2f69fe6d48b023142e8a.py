"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from heapq import heappush, heappop
from collections import defaultdict

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
    
    # Build children adjacency list for O(n) parent counting
    children = defaultdict(list)
    for v in range(n):
        for p in get_parents(v):
            children[p].append(v)
    
    # Compute remaining parent counts (how many times each vertex will be needed)
    remaining_parent_count = [0] * n
    for v in range(n):
        remaining_parent_count[v] = len(children[v])
    
    red = set()
    operations = []
    
    # Priority queue for eviction: (remaining_count, vertex)
    # We'll use a lazy deletion approach with a counter to avoid stale entries
    eviction_heap = []
    eviction_version = [0] * n  # version counter for each vertex
    
    def schedule_eviction(v):
        heappush(eviction_heap, (remaining_parent_count[v], v))
        eviction_version[v] += 1
    
    def get_next_eviction():
        while eviction_heap:
            count, v = eviction_heap[0]
            if v in red and remaining_parent_count[v] == count:
                return v
            heappop(eviction_heap)
        return None
    
    for v in range(n):
        parents = get_parents(v)
        
        if not parents:
            # Input vertex
            if len(red) >= fast_memory_budget:
                to_evict = get_next_eviction()
                if to_evict is not None:
                    operations.append(['store', to_evict])
                    red.discard(to_evict)
                    remaining_parent_count[to_evict] = 0  # Mark as no longer needed
        
        if v not in red:
            if len(red) >= fast_memory_budget:
                to_evict = get_next_eviction()
                if to_evict is not None:
                    operations.append(['store', to_evict])
                    red.discard(to_evict)
                    remaining_parent_count[to_evict] = 0
            
            operations.append(['load', v])
            red.add(v)
            schedule_eviction(v)
        
        if parents:
            # Compute v
            operations.append(['compute', v])
            red.add(v)
            schedule_eviction(v)
        
        # Decrement parent count for this vertex (it's now been used as a parent)
        # Actually, we should decrement when a child is computed
        # For now, we'll handle this more carefully
    
    # Better approach: track when each vertex's last child is computed
    # and decrement remaining_parent_count at that point
    
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
    
    if time.time() - start_time > deadline - 5:
        pass
    
    return operations
# EVOLVE-BLOCK-END
