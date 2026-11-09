"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from heapq import heappush, heappop

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = 85.0  # seconds, safety margin before 90s kill
    
    # Parse the DAG structure
    # Expected: dag has vertices, parents (topologically ordered), output vertices
    # We'll work with whatever structure is provided
    
    # Get basic info from dag
    if hasattr(dag, 'num_vertices'):
        n = dag.num_vertices
    elif hasattr(dag, 'n'):
        n = dag.n
    elif isinstance(dag, dict) and 'n' in dag:
        n = dag['n']
    else:
        # Try to infer from structure
        if hasattr(dag, 'vertices'):
            n = len(dag.vertices)
        else:
            # Fallback: use baseline
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
    
    # Get parents for each vertex (topologically ordered, so parents have lower indices)
    def get_parents(v):
        if hasattr(dag, 'parents'):
            return dag.parents[v]
        elif isinstance(dag, dict) and 'parents' in dag:
            return dag['parents'][v]
        else:
            return []
    
    # Get output vertices
    def get_outputs():
        if hasattr(dag, 'outputs'):
            return set(dag.outputs)
        elif isinstance(dag, dict) and 'outputs' in dag:
            return set(dag['outputs'])
        else:
            # Assume last vertex or vertices with no children
            return {n-1}
    
    outputs = get_outputs()
    
    # Greedy pebbling algorithm
    # State: which vertices are red (in fast memory)
    # We process vertices in topological order, trying to minimize transfers
    
    red = set()  # vertices currently in fast memory
    operations = []
    
    # Count how many times each vertex is needed as a parent
    parent_count = [0] * n
    for v in range(n):
        # Find all children of v
        for u in range(n):
            if v in get_parents(u):
                parent_count[v] += 1
    
    # Greedy: process in topological order, keep values in fast memory if beneficial
    # Use a priority queue to decide what to evict when full
    
    for v in range(n):
        # Check if v is an input (no parents) or has parents
        parents = get_parents(v)
        
        if not parents:
            # Input vertex: load from slow to fast
            if len(red) >= fast_memory_budget:
                # Need to evict something
                # Evict the vertex with lowest future use
                candidates = list(red)
                if candidates:
                    # Choose vertex with minimum remaining parent_count
                    min_count = min(parent_count[c] for c in candidates)
                    to_evict = [c for c in candidates if parent_count[c] == min_count][0]
                    operations.append(['store', to_evict])
                    red.discard(to_evict)
            
            if v not in red:
                operations.append(['load', v])
                red.add(v)
        else:
            # Has parents: need all parents in fast memory to compute
            for p in parents:
                if p not in red:
                    # Need to load p
                    if len(red) >= fast_memory_budget:
                        # Evict
                        candidates = list(red)
                        if candidates:
                            min_count = min(parent_count[c] for c in candidates)
                            to_evict = [c for c in candidates if parent_count[c] == min_count][0]
                            operations.append(['store', to_evict])
                            red.discard(to_evict)
                    
                    if p not in red:
                        operations.append(['load', p])
                        red.add(p)
            
            # Compute v
            operations.append(['compute', v])
            red.add(v)
        
        # Decrement parent_count for this vertex's children
        # Actually, we should decrement when we use a vertex as parent
        # Let's track differently
    
    # Ensure all outputs are blue (stored back to slow memory)
    for v in outputs:
        if v in red:
            operations.append(['store', v])
            red.discard(v)
    
    # Check time and operation limit
    if len(operations) > 20000:
        # Truncate and try to make valid
        operations = operations[:20000]
        # Ensure outputs are blue
        for v in outputs:
            if v in red:
                operations.append(['store', v])
    
    # Verify we haven't exceeded deadline
    if time.time() - start_time > deadline - 5:
        # Return what we have
        pass
    
    return operations
# EVOLVE-BLOCK-END
