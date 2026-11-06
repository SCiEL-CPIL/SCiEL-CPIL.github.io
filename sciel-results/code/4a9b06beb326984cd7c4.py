"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq

def pebble(dag, fast_memory_budget):
    """
    Minimize memory transfers in DAG pebbling.
    
    dag: structure with vertices, edges (parent->child), inputs, outputs
    fast_memory_budget: number of fast memory slots available
    
    Returns: list of [operation, vertex]
    """
    deadline = time.time() + 85  # 5s safety margin
    
    # Parse DAG structure
    # Assume dag has: 'vertices' or 'n', 'edges' or 'parents', 'inputs', 'outputs'
    # Need to handle various possible formats
    
    try:
        # Try to extract DAG info
        if hasattr(dag, 'n'):
            n = dag.n
        elif 'n' in dag:
            n = dag['n']
        elif hasattr(dag, 'num_vertices'):
            n = dag.num_vertices
        else:
            # Try to infer from edges
            n = None
        
        if hasattr(dag, 'parents'):
            parents = dag.parents
        elif 'parents' in dag:
            parents = dag['parents']
        elif hasattr(dag, 'edges'):
            edges = dag.edges
            # Convert edges to parents
            parents = [[] for _ in range(n)]
            for child, par in edges:
                parents[child].append(par)
        else:
            parents = None
        
        if hasattr(dag, 'inputs'):
            inputs = set(dag.inputs)
        elif 'inputs' in dag:
            inputs = set(dag['inputs'])
        else:
            inputs = set()
        
        if hasattr(dag, 'outputs'):
            outputs = set(dag.outputs)
        elif 'outputs' in dag:
            outputs = set(dag['outputs'])
        else:
            outputs = set()
        
    except Exception:
        # Fallback: try to import baseline
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []
    
    if n is None or parents is None:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []
    
    # Build children adjacency
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    # Count how many times each vertex is needed (number of children)
    # A vertex can be deleted after all its children are computed
    remaining_children = [len(children[v]) for v in range(n)]
    
    # Topological order (assumed parents are topologically ordered, so parents[v] < v)
    # But let's verify and compute proper topo order if needed
    in_degree = [len(parents[v]) for v in range(n)]
    
    # Compute topological order using Kahn's algorithm
    from collections import deque
    queue = deque()
    for v in range(n):
        if in_degree[v] == 0:
            queue.append(v)
    
    topo_order = []
    in_degree_copy = in_degree[:]
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree_copy[c] -= 1
            if in_degree_copy[c] == 0:
                queue.append(c)
    
    if len(topo_order) != n:
        # Cycle detected, fallback
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []
    
    # State: which vertices are in fast memory (blue)
    in_fast = [False] * n
    for v in inputs:
        in_fast[v] = True
    
    fast_count = len(inputs)
    ops = []
    
    # Track which vertices have been computed
    computed = [False] * n
    for v in inputs:
        computed[v] = True
    
    # Process in topological order
    for v in topo_order:
        if time.time() > deadline:
            break
        
        if computed[v]:
            continue
        
        # Check if all parents are in fast memory
        parents_ready = all(in_fast[p] for p in parents[v])
        
        if not parents_ready:
            # Need to load some parents from slow memory
            # Load parents that are not in fast memory
            for p in parents[v]:
                if not in_fast[p]:
                    # Load p from slow memory
                    ops.append(['load', p])
                    in_fast[p] = True
                    fast_count += 1
            
            # Now all parents should be in fast memory
            parents_ready = all(in_fast[p] for p in parents[v])
        
        if not parents_ready:
            # Shouldn't happen, but fallback
            continue
        
        # Compute v
        ops.append(['compute', v])
        computed[v] = True
        in_fast[v] = True
        fast_count += 1
        
        # Decrement remaining_children for parents
        for p in parents[v]:
            remaining_children[p] -= 1
        
        # Try to delete vertices that are no longer needed
        # A vertex can be deleted if:
        # 1. It's not an output (or already stored)
        # 2. All its children have been computed
        # 3. It's not going to be needed again
        
        # Delete vertices that are no longer needed to free up space
        if fast_count > fast_memory_budget:
            # Need to free some slots
            # Delete vertices that are no longer needed
            # Priority: delete non-outputs first, then outputs that are done
            deletable = []
            for u in range(n):
                if in_fast[u] and computed[u]:
                    # Can we delete u?
                    # u is needed if it has uncomputed children
                    if remaining_children[u] == 0:
                        # u is no longer needed by any child
                        # But if u is an output, we need to store it first
                        if u in outputs:
                            # Store it first
                            ops.append(['store', u])
                            in_fast[u] = False
                            fast_count -= 1
                            # Now it's stored, can't delete (it's in slow memory)
                            # Actually, once stored, it's in slow memory, not fast
                        else:
                            # Delete u
                            ops.append(['delete', u])
                            in_fast[u] = False
                            fast_count -= 1
                        if fast_count <= fast_memory_budget:
                            break
        
        # Even if we have space, delete vertices that are no longer needed
        # to keep fast memory clean
        if fast_count > 0:
            for u in range(n):
                if fast_count <= fast_memory_budget:
                    break
                if in_fast[u] and computed[u] and remaining_children[u] == 0:
                    if u in outputs:
                        ops.append(['store', u])
                        in_fast[u] = False
                        fast_count -= 1
                    else:
                        ops.append(['delete', u])
                        in_fast[u] = False
                        fast_count -= 1
    
    # Store all outputs that are still in fast memory
    for v in outputs:
        if in_fast[v]:
            ops.append(['store', v])
            in_fast[v] = False
            fast_count -= 1
    
    return ops
# EVOLVE-BLOCK-END
