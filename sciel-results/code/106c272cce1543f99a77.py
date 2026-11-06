"""Communication-minimizing DAG pebbling under bounded fast memory."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    dag structure expected:
      - dag['parents']: dict vertex -> list of parent vertices (topologically ordered)
      - dag['outputs']: list of output vertices
      - dag['inputs']: list of input vertices (start blue)
    Returns list of [operation, vertex] where operation in {load, store, compute, delete}.
    """
    start_time = time.time()
    DEADLINE = 85.0  # wall-clock limit from evaluator with safety margin
    MAX_OPS = 20000

    try:
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])
        
        # Build reverse adjacency (children)
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
        
        all_vertices = set(parents.keys()) | set(inputs) | set(outputs)
        
        # Topological order
        in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
        for v in inputs:
            in_degree[v] = 0
        
        queue = deque([v for v in all_vertices if in_degree.get(v, 0) == 0])
        topo_order = []
        while queue:
            u = queue.popleft()
            topo_order.append(u)
            for c in children.get(u, []):
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
        
        # Map vertex to its index in topo order for quick lookup
        topo_index = {v: i for i, v in enumerate(topo_order)}
        
        # State
        fast_mem = set()
        stored_in_slow = set()
        ops = []
        
        # remaining_deps[v] = number of children of v that are not yet processed
        remaining_deps = {v: len(children.get(v, [])) for v in all_vertices}
        
        # next_use[v] = topo_index of the child of v with the smallest topo_index among those not yet processed
        # If no children left, it's infinity
        INF = float('inf')
        next_use = {}
        
        # Initialize next_use
        for v in all_vertices:
            if children.get(v):
                # Find the child with the smallest topo index
                min_idx = min(topo_index[c] for c in children[v])
                next_use[v] = min_idx
            else:
                next_use[v] = INF
                
        output_set = set(outputs)
        
        # Process vertices in topological order
        for v in topo_order:
            if time.time() > DEADLINE:
                break
            if len(ops) >= MAX_OPS:
                break
            
            ps = parents.get(v, [])
            
            # Ensure parents are in fast memory
            for p in ps:
                if p not in fast_mem:
                    if p in stored_in_slow:
                        ops.append(['load', p])
                        fast_mem.add(p)
                        stored_in_slow.discard(p)
                    else:
                        # Should be in slow memory (inputs or stored)
                        # If it's an input, it's effectively in slow memory.
                        # If it's computed and not in stored_in_slow, it's a bug.
                        # We assume inputs are available.
                        ops.append(['load', p])
                        fast_mem.add(p)
            
            # Compute v
            if v not in fast_mem:
                ops.append(['compute', v])
                fast_mem.add(v)
            
            # Update state for parents
            for p in ps:
                remaining_deps[p] -= 1
                # Update next_use[p]
                # The current child v is now processed.
                # If next_use[p] was topo_index[v], we need to find the next smallest child index.
                if next_use[p] == topo_index[v]:
                    # Find next smallest
                    next_min = INF
                    for c in children[p]:
                        if topo_index[c] > topo_index[v]:
                            if topo_index[c] < next_min:
                                next_min = topo_index[c]
                    next_use[p] = next_min

            # Memory Management
            # If we exceed budget, evict vertices
            while len(fast_mem) > fast_memory_budget:
                # Find vertex to evict
                # Strategy:
                # 1. Find any vertex with next_use == INF. Delete it.
                # 2. If none, find vertex with max next_use. Store it.
                
                delete_candidate = None
                store_candidate = None
                max_next_use = -1
                
                for u in fast_mem:
                    nu = next_use[u]
                    if nu == INF:
                        delete_candidate = u
                        break # Found a free delete, best case
                    elif nu > max_next_use:
                        max_next_use = nu
                        store_candidate = u
                
                if delete_candidate is not None:
                    ops.append(['delete', delete_candidate])
                    fast_mem.discard(delete_candidate)
                elif store_candidate is not None:
                    ops.append(['store', store_candidate])
                    ops.append(['delete', store_candidate])
                    fast_mem.discard(store_candidate)
                    stored_in_slow.add(store_candidate)
                else:
                    # Should not happen if fast_mem is not empty
                    break

        # Ensure all outputs are in slow memory (Blue)
        # Task: "Inputs start blue, all outputs must finish blue".
        # Blue = Slow Memory.
        for o in outputs:
            if o in fast_mem:
                ops.append(['store', o])
                fast_mem.discard(o)
                stored_in_slow.add(o)
            elif o not in stored_in_slow:
                # If it's lost, we can't recover easily without recomputation.
                # But if it was an output, it should have been stored or kept.
                # If it was deleted without storing (bug), we try to load (might fail if not in slow).
                ops.append(['load', o])
                fast_mem.add(o)
                ops.append(['store', o])
                fast_mem.discard(o)
                stored_in_slow.add(o)

        # Check operation count
        if len(ops) <= MAX_OPS:
            return ops
        else:
            # Too many ops, truncate
            return ops[:MAX_OPS]
            
    except Exception as e:
        return []
# EVOLVE-BLOCK-END
