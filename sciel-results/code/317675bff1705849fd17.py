"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    """
    Minimize communication (load/store) operations for DAG pebbling.
    Returns: list of [operation, vertex] where operation in {load, store, compute, delete}
    """
    deadline = time.time() + 50  # safety margin before 60s limit

    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('nodes', dag.get('V', None)))
        parents = dag.get('parents', dag.get('edges', dag.get('pred', None)))
        inputs = dag.get('inputs', dag.get('sources', dag.get('input_nodes', None)))
        outputs = dag.get('outputs', dag.get('sinks', dag.get('output_nodes', None)))
    else:
        vertices = None
        parents = None
        inputs = None
        outputs = None

    # If we can't parse, fall back to baseline
    if vertices is None or parents is None:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []

    # Normalize to sets/lists
    if isinstance(vertices, (list, tuple, range)):
        vertex_list = list(vertices)
    elif isinstance(vertices, int):
        vertex_list = list(range(vertices))
    else:
        vertex_list = list(vertices)

    num_vertices = len(vertex_list)

    # Normalize parents to dict
    if isinstance(parents, dict):
        parent_dict = dict(parents)
    elif isinstance(parents, (list, tuple)):
        parent_dict = {i: list(p) for i, p in enumerate(parents)}
    else:
        parent_dict = {}

    for v in vertex_list:
        if v not in parent_dict:
            parent_dict[v] = []

    # Normalize inputs and outputs
    if inputs is None:
        inputs = set(v for v in vertex_list if not parent_dict.get(v, []))
    elif isinstance(inputs, (set, list, tuple)):
        inputs = set(inputs)
    else:
        inputs = set(inputs)

    if outputs is None:
        children = {v: [] for v in vertex_list}
        for v in vertex_list:
            for p in parent_dict.get(v, []):
                children[p].append(v)
        outputs = set(v for v in vertex_list if not children[v])
    elif isinstance(outputs, (set, list, tuple)):
        outputs = set(outputs)
    else:
        outputs = set(outputs)

    # Build children map
    children = {v: [] for v in vertex_list}
    for v in vertex_list:
        for p in parent_dict.get(v, []):
            children[p].append(v)

    # Determine needed vertices (ancestors of outputs)
    needed = set(outputs)
    for v in reversed(vertex_list):
        if v in needed:
            for p in parent_dict.get(v, []):
                needed.add(p)

    needed_vertices = [v for v in vertex_list if v in needed]
    needed_set = set(needed_vertices)
    vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}

    # Compute last use index for each vertex (index in needed_vertices list)
    # We process in topological order (vertex_list order is assumed topological)
    last_child_idx = {v: -1 for v in needed_vertices}
    for v in needed_vertices:
        for child in children.get(v, []):
            if child in needed_set:
                child_idx = vertex_to_idx.get(child, -1)
                if child_idx > last_child_idx[v]:
                    last_child_idx[v] = child_idx

    # Greedy pebbling with smart eviction
    in_fast = set()
    operations = []

    def load(v):
        operations.append(['load', v])
        in_fast.add(v)

    def store(v):
        operations.append(['store', v])
        in_fast.discard(v)

    def compute(v):
        operations.append(['compute', v])
        in_fast.add(v)

    def delete(v):
        operations.append(['delete', v])
        in_fast.discard(v)

    # Process vertices in topological order
    for v in needed_vertices:
        if time.time() > deadline:
            break

        parents = [p for p in parent_dict.get(v, []) if p in needed_set]
        v_idx = vertex_to_idx.get(v, -1)

        # Ensure parents are in fast memory
        for p in parents:
            if p in in_fast:
                continue
            
            # Need to load parent p
            if len(in_fast) >= fast_memory_budget:
                # Evict a vertex
                # Strategy: Evict vertex that is not a parent of v, not an output, 
                # and has the lowest "value".
                # Value heuristic:
                # 1. If it will be used again (last_child_idx > v_idx), it has high value.
                # 2. If it is an output, it has infinite value (must stay until end).
                # 3. If it will NOT be used again, it has low value (can be stored).
                
                best_evict = None
                best_score = float('inf')
                
                for u in in_fast:
                    if u == v or u in parents:
                        continue
                    
                    # If u is an output, we really don't want to evict it unless absolutely necessary
                    # But outputs are sinks, so they don't have children. 
                    # If u is an output, last_child_idx[u] == -1 (or max child which doesn't exist).
                    # Actually, outputs are in needed_set.
                    
                    # Check if u is needed in the future
                    future_use = last_child_idx.get(u, -1) > v_idx
                    
                    if u in outputs:
                        score = 1000000 # Very high cost to evict output
                    elif future_use:
                        # Cost is proportional to how soon it's needed? 
                        # Or just high cost.
                        score = 100000
                    else:
                        # Not needed in future, safe to store
                        score = 0
                        
                    # Tie-breaker: Evict the one with the furthest last use if scores are equal?
                    # Or just pick the first one.
                    if score < best_score:
                        best_score = score
                        best_evict = u

                if best_evict is not None:
                    store(best_evict)
                else:
                    # Should not happen if budget > 0 and we have non-parents
                    # Emergency delete
                    for u in in_fast:
                        if u not in parents and u != v:
                            delete(u)
                            break

            load(p)

        # Compute v
        if v not in in_fast:
            compute(v)
        
        # Proactive cleanup: If v is no longer needed in the future, store it now
        # This frees up space for subsequent vertices
        if v not in outputs:
            if last_child_idx.get(v, -1) < v_idx: # No future children in needed_set
                # v is done. Store it to free space.
                # But wait, if we store it, and we need it again? 
                # last_child_idx is the max index of children. 
                # If current index v_idx > last_child_idx[v], then all children are processed.
                # So v is never needed again.
                store(v)

    # Ensure all outputs are in fast memory at the end
    for v in outputs:
        if v in needed_set and v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                for u in in_fast:
                    if u not in outputs:
                        store(u)
                        break
            load(v)

    return operations
# EVOLVE-BLOCK-END
