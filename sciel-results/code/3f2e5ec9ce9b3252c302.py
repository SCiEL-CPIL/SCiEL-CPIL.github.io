"""Pebbling optimizer: minimize mean transfers vs naive on layered DAGs."""
import time
import math
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # Parse the DAG structure
    # dag is expected to have:
    #   - vertices (list or range)
    #   - parents: for each vertex, list of parent indices (topologically ordered)
    #   - inputs: vertices with no parents (start blue)
    #   - outputs: vertices that must finish blue
    # We need to handle various possible formats.

    # Try to extract structure from dag
    if isinstance(dag, dict):
        vertices = dag.get('vertices', None)
        parents = dag.get('parents', None)
        inputs = dag.get('inputs', None)
        outputs = dag.get('outputs', None)
        n_layers = dag.get('n_layers', None)
    elif isinstance(dag, (list, tuple)):
        # Maybe dag = (vertices, parents, inputs, outputs)
        if len(dag) >= 4:
            vertices, parents, inputs, outputs = dag[0], dag[1], dag[2], dag[3]
            n_layers = dag[4] if len(dag) > 4 else None
        else:
            # Fallback: treat as list of (vertex, parents) pairs
            parents = {}
            vertices_list = []
            for item in dag:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    v = item[0]
                    p = item[1]
                    vertices_list.append(v)
                    parents[v] = list(p)
            vertices = sorted(vertices_list) if vertices_list else list(range(len(dag)))
            inputs = [v for v in vertices if not parents.get(v)]
            outputs = [v for v in vertices if not any(v in parents.get(u, []) for u in vertices)]
            n_layers = None
    else:
        # Assume dag is an object with attributes
        vertices = getattr(dag, 'vertices', None)
        parents = getattr(dag, 'parents', None)
        inputs = getattr(dag, 'inputs', None)
        outputs = getattr(dag, 'outputs', None)
        n_layers = getattr(dag, 'n_layers', None)

    # Normalize to a clean representation
    if vertices is None:
        # Infer from parents
        if parents is not None:
            if isinstance(parents, dict):
                all_v = set(parents.keys())
                for p_list in parents.values():
                    for p in p_list:
                        all_v.add(p)
                vertices = sorted(all_v)
            else:
                vertices = list(range(len(parents)))
        else:
            vertices = list(range(10))

    if parents is None:
        parents = {}

    # Convert parents to dict: vertex -> list of parent vertices
    if not isinstance(parents, dict):
        # Maybe it's a list of lists
        p_dict = {}
        for i, v in enumerate(vertices):
            if i < len(parents):
                p_dict[v] = list(parents[i])
        parents = p_dict

    # Ensure all vertices have an entry
    for v in vertices:
        if v not in parents:
            parents[v] = []

    if inputs is None:
        inputs = [v for v in vertices if len(parents.get(v, [])) == 0]

    if outputs is None:
        # Outputs are vertices that are not parents of any other vertex
        child_set = set()
        for v in vertices:
            for p in parents.get(v, []):
                child_set.add(p)
        outputs = [v for v in vertices if v not in child_set]

    n = len(vertices)
    if n == 0:
        return []

    # Build topological order
    # Parents are topologically ordered, so we can use Kahn's algorithm
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children_map[p].append(v)

    topo_order = []
    queue = [v for v in vertices if in_degree[v] == 0]
    queue.sort()  # deterministic
    while queue:
        v = queue.pop(0)
        topo_order.append(v)
        for c in children_map[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
        queue.sort()

    if len(topo_order) != n:
        # Cycle or issue; just use all vertices in given order
        topo_order = list(vertices)

    # Compute depth (longest path from any input)
    depth = {}
    for v in topo_order:
        if not parents.get(v):
            depth[v] = 0
        else:
            depth[v] = 1 + max(depth.get(p, 0) for p in parents[v])

    # Compute "recomputation cost" = number of times a value would need to be
    # recomputed if we only keep it until its last use. Use a proxy:
    # the number of descendants (children in transitive closure), weighted by depth.
    # For layered DAGs, this correlates with how much work is saved by keeping it.
    # We'll compute: for each vertex, the number of output vertices that depend on it.
    # This is the "fan-out to outputs" metric.

    # Build reverse: for each vertex, which outputs depend on it
    # Do a DFS/BFS from each output backwards, or forward from each vertex to see which outputs it reaches.
    # For efficiency, do a single pass: compute for each vertex the set of outputs it can reach.
    # Use bitsets if n is small, otherwise use counts.

    # Compute reachability to outputs via reverse topological order
    # reach[v] = set of outputs reachable from v (including v if v is an output)
    # For large n, use a list of booleans or just count.
    output_set = set(outputs)

    # For each vertex, compute number of distinct outputs it can reach
    # Process in reverse topological order
    reachable_outputs_count = {}
    # For memory efficiency with large n, we use a list of sets only if n is small
    if n <= 5000:
        reachable = {}
        for v in reversed(topo_order):
            if v in output_set:
                reachable[v] = {v}
            else:
                r = set()
                for c in children_map.get(v, []):
                    if c in reachable:
                        r |= reachable[c]
                reachable[v] = r
            reachable_outputs_count[v] = len(reachable[v])
    else:
        # Approximate: use depth-based heuristic
        for v in topo_order:
            reachable_outputs_count[v] = max(1, len(outputs) - depth.get(v, 0))

    # Now, the key strategy:
    # We want to minimize total transfers (load + store operations).
    # Naive schedule: for each vertex in topo order, load all parents (if not in memory),
    # compute, store result if it will be used again, else delete immediately.
    #
    # Our improvement: use a priority-based eviction policy.
    # When fast memory is full, evict the vertex with the lowest "value" (recomputation cost).
    # Value = number of outputs it can reach * log(1 + depth)
    #
    # We simulate the pebbling process:
    # - State: which vertices are in fast memory (red)
    # - For each vertex in topo order:
    #   1. Ensure all parents are in fast memory (load if needed)
    #   2. Compute the vertex (place a red pebble)
    #   3. Decide whether to keep it or delete it based on future needs
    #
    # The decision to keep or delete:
    # - If the vertex will be needed again (has children not yet processed), keep it
    #   unless memory is full, in which case evict the least valuable one.
    # - If it won't be needed again, delete it immediately (saves a store).
    #
    # Actually, let's think about what "transfers" means:
    # - load: move from slow to fast (1 transfer)
    # - store: move from fast to slow (1 transfer)
    # - compute: no transfer
    # - delete: free a slot, no transfer
    #
    # Total transfers = number of loads + number of stores
    #
    # Naive: load each parent every time it's needed (no caching), store each result.
    # Our approach: cache in fast memory to avoid reloads, but must manage limited slots.

    # Compute for each vertex: last use position (index in topo_order where it's last needed as a parent)
    last_use = {}
    for i, v in enumerate(topo_order):
        last_use[v] = i  # at least its own position
    for v in topo_order:
        for c in children_map.get(v, []):
            if c in last_use:
                last_use[v] = max(last_use[v], last_use[c])

    # Also, outputs must be stored (they need to finish blue, meaning they must be in slow memory at the end)
    # So for outputs, they must be stored at the end.

    # Value function for keeping a vertex in fast memory
    def vertex_value(v):
        # Higher value = more important to keep
        # Based on: number of outputs reachable, depth, and whether it's still needed
        ro = reachable_outputs_count.get(v, 0)
        d = depth.get(v, 0)
        return ro * (1.0 + math.log(1 + d))

    # Simulate the pebbling
    fast_memory = set()  # vertices currently in fast memory (red pebbles)
    operations = []
    B = fast_memory_budget

    for i, v in enumerate(topo_order):
        if time.time() - t0 > DEADLINE:
            break

        # Step 1: Load all parents that are not in fast memory
        for p in parents.get(v, []):
            if p not in fast_memory:
                operations.append(["load", p])
                fast_memory.add(p)

        # Step 2: Check if we have room for the computed result
        # After loading parents, we need 1 more slot for v
        # If memory is full, we may need to evict
        # But we can only evict vertices that won't be needed again, OR we evict the least valuable
        # Actually, we must keep all parents until computation is done, but after computation,
        # parents may be evicted if not needed again.

        # Let's first check: do we have space?
        if len(fast_memory) >= B:
            # Need to free a slot. Evict the least valuable vertex that is NOT a parent of v
            # and is not v itself.
            # Prefer to evict vertices that are no longer needed (last_use < i)
            candidates = []
            for x in fast_memory:
                if x == v:
                    continue
                if x in parents.get(v, []):
                    # Still needed for current computation, can't evict yet
                    continue
                candidates.append(x)

            if candidates:
                # Find the one with lowest value
                best_to_evict = min(candidates, key=lambda x: vertex_value(x))
                operations.append(["delete", best_to_evict])
                fast_memory.discard(best_to_evict)

        # Step 3: Compute v
        operations.append(["compute", v])
        fast_memory.add(v)

        # Step 4: Decide whether to keep v or delete it
        # v is needed again if it has children not yet processed
        # i.e., if last_use[v] > i
        if last_use.get(v, i) > i:
            # Keep it (it will be needed again)
            pass
        else:
            # Not needed again. But if v is an output, it must be stored.
            if v in output_set:
                # Must store it
                operations.append(["store", v])
                fast_memory.discard(v)
            else:
                # Delete it
                operations.append(["delete", v])
                fast_memory.discard(v)

    # At the end, any remaining vertices in fast memory that are outputs must be stored
    for v in list(fast_memory):
        if v in output_set:
            operations.append(["store", v])
            fast_memory.discard(v)
        else:
            operations.append(["delete", v])
            fast_memory.discard(v)

    # Verify: all outputs must finish blue (in slow memory)
    # In our model, "blue" means in slow memory. Outputs are stored, so they're blue.
    # Inputs start blue.

    # Count operations
    if len(operations) > 20000:
        # Truncate? This shouldn't happen for reasonable inputs.
        # Just return what we have.
        pass

    return operations
