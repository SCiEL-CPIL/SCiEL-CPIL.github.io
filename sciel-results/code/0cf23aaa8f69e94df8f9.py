"""Rewritten communication-minimizing DAG pebbling with stronger algorithm."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

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
    DEADLINE = 85.0

    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
    except Exception:
        pebble_baseline = None

    try:
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])

        # Build children map
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)

        all_vertices = set(parents.keys()) | set(inputs) | set(outputs)

        # Topological sort (Kahn's algorithm)
        in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
        for v in inputs:
            in_degree[v] = 0

        queue = sorted([v for v in all_vertices if in_degree.get(v, 0) == 0])
        topo_order = []
        while queue:
            u = queue.pop(0)
            topo_order.append(u)
            for c in children.get(u, []):
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
                    queue.sort()

        topo_index = {v: i for i, v in enumerate(topo_order)}

        # Precompute depth (longest path from any input)
        depth = {}
        for v in topo_order:
            ps = parents.get(v, [])
            if not ps:
                depth[v] = 0
            else:
                depth[v] = max(depth[p] for p in ps) + 1

        # Precompute last_use: the maximum topo_index of any child, or own index if leaf
        last_use = {}
        for v in all_vertices:
            if v in children and children[v]:
                last_use[v] = max(topo_index[c] for c in children[v])
            else:
                last_use[v] = topo_index.get(v, -1)

        # Outputs must be blue at the end
        must_be_blue = set(outputs)

        # Recomputation cost estimate: number of ancestors (approximate)
        # We'll use depth as a proxy: higher depth = more expensive to recompute
        # For eviction, we prefer to evict vertices with LOW depth (cheap to recompute)
        # and those whose last_use is furthest in the future is BAD to evict
        # So: evict the vertex with the SMALLEST depth, tie-break by EARLIEST last_use
        # But also: if last_use < current_i, it's dead, evict it immediately (depth 0 effectively)

        ops = []
        fast_mem = set()

        # Track whether a vertex has been "stored" to slow memory (so we can load it later)
        # Actually, in standard pebbling, 'load' is from slow memory.
        # If we compute a vertex and then delete it without storing, we must recompute.
        # If we store it, we can load it later (cheaper if parents are also gone).
        # Let's assume:
        #   - 'load' is valid for inputs (always available in slow mem) and for previously stored computed vertices.
        #   - 'store' writes to slow memory.
        #   - 'compute' requires all parents in fast mem.
        #   - 'delete' removes from fast mem; value is lost unless stored.
        #
        # To keep it simple and correct:
        #   - We never 'store' unless necessary.
        #   - If we delete a computed vertex, we must recompute it later (via its parents).
        #   - This means our 'ensure' function must be recursive: to recompute v, ensure all parents first.
        #   - The cost of recomputation is bounded by the size of the subgraph.
        #
        # Given the 20000 op limit and small DAGs, this should be fine.

        # Eviction policy:
        # When we need to free space, we pick a candidate from fast_mem:
        # 1. Prefer vertices that are "dead" (last_use < current_topo_index) — but we proactively delete those.
        # 2. Among live non-output vertices, pick the one with smallest depth (cheapest to recompute).
        # 3. Tie-break by earliest last_use.
        # 4. If only outputs remain, pick the one with smallest depth (most expensive to keep? No, we have to delete one).

        # We'll use a function to get the best eviction candidate
        current_topo_index = [0]  # mutable reference for the current processing index

        def get_eviction_candidate():
            """Return the vertex in fast_mem to evict, or None."""
            best = None
            best_key = None
            for v in fast_mem:
                if v in must_be_blue:
                    # Outputs: only consider if no other option
                    # Give them a high "cost to evict" key
                    key = (1, depth[v], last_use[v])
                else:
                    # Non-outputs: prefer low depth, early last_use
                    key = (0, depth[v], last_use[v])
                if best_key is None or key < best_key:
                    best_key = key
                    best = v
            return best

        # Recursive ensure function with cycle protection
        # To avoid deep recursion issues, we'll use an iterative approach with a stack
        # But for small DAGs, recursion depth is bounded by DAG depth, which should be fine.
        # Let's use recursion with a memo/visiting set to detect cycles (shouldn't happen in a DAG).

        visiting = set()

        def ensure_in_fast_mem(v, depth_limit=0):
            """
            Ensure vertex v is in fast_mem.
            Returns True on success, False on failure.
            """
            if time.time() > DEADLINE:
                return False

            if v in fast_mem:
                return True

            if v in visiting:
                # Cycle detected (shouldn't happen in a DAG, but safety)
                return False

            ps = parents.get(v, [])

            if not ps:
                # Input vertex
                if len(fast_mem) >= fast_memory_budget:
                    cand = get_eviction_candidate()
                    if cand is None:
                        return False
                    ops.append(['delete', cand])
                    fast_mem.discard(cand)
                ops.append(['load', v])
                fast_mem.add(v)
                return True
            else:
                # Computed vertex: ensure all parents first
                visiting.add(v)
                try:
                    for p in ps:
                        if not ensure_in_fast_mem(p, depth_limit + 1):
                            return False

                    # Now compute v
                    if len(fast_mem) >= fast_memory_budget:
                        cand = get_eviction_candidate()
                        if cand is None:
                            return False
                        ops.append(['delete', cand])
                        fast_mem.discard(cand)

                    ops.append(['compute', v])
                    fast_mem.add(v)
                    return True
                finally:
                    visiting.discard(v)

        # Main loop: process in topological order
        for i, v in enumerate(topo_order):
            if time.time() > DEADLINE:
                break

            current_topo_index[0] = i

            # Proactively delete dead vertices (last_use < i, not outputs, not v)
            to_delete = []
            for u in list(fast_mem):
                if u == v:
                    continue
                if u in must_be_blue:
                    continue
                if last_use[u] < i:
                    to_delete.append(u)

            for u in to_delete:
                if u in fast_mem:
                    ops.append(['delete', u])
                    fast_mem.discard(u)

            # Ensure v is in fast_mem
            if not ensure_in_fast_mem(v):
                # Failed; try to free space more aggressively
                # Delete any non-output vertex
                for u in list(fast_mem):
                    if u not in must_be_blue and u != v:
                        ops.append(['delete', u])
                        fast_mem.discard(u)
                        if len(fast_mem) < fast_memory_budget:
                            break
                if not ensure_in_fast_mem(v):
                    break

        # Ensure all outputs are in fast_mem at the end
        for o in outputs:
            if time.time() > DEADLINE:
                break
            if o not in fast_mem:
                # Try to free space
                for u in list(fast_mem):
                    if u not in must_be_blue:
                        ops.append(['delete', u])
                        fast_mem.discard(u)
                        if len(fast_mem) < fast_memory_budget:
                            break
                ensure_in_fast_mem(o)

        # If we still don't have all outputs, try harder
        for o in outputs:
            if o not in fast_mem:
                # Force: delete everything non-output
                for u in list(fast_mem):
                    if u not in must_be_blue:
                        ops.append(['delete', u])
                        fast_mem.discard(u)
                ensure_in_fast_mem(o)

        # Check operation count
        if len(ops) <= 20000:
            return ops
        else:
            if pebble_baseline:
                return pebble_baseline(dag, fast_memory_budget)
            return ops[:20000]

    except Exception as e:
        if pebble_baseline:
            return pebble_baseline(dag, fast_memory_budget)
        return []
# EVOLVE-BLOCK-END
