"""Communication-minimizing DAG pebbling with smooth eviction scoring and usage-aware store heuristics."""
# EVOLVE-BLOCK-START
import time
import math
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
        n_vertices = len(topo_order)

        # Precompute depth (longest path from any input)
        depth = {}
        for v in topo_order:
            ps = parents.get(v, [])
            if not ps:
                depth[v] = 0
            else:
                depth[v] = max(depth[p] for p in ps) + 1

        # Precompute distance to sink (longest path from v to any output)
        dist_to_sink = {}
        for v in reversed(topo_order):
            ch = children.get(v, [])
            if not ch:
                dist_to_sink[v] = 1 if v in set(outputs) else 0
            else:
                dist_to_sink[v] = 1 + max(dist_to_sink[c] for c in ch)

        # Precompute remaining usage count: how many times each vertex is needed as a parent
        remaining_uses = {}
        for v in all_vertices:
            remaining_uses[v] = len(children.get(v, []))

        # Precompute recompute_cost: number of ancestors (approximate)
        recompute_cost = {}
        for v in topo_order:
            ps = parents.get(v, [])
            if not ps:
                recompute_cost[v] = 1
            else:
                recompute_cost[v] = 1 + max(recompute_cost[p] for p in ps)

        # Precompute last_use: the maximum topo_index of any child, or own index if leaf
        last_use = {}
        for v in all_vertices:
            if v in children and children[v]:
                last_use[v] = max(topo_index[c] for c in children[v])
            else:
                last_use[v] = topo_index.get(v, -1)

        # Outputs must be blue at the end
        must_be_blue = set(outputs)

        ops = []
        fast_mem = set()
        stored_vertices = set()

        # Track remaining uses dynamically
        rem_uses = dict(remaining_uses)

        def get_eviction_candidate(cur_idx):
            """Return the vertex in fast_mem to evict, using a smooth scoring function."""
            best = None
            best_key = None
            
            for v in fast_mem:
                if v in must_be_blue:
                    # Outputs: very high cost to evict
                    key = (2, float(recompute_cost[v]), 0.0)
                else:
                    if last_use[v] < cur_idx:
                        # Dead vertex, evict first
                        key = (0, 0.0, 0.0)
                    else:
                        # Smooth eviction score:
                        # - High recompute cost / remaining uses = expensive to recompute relative to how many times needed
                        # - High dist_to_sink = valuable for far-future computations
                        # - Low remaining uses = less valuable to keep
                        #
                        # Use a soft-min surrogate: we want to minimize the "cost" of evicting
                        # Cost of eviction = recompute_cost (if not stored) or 1 (if stored)
                        # Benefit of keeping = remaining_uses * dist_to_sink
                        #
                        # Eviction score = cost / (benefit + 1)  -- lower is better to evict
                        eff_cost = 1.0 if v in stored_vertices else float(recompute_cost[v])
                        benefit = max(1, rem_uses.get(v, 0)) * max(1, dist_to_sink.get(v, 1))
                        
                        # Smooth score: use log-sum-exp style combination
                        # score = eff_cost / (benefit + 1.0) + 0.1 * eff_cost / (rem_uses.get(v, 0) + 1.0)
                        score = eff_cost / (benefit + 1.0) + 0.1 * eff_cost / (rem_uses.get(v, 0) + 1.0)
                        key = (1, score, recompute_cost[v])
                
                if best_key is None or key < best_key:
                    best_key = key
                    best = v
            return best

        def make_space(needed, cur_idx):
            """Ensure at least 'needed' slots are free in fast_mem."""
            while len(fast_mem) >= fast_memory_budget:
                cand = get_eviction_candidate(cur_idx)
                if cand is None:
                    return False
                ops.append(['delete', cand])
                fast_mem.discard(cand)
            return True

        # Recursive ensure function with cycle protection
        visiting = set()

        def ensure_in_fast_mem(v, cur_idx):
            """
            Ensure vertex v is in fast_mem.
            Returns True on success, False on failure.
            """
            if time.time() > DEADLINE:
                return False

            if v in fast_mem:
                return True

            if v in visiting:
                return False

            ps = parents.get(v, [])

            if not ps:
                # Input vertex
                if not make_space(1, cur_idx):
                    return False
                ops.append(['load', v])
                fast_mem.add(v)
                return True
            else:
                # Computed vertex: check if it was stored
                if v in stored_vertices:
                    if not make_space(1, cur_idx):
                        return False
                    ops.append(['load', v])
                    fast_mem.add(v)
                    return True
                else:
                    # Computed vertex: ensure all parents first
                    visiting.add(v)
                    try:
                        for p in ps:
                            if not ensure_in_fast_mem(p, cur_idx):
                                return False

                        # Now compute v
                        if not make_space(1, cur_idx):
                            return False

                        ops.append(['compute', v])
                        fast_mem.add(v)
                        
                        # Decide whether to store
                        # Store if: remaining uses > 1 AND recompute cost is high
                        # OR if it's needed soon (within 5 topo steps) and expensive to recompute
                        num_rem = rem_uses.get(v, 0)
                        next_dist = max(0, last_use[v] - cur_idx)
                        should_store = (
                            (num_rem >= 2 and recompute_cost[v] >= 3) or
                            (num_rem >= 1 and recompute_cost[v] >= 5 and next_dist <= 10) or
                            (v in must_be_blue)
                        )
                        if should_store:
                            ops.append(['store', v])
                            stored_vertices.add(v)
                        
                        return True
                    finally:
                        visiting.discard(v)

        # Process outputs in topological order
        sorted_outputs = sorted(outputs, key=lambda o: topo_index.get(o, 0))
        
        for i, o in enumerate(sorted_outputs):
            if time.time() > DEADLINE:
                break
            cur_idx = topo_index.get(o, i)
            if not ensure_in_fast_mem(o, cur_idx):
                # Failed; try to free space more aggressively
                for u in list(fast_mem):
                    if u not in must_be_blue and u != o:
                        ops.append(['delete', u])
                        fast_mem.discard(u)
                        if len(fast_mem) < fast_memory_budget:
                            break
                if not ensure_in_fast_mem(o, cur_idx):
                    break

        # Ensure all outputs are in fast_mem at the end
        for o in sorted_outputs:
            if time.time() > DEADLINE:
                break
            if o not in fast_mem:
                for u in list(fast_mem):
                    if u not in must_be_blue:
                        ops.append(['delete', u])
                        fast_mem.discard(u)
                        if len(fast_mem) < fast_memory_budget:
                            break
                ensure_in_fast_mem(o, topo_index.get(o, 0))

        # Final check: all outputs must be in fast_mem
        for o in sorted_outputs:
            if o not in fast_mem:
                for u in list(fast_mem):
                    if u not in must_be_blue:
                        ops.append(['delete', u])
                        fast_mem.discard(u)
                ensure_in_fast_mem(o, topo_index.get(o, 0))

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
