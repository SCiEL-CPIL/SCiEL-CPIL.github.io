"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Strategy:
    - Topological-order pebbling with cost-aware eviction.
    - Eviction cost = (recompute cost of subtree) / (distance to next use).
    - Recompute cost approximated by subtree size (number of nodes that would
      need to be reloaded/recomputed if this vertex is evicted and later needed).
    - Outputs are never evicted (they must end in fast memory).
    - Final polish: if budget >= len(outputs), ensure all outputs are fast.
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin (evaluator kills at 90s)

    # ---- Parse DAG structure ----
    vertices = []
    edges = []
    outputs = []

    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('nodes', []))
        edges = dag.get('edges', dag.get('arcs', []))
        outputs = dag.get('outputs', dag.get('sinks', []))
    elif isinstance(dag, (list, tuple)):
        for item in dag:
            if isinstance(item, (list, tuple)):
                if len(item) >= 3:
                    v, parents, is_out = item[0], item[1], item[2]
                    vertices.append(v)
                    if is_out:
                        outputs.append(v)
                    for p in parents:
                        edges.append((p, v))
                elif len(item) == 2:
                    v, parents = item[0], item[1]
                    vertices.append(v)
                    for p in parents:
                        edges.append((p, v))
    else:
        # Fallback: try baseline
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []

    if not vertices:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []

    # ---- Build adjacency ----
    parents_map = defaultdict(list)
    children_map = defaultdict(list)
    all_vertices = set(vertices)
    output_set = set(outputs)

    for edge in edges:
        if len(edge) >= 2:
            p, c = edge[0], edge[1]
            parents_map[c].append(p)
            children_map[p].append(c)
            all_vertices.add(p)
            all_vertices.add(c)

    for v in outputs:
        all_vertices.add(v)

    if not all_vertices:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []

    # ---- Topological sort ----
    in_degree = {v: 0 for v in all_vertices}
    for v in all_vertices:
        for p in parents_map.get(v, []):
            in_degree[v] += 1

    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children_map.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)

    if len(topo_order) != len(all_vertices):
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []

    if time.time() > deadline:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []

    # ---- Identify inputs and outputs ----
    input_vertices = [v for v in topo_order if not parents_map.get(v)]
    output_vertices = [v for v in topo_order if v in output_set]
    if not output_vertices:
        output_vertices = [v for v in topo_order if not children_map.get(v)]

    n = len(topo_order)
    topo_pos = {v: i for i, v in enumerate(topo_order)}

    # ---- Precompute children positions and subtree sizes ----
    children_pos = {}
    for v in all_vertices:
        if v in children_map:
            children_pos[v] = sorted([topo_pos[c] for c in children_map[v]])
        else:
            children_pos[v] = []

    # Subtree size: number of nodes in the sub-DAG rooted at v (including v).
    # This approximates the recompute cost if v is evicted and later needed.
    subtree_size = {}
    # Process in reverse topological order
    for v in reversed(topo_order):
        s = 1
        for c in children_map.get(v, []):
            if c in subtree_size:
                s += subtree_size[c]
        subtree_size[v] = s

    # Cap subtree size to avoid overflow in scoring
    max_subtree = max(subtree_size.values()) if subtree_size else 1

    # Fan-out (direct children count)
    fan_out = {v: len(children_map.get(v, [])) for v in all_vertices}

    # Remaining uses (how many times v will be needed as a parent going forward)
    remaining_uses = {v: len(children_pos.get(v, [])) for v in all_vertices}

    # ---- State tracking ----
    state = {v: 'uncomputed' for v in all_vertices}
    fast_mem = set()

    # Child pointer for efficient next-use lookup
    child_ptr = {v: 0 for v in all_vertices}

    operations = []
    op_count = 0
    MAX_OPS = 20000

    def get_next_use(v, current_idx):
        """Get the next topo position where v is used as a parent, after current_idx."""
        cps = children_pos.get(v, [])
        ptr = child_ptr.get(v, 0)
        while ptr < len(cps) and cps[ptr] <= current_idx:
            ptr += 1
        child_ptr[v] = ptr
        if ptr < len(cps):
            return cps[ptr]
        return None

    def eviction_score(v, current_idx):
        """
        Score for evicting vertex v. Lower score = better to evict.
        We want to evict vertices with low recompute cost and far next use.
        
        Score = recompute_cost / max(1, next_use_distance)
        Lower is better to evict.
        """
        if v in output_set:
            # Never evict outputs (unless absolutely forced, but we try to avoid)
            return float('inf')
        
        rem = remaining_uses.get(v, 0)
        if rem <= 0:
            # No more uses - ideal to evict
            return -1000.0
        
        next_pos = get_next_use(v, current_idx)
        if next_pos is None:
            return -500.0
        
        dist = next_pos - current_idx
        if dist <= 0:
            dist = 1
        
        # Recompute cost: use subtree size as proxy
        # Normalize to [0, 1] range
        cost = subtree_size.get(v, 1) / max(1, max_subtree)
        
        # Score: high cost / far distance = low priority to keep = high priority to evict
        # We want LOW score to evict, so: score = cost / dist
        # But we also want to prefer evicting things with no more uses
        score = cost / dist
        
        # Penalize evicting things that are used very soon (high cost, small dist)
        # This is already captured: small dist -> large score -> less likely to evict
        
        return score

    # ---- Main pebbling loop ----
    for idx, v in enumerate(topo_order):
        if time.time() > deadline:
            break
        if op_count >= MAX_OPS:
            break

        # Load parents that are not in fast memory
        for p in parents_map.get(v, []):
            if state[p] != 'fast':
                operations.append(['load', p])
                op_count += 1
                state[p] = 'fast'
                fast_mem.add(p)
                if op_count >= MAX_OPS:
                    break

        if op_count >= MAX_OPS:
            break

        # Compute v
        operations.append(['compute', v])
        op_count += 1
        state[v] = 'fast'
        fast_mem.add(v)

        # Decrement remaining uses for parents
        for p in parents_map.get(v, []):
            if p in remaining_uses:
                remaining_uses[p] -= 1

        # Evict if over budget
        while len(fast_mem) > fast_memory_budget:
            if op_count >= MAX_OPS:
                break

            # Find best vertex to evict
            best_to_evict = None
            best_score = float('inf')

            for cand in fast_mem:
                if cand in output_set:
                    continue  # Skip outputs
                s = eviction_score(cand, idx)
                if s < best_score:
                    best_score = s
                    best_to_evict = cand

            if best_to_evict is None:
                # All remaining are outputs; forced to evict one
                # Pick the one with lowest remaining uses
                best_to_evict = None
                best_rem = float('inf')
                for cand in fast_mem:
                    rem = remaining_uses.get(cand, 0)
                    if rem < best_rem:
                        best_rem = rem
                        best_to_evict = cand
                if best_to_evict is None:
                    break

            operations.append(['store', best_to_evict])
            op_count += 1
            state[best_to_evict] = 'slow'
            fast_mem.discard(best_to_evict)

    # ---- Final: ensure all outputs are in fast memory ----
    if time.time() < deadline and op_count < MAX_OPS:
        for v in output_vertices:
            if state[v] != 'fast':
                operations.append(['load', v])
                op_count += 1
                state[v] = 'fast'
                fast_mem.add(v)

                # If over budget, evict non-outputs
                while len(fast_mem) > fast_memory_budget and op_count < MAX_OPS:
                    best_to_evict = None
                    best_score = float('inf')
                    for cand in fast_mem:
                        if cand in output_set:
                            continue
                        rem = remaining_uses.get(cand, 0)
                        if rem <= 0:
                            s = -1000.0
                        else:
                            s = subtree_size.get(cand, 1) / max(1, max_subtree)
                        if s < best_score:
                            best_score = s
                            best_to_evict = cand
                    if best_to_evict is None:
                        break
                    operations.append(['store', best_to_evict])
                    op_count += 1
                    state[best_to_evict] = 'slow'
                    fast_mem.discard(best_to_evict)

    # ---- Final cleanup: store non-output vertices ----
    if time.time() < deadline and op_count < MAX_OPS:
        for v in list(fast_mem):
            if v not in output_set and op_count < MAX_OPS:
                operations.append(['store', v])
                op_count += 1
                state[v] = 'slow'
                fast_mem.discard(v)

    # Limit operations
    if len(operations) > MAX_OPS:
        operations = operations[:MAX_OPS]

    return operations
# EVOLVE-BLOCK-END
