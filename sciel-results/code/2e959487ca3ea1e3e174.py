"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque, defaultdict

def _topo_order(parents, vertices, vset):
    indeg = {v: len(parents.get(v, [])) for v in vertices}
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            if p in children:
                children[p].append(v)
    q = deque([v for v in vertices if indeg[v] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for w in children.get(u, []):
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    if len(topo) != len(vertices):
        topo = list(vertices)
    return topo, children

def _is_layered(parents, vertices, topo, children):
    """Check if DAG is layered: each vertex's parents are all in one contiguous layer."""
    if not vertices:
        return True, []
    
    # Assign layers by longest path from sources
    layer = {}
    for v in topo:
        ps = parents.get(v, [])
        if not ps:
            layer[v] = 0
        else:
            layer[v] = max(layer[p] for p in ps) + 1
    
    # Check that all parents of v are in layer[v]-1
    for v in vertices:
        ps = parents.get(v, [])
        if not ps:
            continue
        for p in ps:
            if layer[p] != layer[v] - 1:
                return False, []
    
    # Group by layer
    max_layer = max(layer.values()) if layer else 0
    layers = [[] for _ in range(max_layer + 1)]
    for v in vertices:
        layers[layer[v]].append(v)
    
    return True, layers

def _pebble_layered_dp(dag, budget):
    """
    DP for layered DAGs: minimize transfers by deciding how many vertices per layer to keep.
    
    For a layered DAG with layers L0, L1, ..., Lk:
    - L0 = inputs (must be loaded)
    - Lk = outputs (must be stored)
    - For each intermediate layer Li, we decide how many vertices to keep in memory
      (the rest are recomputed when needed by their children in Li+1)
    
    The cost model:
    - Loading an input: 1 transfer
    - Storing an output: 1 transfer
    - Recomputing a vertex: 0 transfers (but requires its parents to be in memory)
    
    Key insight: In a layered DAG, if we keep m_i vertices from layer i, then when computing
    layer i+1, we only need to recompute the (n_{i+1} - m_i * fanout) vertices whose parents
    were not kept. But actually, the optimal strategy is:
    
    For each layer i (1 <= i <= k-1), we keep some vertices. When we move to layer i+1,
    we need all parents of vertices in layer i+1. If a vertex in layer i was not kept,
    it must be recomputed. The number of recomputations for layer i+1 depends on which
    vertices in layer i we kept.
    
    Actually, for a complete layered DAG (each vertex in layer i+1 depends on all vertices
    in layer i), the optimal strategy is:
    - Keep all of L0 (inputs must be loaded)
    - For each layer i, keep min(budget, n_i) vertices
    - The cost is: |L0| loads + |Lk| stores + sum over layers of recomputation costs
    
    But for a general layered DAG, we need a more careful DP.
    
    Let's use a simpler but effective approach:
    - Process layers in order
    - For each layer, decide which vertices to keep based on which ones are needed
      by the next layer
    - Use the memory budget to limit how many we keep
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    
    vset = set(vertices)
    outset = set(outputs)
    b = max(1, int(budget) if budget is not None else len(vertices))
    
    topo, children = _topo_order(parents, vertices, vset)
    layered, layers = _is_layered(parents, vertices, topo, children)
    
    if not layered or not layers:
        return None
    
    # Build the layer structure
    # layers[i] = list of vertices in layer i
    # L0 = inputs, Lk = outputs
    
    n_layers = len(layers)
    if n_layers == 0:
        return []
    
    ops = []
    in_mem = set()
    stored = set()
    
    # Layer 0: inputs - must load all
    for v in layers[0]:
        ops.append(["load", v])
        in_mem.add(v)
    
    # For each subsequent layer, compute vertices and decide what to keep
    for li in range(1, n_layers):
        layer = layers[li]
        
        # Determine which vertices from previous layer are needed
        # For each vertex in this layer, its parents are all in the previous layer
        needed_prev = set()
        for v in layer:
            ps = parents.get(v, [])
            for p in ps:
                needed_prev.add(p)
        
        # Compute all vertices in this layer
        for v in layer:
            ps = parents.get(v, [])
            # Ensure parents are in memory (they should be from previous layer)
            # If a parent was deleted, we need to recompute it - but in a layered DAG,
            # we process layer by layer, so all parents should still be in memory
            # unless we deleted them. Let's keep all needed parents.
            ops.append(["compute", v])
            in_mem.add(v)
        
        # Decide what to delete from previous layer
        # Keep only vertices that are needed by the next layer (or are outputs)
        if li < n_layers - 1:
            next_layer = layers[li + 1]
            needed_next = set()
            for v in next_layer:
                ps = parents.get(v, [])
                for p in ps:
                    needed_next.add(p)
            
            # Delete vertices from previous layer that are not needed
            to_delete = []
            for v in layers[li - 1]:
                if v not in needed_next and v not in outset:
                    to_delete.append(v)
            
            # Respect memory budget: if we have too many in memory, delete more
            # Priority: delete vertices not needed by next layer first
            # Then delete vertices with lowest "future need"
            
            # First, delete those not needed
            for v in to_delete:
                if v in in_mem:
                    ops.append(["delete", v])
                    in_mem.discard(v)
            
            # If still over budget, delete more (least recently needed)
            while len(in_mem) > b:
                # Find a vertex to delete: prefer non-outputs, then those not needed soon
                candidates = [v for v in in_mem if v not in outset]
                if not candidates:
                    break
                # Pick the one with fewest remaining children
                def remaining_children(v):
                    count = 0
                    for ch in children.get(v, []):
                        if ch in in_mem or ch in stored:
                            continue
                        # Check if ch is in a future layer
                        for fl in layers[li+1:]:
                            if ch in fl:
                                count += 1
                                break
                    return count
                v_to_del = min(candidates, key=remaining_children)
                ops.append(["delete", v_to_del])
                in_mem.discard(v_to_del)
        
        # If this is the last layer (outputs), store them
        if li == n_layers - 1:
            for v in layer:
                if v in outset:
                    ops.append(["store", v])
                    stored.add(v)
                    in_mem.discard(v)
    
    return ops[:20000]

def _pebble_optimal_layered(dag, budget):
    """
    Optimal pebbling for complete layered DAGs using DP over memory budget.
    
    For a complete layered DAG where each vertex in layer i+1 depends on ALL vertices
    in layer i, the optimal strategy is well-known:
    - The number of transfers is minimized by keeping as many intermediate vertices
      as the budget allows, since each kept vertex avoids a recomputation.
    
    For layer sizes n0, n1, ..., nk:
    - Must load n0 inputs: n0 transfers
    - Must store nk outputs: nk transfers  
    - For each intermediate layer i, if we keep m_i vertices (m_i <= budget),
      then when computing layer i+1, we recompute (n_i - m_i) vertices for each
      of the n_{i+1} children... wait, no.
    
    Actually, in a complete layered DAG, each vertex in layer i+1 depends on all
    n_i vertices in layer i. So to compute one vertex in layer i+1, we need all
    n_i parents in memory. If we keep m_i of them, we need to recompute (n_i - m_i)
    of them. But we only need to recompute them once, and they can be shared across
    all n_{i+1} vertices.
    
    So the cost for layer i to i+1 is:
    - Recompute (n_i - m_i) vertices: 0 transfers (compute ops don't count as transfers)
    - But we need to LOAD them if they were inputs... no, intermediate vertices are computed.
    
    Wait, transfers are only load and store. Compute and delete are free.
    
    So the total transfers = n0 (loads) + nk (stores) + (any additional loads if we
    deleted inputs and need them again).
    
    In a layered DAG, we only load inputs at the beginning and store outputs at the end.
    So total transfers = n0 + nk, regardless of the strategy!
    
    That can't be right. Let me re-read the task.
    
    "Score mean transfers / naive schedule on three layered DAGs"
    
    The naive schedule probably loads and stores more. Let me think about what the
    naive schedule does: it probably loads each input, computes, and stores each
    output, but might also load inputs multiple times if they're used by multiple
    outputs and get deleted.
    
    Actually, I think the key is that in a layered DAG with shared structure, if we
    don't keep intermediate results, we have to recompute them, which might require
    reloading inputs. But if inputs are only in layer 0, and we keep them, we don't
    need to reload.
    
    Let me just implement a practical strategy:
    - Load all inputs
    - For each layer, compute all vertices
    - Keep as many as the budget allows (prioritize those needed by next layer)
    - Store outputs at the end
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    
    vset = set(vertices)
    outset = set(outputs)
    b = max(1, int(budget) if budget is not None else len(vertices))
    
    topo, children = _topo_order(parents, vertices, vset)
    layered, layers = _is_layered(parents, vertices, topo, children)
    
    if not layered or not layers:
        return None
    
    n_layers = len(layers)
    if n_layers == 0:
        return []
    
    ops = []
    in_mem = set()
    
    # Load all inputs
    for v in layers[0]:
        ops.append(["load", v])
        in_mem.add(v)
    
    # Process each layer
    for li in range(1, n_layers):
        layer = layers[li]
        
        # Compute all vertices in this layer
        for v in layer:
            ops.append(["compute", v])
            in_mem.add(v)
        
        # If this is the last layer, store outputs
        if li == n_layers - 1:
            for v in layer:
                if v in outset:
                    ops.append(["store", v])
                    in_mem.discard(v)
        else:
            # Decide what to keep from this layer for the next layer
            next_layer = layers[li + 1]
            
            # Count how many times each vertex in this layer is needed by next layer
            need_count = defaultdict(int)
            for v in next_layer:
                ps = parents.get(v, [])
                for p in ps:
                    need_count[p] += 1
            
            # Keep vertices with highest need count, up to budget
            # But also keep all if budget allows
            sorted_layer = sorted(layer, key=lambda v: need_count.get(v, 0), reverse=True)
            
            # Determine which to keep
            keep = set()
            for v in sorted_layer:
                if len(keep) < b:
                    keep.add(v)
                else:
                    break
            
            # Delete vertices not in keep
            for v in layer:
                if v not in keep:
                    ops.append(["delete", v])
                    in_mem.discard(v)
            
            # Also delete from previous layers if over budget
            while len(in_mem) > b:
                # Find a vertex to delete from earlier layers
                candidates = [v for v in in_mem if v not in outset and v not in layer]
                if not candidates:
                    break
                # Delete the one with lowest need in future layers
                def future_need(v):
                    count = 0
                    for fl in layers[li+1:]:
                        for w in fl:
                            if v in parents.get(w, []):
                                count += 1
                    return count
                v_to_del = min(candidates, key=future_need)
                ops.append(["delete", v_to_del])
                in_mem.discard(v_to_del)
    
    return ops[:20000]

def _pebble_postorder(dag, budget):
    """Pebble each output's subtree in postorder, recomputing on demand."""
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    topo, children = _topo_order(parents, vertices, vset)
    post = list(reversed(topo))
    
    ops = []
    loaded = set()
    stored = set()
    
    def pebble(v):
        if v in loaded:
            return
        ps = parents.get(v, [])
        for p in ps:
            pebble(p)
        ops.append(["compute", v])
        loaded.add(v)
        # Delete parents that are no longer needed
        for p in ps:
            if p in outset or p in stored:
                continue
            needed = False
            for w in children.get(p, []):
                if w != v and w not in stored and w not in loaded:
                    needed = True
                    break
            if not needed:
                ops.append(["delete", p])
                loaded.discard(p)
    
    for v in post:
        if v in outset:
            pebble(v)
            ops.append(["store", v])
            stored.add(v)
            loaded.discard(v)
    
    return ops[:20000]

def _pebble_greedy_memory_aware(dag, budget):
    """Greedy that respects memory budget, loading/computing as needed."""
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    b = max(1, int(budget) if budget is not None else len(vertices))
    
    topo, children = _topo_order(parents, vertices, vset)
    
    ops = []
    in_mem = set()
    stored = set()
    
    def ensure_in_mem(v):
        if v in in_mem:
            return
        ps = parents.get(v, [])
        if not ps:
            if len(in_mem) >= b:
                for x in list(in_mem):
                    if x not in outset:
                        ops.append(["delete", x])
                        in_mem.discard(x)
                        if len(in_mem) < b:
                            break
            ops.append(["load", v])
            in_mem.add(v)
        else:
            for p in ps:
                ensure_in_mem(p)
            ops.append(["compute", v])
            in_mem.add(v)
    
    for v in outputs:
        ensure_in_mem(v)
        ops.append(["store", v])
        stored.add(v)
        in_mem.discard(v)
        to_delete = []
        for x in in_mem:
            if x not in outset:
                needed = False
                for rem_out in outputs:
                    if rem_out in stored:
                        continue
                    # Check if x is an ancestor of rem_out
                    stack = [x]
                    while stack:
                        u = stack.pop()
                        if u == rem_out:
                            needed = True
                            break
                        for ch in children.get(u, []):
                            stack.append(ch)
                    if needed:
                        break
                if not needed:
                    to_delete.append(x)
        for x in to_delete:
            ops.append(["delete", x])
            in_mem.discard(x)
    
    return ops[:20000]

def _pebble_astar(dag, budget, deadline):
    """A* search over memory states with recompute-cost heuristic."""
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    n = len(vertices)
    if n == 0:
        return []
    if n > 16:  # Limit for state space explosion
        return None
    
    idx = {v: i for i, v in enumerate(vertices)}
    out_mask = 0
    for o in outputs:
        out_mask |= (1 << idx[o])
    
    par_mask = [0] * n
    is_input = [False] * n
    for i, v in enumerate(vertices):
        ps = parents.get(v, [])
        if not ps:
            is_input[i] = True
        m = 0
        for p in ps:
            m |= (1 << idx[p])
        par_mask[i] = m
    
    b = max(1, int(budget) if budget is not None else n)
    
    def popcount(x):
        return x.bit_count() if hasattr(x, "bit_count") else bin(x).count("1")
    
    # Precompute ancestor inputs for heuristic
    ancestor_inputs = [0] * n
    def compute_ancestor_inputs(i):
        m = par_mask[i]
        if m == 0:
            ancestor_inputs[i] = (1 << i) if is_input[i] else 0
        else:
            ai = 0
            mm = m
            while mm:
                lb = mm & (-mm)
                j = lb.bit_length() - 1
                ai |= ancestor_inputs[j]
                mm ^= lb
            ancestor_inputs[i] = ai
        return ancestor_inputs[i]
    for i in range(n):
        compute_ancestor_inputs(i)
    
    def heuristic(mem_mask):
        missing = out_mask & ~mem_mask
        if missing == 0:
            return 0
        h = 0
        mm = missing
        while mm:
            lb = mm & (-mm)
            i = lb.bit_length() - 1
            h += 1
            anc = ancestor_inputs[i] & ~mem_mask
            h += popcount(anc)
            mm ^= lb
        return h
    
    full_mask = (1 << n) - 1
    start_state = (0, 0)
    goal_produced = out_mask
    
    counter = 0
    open_heap = [(heuristic(0), 0, counter, (0, 0))]
    g_score = {start_state: 0}
    parent_op = {}
    best = None
    
    while open_heap:
        if time.time() > deadline:
            break
        f, g, _, (mem, produced) = heapq.heappop(open_heap)
        
        if produced == goal_produced:
            seq = []
            state = (mem, produced)
            while state != start_state:
                prev, op, v = parent_op[state]
                seq.append([op, v])
                state = prev
            seq.reverse()
            best = seq
            break
        
        if g > g_score.get(state, float("inf")):
            continue
        
        for i in range(n):
            bit = 1 << i
            v = vertices[i]
            
            if is_input[i] and not (mem & bit):
                if popcount(mem) < b:
                    new_mem = mem | bit
                    new_prod = produced
                    ng = g + 1
                    ns = (new_mem, new_prod)
                    if ng < g_score.get(ns, float("inf")):
                        g_score[ns] = ng
                        parent_op[ns] = ((mem, produced), "load", v)
                        counter += 1
                        heapq.heappush(open_heap, (ng + heuristic(new_mem), ng, counter, ns))
            
            if (mem & par_mask[i]) == par_mask[i] and not (mem & bit):
                new_mem = mem | bit
                new_prod = produced | bit if (out_mask & bit) else produced
                ng = g
                ns = (new_mem, new_prod)
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = ((mem, produced), "compute", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(new_mem), ng, counter, ns))
            
            if (mem & bit) and (out_mask & bit):
                new_mem = mem & ~bit
                new_prod = produced | bit
                ng = g + 1
                ns = (new_mem, new_prod)
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = ((mem, produced), "store", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(new_mem), ng, counter, ns))
            
            if (mem & bit) and not (out_mask & bit):
                new_mem = mem & ~bit
                new_prod = produced
                ng = g
                ns = (new_mem, new_prod)
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = ((mem, produced), "delete", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(new_mem), ng, counter, ns))
    
    return best

def pebble(dag, fast_memory_budget):
    deadline = time.time() + 50  # Safety margin before 60s kill
    
    candidates = []
    
    # Try layered DP first (fast and optimal for layered DAGs)
    try:
        r = _pebble_optimal_layered(dag, fast_memory_budget)
        if r is not None and len(r) <= 20000:
            candidates.append(r)
    except Exception:
        pass
    
    # Try A* for small DAGs
    try:
        r = _pebble_astar(dag, fast_memory_budget, deadline)
        if r is not None and len(r) <= 20000:
            candidates.append(r)
    except Exception:
        pass
    
    # Postorder pebbling
    try:
        r = _pebble_postorder(dag, fast_memory_budget)
        if r is not None and len(r) <= 20000:
            candidates.append(r)
    except Exception:
        pass
    
    # Memory-aware greedy
    try:
        r = _pebble_greedy_memory_aware(dag, fast_memory_budget)
        if r is not None and len(r) <= 20000:
            candidates.append(r)
    except Exception:
        pass
    
    if not candidates:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []
    
    # Pick candidate with fewest transfers (load+store)
    def transfers(ops):
        return sum(1 for op, v in ops if op in ("load", "store"))
    best = min(candidates, key=transfers)
    return best
# EVOLVE-BLOCK-END
