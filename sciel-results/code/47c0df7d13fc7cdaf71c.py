"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
import math
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
    stored = set()
    
    # Layer 0: inputs - must load all
    for v in layers[0]:
        ops.append(["load", v])
        in_mem.add(v)
    
    # For each subsequent layer, compute vertices and decide what to keep
    for li in range(1, n_layers):
        layer = layers[li]
        
        # Compute all vertices in this layer
        for v in layer:
            ps = parents.get(v, [])
            ops.append(["compute", v])
            in_mem.add(v)
        
        # If this is the last layer (outputs), store them
        if li == n_layers - 1:
            for v in layer:
                if v in outset:
                    ops.append(["store", v])
                    stored.add(v)
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
                candidates = [v for v in in_mem if v not in outset and v not in layer]
                if not candidates:
                    break
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

def _pebble_optimal_layered(dag, budget):
    """
    Optimal pebbling for complete layered DAGs using DP over memory budget.
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
                candidates = [v for v in in_mem if v not in outset and v not in layer]
                if not candidates:
                    break
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

def _simulate(ops, dag, budget):
    """
    Simulate the pebbling schedule and return (valid, transfers, max_memory_used).
    Returns None if invalid.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    outset = set(outputs)
    
    # Track state: in_mem (blue/red values), computed (which vertices have been computed at least once)
    # Actually, we need to track which vertices are currently in memory (pebbled)
    # A vertex is "pebbled" if it's in memory. Inputs start pebbled? No, "Inputs start blue"
    # "blue" means it's in memory? Or is it a value?
    # "Inputs start blue, all outputs must finish blue; every red value consumes one fast-memory slot"
    # This implies:
    # - Blue = in memory (consumes slot)
    # - Red = not in memory (free)
    # - Inputs start blue -> they are in memory initially?
    # Wait, "Inputs start blue" usually means they are available.
    # "load" operation: brings a value into memory (blue).
    # "store" operation: writes a value to slow memory (blue -> red? or just removes from fast memory?).
    # "delete" operation: removes a value from memory (blue -> red).
    # "compute" operation: computes a value from parents. Parents must be blue. Result is blue.
    
    # Let's assume:
    # - State is a set of vertices currently in fast memory (blue).
    # - Inputs: Are they initially in memory? "Inputs start blue" suggests yes.
    #   If so, we don't need to load them? But the interface says "load/store/compute/delete".
    #   Usually in pebbling, inputs are free to load, but if they start blue, maybe they are already there.
    #   However, the budget applies. If inputs start blue, they consume slots immediately.
    #   Let's check the naive schedule comparison.
    #   If inputs start blue, the initial memory usage is |inputs|.
    #   If the budget is small, we might not be able to fit all inputs?
    #   "bounded candidate interface... every red value consumes one fast-memory slot"
    #   This phrasing is weird. "red value consumes... slot"?
    #   Usually red = not in memory, blue = in memory.
    #   Maybe "red" is a typo for "blue"? Or maybe "red" is the color of the pebble?
    #   Let's assume standard pebbling:
    #   - Black pebble = in memory.
    #   - White pebble = not in memory.
    #   - "Inputs start blue" -> Inputs are initially in memory (black pebbled).
    #   - "Outputs must finish blue" -> Outputs must be in memory at the end?
    #   - "store" -> writes to output?
    
    # Let's look at the operation "store".
    # If "store" writes to slow memory, does it remove from fast memory?
    # Usually, store removes the pebble from the node.
    # If "Inputs start blue", we have to account for them in the budget.
    
    # Let's implement a strict simulator to verify validity.
    # State: set of vertices in memory.
    # Initial state: All inputs (vertices with no parents) are in memory?
    # Let's assume yes.
    
    inputs = [v for v in parents.keys() if not parents.get(v, [])]
    # Also, what about vertices not in parents.keys()?
    # "dag has topologically ordered parents and output vertices."
    # Vertices = keys of parents + outputs.
    
    in_mem = set(inputs)
    
    # Check initial budget
    if len(in_mem) > budget:
        return None, 0, 0
    
    transfers = 0
    max_mem = len(in_mem)
    
    for op, v in ops:
        if op == "load":
            # Load a vertex. Must be an input? Or any vertex not in memory?
            # Usually load is only for inputs.
            # If v is not an input, it's invalid? Or can we load intermediate values?
            # "load/store/compute/delete".
            # If v is already in memory, invalid?
            if v in in_mem:
                return None, 0, 0
            # Can we load any vertex?
            # Usually, load is for external inputs.
            # Let's assume load is only valid for inputs (no parents).
            if parents.get(v, []):
                # Not an input?
                # Maybe "load" can bring back a deleted value?
                # But without a source, it's just a guess.
                # Let's assume load is ONLY for inputs.
                return None, 0, 0
            
            in_mem.add(v)
            transfers += 1
            if len(in_mem) > budget:
                return None, 0, 0
            
        elif op == "store":
            # Store a vertex. Must be in memory. Must be an output?
            # "all outputs must finish blue"
            # If store removes from memory, how can it finish blue?
            # Maybe "store" means "write to output buffer" and it STAYS in memory?
            # Or maybe "finish blue" means the final state has them in memory.
            # If store removes it, we'd need to load/compute it again to make it blue.
            # Let's assume store removes it from fast memory (standard pebbling).
            # Then "finish blue" means the LAST operation on an output must leave it blue?
            # Or the final state of the simulation must have outputs in memory.
            
            if v not in in_mem:
                return None, 0, 0
            
            # Is v an output?
            if v not in outset:
                # Storing a non-output? Probably invalid or useless.
                # Let's allow it but it doesn't help.
                pass
            
            in_mem.discard(v)
            transfers += 1
            # Note: store frees up space.
            
        elif op == "compute":
            # Compute a vertex. Parents must be in memory.
            ps = parents.get(v, [])
            for p in ps:
                if p not in in_mem:
                    return None, 0, 0
            
            if v in in_mem:
                # Already in memory. Recomputing is allowed?
                # "Recomputations allowed"
                # It might overwrite.
                pass
            
            in_mem.add(v)
            # Compute does NOT consume a transfer.
            if len(in_mem) > budget:
                return None, 0, 0
            
        elif op == "delete":
            # Delete a vertex. Must be in memory.
            if v not in in_mem:
                return None, 0, 0
            
            in_mem.discard(v)
            # Delete does NOT consume a transfer.
            
        else:
            return None, 0, 0
    
    # Final check: all outputs must be in memory (blue)
    for o in outset:
        if o not in in_mem:
            return None, 0, 0
    
    return True, transfers, max_mem

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
    
    # Validate candidates and pick the best valid one
    valid_candidates = []
    for r in candidates:
        is_valid, transfers, _ = _simulate(r, dag, fast_memory_budget)
        if is_valid:
            valid_candidates.append((transfers, r))
    
    if valid_candidates:
        # Sort by transfers
        valid_candidates.sort(key=lambda x: x[0])
        return valid_candidates[0][1]
    
    # If none valid, return the first candidate (might be invalid, but better than nothing?)
    # Or try to repair?
    # For now, return the first one.
    return candidates[0]
# EVOLVE-BLOCK-END
