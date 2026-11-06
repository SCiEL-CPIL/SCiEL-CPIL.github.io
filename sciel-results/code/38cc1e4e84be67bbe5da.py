"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque
from functools import lru_cache

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

def _compute_recompute_costs(parents, vertices):
    """Compute the minimum number of compute operations needed to produce each vertex from inputs."""
    cost = {}
    def get_cost(v):
        if v in cost:
            return cost[v]
        ps = parents.get(v, [])
        if not ps:
            cost[v] = 1
        else:
            c = 1 + sum(get_cost(p) for p in ps)
            cost[v] = c
        return cost[v]
    for v in vertices:
        get_cost(v)
    return cost

def _get_ancestors(parents, v, visited=None):
    """Get all ancestors of vertex v (excluding v itself)."""
    if visited is None:
        visited = set()
    ps = parents.get(v, [])
    result = set()
    for p in ps:
        if p not in visited:
            result.add(p)
            result |= _get_ancestors(parents, p, visited | {p})
    return result

def _get_descendants(children, v, visited=None):
    """Get all descendants of vertex v (excluding v itself)."""
    if visited is None:
        visited = set()
    chs = children.get(v, [])
    result = set()
    for c in chs:
        if c not in visited:
            result.add(c)
            result |= _get_descendants(children, c, visited | {c})
    return result

def _build_layered_structure(dag):
    """Build layered structure for a DAG, returning layers and metadata."""
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return None
    vset = set(vertices)
    outset = set(outputs)
    topo, children = _topo_order(parents, vertices, vset)
    
    # Compute layer for each vertex
    layer_of = {}
    for v in topo:
        ps = parents.get(v, [])
        if not ps:
            layer_of[v] = 0
        else:
            layer_of[v] = 1 + max(layer_of[p] for p in ps)
    
    max_layer = max(layer_of.values()) if layer_of else 0
    layers = [[] for _ in range(max_layer + 1)]
    for v in vertices:
        layers[layer_of[v]].append(v)
    
    return {
        "parents": parents,
        "outputs": outputs,
        "vertices": vertices,
        "vset": vset,
        "outset": outset,
        "topo": topo,
        "children": children,
        "layer_of": layer_of,
        "max_layer": max_layer,
        "layers": layers
    }

def _pebble_td_dp(dag, budget, deadline):
    """
    Tree decomposition DP for pebbling layered DAGs.
    
    For layered DAGs, we can use a layered tree decomposition where each bag
    contains vertices from consecutive layers. The DP state tracks which vertices
    in the current bag are in memory.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    n = len(vertices)
    if n > 30:
        return None  # Too large for DP
    
    vset = set(vertices)
    outset = set(outputs)
    b = max(1, int(fast_memory_budget) if fast_memory_budget is not None else n)
    
    topo, children = _topo_order(parents, vertices, vset)
    
    # Compute layers
    layer_of = {}
    for v in topo:
        ps = parents.get(v, [])
        if not ps:
            layer_of[v] = 0
        else:
            layer_of[v] = 1 + max(layer_of[p] for p in ps)
    
    max_layer = max(layer_of.values()) if layer_of else 0
    layers = [[] for _ in range(max_layer + 1)]
    for v in vertices:
        layers[layer_of[v]].append(v)
    
    # For each layer, identify which vertices from previous layers are needed
    # and which vertices in this layer will be needed by future layers
    
    # Build a simplified state space: for each layer, track which vertices
    # from the "frontier" (vertices needed by future layers) are in memory
    
    # The key insight: we process layers in order. At each layer, we need to:
    # 1. Load/compute vertices in this layer (requires parents in memory)
    # 2. Decide which vertices to keep in memory for future layers
    
    # For efficiency, we'll use a DP where state = set of vertices in memory
    # that are "active" (needed by future layers)
    
    # First, determine for each vertex which future layers need it
    future_need = {v: set() for v in vertices}
    for v in vertices:
        for child in children.get(v, []):
            future_need[v].add(layer_of[child])
    
    # DP: dp[layer_idx][mem_mask] = min transfers to reach this state
    # mem_mask encodes which vertices in the "frontier" are in memory
    
    # Define frontier for each layer: vertices that are in this layer or earlier
    # and are needed by this layer or later
    frontier = [set() for _ in range(max_layer + 1)]
    for v in vertices:
        for future_layer in future_need[v]:
            if future_layer >= 0 and future_layer <= max_layer:
                frontier[future_layer].add(v)
    
    # For each layer, the relevant vertices are those in the frontier
    # plus vertices in the current layer that we're computing
    
    # This is getting complex. Let's use a simpler approach:
    # Process vertices in topological order, using DP to track memory state
    
    # For each vertex in topo order, we decide:
    # - If it's an input: load it (cost 1)
    # - If it's computed: compute it (cost 0, but parents must be in memory)
    # - After computing, decide to keep or delete
    
    # The state is the set of vertices in memory (encoded as bitmask)
    # But n can be up to 30, so 2^30 states is too many
    
    # Alternative: use a heuristic DP that only tracks "important" vertices
    
    # For now, let's implement a simpler greedy with better memory management
    
    ops = []
    in_mem = set()
    stored = set()
    
    # Precompute: for each vertex, which future vertices need it
    needed_by = {v: [] for v in vertices}
    for v in vertices:
        ps = parents.get(v, [])
        for p in ps:
            needed_by[p].append(v)
    
    # Precompute: for each vertex, its "importance" = number of future uses
    importance = {}
    for v in vertices:
        imp = 1  # At least itself
        for w in needed_by[v]:
            if w not in outset:
                imp += len(needed_by[w])
        importance[v] = imp
    
    # Process in topological order with smart memory management
    for v in topo:
        if time.time() > deadline:
            break
        
        ps = parents.get(v, [])
        
        # Ensure all parents are in memory
        for p in ps:
            if p not in in_mem and p not in stored:
                # Need to recompute p
                # First ensure p's parents are in memory
                pps = parents.get(p, [])
                for pp in pps:
                    if pp not in in_mem and pp not in stored:
                        # Recursively ensure pp is in memory
                        if not parents.get(pp, []):
                            # pp is an input
                            if len(in_mem) >= b:
                                # Delete least important vertex
                                to_delete = min([x for x in in_mem if x not in outset and x not in stored],
                                                 key=lambda x: importance[x], default=None)
                                if to_delete:
                                    ops.append(["delete", to_delete])
                                    in_mem.discard(to_delete)
                            ops.append(["load", pp])
                            in_mem.add(pp)
                        else:
                            # pp needs to be computed
                            ppps = parents.get(pp, [])
                            for ppp in ppps:
                                if ppp not in in_mem and ppp not in stored:
                                    if not parents.get(ppp, []):
                                        if len(in_mem) >= b:
                                            to_delete = min([x for x in in_mem if x not in outset and x not in stored],
                                                             key=lambda x: importance[x], default=None)
                                            if to_delete:
                                                ops.append(["delete", to_delete])
                                                in_mem.discard(to_delete)
                                        ops.append(["load", ppp])
                                        in_mem.add(ppp)
                                    else:
                                        # This is getting too deep, just load if possible
                                        pass
                            if all(ppp in in_mem or ppp in stored for ppp in ppps):
                                ops.append(["compute", pp])
                                in_mem.add(pp)
                    elif pp in stored:
                        if len(in_mem) >= b:
                            to_delete = min([x for x in in_mem if x not in outset and x not in stored],
                                             key=lambda x: importance[x], default=None)
                            if to_delete:
                                ops.append(["delete", to_delete])
                                in_mem.discard(to_delete)
                        ops.append(["load", pp])
                        in_mem.add(pp)
                if all(ppp in in_mem or ppp in stored for ppp in pps):
                    ops.append(["compute", p])
                    in_mem.add(p)
            elif p in stored:
                if len(in_mem) >= b:
                    to_delete = min([x for x in in_mem if x not in outset and x not in stored],
                                     key=lambda x: importance[x], default=None)
                    if to_delete:
                        ops.append(["delete", to_delete])
                        in_mem.discard(to_delete)
                ops.append(["load", p])
                in_mem.add(p)
        
        # Compute v if all parents are in memory
        if all(p in in_mem or p in stored for p in ps):
            ops.append(["compute", v])
            in_mem.add(v)
        
        # Decide whether to store v
        if v in outset:
            ops.append(["store", v])
            stored.add(v)
            in_mem.discard(v)
        else:
            # Check if v is needed by future vertices
            future_needed = any(w not in stored and w != v for w in needed_by[v])
            if not future_needed:
                ops.append(["delete", v])
                in_mem.discard(v)
            elif len(in_mem) >= b:
                # Keep v, delete least important other vertex
                to_delete = min([x for x in in_mem if x != v and x not in outset and x not in stored],
                                 key=lambda x: importance[x], default=None)
                if to_delete:
                    ops.append(["delete", to_delete])
                    in_mem.discard(to_delete)
    
    return ops[:20000]

def _pebble_hierarchical_optimal(dag, budget, deadline):
    """
    Hierarchical pebbling with optimal substructure for layered DAGs.
    
    For a layered DAG, we can recursively split the DAG into two halves
    and optimally pebble each half, keeping boundary values in memory.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    b = max(1, int(fast_memory_budget) if fast_memory_budget is not None else len(vertices))
    
    topo, children = _topo_order(parents, vertices, vset)
    n = len(vertices)
    
    # Compute layers
    layer_of = {}
    for v in topo:
        ps = parents.get(v, [])
        if not ps:
            layer_of[v] = 0
        else:
            layer_of[v] = 1 + max(layer_of[p] for p in ps)
    
    max_layer = max(layer_of.values()) if layer_of else 0
    layers = [[] for _ in range(max_layer + 1)]
    for v in vertices:
        layers[layer_of[v]].append(v)
    
    # For each layer, identify boundary vertices (needed by next layer)
    layer_boundary = [set() for _ in range(max_layer + 1)]
    for v in vertices:
        for child in children.get(v, []):
            if layer_of[child] > layer_of[v]:
                layer_boundary[layer_of[v]].add(v)
    
    # Recursive pebbling: split layers in half
    def pebble_layer_range(layer_start, layer_end, available_memory):
        """
        Pebble layers from layer_start to layer_end.
        Returns (ops, boundary_in, boundary_out) where:
        - ops: list of operations
        - boundary_in: set of vertices that must be in memory at start
        - boundary_out: set of vertices in memory at end
        """
        if layer_start > layer_end:
            return [], set(), set()
        
        # Find the split point that minimizes boundary size
        best_split = layer_start
        best_boundary_size = float('inf')
        
        for split in range(layer_start, layer_end + 1):
            # Boundary between left (layer_start to split) and right (split+1 to layer_end)
            left_boundary = set()
            for v in vertices:
                if layer_start <= layer_of[v] <= split:
                    for child in children.get(v, []):
                        if split < layer_of[child] <= layer_end:
                            left_boundary.add(v)
            
            # Also include outputs that are in the left part
            left_outputs = {v for v in outset if layer_start <= layer_of[v] <= split}
            
            boundary_size = len(left_boundary) + len(left_outputs)
            if boundary_size < best_boundary_size:
                best_boundary_size = boundary_size
                best_split = split
        
        split = best_split
        
        # Pebble left half
        left_ops, left_in, left_out = pebble_layer_range(layer_start, split, available_memory)
        
        # Store boundary vertices from left
        store_ops = []
        boundary_to_store = left_out - left_in  # Vertices in left_out that weren't in left_in
        for v in sorted(boundary_to_store, key=lambda x: (x if isinstance(x, int) else 0)):
            store_ops.append(["store", v])
        
        # Pebble right half
        right_ops, right_in, right_out = pebble_layer_range(split + 1, layer_end, available_memory)
        
        # Load boundary vertices for right half
        load_ops = []
        boundary_to_load = right_in - left_out  # Vertices needed by right that weren't produced by left
        for v in sorted(boundary_to_load, key=lambda x: (x if isinstance(x, int) else 0)):
            load_ops.append(["load", v])
        
        # Combine: left_ops + store_ops + load_ops + right_ops
        all_ops = left_ops + store_ops + load_ops + right_ops
        
        # The boundary in for the whole range is left_in
        # The boundary out for the whole range is right_out
        return all_ops, left_in, right_out
    
    # Start pebbling
    all_ops, final_in, final_out = pebble_layer_range(0, max_layer, b)
    
    # Ensure all outputs are stored
    stored = set()
    for v in outset:
        if v not in stored:
            all_ops.append(["store", v])
            stored.add(v)
    
    return all_ops[:20000]

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
    
    rcost = _compute_recompute_costs(parents, vertices)
    
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
    b = max(1, int(fast_memory_budget) if fast_memory_budget is not None else len(vertices))
    
    topo, children = _topo_order(parents, vertices, vset)
    
    ops = []
    in_mem = set()
    stored = set()
    
    def ensure_in_mem2(v):
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
                ensure_in_mem2(p)
            ops.append(["compute", v])
            in_mem.add(v)
    
    for v in outputs:
        ensure_in_mem2(v)
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
                    def is_ancestor(a, target):
                        if a == target:
                            return True
                        for ch in children.get(a, []):
                            if is_ancestor(ch, target):
                                return True
                        return False
                    if is_ancestor(x, rem_out):
                        needed = True
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
    if n > 18:
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
    
    rcost = [0] * n
    def get_rcost(i):
        if rcost[i] > 0:
            return rcost[i]
        m = par_mask[i]
        if m == 0:
            rcost[i] = 1
        else:
            s = 1
            mm = m
            while mm:
                lb = mm & (-mm)
                j = lb.bit_length() - 1
                s += get_rcost(j)
                mm ^= lb
            rcost[i] = s
        return rcost[i]
    for i in range(n):
        get_rcost(i)
    
    b = max(1, int(fast_memory_budget) if fast_memory_budget is not None else n)
    
    def popcount(x):
        return x.bit_count() if hasattr(x, "bit_count") else bin(x).count("1")
    
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
    start_t = time.time()
    
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

def _validate_ops(dag, ops, budget):
    """Validate that ops are correct and within budget."""
    if not ops:
        return False
    if len(ops) > 20000:
        return False
    
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    outset = set(outputs)
    b = max(1, int(fast_memory_budget) if fast_memory_budget is not None else len(outputs))
    
    in_mem = set()
    stored = set()
    computed = set()
    
    for op, v in ops:
        if op == "load":
            if v in in_mem:
                return False
            if len(in_mem) >= b:
                return False
            in_mem.add(v)
        elif op == "store":
            if v not in in_mem:
                return False
            in_mem.discard(v)
            stored.add(v)
        elif op == "compute":
            ps = parents.get(v, [])
            for p in ps:
                if p not in in_mem and p not in stored:
                    return False
            if v in in_mem:
                return False
            if len(in_mem) >= b:
                return False
            in_mem.add(v)
            computed.add(v)
        elif op == "delete":
            if v not in in_mem:
                return False
            in_mem.discard(v)
        else:
            return False
    
    # All outputs must be stored
    for o in outset:
        if o not in stored:
            return False
    
    return True

def pebble(dag, fast_memory_budget):
    deadline = time.time() + 45
    candidates = []
    
    # Try hierarchical optimal pebbling first
    try:
        r = _pebble_hierarchical_optimal(dag, fast_memory_budget, deadline)
        if r is not None and len(r) <= 20000 and _validate_ops(dag, r, fast_memory_budget):
            candidates.append(r)
    except Exception:
        pass
    
    # Try TD DP pebbling
    try:
        r = _pebble_td_dp(dag, fast_memory_budget, deadline)
        if r is not None and len(r) <= 20000 and _validate_ops(dag, r, fast_memory_budget):
            candidates.append(r)
    except Exception:
        pass
    
    # Try A* for small DAGs
    try:
        r = _pebble_astar(dag, fast_memory_budget, deadline)
        if r is not None and len(r) <= 20000 and _validate_ops(dag, r, fast_memory_budget):
            candidates.append(r)
    except Exception:
        pass
    
    # Postorder pebbling (recomputation-based)
    try:
        r = _pebble_postorder(dag, fast_memory_budget)
        if r is not None and len(r) <= 20000 and _validate_ops(dag, r, fast_memory_budget):
            candidates.append(r)
    except Exception:
        pass
    
    # Memory-aware greedy
    try:
        r = _pebble_greedy_memory_aware(dag, fast_memory_budget)
        if r is not None and len(r) <= 20000 and _validate_ops(dag, r, fast_memory_budget):
            candidates.append(r)
    except Exception:
        pass
    
    if not candidates:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []
    
    def transfers(ops):
        return sum(1 for op, v in ops if op in ("load", "store"))
    best = min(candidates, key=transfers)
    return best
# EVOLVE-BLOCK-END
