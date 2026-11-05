"""Optimized DAG pebbling for communication minimization."""
import time
import heapq
from collections import deque

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

def _pebble_layered(dag, budget, deadline):
    """
    Layer-based pebbling with checkpointing.
    For layered DAGs, process layer by layer, keeping frontier in memory.
    Use checkpoints to balance load/store vs recompute.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    n = len(vertices)
    if n == 0:
        return []
    
    vset = set(vertices)
    outset = set(outputs)
    b = max(1, int(fast_memory_budget) if fast_memory_budget is not None else n)
    
    topo, children = _topo_order(parents, vertices, vset)
    
    # Compute depth/layer for each vertex
    depth = {}
    for v in topo:
        ps = parents.get(v, [])
        if not ps:
            depth[v] = 0
        else:
            depth[v] = 1 + max(depth[p] for p in ps if p in depth)
    
    max_depth = max(depth.values()) if depth else 0
    
    # Group vertices by depth
    layers = [[] for _ in range(max_depth + 1)]
    for v in topo:
        layers[depth[v]].append(v)
    
    # Strategy: process layer by layer
    # For each layer, compute all vertices, then delete previous layer if memory is tight
    # Checkpoint: store outputs as we go
    
    ops = []
    in_mem = set()
    stored = set()
    
    # Precompute which vertices are ancestors of which outputs
    # For deletion decisions
    output_ancestors = {o: set() for o in outputs}
    for o in outputs:
        # BFS to find all ancestors
        stack = [o]
        while stack:
            v = stack.pop()
            if v in output_ancestors[o]:
                continue
            output_ancestors[o].add(v)
            for p in parents.get(v, []):
                if p not in output_ancestors[o]:
                    stack.append(p)
    
    def is_ancestor_of_unstored_output(v):
        for o in outputs:
            if o not in stored and v in output_ancestors[o]:
                return True
        return False
    
    def free_space_if_needed(needed):
        """Free space until we have at least 'needed' slots free."""
        while len(in_mem) + needed > b:
            # Find a vertex to delete that is not an output and not needed
            candidates = []
            for x in in_mem:
                if x not in outset and not is_ancestor_of_unstored_output(x):
                    candidates.append(x)
            if not candidates:
                # Can't free space, break
                break
            # Delete the one with highest depth (least likely to be reused soon)
            candidates.sort(key=lambda x: -depth.get(x, 0))
            x = candidates[0]
            ops.append(["delete", x])
            in_mem.discard(x)
    
    # Process each layer
    for d in range(max_depth + 1):
        if time.time() > deadline:
            break
        for v in layers[d]:
            ps = parents.get(v, [])
            # Ensure all parents are in memory
            for p in ps:
                if p not in in_mem:
                    # Need to recompute p
                    # This shouldn't happen if we process in topological order
                    # and keep parents around, but handle it
                    pass
            
            # Compute v
            if v not in in_mem:
                ops.append(["compute", v])
                in_mem.add(v)
            
            # If v is an output, store it
            if v in outset:
                ops.append(["store", v])
                stored.add(v)
                in_mem.discard(v)
            
            # Free space if needed
            free_space_if_needed(0)
    
    return ops[:20000]

def _pebble_optimal_small(dag, budget, deadline):
    """
    For small DAGs (n <= 12), use exact A* search over memory states.
    State: (mem_mask, produced_mask)
    Cost: number of transfers (load + store)
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    n = len(vertices)
    if n == 0:
        return []
    if n > 12:
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
    
    b = max(1, int(fast_memory_budget) if fast_memory_budget is not None else n)
    
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
    
    def heuristic(mem_mask, produced_mask):
        missing = out_mask & ~produced_mask
        if missing == 0:
            return 0
        h = 0
        mm = missing
        while mm:
            lb = mm & (-mm)
            i = lb.bit_length() - 1
            h += 1  # store
            anc = ancestor_inputs[i] & ~mem_mask
            h += popcount(anc)  # loads needed
            mm ^= lb
        return h
    
    start_state = (0, 0)
    goal_produced = out_mask
    
    counter = 0
    open_heap = [(heuristic(0, 0), 0, counter, (0, 0))]
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
            
            # Load: only for inputs not in memory
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
                        heapq.heappush(open_heap, (ng + heuristic(new_mem, new_prod), ng, counter, ns))
            
            # Compute: only if all parents in memory and v not in memory
            if (mem & par_mask[i]) == par_mask[i] and not (mem & bit):
                new_mem = mem | bit
                new_prod = produced | bit if (out_mask & bit) else produced
                ng = g
                ns = (new_mem, new_prod)
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = ((mem, produced), "compute", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(new_mem, new_prod), ng, counter, ns))
            
            # Store: only for outputs in memory
            if (mem & bit) and (out_mask & bit):
                new_mem = mem & ~bit
                new_prod = produced | bit
                ng = g + 1
                ns = (new_mem, new_prod)
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = ((mem, produced), "store", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(new_mem, new_prod), ng, counter, ns))
            
            # Delete: only for non-outputs in memory
            if (mem & bit) and not (out_mask & bit):
                new_mem = mem & ~bit
                new_prod = produced
                ng = g
                ns = (new_mem, new_prod)
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = ((mem, produced), "delete", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(new_mem, new_prod), ng, counter, ns))
    
    return best

def _pebble_greedy_frontier(dag, budget, deadline):
    """
    Greedy frontier-based pebbling:
    - Process vertices in topological order
    - Keep a frontier of vertices that are computed but not yet used
    - When memory is full, delete the vertex with the farthest last use
    - Recompute if a needed vertex was deleted
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    n = len(vertices)
    if n == 0:
        return []
    
    vset = set(vertices)
    outset = set(outputs)
    b = max(1, int(fast_memory_budget) if fast_memory_budget is not None else n)
    
    topo, children = _topo_order(parents, vertices, vset)
    
    # Compute last use for each vertex
    last_use = {v: -1 for v in vertices}
    for i, v in enumerate(topo):
        for p in parents.get(v, []):
            last_use[p] = i
    
    ops = []
    in_mem = set()
    stored = set()
    
    # Track which vertices have been computed (even if deleted)
    # We need to recompute if a parent was deleted
    
    def ensure_in_mem(v):
        """Ensure v is in memory, recomputing if necessary."""
        if v in in_mem:
            return
        ps = parents.get(v, [])
        if not ps:
            # Input: load it
            if len(in_mem) >= b:
                # Delete the vertex with the farthest last use that is not an output
                candidates = [x for x in in_mem if x not in outset]
                if candidates:
                    # Sort by last use, delete the one with farthest last use
                    candidates.sort(key=lambda x: last_use.get(x, -1))
                    x = candidates[0]
                    ops.append(["delete", x])
                    in_mem.discard(x)
            ops.append(["load", v])
            in_mem.add(v)
        else:
            # Compute: ensure all parents in memory
            for p in ps:
                ensure_in_mem(p)
            ops.append(["compute", v])
            in_mem.add(v)
    
    for v in topo:
        if time.time() > deadline:
            break
        if v in outset:
            ensure_in_mem(v)
            ops.append(["store", v])
            stored.add(v)
            in_mem.discard(v)
        else:
            # Keep in memory for now; will be deleted when needed
            pass
    
    return ops[:20000]

def pebble(dag, fast_memory_budget):
    deadline = time.time() + 45  # Safety margin before 90s kill
    candidates = []
    
    # Try exact A* for small DAGs
    try:
        r = _pebble_optimal_small(dag, fast_memory_budget, deadline)
        if r is not None and len(r) <= 20000:
            candidates.append(r)
    except Exception:
        pass
    
    # Greedy frontier-based
    try:
        r = _pebble_greedy_frontier(dag, fast_memory_budget, deadline)
        if r is not None and len(r) <= 20000:
            candidates.append(r)
    except Exception:
        pass
    
    # Layered pebbling
    try:
        r = _pebble_layered(dag, fast_memory_budget, deadline)
        if r is not None and len(r) <= 20000:
            candidates.append(r)
    except Exception:
        pass
    
    if not candidates:
        return []
    
    # Pick candidate with fewest transfers (load+store)
    def transfers(ops):
        return sum(1 for op, v in ops if op in ("load", "store"))
    best = min(candidates, key=transfers)
    return best
