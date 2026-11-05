"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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

def _compute_recompute_costs(parents, vertices):
    """Compute the minimum number of compute operations needed to produce each vertex from inputs."""
    cost = {}
    def get_cost(v):
        if v in cost:
            return cost[v]
        ps = parents.get(v, [])
        if not ps:
            cost[v] = 1  # just compute the input itself (or it's free?)
        else:
            c = 1 + sum(get_cost(p) for p in ps)
            cost[v] = c
        return cost[v]
    for v in vertices:
        get_cost(v)
    return cost

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
    
    # Compute recompute costs for pruning
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
        # Delete parents that are no longer needed
        for p in ps:
            if p in outset or p in stored:
                continue
            # Check if p is needed for any other child not yet computed/stored
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
    
    # Compute last-use for each vertex
    last_use = {v: -1 for v in vertices}
    for i, v in enumerate(topo):
        for p in parents.get(v, []):
            last_use[p] = i
    
    ops = []
    in_mem = set()
    stored = set()
    
    def ensure_in_mem(v):
        """Ensure v is in memory, loading/computing as needed."""
        if v in in_mem:
            return
        ps = parents.get(v, [])
        if not ps:
            # Input: load it
            if len(in_mem) >= b:
                # Need to free space
                for x in list(in_mem):
                    if x not in outset and x not in stored:
                        ops.append(["delete", x])
                        in_mem.discard(x)
                        if len(in_mem) < b:
                            break
            ops.append(["load", v])
            in_mem.add(v)
        else:
            # Compute: ensure all parents in memory
            for p in ps:
                ensure_in_mem(p)
            ops.append(["compute", v])
            in_mem.add(v)
    
    for v in topo:
        if v in outset:
            ensure_in_mem(v)
            ops.append(["store", v])
            stored.add(v)
            in_mem.discard(v)
        else:
            # Keep in memory if budget allows, else compute on demand
            pass
    
    # For non-output vertices, we compute them when needed by their children
    # This is handled by ensure_in_mem during output pebbling
    
    # Actually, let's re-do this: process outputs one by one
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
        # Clean up: delete vertices no longer needed
        to_delete = []
        for x in in_mem:
            if x not in outset:
                # Check if x is needed for any remaining output
                needed = False
                for rem_out in outputs:
                    if rem_out in stored:
                        continue
                    # Check if x is an ancestor of rem_out
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
    if n > 18:  # Limit for state space explosion
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
    
    # Recompute costs: min ops to produce v from nothing
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
    
    # Heuristic: for each output not in memory, we need at least 1 store + 
    # loads for inputs not in memory that are ancestors. Lower bound: count of missing outputs.
    # Better: sum over missing outputs of (1 + number of input ancestors not in memory)
    input_mask = 0
    for i in range(n):
        if is_input[i]:
            input_mask |= (1 << i)
    
    # Precompute for each vertex, which inputs are its ancestors
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
            # Need to store this output: 1 op
            h += 1
            # Need to have all its input ancestors in memory: count those not in mem
            anc = ancestor_inputs[i] & ~mem_mask
            h += popcount(anc)
            mm ^= lb
        return h
    
    start = 0
    goal = out_mask  # All outputs in memory (we'll store them after)
    # Actually, we need to store outputs, which removes them. So goal is: we can store all outputs.
    # Let's use a different approach: track stored outputs separately.
    # State: (mem_mask, stored_mask) where stored_mask tracks which outputs have been stored.
    # But that doubles state space. Alternative: since we can always recompute, 
    # the optimal is to compute each output exactly once. So we can just find the min-cost 
    # sequence that produces each output at least once.
    
    # Simpler: use Dijkstra where cost = number of transfers (load+store), 
    # and we track which outputs have been produced.
    # State: (mem_mask, produced_mask)
    # Goal: produced_mask == out_mask
    
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
            # Reconstruct
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
        
        # Generate successors
        for i in range(n):
            bit = 1 << i
            v = vertices[i]
            
            # Load: only for inputs not in memory
            if is_input[i] and not (mem & bit):
                if popcount(mem) < b:
                    new_mem = mem | bit
                    new_prod = produced
                    ng = g + 1  # load is a transfer
                    ns = (new_mem, new_prod)
                    if ng < g_score.get(ns, float("inf")):
                        g_score[ns] = ng
                        parent_op[ns] = ((mem, produced), "load", v)
                        counter += 1
                        heapq.heappush(open_heap, (ng + heuristic(new_mem), ng, counter, ns))
            
            # Compute: only if all parents in memory and v not in memory
            if (mem & par_mask[i]) == par_mask[i] and not (mem & bit):
                new_mem = mem | bit
                new_prod = produced | bit if (out_mask & bit) else produced
                ng = g  # compute is not a transfer
                ns = (new_mem, new_prod)
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = ((mem, produced), "compute", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(new_mem), ng, counter, ns))
            
            # Store: only for outputs in memory
            if (mem & bit) and (out_mask & bit):
                new_mem = mem & ~bit
                new_prod = produced | bit
                ng = g + 1  # store is a transfer
                ns = (new_mem, new_prod)
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = ((mem, produced), "store", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(new_mem), ng, counter, ns))
            
            # Delete: only for non-outputs in memory
            if (mem & bit) and not (out_mask & bit):
                new_mem = mem & ~bit
                new_prod = produced
                ng = g  # delete is not a transfer
                ns = (new_mem, new_prod)
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = ((mem, produced), "delete", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(new_mem), ng, counter, ns))
    
    return best

def pebble(dag, fast_memory_budget):
    deadline = time.time() + 45  # Safety margin before 60s/90s kill
    candidates = []
    
    # Try A* for small DAGs
    try:
        r = _pebble_astar(dag, fast_memory_budget, deadline)
        if r is not None and len(r) <= 20000:
            candidates.append(r)
    except Exception:
        pass
    
    # Postorder pebbling (recomputation-based)
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
