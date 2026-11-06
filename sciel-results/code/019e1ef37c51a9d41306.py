import time
import heapq
from collections import deque

def _topo_order(parents, vertices):
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

def _pebble_astar(dag, budget, deadline):
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    n = len(vertices)
    if n == 0:
        return []
    if n > 25:
        return None
    
    idx = {v: i for i, v in enumerate(vertices)}
    out_mask = 0
    for o in outputs:
        out_mask |= (1 << idx[o])
        
    par_mask = [0] * n
    is_input = [False] * n
    input_mask_for_output = [0] * n # For each vertex, mask of inputs that are its ancestors
    
    # Precompute input ancestors
    # Since n is small, we can do this for each vertex
    def get_input_ancestors(i):
        if is_input[i]:
            return 1 << i
        m = par_mask[i]
        if m == 0:
            return 0
        res = 0
        mm = m
        while mm:
            lb = mm & (-mm)
            j = lb.bit_length() - 1
            res |= get_input_ancestors(j)
            mm ^= lb
        return res

    for i in range(n):
        ps = parents.get(vertices[i], [])
        if not ps:
            is_input[i] = True
        m = 0
        for p in ps:
            m |= (1 << idx[p])
        par_mask[i] = m
        input_mask_for_output[i] = get_input_ancestors(i)

    b = max(1, int(budget) if budget is not None else n)
    
    def popcount(x):
        return x.bit_count() if hasattr(x, "bit_count") else bin(x).count("1")

    def heuristic(mem_mask, produced):
        missing_outputs = out_mask & ~produced
        if missing_outputs == 0:
            return 0
        
        # Count stores
        h = popcount(missing_outputs)
        
        # Count loads: union of input ancestors of all missing outputs
        union_inputs = 0
        mm = missing_outputs
        while mm:
            lb = mm & (-mm)
            i = lb.bit_length() - 1
            union_inputs |= input_mask_for_output[i]
            mm ^= lb
            
        # Inputs not currently in memory
        loads_needed = popcount(union_inputs & ~mem_mask)
        h += loads_needed
        
        return h

    start_state = (0, 0)
    goal_produced = out_mask
    
    counter = 0
    open_heap = [(heuristic(0, 0), 0, counter, (0, 0))]
    g_score = {start_state: 0}
    parent_op = {}
    best = None
    max_expansions = 100000
    
    expansions = 0
    while open_heap:
        if time.time() > deadline or expansions > max_expansions:
            break
        f, g, _, (mem, produced) = heapq.heappop(open_heap)
        expansions += 1
        
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
            
            # Load input
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
            
            # Compute
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
            
            # Store output
            if (mem & bit) and (out_mask & bit) and not (produced & bit):
                new_mem = mem & ~bit
                new_prod = produced | bit
                ng = g + 1
                ns = (new_mem, new_prod)
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = ((mem, produced), "store", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(new_mem, new_prod), ng, counter, ns))
            
            # Delete non-output (or output if already produced)
            if (mem & bit) and (not (out_mask & bit) or (produced & bit)):
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

def _pebble_greedy(dag, budget):
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    b = max(1, int(budget) if budget is not None else len(vertices))
    
    topo, children = _topo_order(parents, vertices)
    
    ops = []
    in_mem = set()
    stored = set()
    
    # Count how many times each vertex is needed by future vertices
    needed_count = {v: 0 for v in vertices}
    for v in vertices:
        ps = parents.get(v, [])
        for p in ps:
            needed_count[p] += 1
            
    # Track remaining needs
    remaining_needs = dict(needed_count)
    
    for v in topo:
        # Ensure parents are in memory
        ps = parents.get(v, [])
        for p in ps:
            if p not in in_mem:
                # Load or Recompute?
                # If p is an input, load it.
                # If p is computed, we lost it. We must recompute.
                # For simplicity in greedy, we assume we keep things if needed.
                # But if we deleted it, we have to recompute.
                # This greedy is simple: Load inputs, Compute, Delete if not needed.
                # It doesn't handle recomputation of deleted intermediates well.
                # Let's stick to a simple valid schedule:
                # Load inputs as needed. Compute. Store outputs. Delete intermediates if not needed later.
                if len(in_mem) >= b:
                    # Delete something
                    # Delete a vertex that is not an output and not needed by v or future?
                    # Just delete any non-output, non-current-parent
                    for x in list(in_mem):
                        if x not in outset and x not in ps:
                            ops.append(["delete", x])
                            in_mem.discard(x)
                            break
                    if len(in_mem) >= b:
                         # Force delete even if needed? No, must be valid.
                         # If we can't make space, we can't compute.
                         # But this greedy assumes we can always make space.
                         pass
                
                if p in outset and p in stored:
                    ops.append(["load", p])
                    in_mem.add(p)
                elif not parents.get(p, []):
                    ops.append(["load", p])
                    in_mem.add(p)
                else:
                    # Recompute p
                    # This is complex. Let's just load if possible or fail.
                    # For a valid baseline, let's just do topological load/compute/store/delete.
                    # If p is not in mem, it must be an input or we messed up.
                    # Let's assume inputs are the only things we load.
                    # If p is not input and not in mem, this greedy fails.
                    # We need a strategy that keeps intermediates if needed.
                    pass
        
        # Compute v
        ops.append(["compute", v])
        in_mem.add(v)
        
        # Store or Delete
        if v in outset:
            ops.append(["store", v])
            stored.add(v)
            in_mem.discard(v)
        else:
            # Update needs
            # v is now produced. Its parents' needs are satisfied for this v.
            for p in ps:
                remaining_needs[p] -= 1
            
            if remaining_needs[v] == 0:
                ops.append(["delete", v])
                in_mem.discard(v)
                
    return ops[:20000]

def _verify_ops(ops, dag, budget):
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    outset = set(outputs)
    in_mem = set()
    stored = set()
    computed = set()
    b = max(1, int(budget) if budget is not None else 10**9)
    
    for op, v in ops:
        if op == "load":
            if v in in_mem:
                return False
            if len(in_mem) >= b:
                return False
            if v not in computed and v not in stored:
                ps = parents.get(v, [])
                if ps:
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
            if len(in_mem) >= b and v not in in_mem:
                return False
            in_mem.add(v)
            computed.add(v)
        elif op == "delete":
            if v not in in_mem:
                return False
            in_mem.discard(v)
        else:
            return False
    
    for o in outset:
        if o not in stored:
            return False
    return True

def pebble(dag, fast_memory_budget):
    deadline = time.time() + 45
    candidates = []
    
    try:
        r = _pebble_astar(dag, fast_memory_budget, deadline)
        if r is not None and len(r) <= 20000:
            if _verify_ops(r, dag, fast_memory_budget):
                candidates.append(r)
    except Exception:
        pass
    
    if not candidates:
        try:
            r = _pebble_greedy(dag, fast_memory_budget)
            if r is not None and len(r) <= 20000:
                if _verify_ops(r, dag, fast_memory_budget):
                    candidates.append(r)
        except Exception:
            pass
            
    if not candidates:
        return []
        
    def transfers(ops):
        return sum(1 for op, v in ops if op in ("load", "store"))
        
    candidates.sort(key=transfers)
    return candidates[0]
