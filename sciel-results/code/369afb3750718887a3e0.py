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
            cost[v] = 1
        else:
            c = 1 + sum(get_cost(p) for p in ps)
            cost[v] = c
        return cost[v]
    for v in vertices:
        get_cost(v)
    return cost

def _hierarchical_pebble(dag, budget, deadline):
    """
    Hierarchical pebbling: partition DAG into balanced sub-DAGs (blocks),
    recursively pebble each block, treating inter-block dependencies as external I/O.
    For layered DAGs, this approximates optimal recursive pebbling.
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
    
    # Precompute usage counts: how many times each vertex is needed by future vertices
    usage_count = {v: 0 for v in vertices}
    for v in vertices:
        ps = parents.get(v, [])
        for p in ps:
            usage_count[p] += 1
    
    # Compute depth (longest path from any input to v) for block partitioning
    depth = {v: 0 for v in vertices}
    for v in topo:
        ps = parents.get(v, [])
        if ps:
            depth[v] = 1 + max(depth[p] for p in ps)
    
    max_depth = max(depth.values()) if depth else 0
    
    # Partition into blocks based on depth layers
    # For layered DAGs, each layer is a natural block
    if max_depth <= 20:
        # Fine-grained: one block per depth level
        blocks_by_depth = {}
        for v in topo:
            d = depth[v]
            if d not in blocks_by_depth:
                blocks_by_depth[d] = []
            blocks_by_depth[d].append(v)
        blocks = [blocks_by_depth[d] for d in sorted(blocks_by_depth.keys())]
    else:
        # Coarser partition: group into ~sqrt(n) blocks
        num_blocks = max(2, min(int(n ** 0.5), 10))
        block_size = (n + num_blocks - 1) // num_blocks
        blocks = [topo[i:i+block_size] for i in range(0, n, block_size)]
    
    ops = []
    in_mem = set()
    stored = set()
    
    def can_delete(v):
        """Check if vertex v can be safely deleted."""
        if v in outset or v in stored:
            return False
        for w in children.get(v, []):
            if w not in stored and w not in in_mem:
                return False
        return True
    
    def make_space(needed):
        """Free up memory slots by deleting vertices that can be deleted."""
        freed = 0
        for v in reversed(topo):
            if v in in_mem and can_delete(v):
                ops.append(["delete", v])
                in_mem.discard(v)
                freed += 1
                if freed >= needed:
                    break
    
    def pebble_block(block_vertices, external_deps):
        """
        Pebble a block of vertices. external_deps is the set of vertices
        outside this block that are needed as inputs.
        """
        for v in block_vertices:
            if time.time() > deadline:
                return
            
            ps = parents.get(v, [])
            
            # Ensure all parents are in memory
            for p in ps:
                if p not in in_mem:
                    if p in stored:
                        if len(in_mem) >= b:
                            make_space(1)
                        ops.append(["load", p])
                        in_mem.add(p)
                    else:
                        # p is not in memory and not stored - need to ensure it's available
                        # If p is an input (no parents), load it
                        pps = parents.get(p, [])
                        if not pps:
                            if len(in_mem) >= b:
                                make_space(1)
                            ops.append(["load", p])
                            in_mem.add(p)
                        else:
                            # Recursively ensure p is computed
                            # This is a fallback for cross-block dependencies
                            for pp in pps:
                                if pp not in in_mem:
                                    if pp in stored:
                                        if len(in_mem) >= b:
                                            make_space(1)
                                        ops.append(["load", pp])
                                        in_mem.add(pp)
                                    else:
                                        ppps = parents.get(pp, [])
                                        if not ppps:
                                            if len(in_mem) >= b:
                                                make_space(1)
                                            ops.append(["load", pp])
                                            in_mem.add(pp)
                                        else:
                                            # Deep recursion fallback - just load if possible
                                            if len(in_mem) >= b:
                                                make_space(1)
                                            ops.append(["load", pp])
                                            in_mem.add(pp)
                            ops.append(["compute", p])
                            in_mem.add(p)
            
            # Compute v
            ops.append(["compute", v])
            in_mem.add(v)
            
            # Decide whether to store v
            if v in outset:
                ops.append(["store", v])
                stored.add(v)
                in_mem.discard(v)
            else:
                future_needed = usage_count.get(v, 0) > 0
                if not future_needed:
                    ops.append(["delete", v])
                    in_mem.discard(v)
                else:
                    if len(in_mem) > b:
                        for x in reversed(list(in_mem)):
                            if x != v and x not in outset and can_delete(x):
                                ops.append(["delete", x])
                                in_mem.discard(x)
                                break
    
    # Process blocks in topological order
    for block in blocks:
        pebble_block(block, set())
    
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
    """
    A* search over memory bitmasks with improved heuristic.
    For small DAGs (n <= 16), finds near-optimal solutions.
    Uses recompute-cost-aware heuristic and memory pressure penalties.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    n = len(vertices)
    if n == 0:
        return []
    if n > 16:  # Limit to 16 vertices for performance
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
    
    # Compute recompute cost
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
    
    # Precompute ancestor masks
    ancestor_set = [0] * n
    for i in range(n):
        m = par_mask[i]
        if m == 0:
            ancestor_set[i] = (1 << i) if is_input[i] else 0
        else:
            ai = 0
            mm = m
            while mm:
                lb = mm & (-mm)
                j = lb.bit_length() - 1
                ai |= ancestor_set[j]
                mm ^= lb
            ancestor_set[i] = ai
    
    # Improved heuristic: estimate minimum transfers using recompute costs
    def heuristic(mem_mask, produced):
        missing = out_mask & ~produced
        if missing == 0:
            return 0
        
        h = 0
        mm = missing
        while mm:
            lb = mm & (-mm)
            i = lb.bit_length() - 1
            h += 1  # store
            # Estimate loads needed: count ancestors not in memory
            anc = ancestor_set[i] & ~mem_mask
            h += popcount(anc)  # loads for missing inputs
            mm ^= lb
        
        # Memory pressure penalty: if we're near capacity, we'll need to delete and reload
        if popcount(mem_mask) >= b - 1:
            h += 3  # higher penalty for being at capacity
        
        return h
    
    start_state = (0, 0)
    goal_produced = out_mask
    
    counter = 0
    open_heap = [(heuristic(0, 0), 0, counter, (0, 0))]
    g_score = {start_state: 0}
    parent_op = {}
    best = None
    max_expansions = 50000
    
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
            
            # Delete non-output
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

def _verify_ops(ops, dag, budget):
    """Verify that the operations sequence is valid."""
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
            # v must be an input or previously computed/stored
            if v not in computed and v not in stored:
                ps = parents.get(v, [])
                if ps:  # not an input
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
    
    # Check all outputs are stored
    for o in outset:
        if o not in stored:
            return False
    
    return True

def _local_search_polish(ops, dag, budget, deadline):
    """
    Local search to reduce transfers by reordering and removing redundant operations.
    """
    if not ops or len(ops) > 500:  # Only polish small sequences
        return ops
    
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    outset = set(outputs)
    b = max(1, int(budget) if budget is not None else 10**9)
    
    current_ops = list(ops)
    best_ops = list(current_ops)
    best_transfers = sum(1 for op, v in current_ops if op in ("load", "store"))
    
    # Try local moves: swap adjacent operations, remove redundant deletes
    for iteration in range(100):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try swapping adjacent operations
        for i in range(len(current_ops) - 1):
            new_ops = list(current_ops)
            new_ops[i], new_ops[i+1] = new_ops[i+1], new_ops[i]
            
            if _verify_ops(new_ops, dag, budget):
                t = sum(1 for op, v in new_ops if op in ("load", "store"))
                if t < best_transfers:
                    best_ops = new_ops
                    best_transfers = t
                    current_ops = new_ops
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_ops

def pebble(dag, fast_memory_budget):
    deadline = time.time() + 45
    candidates = []
    
    # Try A* first for small DAGs (can find optimal)
    try:
        r = _pebble_astar(dag, fast_memory_budget, deadline)
        if r is not None and len(r) <= 20000:
            if _verify_ops(r, dag, fast_memory_budget):
                candidates.append(r)
    except Exception:
        pass
    
    # Try hierarchical pebbling for larger DAGs
    try:
        r = _hierarchical_pebble(dag, fast_memory_budget, deadline)
        if r is not None and len(r) <= 20000:
            if _verify_ops(r, dag, fast_memory_budget):
                candidates.append(r)
    except Exception:
        pass
    
    # Postorder pebbling (recomputation-based)
    try:
        r = _pebble_postorder(dag, fast_memory_budget)
        if r is not None and len(r) <= 20000:
            if _verify_ops(r, dag, fast_memory_budget):
                candidates.append(r)
    except Exception:
        pass
    
    # Memory-aware greedy
    try:
        r = _pebble_greedy_memory_aware(dag, fast_memory_budget)
        if r is not None and len(r) <= 20000:
            if _verify_ops(r, dag, fast_memory_budget):
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
    
    # Sort candidates by transfer count
    candidates.sort(key=transfers)
    best = candidates[0]
    
    # Try local search polish on the best candidate
    if len(best) <= 500 and time.time() < deadline - 5:
        try:
            polished = _local_search_polish(best, dag, fast_memory_budget, deadline)
            if polished and _verify_ops(polished, dag, fast_memory_budget):
                if transfers(polished) < transfers(best):
                    best = polished
        except Exception:
            pass
    
    return best
# EVOLVE-BLOCK-END
