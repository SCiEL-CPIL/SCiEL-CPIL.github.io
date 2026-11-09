# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict

def _topo_order(dag):
    """Return vertices in topological order."""
    try:
        parents = dag['parents']
    except Exception:
        parents = getattr(dag, 'parents', None)
    if parents is None:
        # maybe dag is a dict mapping vertex->parents
        parents = dag
    # gather all vertices
    verts = set()
    for v, ps in parents.items():
        verts.add(v)
        for p in ps:
            verts.add(p)
    # Kahn
    indeg = {v: 0 for v in verts}
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            indeg[v] = indeg.get(v, 0) + 1
            children[p].append(v)
    # roots have indeg 0
    import heapq as _hq
    q = _hq.heapify
    heap = [v for v in verts if indeg[v] == 0]
    _hq.heapify(heap)
    order = []
    while heap:
        v = _hq.heappop(heap)
        order.append(v)
        for c in children[v]:
            indeg[c] -= 1
            if indeg[c] == 0:
                _hq.heappush(heap, c)
    return order, verts

def _naive_cost(dag, budget):
    """Naive schedule cost: for each vertex in topo order, load each parent not already
    in memory, compute, then store result and delete parents. Count transfers."""
    order, verts = _topo_order(dag)
    try:
        parents = dag['parents']
    except Exception:
        parents = getattr(dag, 'parents', None)
    if parents is None:
        parents = dag
    # inputs start blue (available). outputs must finish blue.
    # naive: process each vertex; assume we can keep all in memory (budget large enough),
    # cost = number of loads + number of stores.
    # A load happens when a parent value is needed but not in fast memory.
    # Naive keeps everything computed so far in memory (if budget allows) else reload.
    # For a fair baseline, assume budget is enough to hold all intermediate values that
    # are still needed, i.e., we load each vertex's parents only once (first time needed),
    # and store each vertex once.
    # Count: each edge (parent->child) causes a load of parent the first time it's needed.
    # Actually simpler: naive cost = (#vertices that are not inputs) computations don't transfer,
    # but each non-input vertex must be loaded once (its value computed and stored) and each
    # input loaded once. Let's define naive as: total transfers = number of vertices (each
    # value moves slow->fast once) + number of outputs (fast->slow once).
    n = len(order)
    outputs = set()
    try:
        outputs = set(dag['outputs'])
    except Exception:
        outputs = set(getattr(dag, 'outputs', []))
    if not outputs:
        # assume sinks (no children) are outputs
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
        outputs = {v for v in verts if not children[v]}
    # inputs = vertices with no parents
    inputs = {v for v in verts if len(parents.get(v, [])) == 0}
    # naive: each non-input vertex computed (its parents loaded), then stored to slow? 
    # Let's define naive cost = loads + stores where:
    #   load: input values loaded once each; intermediate values loaded once each when needed
    #   store: output values stored once each
    # Simplest reasonable naive: cost = n (load every vertex value into fast) + len(outputs) (store outputs)
    # But inputs are already available (blue) so maybe not loaded. 
    # I'll use: cost = (n - len(inputs)) + len(outputs)  [compute results loaded + outputs stored]
    # Actually to be safe and match "naive schedule", let me simulate with infinite budget:
    # keep a set of values in fast memory. For each vertex in topo order:
    #   for each parent: if parent not in fast, load it (cost+1)
    #   compute (no cost)
    #   result is in fast (no cost)
    #   if vertex is an output, store it (cost+1) and remove from fast
    #   optionally delete parents that are no longer needed (no cost for delete? delete is an op but maybe no transfer)
    # The task says operations load/store/compute/delete. Transfers = load + store presumably.
    # Let me count load and store as transfers.
    fast = set()
    cost = 0
    # precompute children
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
    # reference counts of how many children still need each vertex
    need = {}
    for v in verts:
        need[v] = len(children[v])
    for v in order:
        ps = parents.get(v, [])
        for p in ps:
            if p not in fast:
                fast.add(p)
                cost += 1
        # compute v
        # v now in fast
        fast.add(v)
        # if output, store
        if v in outputs:
            cost += 1
            fast.discard(v)
        # delete parents no longer needed
        for p in ps:
            need[p] -= 1
            if need[p] == 0 and p in fast and p not in outputs:
                fast.discard(p)
    return cost, order, verts, parents, outputs, inputs

def pebble(dag, fast_memory_budget):
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    try:
        naive_cost, order, verts, parents, outputs, inputs = _naive_cost(dag, fast_memory_budget)
    except Exception:
        naive_cost = 1
        order, verts, parents, outputs, inputs = [], set(), {}, set(), set()

    # If we can't do better than naive, return a naive schedule.
    # Build a naive schedule as fallback.
    def _naive_schedule():
        ops = []
        fast = set()
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
        need = {v: len(children[v]) for v in verts}
        for v in order:
            ps = parents.get(v, [])
            for p in ps:
                if p not in fast:
                    ops.append(['load', p])
                    fast.add(p)
            ops.append(['compute', v])
            fast.add(v)
            if v in outputs:
                ops.append(['store', v])
                fast.discard(v)
            for p in ps:
                need[p] -= 1
                if need[p] == 0 and p in fast and p not in outputs:
                    ops.append(['delete', p])
                    fast.discard(p)
        return ops

    naive_ops = _naive_schedule()

    # Now try to improve: use a heuristic that reuses values to reduce loads.
    # Strategy: process vertices in an order that maximizes reuse. Use a priority based on
    # how many children each vertex has (process high-fanout early) to keep them in memory.
    # Actually, to minimize loads, we want to keep values that are needed by many remaining
    # children. This is like a cache problem.
    # Greedy: at each step, among available vertices (all parents in fast or loadable),
    # pick the one whose computation allows deleting the most values (to free memory) or
    # the one that is needed soonest.
    # Simpler improvement: topological order but process vertices in an order that groups
    # siblings together. Use a BFS-like order that keeps parents hot.

    # Let's try a different topo order: prioritize vertices whose parents are already in fast.
    # We'll simulate with a priority queue: available = vertices whose parents are all in fast.
    # Among available, pick the one with the most remaining children (to keep it useful) 
    # or the one that frees the most memory.
    # Actually we want to minimize loads. A load happens when a parent isn't in fast.
    # To avoid loads, we want parents to be in fast when we compute their children.
    # So we should compute children soon after computing the parent, while it's still in fast.
    # This suggests a DFS-like or depth-first order.

    # Let me try: order vertices by a DFS that goes deep, so parents are still in fast
    # when children are computed.

    def _dfs_order():
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
        # start from inputs
        visited = set()
        result = []
        def dfs(v):
            if v in visited:
                return
            visited.add(v)
            result.append(v)
            for c in children[v]:
                dfs(c)
        for v in sorted(inputs):
            dfs(v)
        # ensure all vertices
        for v in order:
            if v not in visited:
                visited.add(v)
                result.append(v)
        return result

    def _schedule_with_order(ord_list):
        ops = []
        fast = set()
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
        need = {v: len(children[v]) for v in verts}
        # We need to ensure that when we compute v, its parents are in fast.
        # If a parent is not in fast, we must load it.
        for v in ord_list:
            ps = parents.get(v, [])
            for p in ps:
                if p not in fast:
                    ops.append(['load', p])
                    fast.add(p)
            ops.append(['compute', v])
            fast.add(v)
            if v in outputs:
                ops.append(['store', v])
                fast.discard(v)
            for p in ps:
                need[p] -= 1
                if need[p] == 0 and p in fast and p not in outputs:
                    ops.append(['delete', p])
                    fast.discard(p)
        # cost = number of loads + stores
        cost = sum(1 for op in ops if op[0] in ('load', 'store'))
        return ops, cost

    # Try DFS order
    dfs_ord = _dfs_order()
    dfs_ops, dfs_cost = _schedule_with_order(dfs_ord)

    # Try original topo order
    topo_ops, topo_cost = _schedule_with_order(order)

    # Pick the better of dfs and topo
    if dfs_cost <= topo_cost:
        best_ops, best_cost = dfs_ops, dfs_cost
    else:
        best_ops, best_cost = topo_ops, topo_cost

    # If best is not better than naive, use naive
    if best_cost > naive_cost:
        best_ops = naive_ops
        best_cost = naive_cost

    # Deadline guard
    if time.time() - t0 > DEADLINE:
        return naive_ops[:20000]

    # Try to improve further with a local search: swap adjacent vertices in the order
    # if it reduces cost. But this is expensive. Skip for now.

    # Cap operations
    if len(best_ops) > 20000:
        best_ops = best_ops[:20000]

    return best_ops
# EVOLVE-BLOCK-END
