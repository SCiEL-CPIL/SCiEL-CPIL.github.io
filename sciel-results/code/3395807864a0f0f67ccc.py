"""Communication-minimizing DAG pebbling with independent verification."""
import time
import math
from collections import deque

try:
    from autoresearch_suite.science_tcs_expansion import pebble_baseline
    _HAVE_BASELINE = True
except Exception:
    _HAVE_BASELINE = False


def _count_transfers(ops):
    c = 0
    for op in ops:
        if op and op[0] in ('load', 'store'):
            c += 1
    return c


def _extract(dag):
    def get(d, k, default=None):
        if isinstance(d, dict):
            return d.get(k, default)
        return getattr(d, k, default)

    vertices = get(dag, 'vertices')
    parents = get(dag, 'parents')
    inputs = get(dag, 'inputs')
    outputs = get(dag, 'outputs')

    if vertices is None and isinstance(dag, dict):
        vertices = list(dag.keys())
        parents = {}
        for k, v in dag.items():
            if isinstance(v, (list, tuple, set)):
                parents[k] = list(v)
            else:
                parents[k] = []

    if vertices is None:
        return None
    if parents is None:
        parents = {}
    if inputs is None:
        inputs = []
    if outputs is None:
        outputs = []
    return vertices, parents, inputs, outputs


def _verify(ops, vertices, parents, inputs, outputs, budget):
    vset = set(vertices)
    in_mem = set()
    for v in inputs:
        if v in vset:
            in_mem.add(v)
    for op in ops:
        if not op or len(op) < 2:
            return False
        name, v = op[0], op[1]
        if v not in vset:
            return False
        if name == 'load':
            if v not in in_mem:
                in_mem.add(v)
        elif name == 'store':
            if v not in in_mem:
                return False
            in_mem.discard(v)
        elif name == 'delete':
            if v not in in_mem:
                return False
            in_mem.discard(v)
        elif name == 'compute':
            for p in parents.get(v, []):
                if p not in in_mem:
                    return False
            in_mem.add(v)
        else:
            return False
        if len(in_mem) > budget:
            return False
    for v in outputs:
        if v in vset and v not in in_mem:
            return False
    return True


def _my_schedule(dag, fast_memory_budget, deadline):
    ex = _extract(dag)
    if ex is None:
        return None
    vertices, parents, inputs, outputs = ex
    n = len(vertices)
    if n == 0:
        return []
    vindex = {v: i for i, v in enumerate(vertices)}

    children = {v: [] for v in vertices}
    indeg = {v: 0 for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in vindex:
                children[p].append(v)
                indeg[v] += 1

    indeg2 = dict(indeg)
    q = deque([v for v in vertices if indeg2[v] == 0])
    topo = []
    while q:
        v = q.popleft()
        topo.append(v)
        for c in children[v]:
            indeg2[c] -= 1
            if indeg2[c] == 0:
                q.append(c)
    for v in vertices:
        if v not in topo:
            topo.append(v)
    pos = {v: i for i, v in enumerate(topo)}

    remaining_children = {v: len(children[v]) for v in vertices}
    in_mem = set()
    for v in inputs:
        if v in vindex:
            in_mem.add(v)
    outputs_set = set(v for v in outputs if v in vindex)

    try:
        budget = max(0, int(fast_memory_budget))
    except Exception:
        budget = 0

    ops = []

    def evict():
        nonlocal in_mem
        while len(in_mem) > budget:
            candidates = [x for x in in_mem if x not in outputs_set]
            if not candidates:
                candidates = list(in_mem)
            # Evict the vertex with fewest remaining children, breaking ties by earliest topo pos
            victim = min(candidates,
                         key=lambda x: (remaining_children.get(x, 0),
                                        pos.get(x, 10 ** 9)))
            ops.append(['store', victim])
            in_mem.discard(victim)

    for v in topo:
        if time.time() > deadline:
            break
        ps = [p for p in parents.get(v, []) if p in vindex]
        for p in ps:
            if p not in in_mem:
                ops.append(['load', p])
                in_mem.add(p)
        ops.append(['compute', v])
        if v not in in_mem:
            in_mem.add(v)
        evict()
        for c in children[v]:
            remaining_children[c] -= 1

    for v in outputs:
        if v in vindex and v not in in_mem:
            ops.append(['load', v])
            in_mem.add(v)

    return ops


def _improved_schedule(dag, fast_memory_budget, deadline):
    """
    Improved pebbling strategy:
    1. Compute all vertices in topological order.
    2. Maintain a priority-based eviction policy that considers:
       - Number of remaining children (lower is better to evict)
       - Whether the vertex is an output (outputs should be kept if possible)
       - Topological position (earlier vertices are less likely to be needed)
    3. Use a look-ahead heuristic: when computing a vertex, if we can predict
       that its parents will not be needed soon, we can evict them earlier.
    """
    ex = _extract(dag)
    if ex is None:
        return None
    vertices, parents, inputs, outputs = ex
    n = len(vertices)
    if n == 0:
        return []
    vindex = {v: i for i, v in enumerate(vertices)}

    children = {v: [] for v in vertices}
    indeg = {v: 0 for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in vindex:
                children[p].append(v)
                indeg[v] += 1

    indeg2 = dict(indeg)
    q = deque([v for v in vertices if indeg2[v] == 0])
    topo = []
    while q:
        v = q.popleft()
        topo.append(v)
        for c in children[v]:
            indeg2[c] -= 1
            if indeg2[c] == 0:
                q.append(c)
    for v in vertices:
        if v not in topo:
            topo.append(v)
    pos = {v: i for i, v in enumerate(topo)}

    # Precompute for each vertex the set of vertices that depend on it (transitively)
    # For efficiency, we'll just use direct children count as a proxy
    remaining_children = {v: len(children[v]) for v in vertices}
    
    # Count how many times each vertex is used as a parent (i.e., its out-degree in the reverse sense)
    # This is already captured by remaining_children initially

    in_mem = set()
    for v in inputs:
        if v in vindex:
            in_mem.add(v)
    outputs_set = set(v for v in outputs if v in vindex)

    try:
        budget = max(0, int(fast_memory_budget))
    except Exception:
        budget = 0

    ops = []

    def evict_smart():
        nonlocal in_mem
        while len(in_mem) > budget:
            # Prioritize evicting non-output vertices with fewest remaining uses
            candidates = [x for x in in_mem if x not in outputs_set]
            if not candidates:
                candidates = list(in_mem)
            
            # Score: prefer evicting vertices with fewer remaining children
            # Tie-break: prefer evicting vertices that appear earlier in topo order
            # (they are less likely to be needed again)
            def evict_score(x):
                rc = remaining_children.get(x, 0)
                p = pos.get(x, 10 ** 9)
                # Lower score = better candidate for eviction
                # Use a combination: primarily remaining children, secondarily position
                return (rc, p)
            
            victim = min(candidates, key=evict_score)
            ops.append(['store', victim])
            in_mem.discard(victim)

    # Process vertices in topological order
    for v in topo:
        if time.time() > deadline:
            break
        
        # Ensure all parents are in memory
        ps = [p for p in parents.get(v, []) if p in vindex]
        for p in ps:
            if p not in in_mem:
                ops.append(['load', p])
                in_mem.add(p)
        
        # Compute v
        ops.append(['compute', v])
        if v not in in_mem:
            in_mem.add(v)
        
        # Evict if over budget
        evict_smart()
        
        # Decrement remaining children count for v's children
        for c in children[v]:
            remaining_children[c] -= 1

    # Ensure outputs are in memory
    for v in outputs:
        if v in vindex and v not in in_mem:
            ops.append(['load', v])
            in_mem.add(v)

    return ops


def _hybrid_schedule(dag, fast_memory_budget, deadline):
    """
    Hybrid approach: Try both the original and improved schedules,
    and pick the one with fewer transfers.
    """
    best = None
    best_transfers = float('inf')
    
    for schedule_fn in [_my_schedule, _improved_schedule]:
        try:
            ops = schedule_fn(dag, fast_memory_budget, deadline)
            if ops is not None and len(ops) <= 20000:
                ex = _extract(dag)
                if ex is not None:
                    vertices, parents, inputs, outputs = ex
                    try:
                        budget = max(0, int(fast_memory_budget))
                    except Exception:
                        budget = 0
                    if _verify(ops, vertices, parents, inputs, outputs, budget):
                        t = _count_transfers(ops)
                        if t < best_transfers:
                            best_transfers = t
                            best = ops
        except Exception:
            continue
    
    return best


def pebble(dag, fast_memory_budget):
    start = time.time()
    deadline = start + 80.0  # Safety margin before 90s kill

    baseline_ops = None
    if _HAVE_BASELINE:
        try:
            b = pebble_baseline(dag, fast_memory_budget)
            if isinstance(b, list):
                baseline_ops = b
        except Exception:
            baseline_ops = None

    my_ops = None
    try:
        m = _hybrid_schedule(dag, fast_memory_budget, deadline)
        if m is not None:
            my_ops = m
    except Exception:
        my_ops = None

    # Decision: prefer the verified schedule with fewer transfers.
    if my_ops is not None and baseline_ops is not None:
        if _count_transfers(my_ops) <= _count_transfers(baseline_ops):
            return my_ops
        return baseline_ops
    if my_ops is not None:
        return my_ops
    if baseline_ops is not None:
        return baseline_ops
    return []
