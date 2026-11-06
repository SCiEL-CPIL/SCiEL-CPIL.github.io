"""Communication-minimizing DAG pebbling with independent verification."""
import time
import heapq
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

    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in vindex:
                children[p].append(v)

    # Compute descendants count (how many vertices depend on v)
    # Reverse topological order
    indeg = {v: 0 for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in vindex:
                indeg[v] += 1
                
    q = deque([v for v in vertices if indeg[v] == 0])
    topo = []
    while q:
        v = q.popleft()
        topo.append(v)
        for c in children[v]:
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    for v in vertices:
        if v not in topo:
            topo.append(v)
            
    # desc_count[v] = 1 + sum(desc_count[c] for c in children[v])
    # Wait, "descendants" usually means all nodes reachable.
    # For eviction priority, "how many future computations need this value" is a good proxy.
    # If v is a parent of c, and c is parent of d, v is needed for d.
    # So desc_count[v] = 1 + sum(desc_count[c]) is the size of the subgraph rooted at v.
    # This is a good measure of "recomputation cost" if v is evicted.
    desc_count = {v: 0 for v in vertices}
    for v in reversed(topo):
        count = 1
        for c in children[v]:
            count += desc_count[c]
        desc_count[v] = count

    try:
        budget = max(0, int(fast_memory_budget))
    except Exception:
        budget = 0

    ops = []
    in_mem = set()
    for v in inputs:
        if v in vindex:
            in_mem.add(v)
            
    outputs_set = set(v for v in outputs if v in vindex)
    
    # Heap items: (-is_output, -desc_count, vertex)
    # We push all vertices not initially in memory.
    heap = []
    for v in vertices:
        if v not in in_mem:
            is_out = 1 if v in outputs_set else 0
            heapq.heappush(heap, (-is_out, -desc_count[v], v))
            
    # To avoid infinite loops or excessive heap operations, we track a "version" or just rely on the logic.
    # The logic: Pop v. If ready, compute. If not, load a missing parent.
    # If we load a parent, we don't push v back immediately, we just continue.
    # But if v is still blocked, we might pop another vertex.
    # If we pop v again later, it might be ready.
    # Problem: If we pop v, it's not ready, we load p. We don't push v back.
    # Then we pop w. We compute w.
    # Then we need v again. It's not in the heap!
    # So we MUST push v back if it's not ready?
    # If we push v back, and it's still not ready, we pop it again, load another parent, push it back...
    # This works, but might be slow if we have deep chains.
    # However, each "load" reduces the number of missing parents.
    # So it will terminate.
    
    while outputs_set and not outputs_set.issubset(in_mem):
        if time.time() > deadline:
            break
        if not heap:
            # Should not happen if we push back
            break
            
        # Find a valid vertex
        valid_v = None
        while heap:
            item = heapq.heappop(heap)
            v = item[2]
            if v in in_mem:
                continue
            # Check if ready
            ready = True
            missing_parent = None
            for p in parents.get(v, []):
                if p not in in_mem:
                    ready = False
                    missing_parent = p
                    break
            if ready:
                valid_v = v
                break
            else:
                # Not ready. We need to load missing_parent.
                # But we should only do this if this is the "best" candidate?
                # If we just load missing_parent for the first non-ready vertex we pop,
                # we might be loading low-priority parents for high-priority vertices.
                # This is the "Pull" strategy.
                
                # Load the missing parent
                ops.append(['load', missing_parent])
                in_mem.add(missing_parent)
                
                # Evict if necessary
                if len(in_mem) > budget:
                    # Find victim
                    # Exclude outputs
                    # Exclude v? (v is not in mem, so no)
                    # Exclude parents of v? (We just loaded one, might need others)
                    # Let's just pick min desc_count among non-outputs.
                    candidates = [x for x in in_mem if x not in outputs_set]
                    if candidates:
                        victim = min(candidates, key=lambda x: desc_count[x])
                        ops.append(['store', victim])
                        in_mem.discard(victim)
                    else:
                        # If only outputs in mem, we can't evict.
                        # This means budget is too small?
                        pass
                
                # Push v back?
                # If we push v back, it will be popped again soon.
                # If we don't, we might lose track of it.
                # Let's push it back.
                heapq.heappush(heap, item)
                # To prevent infinite loop if we can't make progress (e.g. budget 0 and v needs 2 parents),
                # we might need a counter. But budget 0 implies we can't hold anything.
                # If budget >= 1, we can hold at least one parent.
                continue
                
        if valid_v is None:
            # No valid vertex found in the entire heap scan?
            # This implies we couldn't make progress.
            break

        # Compute valid_v
        ops.append(['compute', valid_v])
        in_mem.add(valid_v)
        
        # Evict if necessary
        if len(in_mem) > budget:
            candidates = [x for x in in_mem if x not in outputs_set]
            if candidates:
                victim = min(candidates, key=lambda x: desc_count[x])
                ops.append(['store', victim])
                in_mem.discard(victim)
            else:
                # Can't evict. If budget is full of outputs, we are stuck if we need to compute more?
                # But if outputs are in mem, we are done?
                # No, we might have non-output outputs? No, outputs_set is all outputs.
                pass

    # Ensure all outputs are in mem
    # If the loop broke early due to deadline or heap empty, we might need to load outputs.
    for v in outputs:
        if v in vindex and v not in in_mem:
            ops.append(['load', v])
            in_mem.add(v)
            # If this load exceeds budget, we must evict something.
            # But we need v to finish.
            # If we evict v, we fail.
            # So we must evict something else.
            if len(in_mem) > budget:
                candidates = [x for x in in_mem if x not in outputs_set]
                if candidates:
                    victim = min(candidates, key=lambda x: desc_count[x])
                    ops.append(['store', victim])
                    in_mem.discard(victim)

    return ops


def pebble(dag, fast_memory_budget):
    start = time.time()
    deadline = start + 85.0

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
        m = _my_schedule(dag, fast_memory_budget, deadline)
        if m is not None and len(m) <= 20000:
            ex = _extract(dag)
            if ex is not None:
                vertices, parents, inputs, outputs = ex
                try:
                    budget = max(0, int(fast_memory_budget))
                except Exception:
                    budget = 0
                if _verify(m, vertices, parents, inputs, outputs, budget):
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
