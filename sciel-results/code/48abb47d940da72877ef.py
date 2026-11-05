"""Optimized DAG pebbling: memory-aware greedy with recompute costs."""
import time
from collections import deque

def _topo_order(parents, vertices):
    """Return topological order and children map."""
    vset = set(vertices)
    indeg = {v: len([p for p in parents.get(v, []) if p in vset]) for v in vertices}
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            if p in vset:
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
        # Fallback: just use given order
        topo = list(vertices)
    return topo, children

def _compute_recompute_costs(parents, vertices):
    """Compute minimum compute operations to produce each vertex from inputs.
    
    For a vertex v with parents ps:
    - If v is an input (no parents): cost = 1 (just load)
    - Otherwise: cost = 1 (compute v) + sum of costs of parents
    """
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

def _pebble_optimal_greedy(dag, budget):
    """
    Greedy pebbling that minimizes transfers by:
    1. Processing outputs in order of highest recompute cost first (most expensive to recompute)
    2. Keeping in memory vertices that are needed by multiple remaining outputs
    3. Deleting vertices only when no longer needed
    
    Key insight: minimize load+store operations (transfers), not total operations.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices or not outputs:
        return []
    
    vset = set(vertices)
    outset = set(outputs)
    b = max(1, int(budget) if budget is not None else len(vertices))
    
    topo, children = _topo_order(parents, vertices)
    rcost = _compute_recompute_costs(parents, vertices)
    
    # Precompute: for each vertex, which outputs it is an ancestor of
    # and how many times it's needed
    output_indices = {o: i for i, o in enumerate(outputs)}
    
    # For each vertex, compute set of outputs it feeds into
    feeds_outputs = {v: set() for v in vertices}
    def compute_feeds(v):
        if feeds_outputs[v]:
            return feeds_outputs[v]
        if v in outset:
            feeds_outputs[v] = {v}
        else:
            result = set()
            for ch in children.get(v, []):
                result |= compute_feeds(ch)
            feeds_outputs[v] = result
        return feeds_outputs[v]
    
    for v in vertices:
        compute_feeds(v)
    
    # Sort outputs by recompute cost descending (most expensive to recompute first)
    sorted_outputs = sorted(outputs, key=lambda o: -rcost.get(o, 0))
    
    ops = []
    in_mem = set()
    stored = set()
    remaining_outputs = set(sorted_outputs)
    
    def ensure_in_mem(v):
        """Ensure v is in memory, loading/computing as needed."""
        if v in in_mem:
            return
        ps = parents.get(v, [])
        if not ps:
            # Input: load it
            if len(in_mem) >= b:
                # Need to free space - delete least useful vertex
                # Score each vertex in memory by how many remaining outputs it feeds
                def usefulness(x):
                    if x in outset:
                        return 0  # Outputs should be stored, not kept
                    fed = feeds_outputs.get(x, set()) & remaining_outputs
                    return len(fed)
                
                # Delete vertex with lowest usefulness (and not needed for current computation)
                to_delete = None
                min_util = float('inf')
                for x in sorted(in_mem, key=lambda x: -usefulness(x)):
                    # Don't delete if it's a parent of v (needed for current computation)
                    if x in ps or (v in outset and x in outset):
                        continue
                    u = usefulness(x)
                    if u < min_util:
                        min_util = u
                        to_delete = x
                    if u == 0:
                        to_delete = x
                        break
                
                if to_delete is None:
                    # All have same utility, delete any non-parent
                    for x in in_mem:
                        if x not in ps:
                            to_delete = x
                            break
                if to_delete is not None:
                    ops.append(["delete", to_delete])
                    in_mem.discard(to_delete)
            
            ops.append(["load", v])
            in_mem.add(v)
        else:
            # Compute: ensure all parents in memory first
            for p in ps:
                ensure_in_mem(p)
            ops.append(["compute", v])
            in_mem.add(v)
    
    # Process each output
    for out in sorted_outputs:
        ensure_in_mem(out)
        ops.append(["store", out])
        stored.add(out)
        in_mem.discard(out)
        remaining_outputs.discard(out)
        
        # Clean up: delete vertices no longer needed for any remaining output
        to_delete = []
        for x in list(in_mem):
            if x in outset:
                continue  # Don't delete outputs (shouldn't be in mem after store)
            fed = feeds_outputs.get(x, set()) & remaining_outputs
            if not fed:
                to_delete.append(x)
        
        # Also delete vertices whose only remaining use is for outputs already processed
        for x in to_delete:
            ops.append(["delete", x])
            in_mem.discard(x)
    
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
    topo, children = _topo_order(parents, vertices)
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

def _pebble_memory_aware_postorder(dag, budget):
    """Postorder pebbling with memory budget awareness.
    
    When budget is tight, we can't keep all ancestors in memory simultaneously.
    Strategy: for each output, compute it bottom-up, but if memory is full,
    delete the vertex with the highest recompute cost (most expensive to recompute).
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices or not outputs:
        return []
    
    vset = set(vertices)
    outset = set(outputs)
    b = max(1, int(budget) if budget is not None else len(vertices))
    
    topo, children = _topo_order(parents, vertices)
    rcost = _compute_recompute_costs(parents, vertices)
    
    # For each vertex, which outputs does it feed?
    feeds_outputs = {v: set() for v in vertices}
    def compute_feeds(v):
        if feeds_outputs[v]:
            return feeds_outputs[v]
        if v in outset:
            feeds_outputs[v] = {v}
        else:
            result = set()
            for ch in children.get(v, []):
                result |= compute_feeds(ch)
            feeds_outputs[v] = result
        return feeds_outputs[v]
    
    for v in vertices:
        compute_feeds(v)
    
    # Sort outputs by recompute cost
    sorted_outputs = sorted(outputs, key=lambda o: -rcost.get(o, 0))
    
    ops = []
    in_mem = set()
    stored = set()
    remaining = set(sorted_outputs)
    
    def get_utility(x):
        """How valuable is keeping x in memory?"""
        if x in outset:
            return 0
        fed = feeds_outputs.get(x, set()) & remaining
        if not fed:
            return -1  # Not needed at all
        # Higher recompute cost = more valuable to keep
        # Also value vertices that feed many remaining outputs
        return rcost.get(x, 0) * len(fed)
    
    def free_space_if_needed():
        """Free up memory space if budget exceeded."""
        while len(in_mem) >= b:
            # Find least useful vertex to delete
            best_del = None
            best_score = float('inf')
            for x in in_mem:
                s = get_utility(x)
                if s < best_score:
                    best_score = s
                    best_del = x
            if best_del is not None:
                ops.append(["delete", best_del])
                in_mem.discard(best_del)
            else:
                break
    
    def compute_vertex(v):
        """Compute v, ensuring parents are in memory."""
        if v in in_mem:
            return
        ps = parents.get(v, [])
        if not ps:
            free_space_if_needed()
            ops.append(["load", v])
            in_mem.add(v)
        else:
            for p in ps:
                compute_vertex(p)
            free_space_if_needed()
            ops.append(["compute", v])
            in_mem.add(v)
    
    for out in sorted_outputs:
        compute_vertex(out)
        ops.append(["store", out])
        stored.add(out)
        in_mem.discard(out)
        remaining.discard(out)
        
        # Clean up unused vertices
        to_delete = [x for x in in_mem if get_utility(x) < 0]
        for x in to_delete:
            ops.append(["delete", x])
            in_mem.discard(x)
    
    return ops[:20000]

def _count_transfers(ops):
    """Count load+store operations."""
    return sum(1 for op, v in ops if op in ("load", "store"))

def _validate_schedule(dag, ops, budget):
    """Verify the schedule is valid."""
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    outset = set(outputs)
    b = max(1, int(budget) if budget is not None else len(parents))
    
    in_mem = set()
    stored = set()
    
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
            if v in in_mem:
                return False
            ps = parents.get(v, [])
            for p in ps:
                if p not in in_mem:
                    return False
            if len(in_mem) >= b:
                return False
            in_mem.add(v)
        elif op == "delete":
            if v not in in_mem:
                return False
            in_mem.discard(v)
    
    # All outputs must be stored
    if not outset.issubset(stored):
        return False
    
    return True

def pebble(dag, fast_memory_budget):
    deadline = time.time() + 40  # Safety margin
    
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    
    if not vertices or not outputs:
        return []
    
    candidates = []
    
    # Strategy 1: Memory-aware postorder with recompute cost ordering
    try:
        r = _pebble_memory_aware_postorder(dag, fast_memory_budget)
        if r and _validate_schedule(dag, r, fast_memory_budget):
            candidates.append(r)
    except Exception:
        pass
    
    # Strategy 2: Optimal greedy
    try:
        r = _pebble_optimal_greedy(dag, fast_memory_budget)
        if r and _validate_schedule(dag, r, fast_memory_budget):
            candidates.append(r)
    except Exception:
        pass
    
    # Strategy 3: Simple postorder
    try:
        r = _pebble_postorder(dag, fast_memory_budget)
        if r and _validate_schedule(dag, r, fast_memory_budget):
            candidates.append(r)
    except Exception:
        pass
    
    if not candidates:
        # Fallback: simple load-compute-store for each output
        ops = []
        in_mem = set()
        b = max(1, int(fast_memory_budget) if fast_memory_budget is not None else len(vertices))
        
        def ensure(v):
            if v in in_mem:
                return
            ps = parents.get(v, [])
            if not ps:
                while len(in_mem) >= b:
                    # Delete something
                    for x in list(in_mem):
                        ops.append(["delete", x])
                        in_mem.discard(x)
                        break
                ops.append(["load", v])
                in_mem.add(v)
            else:
                for p in ps:
                    ensure(p)
                while len(in_mem) >= b:
                    for x in list(in_mem):
                        if x not in ps:
                            ops.append(["delete", x])
                            in_mem.discard(x)
                            break
                    else:
                        break
                ops.append(["compute", v])
                in_mem.add(v)
        
        for out in outputs:
            ensure(out)
            ops.append(["store", out])
            in_mem.discard(out)
            # Clean up
            for x in list(in_mem):
                ops.append(["delete", x])
                in_mem.discard(x)
        
        candidates.append(ops[:20000])
    
    # Pick candidate with fewest transfers
    best = min(candidates, key=_count_transfers)
    
    # Final validation
    if not _validate_schedule(dag, best, fast_memory_budget):
        # Return the first valid one
        for c in candidates:
            if _validate_schedule(dag, c, fast_memory_budget):
                best = c
                break
    
    return best
