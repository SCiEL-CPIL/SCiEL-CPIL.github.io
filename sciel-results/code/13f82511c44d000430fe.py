"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque, defaultdict

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Returns list of [operation, vertex] where operation is one of
    'load', 'store', 'compute', 'delete'.
    """
    start_time = time.time()
    DEADLINE = 85.0
    
    # Parse dag structure
    def _get(d, *keys, default=None):
        for k in keys:
            if isinstance(d, dict):
                if k in d:
                    return d[k]
            elif hasattr(d, k):
                return getattr(d, k)
        return default
    
    vertices = _get(dag, 'vertices', 'V', 'nodes', 'verts', default=None)
    parents = _get(dag, 'parents', 'preds', 'predecessors', 'in_edges', 'input_edges', default=None)
    inputs = _get(dag, 'inputs', 'input', 'sources', 'input_vertices', default=None)
    outputs = _get(dag, 'outputs', 'output', 'sinks', 'output_vertices', default=None)
    
    if vertices is None:
        if parents is not None:
            vertices = list(parents.keys())
        else:
            return []
    
    if parents is None:
        parents = {}
    
    if inputs is None:
        inputs = [v for v in vertices if not parents.get(v)]
    
    if outputs is None:
        parent_set = set()
        for v in vertices:
            for p in parents.get(v, []):
                parent_set.add(p)
        outputs = [v for v in vertices if v not in parent_set]
    
    # Build children map
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Topological sort
    in_deg = {v: len(parents.get(v, [])) for v in vertices}
    queue = deque([v for v in vertices if in_deg[v] == 0])
    topo = []
    while queue:
        v = queue.popleft()
        topo.append(v)
        for c in children[v]:
            in_deg[c] -= 1
            if in_deg[c] == 0:
                queue.append(c)
    
    if len(topo) < len(vertices):
        # Add remaining vertices
        remaining = set(vertices) - set(topo)
        topo.extend(sorted(remaining, key=lambda x: str(x)))
    
    # Pebbling strategy for layered DAGs:
    # Process in topological order.
    # State: 'blue' (in fast mem), 'red' (in external, still counted), None (gone)
    # Constraint: at any time, count(blue) + count(red) <= fast_memory_budget
    # 
    # Strategy: Keep values blue while they're needed. Delete when no longer needed.
    # This minimizes transfers since we only compute once and delete.
    
    state = {}
    for v in inputs:
        state[v] = 'blue'
    
    ops = []
    
    # Track how many children still need each vertex
    remaining_children = {v: len(children[v]) for v in vertices}
    
    # Also track: for each vertex, which of its children have been processed
    # so we know when to delete it
    
    for v in topo:
        if time.time() - start_time > DEADLINE:
            break
        
        # Ensure all parents are available (blue)
        for p in parents.get(v, []):
            if state.get(p) == 'red':
                ops.append(['load', p])
                state[p] = 'blue'
            elif state.get(p) is None:
                # Parent not computed yet - shouldn't happen in topo order
                # But if it does, we have a problem
                pass
        
        # Compute v if not already done
        if state.get(v) is None:
            ops.append(['compute', v])
            state[v] = 'blue'
        
        # Update remaining children count for parents
        for p in parents.get(v, []):
            remaining_children[p] -= 1
            # If p is no longer needed by any child and not an output, delete it
            if remaining_children[p] == 0 and p not in outputs:
                if state.get(p) == 'blue':
                    ops.append(['delete', p])
                    state[p] = None
    
    # Ensure outputs are blue
    for v in outputs:
        if state.get(v) == 'red':
            ops.append(['load', v])
            state[v] = 'blue'
    
    return ops
# EVOLVE-BLOCK-END
