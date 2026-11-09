"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85.0
    
    # Parse DAG
    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('n', 0))
        edges = dag.get('edges', dag.get('arcs', []))
        inputs = set(dag.get('inputs', dag.get('sources', [])))
        outputs = set(dag.get('outputs', dag.get('sinks', [])))
    elif isinstance(dag, (list, tuple)):
        vertices = len(dag)
        edges = []
        for i, adj in enumerate(dag):
            for j in adj:
                edges.append((i, j))
        has_parent = set()
        has_child = set()
        for p, c in edges:
            has_child.add(p)
            has_parent.add(c)
        inputs = set(range(vertices)) - has_parent
        outputs = set(range(vertices)) - has_child
    else:
        return []
    
    if vertices == 0:
        return []
    
    children = defaultdict(list)
    parents = defaultdict(list)
    for p, c in edges:
        children[p].append(c)
        parents[c].append(p)
    
    # Topo order
    in_degree = [0] * vertices
    for i in range(vertices):
        in_degree[i] = len(parents[i])
    
    queue = deque([i for i in range(vertices) if in_degree[i] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != vertices:
        topo_order = list(range(vertices))
    
    MAX_OPS = 20000
    ops = []
    
    # State
    in_memory = set(inputs)  # inputs start blue, in fast memory
    is_blue = set(inputs)
    
    # remaining_uses[v] = number of children of v that haven't been processed yet
    remaining_uses = {v: len(children[v]) for v in range(vertices)}
    
    def elapsed():
        return time.time() - start_time
    
    def make_space():
        """Evict a value to make room if needed."""
        nonlocal in_memory, is_blue, ops
        if len(in_memory) < fast_memory_budget:
            return
        # Find best candidate to evict
        # Prefer: non-outputs, then low remaining uses, then non-blue
        candidates = []
        for x in in_memory:
            if x in outputs:
                continue  # avoid evicting outputs
            candidates.append(x)
        if not candidates:
            candidates = list(in_memory)
        # Sort by remaining uses (evict those with fewest uses first)
        candidates.sort(key=lambda x: remaining_uses[x])
        evict = candidates[0]
        ops.append(['delete', evict])
        in_memory.discard(evict)
        is_blue.discard(evict)
    
    for v in topo_order:
        if elapsed() > DEADLINE:
            break
        if len(ops) >= MAX_OPS:
            break
        
        # Ensure parents are in memory
        for p in parents[v]:
            if p not in in_memory:
                make_space()
                ops.append(['load', p])
                in_memory.add(p)
                # p is now red (loaded, not stored)
        
        # Compute v
        ops.append(['compute', v])
        in_memory.add(v)
        # v is red
        
        # Update remaining uses for parents
        for p in parents[v]:
            remaining_uses[p] -= 1
            # If parent has no more uses, we could evict it
            # But let's be conservative and let make_space handle it
    
    # Ensure outputs are blue
    for v in outputs:
        if elapsed() > DEADLINE:
            break
        if len(ops) >= MAX_OPS:
            break
        
        if v not in in_memory:
            make_space()
            ops.append(['load', v])
            in_memory.add(v)
        
        if v not in is_blue:
            ops.append(['store', v])
            is_blue.add(v)
    
    return ops[:MAX_OPS]

# EVOLVE-BLOCK-END
