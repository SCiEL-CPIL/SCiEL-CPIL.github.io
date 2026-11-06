"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # Parse DAG structure
    if isinstance(dag, dict):
        n = dag.get('n', dag.get('num_vertices', 0))
        edges = dag.get('edges', [])
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
        parents_dict = dag.get('parents', None)
    else:
        n = len(dag) if hasattr(dag, '__len__') else 0
        edges = []
        inputs = set()
        outputs = set()
        parents_dict = None
    
    # Build parent/children maps
    parent_map = defaultdict(list)
    children_map = defaultdict(list)
    
    if parents_dict is not None:
        for v, ps in parents_dict.items():
            for p in ps:
                parent_map[v].append(p)
                children_map[p].append(v)
    else:
        for edge in edges:
            if len(edge) == 2:
                u, v = edge
                parent_map[v].append(u)
                children_map[u].append(v)
            elif len(edge) == 3:
                # (p1, p2, child)
                p1, p2, c = edge
                parent_map[c].extend([p1, p2])
                children_map[p1].append(c)
                children_map[p2].append(c)
    
    # Collect all vertices
    all_vertices = set()
    for v in parent_map:
        all_vertices.add(v)
    for ps in parent_map.values():
        for p in ps:
            all_vertices.add(p)
    for v in children_map:
        all_vertices.add(v)
    for cs in children_map.values():
        for c in cs:
            all_vertices.add(c)
    if inputs:
        all_vertices.update(inputs)
    if outputs:
        all_vertices.update(outputs)
    
    if not inputs:
        inputs = set(v for v in all_vertices if not parent_map.get(v))
    if not outputs:
        outputs = set(v for v in all_vertices if not children_map.get(v))
    
    # Topological sort
    in_degree = {v: len(parent_map.get(v, [])) for v in all_vertices}
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children_map.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # Add any missing vertices
    if len(topo_order) < len(all_vertices):
        remaining = all_vertices - set(topo_order)
        topo_order.extend(sorted(remaining))
    
    # Compute remaining uses for each vertex
    remaining_uses = {}
    for v in reversed(topo_order):
        uses = len(children_map.get(v, []))
        if v in outputs:
            uses += 1
        remaining_uses[v] = uses
    
    # Greedy pebbling
    fast_mem = set()
    computed = set()
    operations = []
    transfers = 0
    
    for v in topo_order:
        if time.time() > deadline:
            break
        
        # If v is an input, it's in slow memory initially
        if v in inputs:
            # Input starts blue in slow memory
            # We'll load it when needed
            pass
        
        if parent_map.get(v):
            # Need to compute v
            # Load all parents into fast memory
            for p in parent_map[v]:
                if p not in fast_mem:
                    # Load from slow memory
                    transfers += 1
                    operations.append(('load', p))
                    fast_mem.add(p)
            
            # Compute v
            operations.append(('compute', v))
            computed.add(v)
            
            # Decide where to put v
            if len(fast_mem) < fast_memory_budget:
                fast_mem.add(v)
            else:
                # Fast memory full
                # Evict the vertex with fewest remaining uses (excluding v)
                candidates = [x for x in fast_mem if x != v]
                if candidates:
                    min_uses = min(remaining_uses.get(x, 0) for x in candidates)
                    # Pick the one with min uses, break ties by topo order
                    evict_v = min(candidates, key=lambda x: (remaining_uses.get(x, 0), topo_order.index(x)))
                    fast_mem.remove(evict_v)
                    transfers += 1
                    operations.append(('store', evict_v))
                    fast_mem.add(v)
                else:
                    # Only v in fast memory, store it
                    transfers += 1
                    operations.append(('store', v))
        
        # If v is an output, ensure it's in slow memory at the end
        if v in outputs:
            if v in fast_mem:
                transfers += 1
                operations.append(('store', v))
                fast_mem.remove(v)
    
    # Safety: limit operations
    if len(operations) > 20000:
        # Truncate or use simpler strategy
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
