"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Parse DAG structure
    def get_vertices(d):
        if hasattr(d, 'vertices'):
            return list(d.vertices)
        elif isinstance(d, dict) and 'vertices' in d:
            return list(d['vertices'])
        elif hasattr(d, 'nodes'):
            return list(d.nodes)
        elif isinstance(d, dict) and 'nodes' in d:
            return list(d['nodes'])
        else:
            if hasattr(d, 'parents'):
                parents = d.parents
            elif isinstance(d, dict) and 'parents' in d:
                parents = d['parents']
            else:
                raise ValueError("Cannot determine vertices from dag")
            verts = set()
            for v, ps in parents.items():
                verts.add(v)
                verts.update(ps)
            return list(verts)
    
    def get_parents(d):
        if hasattr(d, 'parents'):
            return dict(d.parents)
        elif isinstance(d, dict) and 'parents' in d:
            return dict(d['parents'])
        elif hasattr(d, 'edges'):
            parents = {}
            for e in d.edges:
                if len(e) == 2:
                    child, parent = e
                    parents.setdefault(child, []).append(parent)
            return parents
        else:
            raise ValueError("Cannot determine parents from dag")
    
    def get_inputs(d):
        if hasattr(d, 'inputs'):
            return list(d.inputs)
        elif isinstance(d, dict) and 'inputs' in d:
            return list(d['inputs'])
        else:
            parents = get_parents(d)
            return [v for v in get_vertices(d) if not parents.get(v, [])]
    
    def get_outputs(d):
        if hasattr(d, 'outputs'):
            return list(d.outputs)
        elif isinstance(d, dict) and 'outputs' in d:
            return list(d['outputs'])
        else:
            parents = get_parents(d)
            vertices = get_vertices(d)
            children = {}
            for v, ps in parents.items():
                for p in ps:
                    children.setdefault(p, []).append(v)
            return [v for v in vertices if not children.get(v, [])]
    
    try:
        vertices = get_vertices(dag)
        parents_map = get_parents(dag)
        inputs = get_inputs(dag)
        outputs = get_outputs(dag)
    except Exception as e:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []
    
    # Build children map
    children_map = {}
    for v, ps in parents_map.items():
        for p in ps:
            children_map.setdefault(p, []).append(v)
    
    # Topological sort
    in_degree = {v: len(parents_map.get(v, [])) for v in vertices}
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children_map.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(vertices):
        topo_order = list(vertices)
    
    input_set = set(inputs)
    output_set = set(outputs)
    
    MAX_OPS = 20000
    
    # Compute next_uses for lookahead
    next_uses = {v: [] for v in vertices}
    for i, v in enumerate(topo_order):
        for p in parents_map.get(v, []):
            next_uses[p].append(i)
    
    use_index = {v: 0 for v in vertices}
    
    def get_next_use_time(v):
        idx = use_index[v]
        uses = next_uses[v]
        if idx < len(uses):
            return uses[idx]
        return float('inf')
    
    def advance_use(v):
        use_index[v] += 1
    
    # Find bridge vertices: vertices that separate the DAG into sub-components
    # A bridge vertex is one whose removal increases the number of connected components
    # For a DAG, we look for articulation points in the underlying undirected graph
    # But for our purposes, we'll use a simpler heuristic: find vertices with high "betweenness"
    
    # Simpler approach: find vertices that are parents of many vertices across different "layers"
    # For layered DAGs, bridges are vertices that connect different layers
    
    # Compute layer for each vertex (longest path from any input)
    layer = {v: 0 for v in vertices}
    for v in topo_order:
        ps = parents_map.get(v, [])
        if ps:
            layer[v] = max(layer[p] for p in ps) + 1
    
    # Group vertices by layer
    layers = {}
    for v in vertices:
        layers.setdefault(layer[v], []).append(v)
    
    # Find bridge vertices: vertices that have children in multiple "branches"
    # For a layered DAG, a good bridge is a vertex whose children span multiple sub-trees
    
    # Heuristic: find vertices with the most children that are "spread out"
    # For simplicity, let's identify vertices that are parents of vertices in different sub-DAGs
    
    # Alternative approach: use the fact that in a layered DAG, we can split at certain layers
    # and treat the vertices at the split as bridges
    
    # For now, let's use a simpler strategy: if the DAG is small enough, just do the linear schedule
    # But try to optimize by keeping frequently-used vertices in memory
    
    # Count how many times each vertex is used as a parent
    use_count = {v: len(next_uses[v]) for v in vertices}
    
    # Sort vertices by use count (most used first) for priority in memory
    # This helps with eviction decisions
    
    fast_mem = set()
    operations = []
    op_count = 0
    transfer_count = 0
    
    # Precompute: for each vertex, list of (position_in_topo_order) where it's used as a parent
    # next_uses[v] = sorted list of topo indices where v is needed as a parent
    # use_idx[v] = how many uses of v we've already accounted for
    
    def evict_one(exclude=None):
        """Evict one vertex from fast memory. Returns the evicted vertex."""
        nonlocal op_count
        candidates = [v for v in fast_mem if v != exclude]
        if not candidates:
            return None
        # Evict the vertex with the farthest next use (or no future use)
        worst_v = None
        worst_time = -1
        for v in candidates:
            t = get_next_use_time(v)
            if t > worst_time:
                worst_time = t
                worst_v = v
        if worst_v is not None:
            fast_mem.discard(worst_v)
            operations.append(['store', worst_v])
            op_count += 1
            transfer_count += 1
        return worst_v
    
    def ensure_in_memory(v):
        """Ensure vertex v is in fast memory, evicting if necessary."""
        nonlocal op_count, transfer_count
        if v in fast_mem:
            return
        if len(fast_mem) < fast_memory_budget:
            fast_mem.add(v)
            operations.append(['load', v])
            op_count += 1
            transfer_count += 1
        else:
            evict_one(exclude=v)
            fast_mem.add(v)
            operations.append(['load', v])
            op_count += 1
            transfer_count += 1
    
    # Process each vertex in topological order
    for v in topo_order:
        if time.time() > deadline:
            break
        
        if op_count >= MAX_OPS:
            break
        
        if v in input_set:
            # Input vertex: we don't need to load it now; it will be loaded when needed as a parent
            continue
        
        # Non-input vertex: compute it
        # First, ensure all parents are in fast memory
        for p in parents_map.get(v, []):
            if p not in fast_mem:
                ensure_in_memory(p)
            advance_use(p)
        
        # Before compute, if fast_mem is full, we need space for v's result
        # The compute operation will place v into fast memory
        # So we need to evict if full
        if len(fast_mem) >= fast_memory_budget:
            evict_one(exclude=None)
        
        # Compute v - this places v into fast memory
        operations.append(['compute', v])
        op_count += 1
        fast_mem.add(v)
    
    # At the end, ensure all outputs are blue (stored to disk)
    for v in outputs:
        if time.time() > deadline:
            break
        if op_count >= MAX_OPS:
            break
        
        if v in fast_mem:
            operations.append(['store', v])
            op_count += 1
            transfer_count += 1
            fast_mem.discard(v)
    
    # Validate: check we don't exceed max ops
    if op_count > MAX_OPS:
        operations = operations[:MAX_OPS]
    
    return operations

# EVOLVE-BLOCK-END
