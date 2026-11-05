"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    import time
    _start_time = time.time()
    _DEADLINE = 85.0  # safety margin before 90s kill
    
    # Parse DAG structure
    # dag has topologically ordered parents and output vertices
    # We need to figure out the structure from the dag object
    
    # Try to understand the dag structure
    # Common formats: list of (vertex, parents) or dict
    # Let's handle multiple possible formats
    
    vertices = []
    parents = {}  # vertex -> list of parent vertices
    outputs = []
    
    # Try to extract structure from dag
    if hasattr(dag, 'vertices'):
        vertices = list(dag.vertices)
    elif hasattr(dag, 'nodes'):
        vertices = list(dag.nodes)
    elif isinstance(dag, (list, tuple)):
        # Could be list of (vertex, parents) pairs
        if len(dag) > 0 and isinstance(dag[0], (list, tuple)):
            for item in dag:
                if len(item) >= 2:
                    v = item[0]
                    p = item[1] if len(item) > 1 else []
                    vertices.append(v)
                    parents[v] = list(p)
                elif len(item) == 1:
                    vertices.append(item[0])
                    parents[item[0]] = []
        else:
            vertices = list(dag)
            for v in vertices:
                parents[v] = []
    elif isinstance(dag, dict):
        # Could be {vertex: [parents]}
        for v, p in dag.items():
            vertices.append(v)
            parents[v] = list(p) if p else []
    
    # If no vertices found, try other attributes
    if not vertices:
        for attr in ['vertices', 'nodes', 'verts', 'V']:
            if hasattr(dag, attr):
                val = getattr(dag, attr)
                if callable(val):
                    val = val()
                if val:
                    vertices = list(val)
                    break
    
    # Find outputs
    if hasattr(dag, 'outputs'):
        outputs = list(dag.outputs)
    elif hasattr(dag, 'sinks'):
        outputs = list(dag.sinks)
    elif hasattr(dag, 'output_vertices'):
        outputs = list(dag.output_vertices)
    
    # If no explicit outputs, assume vertices with no children are outputs
    if not outputs:
        # Build children map
        children = {v: [] for v in vertices}
        for v in vertices:
            for p in parents.get(v, []):
                if p in children:
                    children[p].append(v)
        # Outputs are vertices with no children
        outputs = [v for v in vertices if not children.get(v, [])]
        if not outputs:
            outputs = vertices[-1:] if vertices else []
    
    # Ensure all parents are in vertices
    all_parent_set = set()
    for v in vertices:
        for p in parents.get(v, []):
            all_parent_set.add(p)
    missing_parents = all_parent_set - set(vertices)
    for p in missing_parents:
        vertices.append(p)
        parents[p] = []
    
    # Build complete parent/children maps
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children:
                children[p].append(v)
    
    # Topological sort (Kahn's algorithm)
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    topo_order = []
    from collections import deque
    queue = deque([v for v in vertices if in_degree[v] == 0])
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # If topo sort didn't include all vertices (cycle), add remaining
    if len(topo_order) < len(vertices):
        remaining = [v for v in vertices if v not in set(topo_order)]
        topo_order.extend(remaining)
    
    # Now generate pebbling schedule
    # Strategy: Process in topological order
    # Keep results in fast memory if budget allows
    # Otherwise, recompute when needed
    
    # Count how many times each vertex's result is needed
    use_count = {v: 0 for v in vertices}
    for v in vertices:
        for c in children.get(v, []):
            use_count[v] += 1
    
    # Greedy strategy: 
    # - Process vertices in topological order
    # - When computing a vertex, its parents must be available
    # - Keep results in memory if use_count > 1 and budget allows
    # - If budget is full, evict the vertex with lowest future use
    
    ops = []
    fast_memory = {}  # vertex -> True (in fast memory)
    
    def load_vertex(v):
        """Load vertex v from slow memory to fast memory"""
        ops.append(['load', v])
        fast_memory[v] = True
    
    def store_vertex(v):
        """Store vertex v from fast memory to slow memory"""
        ops.append(['store', v])
        if v in fast_memory:
            del fast_memory[v]
    
    def compute_vertex(v):
        """Compute vertex v (parents must be in fast memory)"""
        ops.append(['compute', v])
        fast_memory[v] = True
    
    def delete_vertex(v):
        """Delete vertex v from fast memory"""
        ops.append(['delete', v])
        if v in fast_memory:
            del fast_memory[v]
    
    # Input vertices (no parents) start blue (in slow memory)
    # We need to load them when needed
    
    # Process in topological order
    for v in topo_order:
        # Check if we're over deadline
        if time.time() - _start_time > _DEADLINE:
            break
        
        # Check if v is already in fast memory
        if v in fast_memory:
            continue
        
        # Ensure all parents are in fast memory
        for p in parents.get(v, []):
            if p not in fast_memory:
                # Need to load or recompute p
                # If p has no parents, load it
                if not parents.get(p, []):
                    load_vertex(p)
                else:
                    # Recompute p: ensure p's parents are available
                    # This is recursive - handle iteratively
                    # For simplicity, load p if it was previously computed and stored
                    # Otherwise, we need to recompute
                    # For now, just load it (assumes it was stored)
                    load_vertex(p)
        
        # Now compute v
        compute_vertex(v)
        
        # Decide whether to keep v in fast memory or store it
        # Keep if: use_count[v] > 1 and we have budget
        # Or if v is an output
        if v in outputs:
            # Outputs must finish blue - but they start as computed (red)
            # Wait - "all outputs must finish blue" means they need to be in slow memory?
            # Actually, re-reading: "Inputs start blue, all outputs must finish blue"
            # Blue = slow memory, Red = fast memory
            # So outputs need to be stored to slow memory at the end
            pass
        
        # If budget is exceeded, evict something
        while len(fast_memory) > fast_memory_budget:
            # Find vertex to evict: lowest future use count
            # Future use = number of children not yet processed that need this vertex
            best_evict = None
            best_score = float('inf')
            for mem_v in list(fast_memory.keys()):
                # Count how many children of mem_v are still to be processed
                future_use = 0
                for c in children.get(mem_v, []):
                    if c not in fast_memory and c not in [x for x in topo_order[:topo_order.index(v)] if x != v]:
                        # Rough approximation
                        pass
                # Simplified: evict the one with lowest total use_count
                score = use_count.get(mem_v, 0)
                if score < best_score:
                    best_score = score
                    best_evict = mem_v
            if best_evict is not None:
                store_vertex(best_evict)
            else:
                break
    
    # Ensure all outputs are stored (finish blue)
    for v in outputs:
        if v in fast_memory:
            store_vertex(v)
    
    # If we broke early due to deadline, try to complete minimally
    if time.time() - _start_time > _DEADLINE:
        # Just return what we have
        return ops
    
    return ops
# EVOLVE-BLOCK-END
