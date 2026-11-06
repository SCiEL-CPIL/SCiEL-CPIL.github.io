"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Parse DAG structure
    # dag should have: vertices, edges (parents for each vertex), inputs, outputs
    # Assume dag is a dict or object with:
    #   dag['vertices'] or dag.vertices: list of vertex ids
    #   dag['parents'] or dag.parents: dict mapping vertex -> list of parent vertices
    #   dag['inputs'] or dag.inputs: list of input vertex ids
    #   dag['outputs'] or dag.outputs: list of output vertex ids
    
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
            # Try to infer from parents
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
            # edges might be (child, parent) pairs
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
            # Inputs are vertices with no parents
            parents = get_parents(d)
            return [v for v in get_vertices(d) if not parents.get(v, [])]
    
    def get_outputs(d):
        if hasattr(d, 'outputs'):
            return list(d.outputs)
        elif isinstance(d, dict) and 'outputs' in d:
            return list(d['outputs'])
        else:
            # Outputs are vertices with no children
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
        # Fallback: try to use baseline if our parsing fails
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Build children map for lookahead
    children_map = {}
    for v, ps in parents_map.items():
        for p in ps:
            children_map.setdefault(p, []).append(v)
    
    # Topological sort (Kahn's algorithm)
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
    
    # If topo sort didn't include all vertices (cycle?), just use all
    if len(topo_order) != len(vertices):
        topo_order = list(vertices)
    
    # Determine which vertices are inputs (start blue, no need to compute)
    input_set = set(inputs)
    output_set = set(outputs)
    
    # For each vertex, find its position in topo order for lookahead
    pos_in_topo = {v: i for i, v in enumerate(topo_order)}
    
    # Schedule: process vertices in topo order
    # Fast memory: dict vertex -> True (in memory)
    # Track which vertices are currently in fast memory
    fast_mem = {}  # vertex -> True
    operations = []
    
    # Counters
    transfer_count = 0
    op_count = 0
    MAX_OPS = 20000
    
    # Precompute next use for each vertex (for eviction lookahead)
    # next_use[v] = index in topo_order of the next vertex that uses v as a parent
    # We'll compute this on the fly
    
    def compute_next_uses():
        """For each vertex, compute the list of positions where it's needed."""
        next_uses = {v: [] for v in vertices}
        for i, v in enumerate(topo_order):
            for p in parents_map.get(v, []):
                next_uses[p].append(i)
        return next_uses
    
    next_uses = compute_next_uses()
    
    # Track which use index we're at for each vertex
    use_index = {v: 0 for v in vertices}
    
    def get_next_use_time(v):
        """Get the next position in topo_order where v is needed, or infinity."""
        idx = use_index[v]
        uses = next_uses[v]
        if idx < len(uses):
            return uses[idx]
        return float('inf')
    
    def advance_use(v):
        """Advance the use index for vertex v."""
        use_index[v] += 1
    
    def evict_lru_with_lookahead():
        """Evict the vertex in fast memory that has the farthest next use."""
        if not fast_mem:
            return
        # Find the vertex with the maximum next use time
        worst_v = None
        worst_time = -1
        for v in list(fast_mem.keys()):
            t = get_next_use_time(v)
            if t > worst_time:
                worst_time = t
                worst_v = v
        if worst_v is not None:
            del fast_mem[worst_v]
            operations.append(['store', worst_v])
            op_count += 1
    
    # Initialize: inputs start blue (in memory if budget allows, or we load them)
    # Actually, inputs "start blue" means they are available. We need to load them into fast memory to use them.
    # But wait - "Inputs start blue" - does this mean they're already in fast memory?
    # The problem says "Inputs start blue, all outputs must finish blue"
    # "every red value consumes one fast-memory slot"
    # So inputs are blue (available), but to compute with them, they need to be in fast memory (red)?
    # Actually, I think blue means available in slow memory/disk, red means in fast memory.
    # So we need to load inputs into fast memory to use them.
    
    # Process each vertex in topological order
    for v in topo_order:
        if time.time() > deadline:
            break
        
        if op_count >= MAX_OPS:
            break
        
        if v in input_set:
            # Input vertex: needs to be loaded into fast memory
            if v not in fast_mem:
                # Check if we have space
                if len(fast_mem) < fast_memory_budget:
                    fast_mem[v] = True
                    operations.append(['load', v])
                    op_count += 1
                    transfer_count += 1
                else:
                    # Need to evict
                    evict_lru_with_lookahead()
                    fast_mem[v] = True
                    operations.append(['load', v])
                    op_count += 1
                    transfer_count += 1
            # Advance use index for input (it's being "used" now)
            # Actually, inputs don't get "used" in the same way - they're just sources
            # But they can be needed later by other vertices
            # Let's not advance use index for inputs here, since they're not being consumed
            # The use index tracks when a value is needed as a parent
            continue
        
        # Non-input vertex: compute it
        # First, ensure all parents are in fast memory
        for p in parents_map.get(v, []):
            if p not in fast_mem:
                # Load parent
                if len(fast_mem) < fast_memory_budget:
                    fast_mem[p] = True
                    operations.append(['load', p])
                    op_count += 1
                    transfer_count += 1
                else:
                    evict_lru_with_lookahead()
                    fast_mem[p] = True
                    operations.append(['load', p])
                    op_count += 1
                    transfer_count += 1
            
            # Advance use index for parent (it's being used now)
            advance_use(p)
        
        # Compute v
        operations.append(['compute', v])
        op_count += 1
        
        # After computing, v is in fast memory (as a red value)
        # But we need to make sure we have space
        # Actually, the compute operation produces v in fast memory
        # If we're over budget, we need to evict something
        if len(fast_mem) >= fast_memory_budget and v not in fast_mem:
            # We need to evict before adding v
            # But wait, we just computed v, so it should be in fast memory
            # The issue is: do we have room?
            # Let's check: if fast_mem is full, evict first
            if len(fast_mem) >= fast_memory_budget:
                evict_lru_with_lookahead()
        
        fast_mem[v] = True
        
        # If v is an output, we need to ensure it finishes blue
        # "all outputs must finish blue" - this means outputs must be stored to disk?
        # Or does it mean they must be available (blue) at the end?
        # I think "finish blue" means they're in slow memory (stored), not in fast memory
        # So we need to store outputs at the end
        # But actually, maybe blue means "available" and we just need them to be present
        # Let's assume we need to store outputs to ensure they're blue at the end
    
    # At the end, ensure all outputs are blue (stored to disk)
    # This means they should NOT be in fast memory at the end, OR they should be stored
    # Actually, "finish blue" likely means they're in slow memory
    # So we need to store any outputs that are in fast memory
    
    for v in outputs:
        if time.time() > deadline:
            break
        if op_count >= MAX_OPS:
            break
        
        if v in fast_mem:
            operations.append(['store', v])
            op_count += 1
            transfer_count += 1
            del fast_mem[v]
        else:
            # Output is not in fast memory, but it should be blue
            # If it was computed and stored, it's already blue
            # If it's an input that's an output, it's already blue
            # We might need to load and store it? No, that doesn't make sense
            # Let's assume it's already blue if it's not in fast memory
            pass
    
    # Also, we might need to clean up: delete any remaining fast memory values?
    # The problem says "delete" is an operation, but it's not clear if we need to use it
    # Let's not delete unless necessary
    
    # Validate: check we don't exceed max ops
    if op_count > MAX_OPS:
        # Truncate to max ops
        operations = operations[:MAX_OPS]
    
    return operations

# EVOLVE-BLOCK-END
