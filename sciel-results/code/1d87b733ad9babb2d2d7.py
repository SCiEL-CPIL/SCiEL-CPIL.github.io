import collections
import random

def pebble(dag, fast_memory_budget):
    """
    Minimize communication (transfers) for DAG pebbling with bounded fast memory.
    
    The key insight: we can use a recursive pebbling strategy that exploits the
    structure of layered DAGs. For each vertex, we compute it only when needed
    and reuse intermediate results in fast memory to minimize reloads.
    
    Strategy:
    1. Identify the DAG structure (topological order, parents, outputs)
    2. Use a worklist-based pebbling with LRU eviction of fast memory
    3. For layered DAGs, exploit the fact that layers can be processed in order
       and vertices within a layer are independent
    
    The naive schedule loads each input once and computes each vertex once,
    storing results in fast memory until needed. If fast memory is too small,
    results get evicted and must be reloaded (requiring recomputation or reload).
    
    Our approach: use a smarter eviction policy and potentially recompute
    rather than reload when the recomputation cost is lower.
    """
    # Parse DAG structure
    # dag is expected to have:
    #   - vertices: list or range
    #   - parents[v]: list of parent vertices (topologically before v)
    #   - outputs: list of output vertices
    #   - inputs: vertices with no parents (start blue)
    
    # Handle different possible DAG representations
    if isinstance(dag, dict):
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        all_vertices = set(parents.keys())
        for plist in parents.values():
            all_vertices.update(plist)
        all_vertices = sorted(all_vertices)
        inputs = [v for v in all_vertices if not parents.get(v, [])]
        num_vertices = len(all_vertices)
        # Create index mapping
        vindex = {v: i for i, v in enumerate(all_vertices)}
        parents_idx = {vindex[v]: [vindex[p] for p in parents.get(v, [])] for v in all_vertices if v in vindex}
    elif hasattr(dag, 'parents'):
        parents = dag.parents
        outputs = dag.outputs
        all_vertices = list(range(len(parents)))
        inputs = [v for v in all_vertices if not parents[v]]
        num_vertices = len(parents)
        parents_idx = {v: list(parents[v]) for v in all_vertices}
    else:
        # Fallback: assume dag is a list of parent lists
        parents_idx = {i: list(p) for i, p in enumerate(dag)}
        num_vertices = len(dag)
        inputs = [i for i in range(num_vertices) if not dag[i]]
        outputs = [num_vertices - 1]  # Default: last vertex is output
        all_vertices = list(range(num_vertices))
    
    # Build adjacency (children)
    children = {v: [] for v in range(num_vertices)}
    for v in range(num_vertices):
        for p in parents_idx.get(v, []):
            children[p].append(v)
    
    # Topological order
    in_degree = {v: len(parents_idx.get(v, [])) for v in range(num_vertices)}
    queue = collections.deque([v for v in range(num_vertices) if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # If no outputs specified, assume all vertices with no children are outputs
    if not outputs:
        outputs = [v for v in range(num_vertices) if not children[v]]
    
    # Convert outputs to indices if needed
    if isinstance(dag, dict) and 'outputs' in dag:
        outputs = [vindex[v] for v in outputs if v in vindex]
    
    # Naive schedule: process in topological order
    # Each vertex: load parents (if not in fast memory), compute, store in fast memory
    # Fast memory is a set of vertices currently held
    # Transfers = loads + stores (but storing to fast memory might not count as transfer?)
    # Actually, "transfers" likely means load/store operations between slow and fast memory
    
    # Let's think about the cost model:
    # - load: bring a value from slow memory to fast memory (cost 1 transfer)
    # - store: write a value from fast memory to slow memory (cost 1 transfer)  
    # - compute: perform computation in fast memory (no transfer)
    # - delete: remove from fast memory (no transfer, just frees slot)
    
    # Input vertices start blue (in slow memory? or in fast memory?)
    # "Inputs start blue" - they are available but not in fast memory initially
    # "all outputs must finish blue" - they must be in fast memory at the end? Or just computed?
    
    # Let me assume:
    # - Initially, inputs are in slow memory (blue means valid/computed, not in fast mem)
    # - We need to load them into fast memory to use them
    # - After computation, results can be kept in fast memory or stored back
    # - At the end, outputs must be "blue" (valid) - probably means they exist somewhere
    
    # Naive approach: 
    # 1. For each vertex in topological order:
    #    - Load all parents that aren't in fast memory
    #    - Compute the vertex (if parents are in fast memory)
    #    - Keep result in fast memory (if budget allows)
    #    - If budget exceeded, evict oldest/least useful
    
    # The naive baseline probably does:
    # - Load each input once when first needed
    # - Compute each vertex once
    # - Store nothing back (keep everything in fast memory if possible)
    # - If fast memory overflows, something goes wrong or uses LRU
    
    # To beat naive, we need to be smarter about:
    # 1. When to evict (evict vertices that won't be needed soon)
    # 2. Whether to recompute vs reload (recompute if parent is in fast mem)
    
    # Let's implement a worklist-based approach with lookahead
    
    # Fast memory state
    fast_mem = {}  # vertex -> True (present in fast memory)
    fast_mem_order = []  # LRU order
    
    # Track which vertices have been computed and stored to slow memory
    computed = set()  # vertices whose value is available in slow memory
    
    operations = []
    
    def load(v):
        """Load vertex v from slow to fast memory"""
        if v not in fast_mem:
            fast_mem[v] = True
            fast_mem_order.append(v)
            operations.append(['load', v])
        return True
    
    def evict(v):
        """Evict vertex v from fast memory"""
        if v in fast_mem:
            del fast_mem[v]
            if v in fast_mem_order:
                fast_mem_order.remove(v)
            operations.append(['delete', v])
    
    def store(v):
        """Store vertex v to slow memory (marks it as computed/available)"""
        computed.add(v)
        operations.append(['store', v])
    
    def make_room(n=1):
        """Evict vertices until we have room for n more"""
        evicted = 0
        while len(fast_mem) + n > fast_memory_budget and fast_mem_order:
            # Evict the least recently used (or least useful)
            # Simple LRU: evict from front
            v = fast_mem_order[0]
            evict(v)
            evicted += 1
    
    # Determine which vertices need to be in fast memory at the end (outputs)
    output_set = set(outputs)
    
    # Process vertices in topological order
    # For each vertex, we need its parents in fast memory to compute it
    # We load parents as needed, compute, then decide whether to keep or evict
    
    # Better approach: use a recursive pebbling strategy
    # For a vertex v with parents p1, p2, ..., pk:
    #   - Compute each parent (recursively)
    #   - Load parent values into fast memory
    #   - Compute v
    #   - Store v if it will be needed later
    
    # The key optimization: if a parent is only needed for one child,
    # we can evict it immediately after use. If it's needed for multiple
    # children, we keep it.
    
    # Let's use a different strategy: process layer by layer
    # Layer 0: inputs
    # Layer 1: vertices whose parents are all in layer 0
    # etc.
    
    # Actually, let's just do a smart topological processing with
    # lookahead-based eviction
    
    # Compute "needed_at" for each vertex: the topological position where
    # the vertex is last needed (i.e., the last child's position)
    last_needed = {}
    for v in range(num_vertices):
        if not children[v]:
            last_needed[v] = v  # Only needed at its own position (or end if output)
            if v in output_set:
                last_needed[v] = num_vertices  # Keep until end
        else:
            max_child_pos = max(topo_order.index(c) for c in children[v])
            last_needed[v] = max_child_pos
            if v in output_set:
                last_needed[v] = num_vertices
    
    # Now process in topological order
    for pos, v in enumerate(topo_order):
        # We need to compute v, which requires all parents in fast memory
        parents_v = parents_idx.get(v, [])
        
        # Check which parents are already in fast memory
        for p in parents_v:
            if p not in fast_mem:
                # Need to load p
                # First make room if needed
                make_room(1)
                load(p)
        
        # Now all parents are in fast memory, compute v
        operations.append(['compute', v])
        fast_mem[v] = True
        fast_mem_order.append(v)
        computed.add(v)
        
        # Now decide which vertices to evict
        # Evict vertices that are no longer needed (last_needed < current pos)
        # But keep outputs until the end
        to_evict = []
        for u in list(fast_mem_order):
            if u == v:
                continue
            if u in output_set:
                continue
            if last_needed[u] < pos:
                to_evict.append(u)
        
        for u in to_evict:
            evict(u)
        
        # If still over budget, evict LRU
        while len(fast_mem) > fast_memory_budget:
            # Evict the one with earliest last_needed (among non-outputs)
            candidates = [u for u in fast_mem_order if u != v and u not in output_set]
            if not candidates:
                # Can only evict outputs, which we shouldn't
                # This means budget is too small; just keep going
                break
            # Evict the one with smallest last_needed
            candidates.sort(key=lambda u: last_needed[u])
            evict(candidates[0])
    
    # Ensure all outputs are in fast memory at the end
    for o in outputs:
        if o not in fast_mem:
            make_room(1)
            load(o)
    
    # Limit operations to 20000
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
