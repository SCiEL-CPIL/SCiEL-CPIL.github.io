"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Minimize transfers in DAG pebbling.
    
    dag structure (inferred):
    - dag['vertices'] or similar: list of vertices
    - dag['edges'] or adjacency: parent -> children relationships
    - dag['inputs']: input vertices (start blue)
    - dag['outputs']: output vertices (must finish blue)
    - Topologically ordered parents
    
    Returns list of [operation, vertex] where operation is one of:
    'load', 'store', 'compute', 'delete'
    """
    start_time = time.time()
    DEADLINE = start_time + 85  # 5s safety margin
    
    # Parse DAG structure - try common formats
    if isinstance(dag, dict):
        vertices = list(dag.get('vertices', dag.get('nodes', range(dag.get('n', 0)))))
        if 'n' in dag and 'vertices' not in dag and 'nodes' not in dag:
            vertices = list(range(dag['n']))
        edges = dag.get('edges', dag.get('adjacency', {}))
        inputs = set(dag.get('inputs', dag.get('input_vertices', [])))
        outputs = set(dag.get('outputs', dag.get('output_vertices', [])))
    elif isinstance(dag, (list, tuple)):
        # Maybe (n, edges, inputs, outputs)
        if len(dag) >= 4:
            n, edges, inputs, outputs = dag[0], dag[1], dag[2], dag[3]
            vertices = list(range(n))
            inputs = set(inputs)
            outputs = set(outputs)
        else:
            vertices = list(dag)
            edges = {}
            inputs = set()
            outputs = set(vertices)
    else:
        vertices = list(range(len(dag))) if hasattr(dag, '__len__') else [0]
        edges = {}
        inputs = set()
        outputs = set(vertices)
    
    n = len(vertices)
    if n == 0:
        return []
    
    # Build adjacency: parents[v] = list of parents of v, children[v] = list of children
    # edges could be: dict of v -> [parents], or list of (parent, child) pairs, or adjacency matrix
    if isinstance(edges, dict):
        # Could be children[v] = [children] or parents[v] = [parents]
        # Check if keys map to lists
        parent_map = defaultdict(list)
        child_map = defaultdict(list)
        for v, neighbors in edges.items():
            if v in vertices:
                for nb in neighbors:
                    if nb in vertices:
                        # Assume edges[v] = children of v (common convention)
                        child_map[v].append(nb)
                        parent_map[nb].append(v)
    elif isinstance(edges, (list, tuple)):
        # List of (parent, child) pairs or adjacency matrix
        parent_map = defaultdict(list)
        child_map = defaultdict(list)
        if len(edges) > 0:
            first = edges[0]
            if isinstance(first, (list, tuple)) and len(first) == 2:
                # List of (u, v) meaning u is parent of v
                for u, v in edges:
                    if u in vertices and v in vertices:
                        child_map[u].append(v)
                        parent_map[v].append(u)
            else:
                # Maybe adjacency matrix
                for i in range(n):
                    for j in range(n):
                        if edges[i][j]:
                            child_map[i].append(j)
                            parent_map[j].append(i)
    else:
        parent_map = defaultdict(list)
        child_map = defaultdict(list)
    
    # Ensure all vertices have entries
    for v in vertices:
        if v not in parent_map:
            parent_map[v] = []
        if v not in child_map:
            child_map[v] = []
    
    # Topological order
    in_degree = {v: len(parent_map[v]) for v in vertices}
    topo_order = []
    queue = deque([v for v in vertices if in_degree[v] == 0])
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in child_map[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != n:
        # Cycle or issue, add remaining
        remaining = [v for v in vertices if v not in set(topo_order)]
        topo_order.extend(remaining)
    
    # Determine which vertices are inputs (sources) and outputs (sinks)
    if not inputs:
        inputs = set(v for v in vertices if len(parent_map[v]) == 0)
    if not outputs:
        outputs = set(v for v in vertices if len(child_map[v]) == 0)
    
    # Count how many times each vertex's value is needed
    # A vertex's value is needed by each of its children
    # Also, output vertices need to have their value available at the end
    
    # Strategy: Greedy pebbling
    # Process vertices in topological order
    # Keep track of what's in fast memory (set of vertices whose values are there)
    # For each vertex, if its value is in fast memory, we can compute children
    # If not, we need to load it (or recompute)
    
    # Fast memory holds values. Initially, inputs are blue (in fast memory).
    # "Blue" means the value is available in fast memory.
    # "Red" means the value is in fast memory but... wait, let me re-read.
    
    # "Inputs start blue, all outputs must finish blue"
    # "every red value consumes one fast-memory slot"
    
    # I think:
    # - Blue = value is in fast memory (available for computation)
    # - Red = value is stored in fast memory but not actively being used? 
    # Actually, I think blue/red refers to pebbling state:
    # - Blue pebble = value is in fast memory
    # - The fast memory budget limits how many values can be in fast memory
    
    # Let me reconsider: In DAG pebbling:
    # - A vertex is "pebbled" if its value is in fast memory
    # - You can only compute a vertex if all its parents are pebbled
    # - After computing, you can delete parent pebbles to free space
    # - Transfers = loads (putting value into fast memory from slow) + stores (putting value back to slow)
    
    # Actually, for communication-minimizing pebbling:
    # - Slow memory has all values initially (or inputs are in slow memory)
    # - Fast memory has limited capacity
    # - Load: bring value from slow to fast (1 transfer)
    # - Store: put value from fast to slow (1 transfer)  
    # - Compute: compute a vertex from its parents (no transfer cost)
    # - Delete: remove value from fast memory (no transfer cost)
    # - Inputs start in fast memory (blue) - no load needed
    # - Outputs must end in fast memory (blue)
    
    # So the goal is to minimize total loads + stores.
    
    # Simple strategy: 
    # 1. Start with all inputs in fast memory
    # 2. Process vertices in topological order
    # 3. For each vertex, if all parents are in fast memory, compute it
    # 4. If fast memory is full, delete some vertices that won't be needed soon
    # 5. If a parent is not in fast memory, load it (or recompute if possible)
    
    # Better strategy: Use reference counting
    # For each vertex, count how many children still need its value
    # Delete vertices when their last child has been computed
    
    # Let's implement a greedy approach:
    
    # remaining_children[v] = number of children of v that haven't been computed yet
    remaining_children = {v: len(child_map[v]) for v in vertices}
    
    # fast_memory: set of vertices currently in fast memory
    fast_mem = set(inputs)
    
    # We need to track which vertices have been computed
    computed = set(inputs)  # Inputs are already available
    
    # Operations log
    ops = []
    
    # Process in topological order
    # For each vertex, ensure its parents are in fast memory, then compute it
    
    # Priority for deletion: vertices whose remaining_children is 0 can be deleted
    # Vertices with fewer remaining uses should be deleted first
    
    # Let's use a more sophisticated approach:
    # Pre-compute the "last use" of each vertex (the last child in topo order that needs it)
    
    # For each vertex v, last_use[v] = the position in topo_order of the last child of v
    # If v has no children, last_use[v] = its own position (for outputs) or infinity
    
    topo_pos = {v: i for i, v in enumerate(topo_order)}
    
    last_use = {}
    for v in vertices:
        if child_map[v]:
            last_use[v] = max(topo_pos[c] for c in child_map[v])
        else:
            if v in outputs:
                last_use[v] = n  # Must keep until end
            else:
                last_use[v] = topo_pos[v]  # Can delete after computing
    
    # Now process in topological order
    # At each step, we may need to free space by deleting vertices
    
    # Track what's in fast memory and their last use
    # When we need to add a new vertex and fast_mem is full, delete the vertex 
    # with the largest last_use (least urgent) among those not yet needed
    
    # Actually, we want to delete vertices that we won't need for the longest time
    # So delete the one with the LARGEST last_use (furthest in the future)
    
    # Wait, that's wrong. We want to keep vertices we need soon.
    # So delete vertices we need LATER (larger last_use = later = delete first)
    
    for v in topo_order:
        if time.time() > DEADLINE:
            break
        
        if v in inputs:
            # Already in fast memory
            continue
        
        # Check if all parents are in fast memory
        parents = parent_map[v]
        missing_parents = [p for p in parents if p not in fast_mem]
        
        if missing_parents:
            # Need to load missing parents
            for p in missing_parents:
                if time.time() > DEADLINE:
                    break
                # Check if we have space
                if len(fast_mem) >= fast_memory_budget:
                    # Need to delete something
                    # Find a vertex in fast_mem that is not a parent of v and has largest last_use
                    # Prefer deleting vertices not needed for v's computation
                    candidates = [x for x in fast_mem if x not in set(parents)]
                    if not candidates:
                        # All fast mem is parents, need to delete a parent (bad)
                        # Delete the parent with largest last_use
                        candidates = list(parents)
                    # Delete the one with largest last_use (least urgent)
                    del_v = max(candidates, key=lambda x: last_use.get(x, 0))
                    if del_v in fast_mem:
                        fast_mem.discard(del_v)
                        ops.append(['delete', del_v])
                
                # Load the parent
                fast_mem.add(p)
                ops.append(['load', p])
        
        # Now compute v
        if len(fast_mem) >= fast_memory_budget and v not in fast_mem:
            # Need space for v's result
            # Delete a vertex that's not a parent of v and has largest last_use
            candidates = [x for x in fast_mem if x not in set(parents) and x != v]
            if not candidates:
                candidates = [x for x in fast_mem if x != v]
            if candidates:
                del_v = max(candidates, key=lambda x: last_use.get(x, 0))
                fast_mem.discard(del_v)
                ops.append(['delete', del_v])
        
        fast_mem.add(v)
        computed.add(v)
        ops.append(['compute', v])
        
        # Update remaining_children for parents
        for p in parents:
            remaining_children[p] -= 1
    
    # Ensure outputs are in fast memory
    for v in outputs:
        if v not in fast_mem:
            if len(fast_mem) >= fast_memory_budget:
                # Delete something non-output
                candidates = [x for x in fast_mem if x not in outputs]
                if candidates:
                    del_v = max(candidates, key=lambda x: last_use.get(x, 0))
                    fast_mem.discard(del_v)
                    ops.append(['delete', del_v])
            fast_mem.add(v)
            ops.append(['load', v])
    
    return ops
# EVOLVE-BLOCK-END

