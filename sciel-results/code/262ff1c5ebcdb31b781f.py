"""Communication-minimizing DAG pebbling with bounded fast memory."""
import time
import math
from collections import deque, defaultdict

# EVOLVE-BLOCK-START

def pebble(dag, fast_memory_budget):
    """
    Pebble a DAG with bounded fast memory to minimize communication (transfers).
    
    dag: dict or object with:
        - 'vertices' or 'nodes': list of vertex IDs (topologically ordered)
        - 'parents': dict mapping vertex -> list of parent vertices
        - 'inputs' or 'input_vertices': list of input vertices (start blue)
        - 'outputs' or 'output_vertices': list of output vertices (must finish blue)
        Or similar structure.
        
    fast_memory_budget: int, number of red values that can be held in fast memory.
    
    Returns: list of [operation, vertex] where operation is one of 'load', 'store', 'compute', 'delete'.
    """
    t0 = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill
    
    # --- Parse DAG structure ---
    try:
        # Try common attribute patterns
        if hasattr(dag, 'vertices'):
            vertices = list(dag.vertices)
        elif hasattr(dag, 'nodes'):
            vertices = list(dag.nodes)
        elif isinstance(dag, dict) and 'vertices' in dag:
            vertices = list(dag['vertices'])
        elif isinstance(dag, dict) and 'nodes' in dag:
            vertices = list(dag['nodes'])
        else:
            # Assume dag is a dict mapping vertex -> parents, or list
            if isinstance(dag, dict):
                vertices = list(dag.keys())
            else:
                vertices = list(dag)
        
        if hasattr(dag, 'parents'):
            parents = dict(dag.parents)
        elif isinstance(dag, dict) and 'parents' in dag:
            parents = dict(dag['parents'])
        elif isinstance(dag, dict):
            # dag itself might be the parents mapping
            parents = dict(dag)
            # Ensure all vertices are keys
            for v in vertices:
                if v not in parents:
                    parents[v] = []
        else:
            parents = {}
            for v in vertices:
                parents[v] = []
        
        if hasattr(dag, 'inputs'):
            inputs = list(dag.inputs)
        elif isinstance(dag, dict) and 'inputs' in dag:
            inputs = list(dag['inputs'])
        elif hasattr(dag, 'input_vertices'):
            inputs = list(dag.input_vertices)
        elif isinstance(dag, dict) and 'input_vertices' in dag:
            inputs = list(dag['input_vertices'])
        else:
            # Inputs are vertices with no parents
            inputs = [v for v in vertices if not parents.get(v, [])]
        
        if hasattr(dag, 'outputs'):
            outputs = list(dag.outputs)
        elif isinstance(dag, dict) and 'outputs' in dag:
            outputs = list(dag['outputs'])
        elif hasattr(dag, 'output_vertices'):
            outputs = list(dag.output_vertices)
        elif isinstance(dag, dict) and 'output_vertices' in dag:
            outputs = list(dag['output_vertices'])
        else:
            # Outputs are vertices with no children
            children = defaultdict(list)
            for v in vertices:
                for p in parents.get(v, []):
                    children[p].append(v)
            outputs = [v for v in vertices if not children.get(v, [])]
    except Exception:
        # Fallback: assume simple structure
        vertices = list(range(10))
        parents = {v: [] for v in vertices}
        inputs = [0]
        outputs = [9]
    
    # Build children map
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Topological order (vertices should already be topologically ordered)
    # But let's verify/create one
    topo_order = _topological_sort(vertices, parents)
    if not topo_order:
        topo_order = vertices  # fallback
    
    # --- Strategy: Layered pebbling with recomputation ---
    # For layered DAGs, optimal pebbling uses a divide-and-conquer approach.
    # We'll use a bottom-up approach: compute each vertex, store results,
    # and manage fast memory by evicting (storing to slow) when budget exceeded.
    
    # Count in-degrees for scheduling
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    
    # Determine which vertices are "leaves" (inputs) - they start blue
    # and which need to be computed.
    
    # Simple strategy: process in topological order, compute each vertex,
    # keep results in fast memory if budget allows, otherwise store.
    # When we need a parent that's been stored, load it back.
    
    # Better strategy for layered DAGs: use a recursive pebbling algorithm.
    # For now, use a greedy approach with LRU eviction.
    
    ops = []
    fast_mem = set()  # vertices currently in fast memory (red)
    slow_mem = set()  # vertices stored in slow memory
    computed = set()  # vertices that have been computed at least once
    
    # Inputs start blue, so they're "available" without loading
    # But they're blue, meaning they're in slow memory already.
    # When we need them, we load them.
    
    # Process vertices in topological order
    for v in topo_order:
        if time.time() - t0 > DEADLINE:
            break
        
        # Check if all parents are available in fast memory
        parents_of_v = parents.get(v, [])
        
        # Load any parents not in fast memory
        for p in parents_of_v:
            if p not in fast_mem and p not in computed:
                # Should not happen in topo order, but handle
                continue
            if p not in fast_mem:
                # Need to load from slow memory
                # Check if we have room
                if len(fast_mem) >= fast_memory_budget:
                    # Evict a vertex (LRU: evict the one with fewest remaining uses)
                    evict_v = _select_eviction(fast_mem, children, topo_order, v)
                    if evict_v is not None:
                        ops.append(['store', evict_v])
                        fast_mem.discard(evict_v)
                        slow_mem.add(evict_v)
                ops.append(['load', p])
                fast_mem.add(p)
                slow_mem.discard(p)
        
        # Now compute v
        ops.append(['compute', v])
        computed.add(v)
        
        # Decide whether to keep v in fast memory or store it immediately
        # Keep if budget allows, otherwise store
        if len(fast_mem) < fast_memory_budget:
            fast_mem.add(v)
        else:
            # Store immediately
            ops.append(['store', v])
            slow_mem.add(v)
            # Don't add to fast_mem
    
    # Ensure all outputs are in fast memory (blue) at the end
    # Actually, outputs must finish blue. Blue means in slow memory? 
    # Let me re-read: "Inputs start blue, all outputs must finish blue"
    # "every red value consumes one fast-memory slot"
    # So blue = slow memory, red = fast memory.
    # Outputs must be blue = in slow memory at the end.
    
    # So we need to store all outputs that are in fast memory
    for v in outputs:
        if v in fast_mem:
            ops.append(['store', v])
            fast_mem.discard(v)
            slow_mem.add(v)
    
    # Limit operations to 20000
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops


def _topological_sort(vertices, parents):
    """Return topological order of vertices."""
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    queue = deque([v for v in vertices if in_degree[v] == 0])
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(order) != len(vertices):
        return []  # Cycle detected
    return order


def _select_eviction(fast_mem, children, topo_order, current_v):
    """Select vertex to evict from fast memory using a heuristic."""
    # Count remaining uses for each vertex in fast_mem
    # A vertex's remaining use count is the number of times it appears
    # as a parent of a vertex not yet processed.
    
    # Build set of not-yet-processed vertices
    # This is expensive, so use a simpler heuristic: evict the vertex
    # with the fewest children that haven't been processed yet.
    
    # For simplicity, evict the vertex with the smallest topological position
    # that is not an input (inputs can be reloaded cheaply)
    
    # Actually, let's use: evict the vertex with the fewest remaining children
    # that need to be computed.
    
    # Build a set of processed vertices for quick lookup
    # This is O(n) each call, which is expensive. Let's use a simpler approach.
    
    # Evict the vertex that is furthest from current position in topo order
    # but has no remaining children in the unprocessed part.
    
    # Simplest: evict the one with the smallest index in topo_order (oldest)
    min_idx = float('inf')
    best = None
    for v in fast_mem:
        try:
            idx = topo_order.index(v)
        except ValueError:
            idx = 0
        if idx < min_idx:
            min_idx = idx
            best = v
    return best

# EVOLVE-BLOCK-END
