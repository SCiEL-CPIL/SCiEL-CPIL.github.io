import time
import heapq
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    # Parse the DAG structure
    # dag is expected to have:
    #   'n' or 'num_vertices': number of vertices
    #   'parents' or 'edges': list of lists, where parents[i] are the parents of vertex i
    #   'outputs' or 'sinks': list of output vertex indices
    #   'inputs' or 'sources': list of input vertex indices (start blue)
    
    # Try to extract structure
    n = dag.get('n', dag.get('num_vertices', None))
    if n is None:
        parents_list = dag.get('parents', dag.get('edges', []))
        n = len(parents_list)
    
    parents_list = dag.get('parents', dag.get('edges', []))
    outputs = dag.get('outputs', dag.get('sinks', []))
    inputs = dag.get('inputs', dag.get('sources', []))
    
    if n is None:
        n = 0
    
    # Build adjacency
    children = defaultdict(list)
    in_degree = [0] * n
    parents = [list() for _ in range(n)]
    for i in range(n):
        if i < len(parents_list):
            for p in parents_list[i]:
                children[p].append(i)
                in_degree[i] += 1
                parents[i].append(p)
    
    # Identify inputs (sources) and outputs (sinks) if not given
    if not inputs:
        inputs = [i for i in range(n) if in_degree[i] == 0]
    if not outputs:
        outputs = [i for i in range(n) if len(children[i]) == 0]
    
    output_set = set(outputs)
    input_set = set(inputs)
    
    # Topological order
    topo_order = []
    queue = list(inputs)
    in_deg_copy = list(in_degree)
    while queue:
        queue.sort()
        v = queue.pop(0)
        topo_order.append(v)
        for c in children[v]:
            in_deg_copy[c] -= 1
            if in_deg_copy[c] == 0:
                queue.append(c)
    
    # If topo_order doesn't cover all nodes, add remaining
    if len(topo_order) < n:
        remaining = [v for v in range(n) if v not in set(topo_order)]
        topo_order.extend(remaining)
    
    # State tracking
    # fast_mem: set of vertices currently in fast memory (red pebbles)
    # The task says inputs start blue, meaning they are available (in memory or on disk but loadable)
    # "every red value consumes one fast-memory slot, including newly computed results"
    # So fast_memory_budget limits the number of vertices in fast memory at any time.
    
    # We need to produce a list of [operation, vertex]
    # Operations: load, store, compute, delete
    
    # Let's think about the schedule:
    # - Initially, inputs are "blue" (available). But do they occupy fast memory?
    #   The task says "Inputs start blue". Blue typically means available on disk, 
    #   and you need to "load" them to use them, OR they're already in memory.
    #   Actually, re-reading: "all outputs must finish blue; every red value consumes one fast-memory slot"
    #   So "blue" = on disk (not in fast memory), "red" = in fast memory.
    #   But "Inputs start blue" means they start on disk. To compute a vertex, its parents must be red (in fast memory).
    #   So we need to load inputs before computing.
    
    # Wait, let me reconsider. In pebbling:
    # - Black pebble = value available (can be used to compute children)
    # - Red pebble = value in fast memory (limited budget)
    # - White pebble = value not available
    # 
    # But the task uses "blue" and "red":
    # - "Inputs start blue" 
    # - "all outputs must finish blue"
    # - "every red value consumes one fast-memory slot"
    # 
    # So blue = available on disk (not in fast memory), red = in fast memory.
    # To compute a vertex, its parents must be... what? Available. 
    # I think the model is:
    # - A value can be on disk (blue) or in fast memory (red).
    # - To compute vertex v, all its parents must be in fast memory (red).
    # - Computing v puts v in fast memory (red).
    # - You can load a blue value into fast memory (if there's space), making it red.
    # - You can store a red value to disk, making it blue.
    # - You can delete a red value (freeing a slot).
    # - At the end, all outputs must be blue (on disk).
    
    # So the goal is to minimize the number of transfers (load + store operations).
    
    # Let's simulate a topological schedule:
    # - Process vertices in topological order.
    # - Before computing v, ensure all parents of v are in fast memory.
    # - If a parent is not in fast memory, load it (if on disk) or recompute it.
    # - Compute v, putting it in fast memory.
    # - If fast memory is full, we need to evict something (store or delete).
    # - At the end, store all outputs to disk (make them blue).
    
    # This is complex. Let's start with a simple baseline and try to improve.
    
    # For now, let's implement a simple topological schedule:
    # 1. Load all inputs into fast memory.
    # 2. For each vertex in topological order (skipping inputs):
    #    a. Ensure parents are in fast memory (load if needed).
    #    b. Compute the vertex.
    #    c. If fast memory exceeds budget, evict (store) some non-output vertex.
    # 3. Store all outputs.
    
    # But this might not be optimal. Let's try to be smarter about eviction.
    
    # Actually, let's just implement a correct baseline first.
    
    fast_mem = set()  # vertices in fast memory (red)
    on_disk = set()   # vertices on disk (blue)
    computed = set()  # vertices that have been computed at least once
    
    ops = []
    
    # Helper: check if we can add v to fast memory
    def can_add(v):
        return len(fast_mem) < fast_memory_budget
    
    # Helper: evict a vertex from fast memory
    # Strategy: evict a vertex that won't be needed again, or an input
    def evict():
        # Prioritize evicting vertices that are inputs or won't be needed
        # For simplicity, evict the first vertex in fast_mem that is not an output
        # and has already been used (or is an input)
        for v in list(fast_mem):
            if v not in output_set:
                # Check if v will be needed again
                # For now, just evict non-outputs
                fast_mem.discard(v)
                on_disk.add(v)
                ops.append(["store", v])
                return
        # If all are outputs, evict one (shouldn't happen if budget >= num_outputs)
        if fast_mem:
            v = next(iter(fast_mem))
            fast_mem.discard(v)
            on_disk.add(v)
            ops.append(["store", v])
    
    # Step 1: Load all inputs
    for v in inputs:
        if can_add(v):
            fast_mem.add(v)
            on_disk.discard(v)
            ops.append(["load", v])
        else:
            evict()
            fast_mem.add(v)
            on_disk.discard(v)
            ops.append(["load", v])
    
    # Step 2: Compute non-input vertices in topological order
    for v in topo_order:
        if v in input_set:
            continue
        
        # Ensure all parents are in fast memory
        for p in parents[v]:
            if p not in fast_mem:
                if can_add(p):
                    fast_mem.add(p)
                    on_disk.discard(p)
                    ops.append(["load", p])
                else:
                    evict()
                    fast_mem.add(p)
                    on_disk.discard(p)
                    ops.append(["load", p])
        
        # Compute v
        if can_add(v):
            fast_mem.add(v)
            computed.add(v)
            ops.append(["compute", v])
        else:
            evict()
            fast_mem.add(v)
            computed.add(v)
            ops.append(["compute", v])
    
    # Step 3: Store all outputs
    for v in outputs:
        if v in fast_mem:
            fast_mem.discard(v)
            on_disk.add(v)
            ops.append(["store", v])
    
    return ops

# EVOLVE-BLOCK-START
def pebble_wrapper(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85.0  # 85 second deadline
    
    try:
        result = pebble(dag, fast_memory_budget)
        if time.time() > deadline:
            # Return best result found
            return result
        return result
    except Exception as e:
        # Fallback: return empty list or simple schedule
        return []
# EVOLVE-BLOCK-END

# Override the pebble function to use the wrapper
pebble = pebble_wrapper
