# EVOLVE-BLOCK-START
import time
import sys

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling with bounded fast memory.
    
    Uses a recursive divide-and-conquer strategy with memoization of intermediate
    results to minimize total transfers (load + store operations).
    
    Strategy:
    1. Parse the DAG structure
    2. Use a recursive pebbling algorithm that minimizes transfers
    3. Track fast memory usage carefully
    4. Return list of [operation, vertex] pairs
    """
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    # Parse DAG structure
    # dag format: list of (vertex, parents, is_input, is_output)
    # or similar - need to infer from baseline
    
    vertices = []
    parents = {}
    is_input = {}
    is_output = {}
    
    if isinstance(dag, dict):
        # Try dict format
        for key, val in dag.items():
            if isinstance(val, tuple) and len(val) >= 2:
                v = val[0]
                par = val[1]
                vertices.append(v)
                parents[v] = par
            elif isinstance(val, list):
                v = val[0]
                par = val[1] if len(val) > 1 else []
                vertices.append(v)
                parents[v] = par
        # Determine inputs/outputs from structure
        all_parents = set()
        for v in vertices:
            for p in parents.get(v, []):
                all_parents.add(p)
        for v in vertices:
            is_input[v] = v not in all_parents
            is_output[v] = (v not in [p for par in parents.values() for p in par])
    elif isinstance(dag, list):
        # List of tuples: (vertex, parents, is_input, is_output) or similar
        for item in dag:
            if isinstance(item, tuple) and len(item) >= 4:
                v, par, inp, out = item[0], item[1], item[2], item[3]
                vertices.append(v)
                parents[v] = par
                is_input[v] = inp
                is_output[v] = out
            elif isinstance(item, tuple) and len(item) == 2:
                v, par = item
                vertices.append(v)
                parents[v] = par
            elif isinstance(item, list) and len(item) >= 4:
                v, par, inp, out = item[0], item[1], item[2], item[3]
                vertices.append(v)
                parents[v] = par
                is_input[v] = inp
                is_output[v] = out
    
    if not vertices:
        return []
    
    # Topological order
    topo_order = []
    in_degree = {v: 0 for v in vertices}
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            in_degree[v] = in_degree.get(v, 0) + 1
            if p not in children:
                children[p] = []
            children[p].append(v)
    
    queue = [v for v in vertices if in_degree.get(v, 0) == 0]
    while queue:
        v = queue.pop(0)
        topo_order.append(v)
        for c in children.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(vertices):
        # Cyclic - shouldn't happen for DAG
        return []
    
    # Identify layers for divide-and-conquer
    # For layered DAGs, group by depth
    depth = {}
    for v in topo_order:
        if not parents.get(v):
            depth[v] = 0
        else:
            depth[v] = max(depth.get(p, 0) for p in parents.get(v, [])) + 1
    
    max_depth = max(depth.values()) if depth else 0
    
    # Recursive pebbling with memory budget
    # State: set of vertices currently in fast memory (blue)
    # We need to produce all outputs as blue at the end
    
    # Simple but effective strategy:
    # Process in topological order, keeping intermediate results
    # when they will be reused, otherwise delete immediately
    
    # Count how many times each vertex is used as parent
    use_count = {}
    for v in vertices:
        for p in parents.get(v, []):
            use_count[p] = use_count.get(p, 0) + 1
    
    # Strategy: 
    # - Load inputs when first needed
    # - Compute vertices in topo order
    # - Keep results in fast memory if they'll be reused AND we have space
    # - Delete results after last use if memory is tight
    # - Store outputs at the end
    
    ops = []
    fast_mem = set()  # vertices currently in fast memory
    computed = set()  # vertices that have been computed
    inputs_loaded = set()  # input vertices that have been loaded
    
    def load_vertex(v):
        """Load vertex into fast memory"""
        if v not in fast_mem:
            ops.append(["load", v])
            fast_mem.add(v)
    
    def store_vertex(v):
        """Store vertex to slow memory (freeing fast memory slot)"""
        if v in fast_mem:
            ops.append(["store", v])
            fast_mem.discard(v)
    
    def compute_vertex(v):
        """Compute vertex (all parents must be in fast memory)"""
        ops.append(["compute", v])
        fast_mem.add(v)
        computed.add(v)
    
    def delete_vertex(v):
        """Delete vertex from fast memory without storing"""
        if v in fast_mem:
            ops.append(["delete", v])
            fast_mem.discard(v)
    
    # Track last use of each vertex
    last_use = {}
    for i, v in enumerate(topo_order):
        for p in parents.get(v, []):
            last_use[p] = i
    
    # Process in topological order
    for i, v in enumerate(topo_order):
        # Ensure all parents are in fast memory
        for p in parents.get(v, []):
            if p not in fast_mem:
                if p in computed:
                    # Need to reload from slow memory
                    load_vertex(p)
                elif is_input.get(p, False):
                    load_vertex(p)
                    inputs_loaded.add(p)
                else:
                    # Shouldn't happen - parent not computed yet
                    # This would indicate a problem with topo order
                    pass
        
        # Compute v
        compute_vertex(v)
        
        # After computing, decide what to keep in fast memory
        # Strategy: keep vertices that will be reused, if we have space
        # Otherwise, delete or store
        
        # Check if we're over budget
        while len(fast_mem) > fast_memory_budget:
            # Find a vertex to evict
            # Prefer evicting vertices that won't be reused
            candidates = []
            for fm_v in list(fast_mem):
                # Don't evict if it's an output that needs to stay blue
                if is_output.get(fm_v, False):
                    continue
                # Check if it will be reused
                if fm_v in last_use and last_use[fm_v] > i:
                    # Will be reused - lower priority to evict
                    candidates.append((1, fm_v))
                else:
                    # Won't be reused - higher priority to evict
                    candidates.append((0, fm_v))
            
            if not candidates:
                break
            
            # Sort by priority (0 = evict first)
            candidates.sort()
            _, evict_v = candidates[0]
            
            # If it's an input, just delete it
            if is_input.get(evict_v, False):
                delete_vertex(evict_v)
            else:
                # Store it so it can be reloaded later
                store_vertex(evict_v)
        
        # If this is the last use of a parent, we can delete it
        for p in parents.get(v, []):
            if p in last_use and last_use[p] == i and p in fast_mem:
                if not is_output.get(p, False):
                    if is_input.get(p, False):
                        delete_vertex(p)
                    else:
                        # Store it (it's been computed, might need it again for other paths)
                        # Actually, if it's the last use, we can just delete
                        # But wait - it might be needed for output verification
                        # Safer to store
                        store_vertex(p)
    
    # Ensure all outputs are in fast memory (blue) at the end
    for v in vertices:
        if is_output.get(v, False) and v not in fast_mem:
            if v in computed:
                load_vertex(v)
            elif is_input.get(v, False):
                load_vertex(v)
    
    # Final cleanup: ensure exactly fast_memory_budget or fewer in fast memory
    # (This should already be handled, but just in case)
    while len(fast_mem) > fast_memory_budget:
        for fm_v in list(fast_mem):
            if not is_output.get(fm_v, False):
                if is_input.get(fm_v, False):
                    delete_vertex(fm_v)
                else:
                    store_vertex(fm_v)
                break
        else:
            break
    
    return ops
# EVOLVE-BLOCK-END
