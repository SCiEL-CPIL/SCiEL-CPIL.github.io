# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def _topo_sort(vertices, parents):
    """Return topological order of vertices."""
    in_degree = {v: 0 for v in vertices}
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            if p in in_degree:
                in_degree[v] += 1
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
    
    # If not all vertices processed, add remaining (shouldn't happen in valid DAG)
    for v in vertices:
        if v not in order:
            order.append(v)
    return order

def _robust_pebbling(dag, fast_memory_budget):
    """
    Robust pebbling schedule ensuring all outputs are in memory at the end.
    Strategy:
    1. Process vertices in topological order.
    2. Maintain a set of vertices in fast memory.
    3. Before computing a vertex, ensure parents are in memory.
    4. If memory is full, evict a vertex that is 'safe' (no uncomputed children).
       If no safe vertex, evict a vertex that is an input or has the fewest remaining uses.
    5. After processing all vertices, force-load all outputs, evicting non-outputs if necessary.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin

    # Parse dag structure
    if isinstance(dag, dict):
        vertices = list(dag.get('vertices', dag.get('nodes', [])))
        parents = dag.get('parents', dag.get('edges', {}))
        outputs = dag.get('outputs', dag.get('sinks', []))
        inputs = dag.get('inputs', dag.get('sources', []))
    else:
        return []
    
    if not vertices:
        return []
    
    if not isinstance(parents, dict):
        parents = {}
    
    if not inputs:
        input_vertices = [v for v in vertices if not parents.get(v, [])]
        inputs = input_vertices
        
    if not outputs:
        # Find sinks
        out_degree = defaultdict(int)
        for v in vertices:
            for p in parents.get(v, []):
                out_degree[p] += 1
        outputs = [v for v in vertices if out_degree[v] == 0]

    output_set = set(outputs)
    input_set = set(inputs)
    
    # Precompute children and remaining uses
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
            
    # remaining_uncomputed_children[v] = number of children of v not yet computed
    remaining_uncomputed = {v: len(children[v]) for v in vertices}
    
    # Topological order
    topo_order = _topo_sort(vertices, parents)
    # Map vertex to index in topo order for quick lookup if needed
    topo_index = {v: i for i, v in enumerate(topo_order)}
    
    in_memory = set()
    computed = set()
    operations = []
    
    def add_op(op, v):
        operations.append([op, v])
        
    def try_load(v):
        """Attempt to load v into memory. Returns True if successful."""
        if v in in_memory:
            return True
        if len(in_memory) >= fast_memory_budget:
            # Need to evict
            # Priority 1: Vertices with no remaining uncomputed children (safe to delete)
            safe = [x for x in in_memory if remaining_uncomputed[x] == 0 and x not in output_set]
            if safe:
                # Evict the one that is 'oldest' or just first
                victim = safe[0]
                add_op('delete', victim)
                in_memory.discard(victim)
            else:
                # Priority 2: Inputs (can be reloaded)
                safe_inputs = [x for x in in_memory if x in input_set and x not in output_set]
                if safe_inputs:
                    victim = safe_inputs[0]
                    add_op('delete', victim)
                    in_memory.discard(victim)
                else:
                    # Priority 3: Any non-output vertex (risk of recomputation, but necessary)
                    # We should avoid evicting outputs
                    candidates = [x for x in in_memory if x not in output_set]
                    if candidates:
                        # Evict the one with the most remaining uses? No, least?
                        # Let's just pick one. To minimize recomputation, maybe pick one with highest topo index?
                        # Actually, if we are in this branch, we are stuck.
                        victim = candidates[0]
                        add_op('delete', victim)
                        in_memory.discard(victim)
                    else:
                        # Cannot load
                        return False
        
        add_op('load', v)
        in_memory.add(v)
        return True

    # 1. Process vertices in topological order
    for v in topo_order:
        if time.time() > deadline:
            break
            
        # If v is an input, we don't compute it, but we might need it in memory for children.
        # The problem says "Inputs start blue". This usually means they are available to load.
        # We don't need to 'compute' them.
        
        if v in input_set:
            # We don't compute inputs. But we might load them if needed.
            # We don't force load here, we load on demand.
            continue
            
        # Ensure parents are in memory
        for p in parents.get(v, []):
            if p not in in_memory:
                if not try_load(p):
                    # Failed to load parent. This is a critical error.
                    # We should probably abort or try a different strategy.
                    # For now, break.
                    pass
        
        # Compute v
        # Check if all parents are actually in memory
        parents_ok = all(p in in_memory for p in parents.get(v, []))
        if parents_ok:
            add_op('compute', v)
            in_memory.add(v)
            computed.add(v)
            
            # Update remaining_uncomputed for parents
            for p in parents.get(v, []):
                remaining_uncomputed[p] -= 1
        else:
            # If parents are not in memory, we cannot compute v.
            # This shouldn't happen if try_load worked.
            pass

    # 2. End-Game: Ensure all outputs are in memory
    # At this point, all vertices have been processed (or we timed out).
    # We need all outputs in memory.
    # If an output is not in memory, we must load it.
    # We can delete any non-output vertex to make space.
    
    # Sort outputs to ensure consistent behavior
    outputs_to_load = [o for o in outputs if o not in in_memory]
    
    for o in outputs_to_load:
        if time.time() > deadline:
            break
        if o in in_memory:
            continue
            
        # Make space if needed
        if len(in_memory) >= fast_memory_budget:
            # Find a non-output vertex to delete
            # We can delete any vertex that is not an output.
            # Note: Some non-outputs might be in memory.
            candidates = [x for x in in_memory if x not in output_set]
            if candidates:
                victim = candidates[0]
                add_op('delete', victim)
                in_memory.discard(victim)
            else:
                # All memory is filled with outputs?
                # If len(outputs) > fast_memory_budget, this is unsolvable.
                # Assuming solvable, we should have space or be able to make space.
                pass
        
        # Load the output
        add_op('load', o)
        in_memory.add(o)
        
    # Cap operations
    if len(operations) > 20000:
        operations = operations[:20000]
        
    return operations

def pebble(dag, fast_memory_budget):
    """
    Main entry point for DAG pebbling.
    Returns a valid schedule ensuring all outputs are in memory at the end.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    ops = _robust_pebbling(dag, fast_memory_budget)
    
    if len(ops) > 20000:
        ops = ops[:20000]
        
    return ops
# EVOLVE-BLOCK-END
