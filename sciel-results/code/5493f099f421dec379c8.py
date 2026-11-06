"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Solve communication-minimizing DAG pebbling.
    
    dag: dict with keys:
        - 'vertices': list of vertex IDs
        - 'edges': list of (parent, child) pairs
        - 'inputs': list of input vertex IDs (start blue)
        - 'outputs': list of output vertex IDs (must finish blue)
        - 'parents': dict mapping vertex -> list of parent vertices (topologically ordered)
    
    fast_memory_budget: int, max number of red values that can be in fast memory
    
    Returns: list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'
    """
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s limit
    
    # Parse DAG structure
    vertices = dag.get('vertices', [])
    n = len(vertices)
    if n == 0:
        return []
    
    # Build parent/children maps
    parents_map = dag.get('parents', {})
    if not parents_map:
        # Build from edges if parents not given
        edges = dag.get('edges', [])
        parents_map = defaultdict(list)
        for parent, child in edges:
            parents_map[child].append(parent)
    
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
    
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    # Topological Sort
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
    
    if len(topo_order) != n:
        # Cycle or error, return empty
        return []
        
    # State:
    # Blue = Slow Memory (Unlimited)
    # Red = Fast Memory (Limited)
    # Inputs start Blue. Outputs must finish Blue.
    
    slow_set = set(inputs)
    fast_set = set()
    ops = []
    
    # remaining_children: count of children not yet processed
    remaining_children = {v: len(children_map.get(v, [])) for v in vertices}
    
    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            break
        
        if v in inputs:
            # Inputs are already in slow_set.
            # They will be loaded when needed by children.
            continue
        
        # 1. Ensure all parents are in fast_set (Red)
        for p in parents_map.get(v, []):
            if p not in fast_set:
                if p in slow_set:
                    ops.append(['load', p])
                    fast_set.add(p)
                else:
                    # This state should not occur if eviction logic is correct
                    # If it does, we cannot compute v.
                    pass
        
        # 2. Manage Fast Memory Space
        # We need 1 slot for v.
        # If fast_set is full, we must evict.
        while len(fast_set) >= fast_memory_budget:
            # Find best victim to evict
            # Priority:
            # 0: Not output, remaining_children == 0 (Safe to delete)
            # 1: Not output, remaining_children > 0 (Must store)
            # 2: Output (Must store, but avoid if possible)
            
            candidates = []
            for u in fast_set:
                if u in outputs:
                    candidates.append((2, u))
                elif remaining_children[u] == 0:
                    candidates.append((0, u))
                else:
                    candidates.append((1, u))
            
            if not candidates:
                break # Should not happen
            
            candidates.sort()
            priority, victim = candidates[0]
            
            if priority == 0:
                # Safe to delete (no future use)
                ops.append(['delete', victim])
                fast_set.remove(victim)
            else:
                # Must store (future use or output)
                ops.append(['store', victim])
                slow_set.add(victim)
                fast_set.remove(victim)
        
        # 3. Compute v
        ops.append(['compute', v])
        fast_set.add(v)
        
        # 4. Update remaining_children for parents
        # Since v is now processed, its parents have one less remaining child
        for p in parents_map.get(v, []):
            remaining_children[p] -= 1
    
    # 5. Finalize: Ensure all outputs are in slow_set (Blue)
    for out in outputs:
        if out not in slow_set:
            if out in fast_set:
                ops.append(['store', out])
                slow_set.add(out)
                fast_set.remove(out)
            else:
                # Should not happen
                pass
    
    return ops

# EVOLVE-BLOCK-END
