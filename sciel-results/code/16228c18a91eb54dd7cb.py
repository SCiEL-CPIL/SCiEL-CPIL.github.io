"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Compute a pebbling schedule for a DAG minimizing transfers.
    
    dag: dict with keys:
        - 'vertices': list of vertex IDs (topologically ordered)
        - 'parents': dict mapping vertex -> list of parent vertices
        - 'outputs': list of output vertex IDs
    fast_memory_budget: int, max number of red pebbles
    
    Returns: list of [operation, vertex] where operation is 'load', 'store', 'compute', 'delete'
    """
    deadline = time.time() + 55  # safety margin before 60s limit
    
    # Parse DAG structure
    vertices = dag.get('vertices', [])
    parents = dag.get('parents', {})
    outputs = set(dag.get('outputs', []))
    num_vertices = len(vertices)
    
    if num_vertices == 0:
        return []
    
    # Build children map
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Identify inputs (no parents)
    inputs = [v for v in vertices if len(parents.get(v, [])) == 0]
    
    # State
    red = set()      # vertices in fast memory
    blue = set(inputs)  # inputs start in blue
    schedule = []
    
    # Track remaining children
    remaining_children = {v: len(children[v]) for v in vertices}
    
    def is_still_needed(v):
        return remaining_children[v] > 0
    
    def evict_one():
        if not red:
            return
        # Priority 1: not needed, not output
        evictable = [v for v in red if not is_still_needed(v) and v not in outputs]
        if evictable:
            v = evictable[0]
            red.discard(v)
            schedule.append(['delete', v])
            return
        # Priority 2: output (store)
        evictable = [v for v in red if v in outputs]
        if evictable:
            v = evictable[0]
            red.discard(v)
            blue.add(v)
            schedule.append(['store', v])
            return
        # Priority 3: least needed
        v = min(red, key=lambda x: remaining_children[x])
        red.discard(v)
        schedule.append(['delete', v])
    
    def make_room():
        while len(red) >= fast_memory_budget and fast_memory_budget > 0:
            evict_one()
    
    # Process in topological order
    for v in vertices:
        if time.time() > deadline:
            break
        
        # Ensure parents are in red
        for p in parents.get(v, []):
            if p not in red:
                if p in blue:
                    make_room()
                    schedule.append(['load', p])
                    red.add(p)
                    blue.discard(p)
                else:
                    # Parent not available - recompute would be needed
                    # Skip for now (assume budget is sufficient)
                    pass
        
        # Compute v
        if v not in red:
            make_room()
            schedule.append(['compute', v])
            red.add(v)
        
        # Update remaining children
        for p in parents.get(v, []):
            remaining_children[p] -= 1
    
    # Store all outputs to blue
    for v in outputs:
        if time.time() > deadline:
            break
        if v in red:
            schedule.append(['store', v])
            red.discard(v)
            blue.add(v)
    
    # Clean up
    for v in list(red):
        if time.time() > deadline:
            break
        if v not in outputs:
            schedule.append(['delete', v])
            red.discard(v)
    
    return schedule
# EVOLVE-BLOCK-END
