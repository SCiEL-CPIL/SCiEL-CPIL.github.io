"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Minimize communication in DAG pebbling.
    
    dag: dict with 'vertices' (topo-ordered), 'parents', 'outputs'
    fast_memory_budget: max total values (blue + red) allowed
    
    Returns: list of [op, vertex] where op in {load, store, compute, delete}
    """
    start_time = time.time()
    DEADLINE = 85.0
    
    vertices = dag['vertices']
    parents = dag.get('parents', {})
    outputs = set(dag.get('outputs', []))
    n = len(vertices)
    
    if n == 0:
        return []
    
    # Build reverse edges
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Reference count: how many children still need this value
    rem_refs = {v: len(children.get(v, [])) for v in vertices}
    
    # Inputs
    inputs = [v for v in vertices if not parents.get(v)]
    
    # State
    blue = set(inputs)  # in fast memory
    red = set()          # in slow memory  
    computed = set(inputs)
    
    ops = []
    
    def mem_used():
        return len(blue) + len(red)
    
    # Topological processing
    for v in vertices:
        if time.time() - start_time > DEADLINE:
            break
        if v in inputs:
            continue
        
        # Load parents into fast memory if needed
        for p in parents.get(v, []):
            if p not in blue:
                if p in red:
                    ops.append(['load', p])
                    red.discard(p)
                    blue.add(p)
        
        # Compute
        ops.append(['compute', v])
        blue.add(v)
        computed.add(v)
        
        # Decrement ref counts for parents
        for p in parents.get(v, []):
            rem_refs[p] -= 1
        
        # Free memory if needed
        while mem_used() >= fast_memory_budget:
            # Find deletable: not output, no remaining refs, in blue
            deletable = [u for u in blue 
                        if u not in outputs and rem_refs.get(u, 0) <= 0]
            if not deletable:
                # Try storing some blue to red to free blue slots
                # But that doesn't reduce total memory...
                # Actually we need to reduce total memory
                # If we can't delete, we're stuck - but this shouldn't happen
                # if the DAG is pebbleable with this budget
                break
            # Delete the oldest deletable (lowest topo index)
            idx = {v: i for i, v in enumerate(vertices)}
            deletable.sort(key=lambda u: idx.get(u, 0))
            u = deletable[0]
            ops.append(['delete', u])
            blue.discard(u)
    
    # Final: ensure outputs are blue
    for o in outputs:
        if o in red:
            ops.append(['load', o])
            red.discard(o)
            blue.add(o)
    
    return ops

# EVOLVE-BLOCK-END
