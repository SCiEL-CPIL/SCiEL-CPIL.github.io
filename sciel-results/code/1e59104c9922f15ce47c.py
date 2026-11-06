"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Solve DAG pebbling to minimize transfers under fast memory budget.
    
    dag: dict with keys 'vertices', 'parents' (topologically ordered), 'inputs', 'outputs'
    fast_memory_budget: int, max red values in fast memory at once
    
    Returns: list of [operation, vertex]
    """
    start_time = time.time()
    deadline = start_time + 80
    
    # Extract DAG structure robustly
    if isinstance(dag, dict):
        vertices = list(dag.get('vertices', []))
        parents_map = dict(dag.get('parents', {}))
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
    else:
        vertices = list(getattr(dag, 'vertices', []))
        parents_map = dict(getattr(dag, 'parents', {}))
        inputs = set(getattr(dag, 'inputs', []))
        outputs = set(getattr(dag, 'outputs', []))
    
    n = len(vertices)
    if n == 0:
        return []
    
    # Build children map
    children = defaultdict(list)
    for v in vertices:
        ps = parents_map.get(v, [])
        if isinstance(ps, (list, tuple)):
            for p in ps:
                children[p].append(v)
    
    # Identify sources
    has_parents = set()
    for v in vertices:
        ps = parents_map.get(v, [])
        if ps:
            has_parents.add(v)
    
    if inputs:
        sources = inputs
    else:
        sources = set(v for v in vertices if v not in has_parents)
    
    topo_order = vertices
    
    # Count how many times each vertex is used as a parent
    parent_count = defaultdict(int)
    for v in vertices:
        ps = parents_map.get(v, [])
        if isinstance(ps, (list, tuple)):
            for p in ps:
                parent_count[p] += 1
    
    remaining_uses = dict(parent_count)
    for v in vertices:
        if v not in remaining_uses:
            remaining_uses[v] = 0
    
    ops = []
    blue = set(sources)
    computed = set(sources)
    
    def non_input_blue():
        return len(blue - sources)
    
    def can_store(v):
        if v in sources or v in outputs:
            return False
        if remaining_uses.get(v, 0) > 0:
            return False
        return True
    
    def free_space_if_needed():
        while non_input_blue() > fast_memory_budget:
            stored = False
            for b in list(blue):
                if can_store(b):
                    ops.append(['store', b])
                    blue.remove(b)
                    stored = True
                    break
            if not stored:
                for b in list(blue):
                    if b not in sources:
                        ops.append(['store', b])
                        blue.remove(b)
                        break
                if non_input_blue() > fast_memory_budget:
                    break
    
    for v in topo_order:
        if v in computed:
            continue
        
        ps = parents_map.get(v, [])
        if isinstance(ps, (list, tuple)):
            ps_list = list(ps)
        elif ps is not None:
            ps_list = [ps]
        else:
            ps_list = []
        
        for p in ps_list:
            if p not in blue:
                free_space_if_needed()
                ops.append(['load', p])
                blue.add(p)
        
        ops.append(['compute', v])
        blue.add(v)
        computed.add(v)
        
        for p in ps_list:
            if p in remaining_uses:
                remaining_uses[p] -= 1
        
        for p in ps_list:
            if can_store(p) and p in blue:
                ops.append(['store', p])
                blue.remove(p)
    
    for o in outputs:
        if o not in blue:
            free_space_if_needed()
            ops.append(['load', o])
            blue.add(o)
    
    return ops
# EVOLVE-BLOCK-END
