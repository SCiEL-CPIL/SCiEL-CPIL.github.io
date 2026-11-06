"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Solve the communication-minimizing DAG pebbling problem.
    Returns list of [operation, vertex].
    """
    start_time = time.time()
    DEADLINE = 85.0
    
    vertices = dag.get('vertices', [])
    parents = dag.get('parents', {})
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    topo_order = dag.get('topo_order', [])
    
    if not vertices:
        return []
    
    # Build children map
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Count fan-out
    fan_out = {v: len(children.get(v, [])) for v in vertices}
    
    # State tracking
    in_memory = set()
    stored = set()
    computed = set()
    operations = []
    mem_size = 0
    
    def try_evict(prefer_not=None):
        """Evict the least useful item from memory."""
        nonlocal mem_size
        candidates = [x for x in in_memory if x != prefer_not]
        if not candidates:
            return False
        # Prefer to evict items that are no longer needed
        def score(x):
            # Lower score = more likely to evict
            if x in stored:
                return -1000
            if x in outputs:
                return 1000  # Keep outputs
            # Count uncomputed children
            uncomputed_children = sum(1 for c in children.get(x, []) if c not in computed)
            return uncomputed_children
        min_score = min(score(x) for x in candidates)
        to_evict = [x for x in candidates if score(x) == min_score][0]
        operations.append(['delete', to_evict])
        in_memory.discard(to_evict)
        mem_size -= 1
        return True
    
    for v in topo_order:
        if time.time() > DEADLINE:
            break
        
        # Ensure all parents are in memory
        for p in parents.get(v, []):
            if p not in in_memory:
                if mem_size >= fast_memory_budget:
                    if not try_evict(prefer_not=p):
                        continue
                if p in inputs:
                    operations.append(['load', p])
                    in_memory.add(p)
                    mem_size += 1
                else:
                    # p was computed before, need to recompute - complex
                    # For now, skip (this may cause issues)
                    pass
        
        # Compute v if it has parents
        if parents.get(v, []) and v not in computed:
            all_parents_in = all(p in in_memory for p in parents.get(v, []))
            if all_parents_in:
                operations.append(['compute', v])
                computed.add(v)
                if v not in in_memory:
                    in_memory.add(v)
                    mem_size += 1
        
        # Store v if it's an output
        if v in outputs:
            if v in in_memory:
                operations.append(['store', v])
                stored.add(v)
                in_memory.discard(v)
                mem_size -= 1
            else:
                # Need to get v into memory first
                if v in inputs:
                    if mem_size >= fast_memory_budget:
                        try_evict(prefer_not=v)
                    operations.append(['load', v])
                    in_memory.add(v)
                    mem_size += 1
                    operations.append(['store', v])
                    stored.add(v)
                    in_memory.discard(v)
                    mem_size -= 1
    
    return operations
# EVOLVE-BLOCK-END
