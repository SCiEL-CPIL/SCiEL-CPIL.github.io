"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag.get('outputs', []))
    inputs = set(dag.get('inputs', []))
    
    n = len(vertices)
    if n == 0:
        return []
    
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    in_fast = set()
    in_slow = set(inputs.copy())
    computed = set()
    ops = []
    
    def load(v):
        if v in in_fast:
            return
        ops.append(['load', v])
        in_fast.add(v)
    
    def store(v):
        if v not in in_fast:
            return
        ops.append(['store', v])
        in_fast.discard(v)
        in_slow.add(v)
    
    def delete(v):
        if v in in_fast:
            ops.append(['delete', v])
            in_fast.discard(v)
    
    # Precompute: for each vertex, how many times it will be needed (fan-out)
    # and whether it's an input/output
    
    for v in vertices:
        # Load parents if not in fast
        for p in parents.get(v, []):
            if p not in in_fast:
                load(p)
        
        # Compute v
        ops.append(['compute', v])
        in_fast.add(v)
        computed.add(v)
        
        # Check if v needs to be kept for future use
        remaining_children = [c for c in children[v] if c not in computed]
        
        if remaining_children:
            # Will be needed again, store to slow memory
            store(v)
        else:
            # Last use
            if v in outputs:
                # Must end in slow memory
                if v in in_fast:
                    store(v)
            else:
                # Can delete
                if v in in_fast:
                    delete(v)
        
        # Manage memory: if over budget, evict
        while len(in_fast) > fast_memory_budget:
            # Evict vertex with no remaining children, prefer non-outputs
            evicted = False
            for x in list(in_fast):
                rem = [c for c in children[x] if c not in computed]
                if not rem and x not in outputs:
                    delete(x)
                    evicted = True
                    break
            if not evicted:
                for x in list(in_fast):
                    if x not in outputs:
                        delete(x)
                        evicted = True
                        break
            if not evicted and in_fast:
                delete(next(iter(in_fast)))
    
    # Ensure outputs are in slow memory
    for v in outputs:
        if v in in_fast:
            store(v)
    
    return ops
# EVOLVE-BLOCK-END
