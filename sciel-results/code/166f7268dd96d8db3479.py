# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    Strategy: Layer-by-layer with selective caching of high-reuse intermediates.
    """
    t0 = time.time()
    DEADLINE = 75.0
    
    # Extract structure
    vertices = []
    parents = {}
    outputs = set()
    
    if hasattr(dag, 'vertices'):
        vertices = list(dag.vertices)
    elif hasattr(dag, 'nodes'):
        vertices = list(dag.nodes)
    elif isinstance(dag, dict):
        vertices = list(dag.keys())
    elif isinstance(dag, (list, tuple)):
        vertices = list(dag)
    
    n = len(vertices)
    if n == 0:
        return []
    
    # Parents
    if hasattr(dag, 'parents'):
        p_attr = dag.parents
        if callable(p_attr):
            for v in vertices:
                try:
                    parents[v] = list(p_attr(v))
                except:
                    parents[v] = []
        elif isinstance(p_attr, dict):
            for v in vertices:
                parents[v] = list(p_attr.get(v, []))
        else:
            for i, v in enumerate(vertices):
                parents[v] = list(p_attr[i]) if i < len(p_attr) else []
    elif isinstance(dag, dict):
        for v in vertices:
            if v in dag and isinstance(dag[v], dict):
                parents[v] = list(dag[v].get('parents', []))
            elif v in dag and isinstance(dag[v], (list, tuple)):
                parents[v] = list(dag[v])
            else:
                parents[v] = []
    else:
        for v in vertices:
            parents[v] = []
    
    # Outputs
    if hasattr(dag, 'outputs'):
        outputs = set(dag.outputs)
    elif isinstance(dag, dict) and 'outputs' in dag:
        outputs = set(dag['outputs'])
    else:
        # No children = output
        children = defaultdict(set)
        for v in vertices:
            for p in parents.get(v, []):
                children[p].add(v)
        outputs = set(v for v in vertices if not children.get(v, set()))
    
    inputs = set(v for v in vertices if not parents.get(v, []))
    topo = list(vertices)
    
    # Fan-out
    fan_out = defaultdict(int)
    for v in vertices:
        for p in parents.get(v, []):
            fan_out[p] += 1
    
    # Last use index
    last_use = {v: i for i, v in enumerate(topo)}
    for i, v in enumerate(topo):
        for p in parents.get(v, []):
            if i > last_use.get(p, -1):
                last_use[p] = i
    
    B = fast_memory_budget
    if B <= 0:
        ops = []
        for v in topo:
            for p in parents.get(v, []):
                ops.append(['load', p])
            ops.append(['compute', v])
            if v in outputs:
                ops.append(['store', v])
            elif v not in inputs:
                ops.append(['delete', v])
        return ops[:20000]
    
    # Generate schedule
    in_fast = set()
    ops = []
    cnt = 0
    MAX_OPS = 20000
    
    for i, v in enumerate(topo):
        if time.time() - t0 > DEADLINE:
            break
        if cnt >= MAX_OPS:
            break
        
        # Load missing parents
        for p in parents.get(v, []):
            if p not in in_fast:
                ops.append(['load', p])
                in_fast.add(p)
                cnt += 1
                if cnt >= MAX_OPS:
                    break
        if cnt >= MAX_OPS:
            break
        
        # Compute
        ops.append(['compute', v])
        in_fast.add(v)
        cnt += 1
        
        # Evict if over budget
        while len(in_fast) > B:
            # Find victim: not output, lowest remaining-use priority
            candidates = [x for x in in_fast if x not in outputs]
            if not candidates:
                break
            # Priority: last_use - current_idx (lower = more urgent to evict)
            victim = min(candidates, key=lambda x: last_use.get(x, i) - i)
            ops.append(['delete', victim])
            in_fast.discard(victim)
            cnt += 1
            if cnt >= MAX_OPS:
                break
        
        # Store output
        if v in outputs:
            ops.append(['store', v])
            cnt += 1
    
    return ops
# EVOLVE-BLOCK-END
