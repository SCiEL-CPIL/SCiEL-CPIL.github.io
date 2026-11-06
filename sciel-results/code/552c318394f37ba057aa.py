import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = start_time + 55
    
    # Parse DAG
    if hasattr(dag, 'vertices'):
        vertices = list(dag.vertices)
    elif isinstance(dag, dict) and 'vertices' in dag:
        vertices = list(dag['vertices'])
    elif isinstance(dag, (list, tuple)):
        vertices = list(dag)
    else:
        for attr in ['vertices', 'nodes', 'V']:
            if hasattr(dag, attr):
                vertices = list(getattr(dag, attr))
                break
            elif isinstance(dag, dict) and attr in dag:
                vertices = list(dag[attr])
                break
        else:
            raise ValueError("Cannot extract vertices from dag")
    
    n = len(vertices)
    
    if hasattr(dag, 'parents'):
        parents = dict(dag.parents) if not isinstance(dag.parents, dict) else dag.parents
    elif isinstance(dag, dict) and 'parents' in dag:
        parents = dag['parents']
    else:
        parents = {}
    
    if not isinstance(parents, dict):
        if isinstance(parents, (list, tuple)):
            parents = {v: list(parents[i]) for i, v in enumerate(vertices)}
        else:
            parents = {v: [] for v in vertices}
    
    for v in vertices:
        if v not in parents:
            parents[v] = []
    
    if hasattr(dag, 'inputs'):
        inputs = set(dag.inputs)
    elif isinstance(dag, dict) and 'inputs' in dag:
        inputs = set(dag['inputs'])
    else:
        inputs = set(v for v in vertices if not parents[v])
    
    if hasattr(dag, 'outputs'):
        outputs = set(dag.outputs)
    elif isinstance(dag, dict) and 'outputs' in dag:
        outputs = set(dag['outputs'])
    else:
        children = defaultdict(set)
        for v in vertices:
            for p in parents[v]:
                children[p].add(v)
        outputs = set(v for v in vertices if not children[v])
    
    # Compute topological order
    in_degree = {v: len(parents[v]) for v in vertices}
    children = defaultdict(list)
    for v in vertices:
        for p in parents[v]:
            children[p].append(v)
    
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo = []
    while queue:
        v = queue.popleft()
        topo.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # Compute last_use
    last_use = {v: -1 for v in vertices}
    for i, v in enumerate(topo):
        for p in parents[v]:
            last_use[p] = i
    
    # If budget is large enough, do simple topological pass
    if fast_memory_budget >= n:
        ops = []
        in_fast = set()
        
        for v in sorted(inputs):
            ops.append(['load', v])
            in_fast.add(v)
        
        for i, v in enumerate(topo):
            if v in inputs:
                continue
            for p in parents[v]:
                if p not in in_fast:
                    ops.append(['load', p])
                    in_fast.add(p)
            ops.append(['compute', v])
            in_fast.add(v)
            
            for p in parents[v]:
                if p not in outputs and last_use[p] < i:
                    ops.append(['delete', p])
                    in_fast.remove(p)
        
        for v in sorted(outputs):
            if v in in_fast:
                ops.append(['store', v])
                in_fast.remove(v)
        
        return ops
    
    # Memory-constrained: greedy with smart eviction
    ops = []
    in_fast = set()
    
    for i, v in enumerate(topo):
        if time.time() > DEADLINE:
            break
        
        if v in inputs:
            if v not in in_fast:
                if len(in_fast) >= fast_memory_budget:
                    evict_v = min(in_fast, key=lambda x: last_use[x])
                    ops.append(['store', evict_v])
                    in_fast.remove(evict_v)
                ops.append(['load', v])
                in_fast.add(v)
            continue
        
        for p in parents[v]:
            if p not in in_fast:
                if len(in_fast) >= fast_memory_budget:
                    evict_v = min(in_fast, key=lambda x: last_use[x])
                    ops.append(['store', evict_v])
                    in_fast.remove(evict_v)
                ops.append(['load', p])
                in_fast.add(p)
        
        ops.append(['compute', v])
        in_fast.add(v)
        
        for p in parents[v]:
            if p in in_fast and p not in outputs and last_use[p] < i:
                ops.append(['delete', p])
                in_fast.remove(p)
    
    for v in sorted(outputs):
        if v in in_fast:
            ops.append(['store', v])
            in_fast.remove(v)
        else:
            ops.append(['load', v])
            in_fast.add(v)
            ops.append(['store', v])
            in_fast.remove(v)
    
    return ops[:20000]
