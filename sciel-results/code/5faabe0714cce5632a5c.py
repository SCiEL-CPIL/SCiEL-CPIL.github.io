import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85
    
    # Parse DAG
    if isinstance(dag, dict):
        parents = {v: list(p) for v, p in dag.items()}
    elif isinstance(dag, (list, tuple)):
        parents = {}
        for item in dag:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                v, p = item[0], item[1]
                parents[v] = list(p)
            elif isinstance(item, dict):
                for v, p in item.items():
                    parents[v] = list(p)
            else:
                parents[item] = []
    else:
        parents = {}
    
    all_vertices = set(parents.keys())
    
    if not all_vertices:
        return []
    
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
    
    outputs = [v for v in all_vertices if not children[v]]
    
    in_degree = {v: len(parents[v]) for v in all_vertices}
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(all_vertices):
        for v in all_vertices:
            if v not in topo_order:
                topo_order.append(v)
    
    schedule = []
    live = set()
    stored = set()
    
    for v in topo_order:
        if v not in live:
            if not parents[v]:
                schedule.append(('load', v))
                live.add(v)
            else:
                for p in parents[v]:
                    if p not in live:
                        schedule.append(('load', p))
                        live.add(p)
                schedule.append(('compute', v))
                live.add(v)
        
        if v in outputs:
            schedule.append(('store', v))
            live.discard(v)
            stored.add(v)
        else:
            if children[v] and len(live) < fast_memory_budget:
                pass
            else:
                schedule.append(('store', v))
                live.discard(v)
        
        if time.time() > deadline:
            break
    
    for v in outputs:
        if v not in stored:
            if v not in live:
                schedule.append(('load', v))
                live.add(v)
            schedule.append(('store', v))
            live.discard(v)
            stored.add(v)
    
    for v in list(live):
        schedule.append(('delete', v))
        live.discard(v)
    
    return schedule
