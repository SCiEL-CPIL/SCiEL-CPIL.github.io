"""Rewritten program for communication-minimizing DAG pebbling."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85.0
    
    vertices = list(dag.get('vertices', []))
    parents = dag.get('parents', {})
    inputs = list(dag.get('inputs', []))
    outputs = list(dag.get('outputs', []))
    
    if not vertices:
        return []
    
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    queue = [v for v in vertices if in_degree[v] == 0]
    topo = []
    while queue:
        v = queue.pop(0)
        topo.append(v)
        for c in children.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    if len(topo) != len(vertices):
        topo = list(vertices)
    
    input_set = set(inputs)
    output_set = set(outputs)
    
    # Count how many times each vertex's value is needed
    # (number of children + 1 if output)
    need_count = defaultdict(int)
    for v in vertices:
        need_count[v] = len(children.get(v, []))
    for v in outputs:
        need_count[v] += 1
    
    ops = []
    fast = set()
    slow = set()
    
    for v in inputs:
        slow.add(v)
    
    def do_load(v):
        ops.append(['load', v])
        slow.discard(v)
        fast.add(v)
    
    def do_store(v):
        ops.append(['store', v])
        fast.discard(v)
        slow.add(v)
    
    def do_delete(v):
        ops.append(['delete', v])
        fast.discard(v)
    
    def evict_if_full():
        if len(fast) <= fast_memory_budget:
            return
        # Evict non-output with lowest need_count
        cands = [x for x in fast if x not in output_set]
        if not cands:
            cands = list(fast)
        if cands:
            v_out = min(cands, key=lambda x: need_count.get(x, 0))
            do_store(v_out)
    
    for v in topo:
        if time.time() - start_time > DEADLINE:
            break
        if v in input_set:
            continue
        
        # Load parents into fast if not there
        for p in parents.get(v, []):
            if p not in fast:
                if p in slow:
                    do_load(p)
                    evict_if_full()
        
        # Compute
        ops.append(['compute', v])
        fast.add(v)
        evict_if_full()
        
        # If this is the last time v is needed, store it
        # (except outputs, which we store at the end)
        if v not in output_set:
            # Check if all children have been processed
            # Actually, we process in topo order, so children come after
            # We can't know if it's the last use until we process children
            # So we keep it in fast for now; evict_if_full handles budget
            pass
    
    # Store all outputs
    for v in outputs:
        if v in fast:
            do_store(v)
        elif v not in slow:
            # Shouldn't happen
            if v in fast:
                do_store(v)
    
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
# EVOLVE-BLOCK-END
