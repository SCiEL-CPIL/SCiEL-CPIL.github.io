# EVOLVE-BLOCK-START
import time
from collections import deque, defaultdict

def _topological_sort(vertices, parents):
    in_degree = {v: 0 for v in vertices}
    children = {v: [] for v in vertices}
    
    for v in vertices:
        if v in parents:
            for p in parents[v]:
                if p in in_degree:
                    in_degree[v] += 1
                    children[p].append(v)
    
    queue = deque([v for v in vertices if in_degree[v] == 0])
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(order) != len(vertices):
        return list(vertices)
    return order

def _pebbling_schedule(dag, fast_memory_budget):
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    order = _topological_sort(vertices, parents)
    
    # Build children map
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children_map[p].append(v)
    
    # remaining_uses[v] = number of children of v + 1 if v is output
    remaining_uses = {v: len(children_map[v]) + (1 if v in outputs else 0) for v in vertices}
    
    ops = []
    in_memory = set()
    
    def try_evict():
        """Evict a vertex with 0 remaining uses. Returns True if evicted."""
        nonlocal ops, in_memory, remaining_uses
        for v in list(in_memory):
            if remaining_uses[v] == 0:
                if v in outputs:
                    ops.append(['store', v])
                else:
                    ops.append(['delete', v])
                in_memory.remove(v)
                return True
        return False
    
    for v in order:
        is_input = not parents.get(v, [])
        
        if is_input:
            if v not in in_memory:
                while len(in_memory) >= fast_memory_budget:
                    if not try_evict():
                        break
                ops.append(['load', v])
                in_memory.add(v)
        else:
            # Ensure parents are in memory
            for p in parents[v]:
                if p not in in_memory:
                    while len(in_memory) >= fast_memory_budget:
                        if not try_evict():
                            break
                    ops.append(['load', p])
                    in_memory.add(p)
            
            ops.append(['compute', v])
            in_memory.add(v)
            
            # Decrement remaining_uses for parents
            for p in parents[v]:
                remaining_uses[p] -= 1
    
    # Final: store outputs, delete others
    for v in list(in_memory):
        if v in outputs:
            ops.append(['store', v])
        else:
            ops.append(['delete', v])
        in_memory.remove(v)
    
    return ops

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50.0
    
    try:
        schedule = _pebbling_schedule(dag, fast_memory_budget)
        
        if time.time() > deadline or len(schedule) > 20000:
            return _naive_fallback(dag, fast_memory_budget)
        
        return schedule
        
    except Exception:
        try:
            return _naive_fallback(dag, fast_memory_budget)
        except:
            return []

def _naive_fallback(dag, fast_memory_budget):
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    order = _topological_sort(vertices, parents)
    ops = []
    in_memory = set()
    
    for v in order:
        is_input = not parents.get(v, [])
        
        if is_input:
            if v not in in_memory:
                ops.append(['load', v])
                in_memory.add(v)
        else:
            for p in parents[v]:
                if p not in in_memory:
                    ops.append(['load', p])
                    in_memory.add(p)
            
            ops.append(['compute', v])
            in_memory.add(v)
    
    for v in in_memory:
        if v in outputs:
            ops.append(['store', v])
        else:
            ops.append(['delete', v])
    
    return ops

# EVOLVE-BLOCK-END
