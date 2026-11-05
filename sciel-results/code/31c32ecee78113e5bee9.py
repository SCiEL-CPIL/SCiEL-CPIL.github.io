import time
from collections import deque
import sys

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85  # 85s out of 90s, safety margin
    MAX_OPS = 20000
    
    # Increase recursion limit
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(10000)
    
    try:
        # Parse the DAG
        parents = None
        outputs = None
        
        if isinstance(dag, dict):
            parents = dag.get('parents', dag.get('parent', None))
            outputs = dag.get('outputs', dag.get('output', None))
        else:
            parents = getattr(dag, 'parents', getattr(dag, 'parent', None))
            outputs = getattr(dag, 'outputs', getattr(dag, 'output', None))
        
        if parents is None:
            parents = list(dag)
            outputs = []
        
        n = len(parents)
        
        # Determine inputs: vertices with no parents
        inputs = [i for i in range(n) if len(parents[i]) == 0]
        
        # Build children list
        children = [[] for _ in range(n)]
        for i in range(n):
            for p in parents[i]:
                children[p].append(i)
        
        # If outputs is None, assume all vertices with no children are outputs
        if outputs is None:
            outputs = [i for i in range(n) if len(children[i]) == 0]
        
        # Topological sort (Kahn's algorithm)
        in_degree = [0] * n
        for i in range(n):
            for p in parents[i]:
                in_degree[i] += 1
        
        queue = deque([i for i in range(n) if in_degree[i] == 0])
        topo_order = []
        while queue:
            v = queue.popleft()
            topo_order.append(v)
            for c in children[v]:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
        
        if len(topo_order) != n:
            topo_order = list(range(n))
        
        # Map vertex -> position in topo order
        pos = {v: i for i, v in enumerate(topo_order)}
        
        # Compute last_use: for each vertex, the max position of its children
        last_use = [-1] * n
        for v in range(n):
            for c in children[v]:
                if pos[c] > last_use[v]:
                    last_use[v] = pos[c]
        
        is_input = [False] * n
        for i in inputs:
            is_input[i] = True
        
        is_output = [False] * n
        for o in outputs:
            is_output[o] = True
        
        # Greedy pebbling
        in_memory = set()
        ops = []
        transfers = 0
        
        def load(v):
            nonlocal transfers
            ops.append(['load', v])
            in_memory.add(v)
            transfers += 1
        
        def store(v):
            nonlocal transfers
            ops.append(['store', v])
            in_memory.discard(v)
            transfers += 1
        
        def compute(v):
            ops.append(['compute', v])
            in_memory.add(v)
        
        def delete(v):
            ops.append(['delete', v])
            in_memory.discard(v)
        
        def recompute(v):
            # Ensure all parents of v are in memory, then compute v
            for p in parents[v]:
                if p not in in_memory:
                    if is_input[p]:
                        load(p)
                    else:
                        recompute(p)
            compute(v)
        
        for idx, v in enumerate(topo_order):
            if time.time() - start_time > DEADLINE or len(ops) >= MAX_OPS:
                break
            
            if is_input[v]:
                if v not in in_memory:
                    load(v)
            else:
                # Ensure all parents are in memory
                for p in parents[v]:
                    if p not in in_memory:
                        if is_input[p]:
                            load(p)
                        else:
                            recompute(p)
                compute(v)
            
            if is_output[v]:
                store(v)
            
            # Enforce memory budget
            while len(in_memory) > fast_memory_budget:
                if time.time() - start_time > DEADLINE or len(ops) >= MAX_OPS:
                    break
                
                # Find a vertex to delete
                candidates = [u for u in in_memory if last_use[u] < idx]
                if not candidates:
                    def remaining_uses(u):
                        return sum(1 for c in children[u] if pos[c] > idx)
                    candidates = sorted(in_memory, key=lambda u: (is_input[u], remaining_uses(u)))
                
                if candidates:
                    u = candidates[0]
                    delete(u)
                else:
                    break
        
        # Ensure all outputs are stored
        for o in outputs:
            if o in in_memory:
                store(o)
        
        return ops
    finally:
        sys.setrecursionlimit(old_limit)
