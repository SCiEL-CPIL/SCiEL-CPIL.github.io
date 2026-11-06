"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    """
    start_time = time.time()
    DEADLINE = 85.0
    SAFETY_MARGIN = 5.0

    try:
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])
        output_set = set(outputs)
        input_set = set(inputs)

        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)

        all_vertices = set(parents.keys()) | set(inputs) | set(outputs)

        in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
        for v in inputs:
            in_degree[v] = 0

        queue = list(all_vertices)
        queue.sort(key=lambda x: str(x))
        topo_order = []
        while queue:
            next_v = None
            next_idx = -1
            for i, v in enumerate(queue):
                if in_degree[v] == 0:
                    next_v = v
                    next_idx = i
                    break
            if next_v is None:
                break
            queue.pop(next_idx)
            topo_order.append(next_v)
            for c in children.get(next_v, []):
                in_degree[c] -= 1

        topo_index = {v: i for i, v in enumerate(topo_order)}
        n = len(topo_order)

        last_use = {}
        for v in all_vertices:
            max_idx = -1
            for c in children.get(v, []):
                idx = topo_index.get(c, n)
                if idx > max_idx:
                    max_idx = idx
            if v in output_set:
                max_idx = n
            last_use[v] = max_idx

        ops = []
        fast_mem = set()
        slow_blue = set(inputs)
        
        remaining_uses = {}
        for v in all_vertices:
            if v in output_set:
                remaining_uses[v] = n
            else:
                remaining_uses[v] = len(children.get(v, []))
        
        for v in topo_order:
            if time.time() > DEADLINE - SAFETY_MARGIN:
                break
            
            ps = parents.get(v, [])
            
            if v in input_set:
                if v not in fast_mem:
                    if v in slow_blue:
                        ops.append(['load', v])
                        fast_mem.add(v)
                        slow_blue.discard(v)
            else:
                for p in ps:
                    if p not in fast_mem:
                        if p in slow_blue:
                            ops.append(['load', p])
                            fast_mem.add(p)
                            slow_blue.discard(p)
                
                ops.append(['compute', v])
                fast_mem.add(v)
                
                for p in ps:
                    remaining_uses[p] -= 1
                    if remaining_uses[p] == 0 and p not in output_set:
                        ops.append(['delete', p])
                        fast_mem.discard(p)
        
        for o in outputs:
            if o not in fast_mem:
                if o in slow_blue:
                    ops.append(['load', o])
                    fast_mem.add(o)
                    slow_blue.discard(o)
        
        if len(ops) <= 20000:
            return ops
        return ops[:20000]
            
    except Exception as e:
        return []
# EVOLVE-BLOCK-END

