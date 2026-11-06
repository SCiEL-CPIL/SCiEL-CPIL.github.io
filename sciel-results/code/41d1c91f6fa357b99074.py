# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    Minimizes transfers (load + store) by keeping values in fast memory
    as long as they are needed by future computations.
    """
    t0 = time.time()
    
    # Parse DAG structure
    vertices = []
    parents_dict = {}
    inputs = set()
    outputs = set()
    
    if isinstance(dag, dict):
        vertices = list(dag.get('vertices', dag.get('nodes', [])))
        raw_parents = dag.get('parents', dag.get('edges', {}))
        if isinstance(raw_parents, dict):
            parents_dict = {k: list(v) for k, v in raw_parents.items()}
        elif isinstance(raw_parents, list):
            for item in raw_parents:
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    parents_dict[item[0]] = list(item[1])
        inputs = set(dag.get('inputs', dag.get('input_vertices', [])))
        outputs = set(dag.get('outputs', dag.get('output_vertices', [])))
    elif isinstance(dag, (list, tuple)):
        for item in dag:
            if isinstance(item, dict):
                v = item.get('vertex', item.get('id', item.get('name')))
                ps = list(item.get('parents', []))
                vertices.append(v)
                parents_dict[v] = ps
                if item.get('is_input'):
                    inputs.add(v)
                if item.get('is_output'):
                    outputs.add(v)
                elif not ps:
                    inputs.add(v)
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                v, ps = item[0], list(item[1])
                vertices.append(v)
                parents_dict[v] = ps
                if not ps:
                    inputs.add(v)
    
    if not vertices:
        # Try baseline as fallback
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []
    
    # Build children mapping
    children = defaultdict(list)
    for v in vertices:
        for p in parents_dict.get(v, []):
            children[p].append(v)
    
    # Default inputs/outputs if not specified
    if not inputs:
        for v in vertices:
            if not parents_dict.get(v, []):
                inputs.add(v)
    if not outputs:
        for v in vertices:
            if not children.get(v, []):
                outputs.add(v)
    
    # Topological sort
    in_deg = {v: len(parents_dict.get(v, [])) for v in vertices}
    topo = []
    q = deque(v for v in vertices if in_deg[v] == 0)
    while q:
        v = q.popleft()
        topo.append(v)
        for c in children.get(v, []):
            in_deg[c] -= 1
            if in_deg[c] == 0:
                q.append(c)
    
    if len(topo) != len(vertices):
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []
    
    # Track remaining children for each vertex (for need analysis)
    remaining_children = {v: list(children.get(v, [])) for v in vertices}
    
    fast_mem = set()
    slow_mem = set()
    ops = []
    
    def is_still_needed(v):
        """Check if v has any unprocessed children."""
        return len(remaining_children[v]) > 0
    
    for v in topo:
        if time.time() - t0 > 85:
            break
        
        ps = parents_dict.get(v, [])
        
        # Ensure all parents are in fast memory
        for p in ps:
            if p not in fast_mem:
                if p in slow_mem:
                    if len(fast_mem) >= fast_memory_budget:
                        # Evict a vertex that is no longer needed
                        evicted = False
                        for fm_v in list(fast_mem):
                            if not is_still_needed(fm_v) and fm_v not in outputs:
                                fast_mem.discard(fm_v)
                                ops.append(['delete', fm_v])
                                evicted = True
                                break
                        if not evicted:
                            for fm_v in list(fast_mem):
                                if fm_v not in outputs:
                                    fast_mem.discard(fm_v)
                                    ops.append(['delete', fm_v])
                                    break
                    if len(fast_mem) < fast_memory_budget:
                        fast_mem.add(p)
                        ops.append(['load', p])
        
        # Compute v
        if v not in fast_mem:
            if len(fast_mem) >= fast_memory_budget:
                evicted = False
                for fm_v in list(fast_mem):
                    if not is_still_needed(fm_v) and fm_v not in outputs:
                        fast_mem.discard(fm_v)
                        ops.append(['delete', fm_v])
                        evicted = True
                        break
                if not evicted:
                    for fm_v in list(fast_mem):
                        if fm_v not in outputs:
                            fast_mem.discard(fm_v)
                            ops.append(['delete', fm_v])
                            break
            if len(fast_mem) < fast_memory_budget:
                fast_mem.add(v)
                ops.append(['compute', v])
        
        # Update remaining children for parents
        for p in ps:
            if v in remaining_children[p]:
                remaining_children[p].remove(v)
        
        # Decide fate of v in fast memory
        if v in fast_mem:
            if v in outputs:
                pass  # Must remain in fast memory
            else:
                if not is_still_needed(v):
                    # No more children to process, delete
                    fast_mem.discard(v)
                    ops.append(['delete', v])
                else:
                    # Still needed, but check if we should store
                    if len(fast_mem) >= fast_memory_budget:
                        slow_mem.add(v)
                        fast_mem.discard(v)
                        ops.append(['store', v])
    
    # Ensure all outputs are in fast memory
    for v in outputs:
        if v not in fast_mem:
            if v in slow_mem:
                if len(fast_mem) >= fast_memory_budget:
                    for fm_v in list(fast_mem):
                        if fm_v not in outputs:
                            fast_mem.discard(fm_v)
                            ops.append(['delete', fm_v])
                            break
                if len(fast_mem) < fast_memory_budget:
                    fast_mem.add(v)
                    ops.append(['load', v])
    
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
# EVOLVE-BLOCK-END
