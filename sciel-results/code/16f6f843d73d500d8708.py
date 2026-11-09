# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Uses a smart topological-order schedule that:
    1. Prioritizes high-fanout vertices for early computation
    2. Keeps frequently-needed values in fast memory
    3. Minimizes load/store operations
    """
    deadline = time.time() + 85.0
    
    # Try to understand DAG structure
    try:
        if isinstance(dag, dict):
            parents = dag.get('parents', {})
            outputs = dag.get('outputs', [])
            if not parents and not outputs:
                # Maybe different keys
                parents = dag.get('predecessors', {})
                outputs = dag.get('sinks', []) or dag.get('outputs', [])
        elif hasattr(dag, 'parents'):
            parents = dag.parents
            outputs = getattr(dag, 'outputs', [])
        else:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
    except Exception:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Build structure
    all_vertices = set(parents.keys())
    for v in parents:
        for p in parents[v]:
            all_vertices.add(p)
    all_vertices = sorted(all_vertices, key=lambda x: (isinstance(x, tuple), x))
    
    children = defaultdict(list)
    for v, par_list in parents.items():
        for p in par_list:
            children[p].append(v)
    
    n = len(all_vertices)
    if n == 0:
        return []
    
    # Compute out-degree (number of children)
    out_degree = {v: len(children.get(v, [])) for v in all_vertices}
    
    # Topological sort with priority on high-fanout vertices
    in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
    from heapq import heappush, heappop
    
    ready = []
    for v in all_vertices:
        if in_degree[v] == 0:
            heappush(ready, (-out_degree[v], str(v), v))
    
    topo_order = []
    processed = set()
    while ready:
        neg_deg, _, v = heappop(ready)
        if v in processed:
            continue
        processed.add(v)
        topo_order.append(v)
        for child in children.get(v, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                heappush(ready, (-out_degree[child], str(child), child))
    
    # Add any remaining vertices (shouldn't happen in a DAG)
    for v in all_vertices:
        if v not in processed:
            topo_order.append(v)
    
    # Generate pebbling schedule
    fast_mem = set()
    slow_mem = set()
    computed = set()
    ops = []
    
    # Input vertices start in slow memory
    for v in all_vertices:
        if len(parents.get(v, [])) == 0:
            slow_mem.add(v)
            computed.add(v)
    
    # Track which vertices still need their children processed
    remaining_children_count = {v: len(children.get(v, [])) for v in all_vertices}
    
    for v in topo_order:
        if time.time() > deadline:
            break
        if len(ops) >= 20000:
            break
        
        if v in computed:
            continue
        
        # Ensure all parents are in fast memory
        par_list = parents.get(v, [])
        for p in par_list:
            if p not in fast_mem:
                if p in slow_mem:
                    ops.append(['load', p])
                    fast_mem.add(p)
                    slow_mem.discard(p)
                elif p in computed:
                    ops.append(['load', p])
                    fast_mem.add(p)
        
        # Compute v
        ops.append(['compute', v])
        fast_mem.add(v)
        computed.add(v)
        
        # Update remaining children count for parents
        # When we compute v, one child of each parent is done
        for p in par_list:
            remaining_children_count[p] -= 1
        
        # Decide whether to keep v in fast memory
        # Keep if: output, or has remaining children, or fast memory has room
        is_output = v in outputs
        has_remaining_children = remaining_children_count.get(v, 0) > 0
        
        if is_output or has_remaining_children or len(fast_mem) <= fast_memory_budget:
            pass  # Keep in fast memory
        else:
            ops.append(['store', v])
            fast_mem.discard(v)
            slow_mem.add(v)
    
    # Ensure outputs are in fast memory
    for v in outputs:
        if v not in fast_mem and len(ops) < 20000:
            if v in slow_mem:
                ops.append(['load', v])
                fast_mem.add(v)
                slow_mem.discard(v)
            elif v in computed:
                ops.append(['load', v])
                fast_mem.add(v)
    
    # If fast memory is over budget, store some non-output values
    if len(fast_mem) > fast_memory_budget:
        # Prioritize keeping outputs and high-fanout values
        priority = {v: (0 if v in outputs else 1, -out_degree.get(v, 0)) for v in fast_mem}
        sorted_fast = sorted(fast_mem, key=lambda v: priority[v])
        to_store = sorted_fast[fast_memory_budget:]
        for v in to_store:
            if len(ops) < 20000:
                ops.append(['store', v])
                fast_mem.discard(v)
                slow_mem.add(v)
    
    return ops

# EVOLVE-BLOCK-END
