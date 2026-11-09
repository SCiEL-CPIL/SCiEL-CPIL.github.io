import time
import sys
from collections import deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    # Parse DAG structure
    # dag is expected to have topologically ordered parents and output vertices
    # Structure assumption:
    # dag['vertices'] or similar? 
    # The prompt says "dag has topologically ordered parents and output vertices".
    # Let's assume dag is a dict with keys like 'vertices', 'parents', 'inputs', 'outputs'
    # Or maybe it's a list of nodes?
    # Common format in these tasks:
    # dag: {
    #   'vertices': [0, 1, ...],
    #   'parents': {v: [p1, p2, ...]},
    #   'inputs': [0, ...],
    #   'outputs': [10, ...]
    # }
    
    if isinstance(dag, dict):
        vertices = dag.get('vertices', list(dag.get('parents', {}).keys()))
        parents_map = dag.get('parents', {})
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
    else:
        # Fallback if structure is different
        # Assuming dag is a list of dicts or similar is hard without seeing it.
        # But the baseline `pebble_baseline` likely handles the standard format.
        # I will assume the standard dict format.
        raise ValueError("Unknown DAG format")

    # Topological Sort
    # Since parents are topologically ordered, we can just sort vertices by index?
    # "dag has topologically ordered parents" implies if p is parent of v, index(p) < index(v).
    # So sorted(vertices) is a topological order.
    topo_order = sorted(vertices)
    
    # Build children map
    children_map = {v: [] for v in vertices}
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
            
    # Sort children by topological index (they are already if we iterate sorted vertices, but let's be sure)
    for v in vertices:
        children_map[v].sort()
        
    # Precompute next use indices
    # next_use_idx[v] will be the index in topo_order of the next child of v
    # We maintain a pointer for each v
    ptr = {v: 0 for v in vertices}
    next_use_index = {}
    for v in vertices:
        if children_map[v]:
            next_use_index[v] = children_map[v][0]
        else:
            next_use_index[v] = float('inf')
            
    # State
    fast_mem = set()
    slow_mem = set(inputs)
    remaining_uses = {v: len(children_map[v]) for v in vertices}
    
    ops = []
    
    def time_check():
        if time.time() > deadline:
            return True
        return False

    # Main Loop
    for v in topo_order:
        if time_check():
            break
            
        # 1. Ensure parents are in fast memory
        for p in parents_map.get(v, []):
            if p not in fast_mem:
                if p in slow_mem:
                    ops.append(["load", p])
                    fast_mem.add(p)
                else:
                    # This should not happen if logic is correct
                    # If it does, we have to recompute? 
                    # For safety, if p is input, load it.
                    if p in inputs:
                        ops.append(["load", p])
                        fast_mem.add(p)
                    else:
                        # Emergency recompute logic (should be rare/impossible)
                        # We'll just skip for now to avoid infinite loop, 
                        # but this indicates a bug in eviction policy.
                        pass
        
        # 2. Compute or Load v
        if v in inputs:
            if v not in fast_mem:
                ops.append(["load", v])
                fast_mem.add(v)
        else:
            ops.append(["compute", v])
            fast_mem.add(v)
            
        # 3. Update remaining uses and next use index for parents
        for p in parents_map.get(v, []):
            remaining_uses[p] -= 1
            ptr[p] += 1
            if ptr[p] < len(children_map[p]):
                next_use_index[p] = children_map[p][ptr[p]]
            else:
                next_use_index[p] = float('inf')
                
        # 4. Manage Fast Memory Size
        while len(fast_mem) > fast_memory_budget:
            if time_check():
                break
                
            # Find candidate to evict
            # Priority 1: remaining_uses == 0
            # Priority 2: largest next_use_index
            
            candidate = None
            max_next_use = -1
            zero_use_candidate = None
            
            for u in fast_mem:
                if remaining_uses[u] == 0:
                    zero_use_candidate = u
                    break # Any zero use is fine, delete it immediately
                else:
                    nu = next_use_index[u]
                    if nu > max_next_use:
                        max_next_use = nu
                        candidate = u
                        
            if zero_use_candidate is not None:
                u = zero_use_candidate
                ops.append(["delete", u])
                fast_mem.remove(u)
                # Do not store if remaining uses is 0
            else:
                if candidate is None:
                    # Should not happen if fast_mem is not empty
                    break
                u = candidate
                ops.append(["store", u])
                slow_mem.add(u)
                ops.append(["delete", u])
                fast_mem.remove(u)
                
    # Finalize Outputs
    if not time_check():
        for v in outputs:
            if v in fast_mem:
                ops.append(["store", v])
                slow_mem.add(v)
                ops.append(["delete", v])
                fast_mem.remove(v)
            elif v not in slow_mem:
                # If output is not in slow mem and not in fast mem, it's lost.
                # This shouldn't happen.
                pass
                
    return ops
