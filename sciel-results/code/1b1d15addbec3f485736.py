"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    dag structure expected:
      - dag['parents']: dict vertex -> list of parent vertices (topologically ordered)
      - dag['outputs']: list of output vertices
      - dag['inputs']: list of input vertices (start blue)
    Returns list of [operation, vertex] where operation in {load, store, compute, delete}.
    """
    start_time = time.time()
    DEADLINE = 85.0
    SAFETY_MARGIN = 3.0
    
    try:
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])
        
        if not parents and not inputs and not outputs:
            return []
        
        # Build children map
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
        
        all_vertices = set(parents.keys()) | set(inputs) | set(outputs)
        
        # Compute level of each vertex (longest path from any input)
        # Use topological order
        in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
        for v in inputs:
            in_degree[v] = 0
        
        # Topological sort using Kahn's algorithm
        queue = deque()
        for v in all_vertices:
            if in_degree.get(v, 0) == 0:
                queue.append(v)
        
        topo_order = []
        level = {}
        for v in inputs:
            level[v] = 0
        
        while queue:
            u = queue.popleft()
            topo_order.append(u)
            if u not in level:
                level[u] = 0
            for c in children.get(u, []):
                # Update level of child
                new_level = level.get(u, 0) + 1
                if c not in level or level[c] < new_level:
                    level[c] = new_level
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
        
        # Group vertices by level
        levels = defaultdict(list)
        for v in topo_order:
            lv = level.get(v, 0)
            levels[lv].append(v)
        
        max_level = max(levels.keys()) if levels else 0
        
        # State
        fast_mem = set()
        stored_in_slow = set()
        ops = []
        
        # Process level by level
        for lv in range(max_level + 1):
            if time.time() > DEADLINE - SAFETY_MARGIN:
                break
            
            vertices_at_level = levels.get(lv, [])
            
            # For each vertex at this level, ensure its parents are in fast memory
            for v in vertices_at_level:
                ps = parents.get(v, [])
                for p in ps:
                    if p not in fast_mem:
                        if p in stored_in_slow:
                            ops.append(['load', p])
                            fast_mem.add(p)
                            stored_in_slow.discard(p)
                        elif p in inputs:
                            ops.append(['load', p])
                            fast_mem.add(p)
                        else:
                            # Should not happen if we stored properly
                            ops.append(['load', p])
                            fast_mem.add(p)
                
                # Compute v
                if v not in fast_mem:
                    ops.append(['compute', v])
                    fast_mem.add(v)
            
            # After computing all vertices at this level, evict vertices that are no longer needed
            # A vertex is no longer needed if all its children have been computed
            # But we need to be careful: we should only evict if we're over budget
            
            if len(fast_mem) > fast_memory_budget:
                # Find vertices to evict
                # Priority: evict vertices with no remaining children first (delete)
                # Then evict vertices with fewest remaining children (store)
                
                # Count remaining children for each vertex in fast_mem
                remaining_children = {}
                for v in fast_mem:
                    rem = 0
                    for c in children.get(v, []):
                        # If child hasn't been computed yet, it still needs v
                        if c not in fast_mem and c not in stored_in_slow:
                            rem += 1
                        # If child is in fast_mem, it was just computed, so v is no longer needed by c
                        # But we need to check if c has other parents... actually no, once c is computed, v is done for c
                    remaining_children[v] = rem
                
                # Sort by remaining_children, then by level (lower level first, as they're older)
                evict_candidates = sorted(fast_mem, key=lambda v: (remaining_children.get(v, 0), level.get(v, 0)))
                
                evicted = 0
                for v in evict_candidates:
                    if len(fast_mem) <= fast_memory_budget:
                        break
                    if remaining_children.get(v, 0) == 0:
                        ops.append(['delete', v])
                        fast_mem.discard(v)
                    else:
                        ops.append(['store', v])
                        ops.append(['delete', v])
                        fast_mem.discard(v)
                        stored_in_slow.add(v)
                    evicted += 1
        
        # Ensure all outputs are in fast memory
        for o in outputs:
            if o not in fast_mem:
                if o in stored_in_slow:
                    ops.append(['load', o])
                    fast_mem.add(o)
                    stored_in_slow.discard(o)
                else:
                    ops.append(['load', o])
                    fast_mem.add(o)
        
        # Check operation count
        if len(ops) <= 20000:
            return ops
        else:
            # If too many ops, try to simplify
            # Remove redundant loads/stores
            simplified = []
            in_fast = set()
            in_slow = set()
            for op, v in ops:
                if op == 'load':
                    if v not in in_fast:
                        simplified.append([op, v])
                        in_fast.add(v)
                        in_slow.discard(v)
                elif op == 'store':
                    if v in in_fast and v not in in_slow:
                        simplified.append([op, v])
                        in_slow.add(v)
                elif op == 'delete':
                    if v in in_fast:
                        simplified.append([op, v])
                        in_fast.discard(v)
                elif op == 'compute':
                    simplified.append([op, v])
                    in_fast.add(v)
            
            if len(simplified) <= 20000:
                return simplified
            return ops[:20000]
            
    except Exception as e:
        return []
# EVOLVE-BLOCK-END
