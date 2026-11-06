"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Try to extract DAG structure - handle multiple possible formats
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        parents_map = dag.get('parents', {})
        outputs = dag.get('outputs', [])
    else:
        # Fallback: assume dag is a list of vertices
        vertices = list(dag)
        parents_map = {}
        outputs = []
    
    if not vertices:
        return []
    
    n = len(vertices)
    vidx = {v: i for i, v in enumerate(vertices)}
    
    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents_map.get(v, []):
            if p in vidx:
                children[p].append(v)
    
    inputs = set(v for v in vertices if not parents_map.get(v, []))
    outputs_set = set(outputs)
    
    # Count remaining uses for each vertex
    remaining_uses = {v: 0 for v in vertices}
    for v in vertices:
        for p in parents_map.get(v, []):
            remaining_uses[p] += 1
    
    fast_mem = set()
    operations = []
    
    # Inputs start blue (in fast memory)
    for v in inputs:
        fast_mem.add(v)
    
    for v in vertices:
        if time.time() > deadline:
            break
        
        # Load parents if not in fast memory
        for p in parents_map.get(v, []):
            if p not in fast_mem:
                # Make space if needed
                while len(fast_mem) >= fast_memory_budget and p not in fast_mem:
                    # Find best vertex to evict
                    best_evict = None
                    best_priority = float('inf')
                    for cand in fast_mem:
                        if cand == v:
                            continue
                        priority = remaining_uses.get(cand, 0)
                        if cand in outputs_set:
                            priority += 10000  # strongly prefer keeping outputs
                        if priority < best_priority:
                            best_priority = priority
                            best_evict = cand
                    if best_evict is not None:
                        operations.append(['delete', best_evict])
                        fast_mem.discard(best_evict)
                    else:
                        break
                
                if p not in fast_mem:
                    operations.append(['load', p])
                    fast_mem.add(p)
        
        # Compute v
        operations.append(['compute', v])
        fast_mem.add(v)
        
        # Update remaining uses for parents
        for p in parents_map.get(v, []):
            remaining_uses[p] = max(0, remaining_uses[p] - 1)
        
        # Proactively delete vertices with no remaining uses to free space
        if len(fast_mem) > fast_memory_budget:
            while len(fast_mem) > fast_memory_budget:
                best_evict = None
                best_priority = float('inf')
                for cand in list(fast_mem):
                    if cand == v:
                        continue
                    priority = remaining_uses.get(cand, 0)
                    if cand in outputs_set:
                        priority += 10000
                    if priority < best_priority:
                        best_priority = priority
                        best_evict = cand
                if best_evict is not None:
                    operations.append(['delete', best_evict])
                    fast_mem.discard(best_evict)
                else:
                    break
    
    # Ensure outputs are in fast memory at the end
    for v in outputs_set:
        if v not in fast_mem:
            while len(fast_mem) >= fast_memory_budget and v not in fast_mem:
                best_evict = None
                best_priority = float('inf')
                for cand in fast_mem:
                    if cand in outputs_set:
                        continue
                    priority = remaining_uses.get(cand, 0)
                    if priority < best_priority:
                        best_priority = priority
                        best_evict = cand
                if best_evict is not None:
                    operations.append(['delete', best_evict])
                    fast_mem.discard(best_evict)
                else:
                    break
            if v not in fast_mem:
                operations.append(['load', v])
                fast_mem.add(v)
    
    return operations
# EVOLVE-BLOCK-END

