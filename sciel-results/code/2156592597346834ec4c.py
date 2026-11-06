"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling using recursive layer-based strategy.
    """
    start_time = time.time()
    DEADLINE = 85.0
    
    def time_check():
        return (time.time() - start_time) > DEADLINE
    
    vertices = dag['vertices']
    parents_map = dag['parents']
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    n = len(vertices)
    if n == 0:
        return []
    
    # Build children map
    children = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children[p].append(v)
    
    # Compute layer (longest path from any source)
    layer = {}
    for v in vertices:
        ps = parents_map.get(v, [])
        if not ps or v in inputs:
            layer[v] = 0
        else:
            layer[v] = max(layer[p] for p in ps) + 1
    
    max_layer = max(layer.values()) if layer else 0
    
    # Group vertices by layer
    layers = defaultdict(list)
    for v in vertices:
        layers[layer[v]].append(v)
    
    # For each vertex, compute next_use: the earliest index in topo_order
    # where this vertex is needed as a parent
    next_use = {}
    for i, v in enumerate(vertices):
        for p in parents_map.get(v, []):
            if p not in next_use or i < next_use[p]:
                next_use[p] = i
    
    # Recursive pebbling strategy:
    # Process layers in order. For each layer, we need all parents from previous layers.
    # Use a priority queue to evict vertices with farthest next use when memory is full.
    
    ops = []
    in_memory = set()
    
    # We'll use a heap: (next_use_value, vertex) - evict farthest first
    # But we need to be able to mark vertices as "do not evict" temporarily
    
    # Simple greedy: process in topological order, evict farthest-next-use when full
    
    for i, v in enumerate(vertices):
        if time_check():
            break
        
        # Ensure all parents are in memory
        for p in parents_map.get(v, []):
            if p not in in_memory:
                # Make room if needed
                while len(in_memory) >= fast_memory_budget:
                    # Find evictable vertex with farthest next use
                    best_evict = None
                    best_nu = -1
                    for m in in_memory:
                        # Don't evict if needed for current vertex
                        if m in parents_map.get(v, []):
                            continue
                        nu = next_use.get(m, n)
                        if nu > best_nu:
                            best_nu = nu
                            best_evict = m
                    
                    if best_evict is None:
                        # Can't evict, break
                        break
                    
                    ops.append(['store', best_evict])
                    in_memory.remove(best_evict)
                
                if p not in in_memory:
                    ops.append(['load', p])
                    in_memory.add(p)
        
        # Compute v
        if v in in_memory:
            ops.append(['delete', v])
            in_memory.remove(v)
        
        # Make room for v
        while len(in_memory) >= fast_memory_budget and v not in in_memory:
            best_evict = None
            best_nu = -1
            for m in in_memory:
                if m in parents_map.get(v, []):
                    continue
                nu = next_use.get(m, n)
                if nu > best_nu:
                    best_nu = nu
                    best_evict = m
            
            if best_evict is None:
                break
            
            ops.append(['store', best_evict])
            in_memory.remove(best_evict)
        
        ops.append(['compute', v])
        in_memory.add(v)
    
    # Store all outputs
    for v in vertices:
        if v in outputs and v in in_memory:
            if time_check():
                break
            ops.append(['store', v])
            in_memory.remove(v)
    
    # Cleanup remaining
    for v in list(in_memory):
        if time_check():
            break
        ops.append(['delete', v])
        in_memory.remove(v)
    
    return ops
# EVOLVE-BLOCK-END
