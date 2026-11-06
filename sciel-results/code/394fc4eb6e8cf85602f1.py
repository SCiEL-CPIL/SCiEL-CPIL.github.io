"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = 85.0
    
    vertices = dag.get('vertices', [])
    parents = dag.get('parents', {})
    outputs = set(dag.get('outputs', []))
    
    if not vertices:
        return []
    
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    inputs = [v for v in vertices if not parents.get(v, [])]
    
    # Determine layers (longest path from any input)
    layer = {}
    for v in vertices:
        if not parents.get(v, []):
            layer[v] = 0
        else:
            layer[v] = max(layer.get(p, -1) for p in parents[v]) + 1
    
    max_layer = max(layer.values()) if layer else 0
    layers = defaultdict(list)
    for v in vertices:
        layers[layer[v]].append(v)
    
    operations = []
    
    def time_check():
        return (time.time() - start_time) > deadline
    
    # Process layer by layer
    # Keep current layer in fast memory
    fast_memory = set()
    
    for li in range(max_layer + 1):
        if time_check():
            break
        
        current_layer = layers[li]
        
        # Ensure all vertices in current layer are in fast memory
        for v in current_layer:
            if v not in fast_memory:
                if li == 0:
                    # Input: load from slow memory
                    operations.append(['load', v])
                else:
                    # Should have been computed in previous iteration
                    # If not, recompute (simplified: just compute)
                    operations.append(['compute', v])
                fast_memory.add(v)
        
        # If this is the last layer, store all outputs
        if li == max_layer:
            for v in current_layer:
                if v in outputs:
                    operations.append(['store', v])
                    fast_memory.discard(v)
            break
        
        # Compute next layer
        next_layer = layers[li + 1]
        for v in next_layer:
            if time_check():
                break
            # All parents should be in fast memory (they're in current layer)
            operations.append(['compute', v])
            fast_memory.add(v)
        
        # Discard current layer (non-output vertices)
        for v in current_layer:
            if v not in outputs and v not in fast_memory:
                pass  # Already not in fast memory
            elif v not in outputs:
                operations.append(['delete', v])
                fast_memory.discard(v)
    
    # Store any remaining outputs
    for v in outputs:
        if v in fast_memory:
            operations.append(['store', v])
            fast_memory.discard(v)
        else:
            # Fallback: recompute and store
            operations.append(['compute', v])
            operations.append(['store', v])
    
    return operations

# EVOLVE-BLOCK-END
