"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Args:
        dag: dict with keys 'parents' (list of lists, topologically ordered) and 'outputs' (list of vertex indices)
        fast_memory_budget: int, number of red values that can be held simultaneously
    
    Returns:
        list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'
    """
    start_time = time.time()
    DEADLINE = 85.0  # 5s safety margin before 90s kill
    
    # Parse DAG structure
    parents = dag.get('parents', [])
    outputs = set(dag.get('outputs', []))
    n = len(parents)
    
    if n == 0:
        return []
    
    # Build adjacency (children) and in-degree
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    # Identify input vertices (no parents) and output vertices
    is_input = [len(parents[v]) == 0 for v in range(n)]
    
    # Topological order is already given by vertex index (0..n-1)
    
    # Strategy: Use a pebbling algorithm that minimizes transfers.
    # Key insight: We want to minimize the number of load/store operations.
    # A good strategy is to process the DAG in a way that keeps values in fast memory
    # as long as possible, and only store to slow memory when fast memory is full.
    
    # First, let's try a simple but effective strategy:
    # 1. Load all inputs that are needed
    # 2. Compute vertices in topological order
    # 3. Keep values in fast memory when possible
    # 4. Store to slow memory when fast memory is full
    
    # Let's implement a pebbling strategy based on the "black-red pebbling" approach
    # where we try to minimize the number of transfers between slow and fast memory.
    
    # A simple effective strategy:
    # - Process vertices in topological order
    # - For each vertex, ensure its parents' values are available (in fast memory or loaded)
    # - Compute the vertex
    # - Decide whether to keep the result in fast memory or store it
    
    # Let's use a strategy based on the "pebbling number" concept:
    # We'll try to keep as many values in fast memory as possible.
    
    # First, let's determine which vertices need to be kept in fast memory.
    # A vertex needs to be kept if it has multiple children that haven't been processed yet.
    
    # Let's use a greedy strategy:
    # 1. Process in topological order
    # 2. When computing a vertex, load its parents if not already in fast memory
    # 3. After computing, if fast memory is full, store the least-needed value
    
    # Let's track:
    # - fast_memory: set of vertex indices currently in fast memory (red pebbles)
    # - slow_memory: set of vertex indices currently in slow memory (blue pebbles)
    # - operations: list of operations performed
    
    operations = []
    fast_memory = set()
    slow_memory = set()  # Values in slow memory (initially all inputs are here)
    
    # Initially, all input vertices are in slow memory (blue)
    for v in range(n):
        if is_input[v]:
            slow_memory.add(v)
    
    # Count how many times each vertex is needed (number of children)
    # This helps decide which values to keep in fast memory
    remaining_children = [len(children[v]) for v in range(n)]
    
    # Process vertices in topological order
    for v in range(n):
        if time.time() > DEADLINE:
            break
        
        # If this vertex is an input, it's already in slow memory
        if is_input[v]:
            # Load it into fast memory
            if v not in fast_memory:
                operations.append(['load', v])
                fast_memory.add(v)
                slow_memory.discard(v)
            continue
        
        # For non-input vertices, we need to compute them
        # First, ensure all parents are in fast memory
        for p in parents[v]:
            if p not in fast_memory:
                # Load parent from slow memory
                if p in slow_memory:
                    operations.append(['load', p])
                    fast_memory.add(p)
                    slow_memory.discard(p)
                else:
                    # Parent was deleted, need to recompute or reload
                    # For now, assume it's in slow memory (shouldn't happen with proper strategy)
                    operations.append(['load', p])
                    fast_memory.add(p)
        
        # Compute the vertex
        operations.append(['compute', v])
        fast_memory.add(v)
        
        # Now, manage fast memory: if over budget, evict a value
        while len(fast_memory) > fast_memory_budget:
            # Find the vertex to evict: the one with fewest remaining children
            # that is not an output and not needed soon
            best_to_evict = None
            best_score = float('inf')
            for u in fast_memory:
                # Don't evict if it's an output that still needs to be stored
                # Score: prefer evicting vertices with fewer remaining uses
                score = remaining_children[u]
                # Penalize evicting outputs
                if u in outputs:
                    score += 1000
                if score < best_score:
                    best_score = score
                    best_to_evict = u
            
            if best_to_evict is not None:
                # Store to slow memory
                operations.append(['store', best_to_evict])
                fast_memory.discard(best_to_evict)
                slow_memory.add(best_to_evict)
            else:
                break
    
    # After processing all vertices, ensure all outputs are in fast memory (blue)
    # Actually, the requirement is that all outputs must finish blue.
    # Blue means in slow memory. So we need to store all outputs.
    
    # Wait, let me re-read: "Inputs start blue, all outputs must finish blue"
    # Blue = slow memory, Red = fast memory
    # So outputs need to be in slow memory at the end.
    
    # Let's store any outputs that are in fast memory
    for v in list(fast_memory):
        if v in outputs:
            operations.append(['store', v])
            fast_memory.discard(v)
            slow_memory.add(v)
    
    # Also, any non-output values in fast memory can be deleted
    for v in list(fast_memory):
        if v not in outputs:
            operations.append(['delete', v])
            fast_memory.discard(v)
    
    # Verify: all outputs should be in slow_memory
    # (This is a sanity check, not part of the output)
    
    # Trim operations to at most 20000
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
